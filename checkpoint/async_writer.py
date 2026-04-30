"""
Async checkpoint writer — training doesn't pause while checkpoints are saved.

For FSDP models: uses SHARDED_STATE_DICT + torch.distributed.checkpoint so each
rank writes only its own shard locally — no all-gather across GPUs, no NIC saturation.

For non-FSDP models: original rank-0-only async write path.
"""

import json
import hashlib
import shutil
import threading
from pathlib import Path
from datetime import datetime

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    StateDictType,
)


class AsyncCheckpointWriter:
    def __init__(self, base_dir, s3_bucket=None, s3_prefix=None, max_kept=5):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.max_kept = max_kept
        self._write_thread = None
        self._last_error = None
        self._thread_lock = threading.Lock()

    def save(self, model, optimizer, scheduler, step, metrics=None, rank=0, model_name=None):
        if isinstance(model, FSDP):
            self._save_sharded(model, optimizer, scheduler, step, metrics, rank, model_name)
        else:
            self._save_full(model, optimizer, scheduler, step, metrics, rank, model_name)

    # ------------------------------------------------------------------
    # Sharded save — FSDP only, each rank writes its own shard locally
    # ------------------------------------------------------------------

    def _save_sharded(self, model, optimizer, scheduler, step, metrics, rank, model_name):
        import torch.distributed.checkpoint as dist_cp

        staging_dir = self.base_dir / f".staging-{step}"
        final_dir = self.base_dir / f"checkpoint-{step}"

        # Rank 0 clears any stale staging dir; barrier before any rank creates dirs
        if rank == 0 and staging_dir.exists():
            shutil.rmtree(staging_dir)
        self._barrier()
        staging_dir.mkdir(parents=True, exist_ok=True)

        # 1. Model — sharded save via DCP, purely local I/O on each rank
        model_dir = staging_dir / "model"
        model_dir.mkdir(exist_ok=True)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            model_sd = model.state_dict()
        dist_cp.save({"model": model_sd}, storage_writer=dist_cp.FileSystemWriter(str(model_dir)))

        # 2. Optimizer — each rank saves its own flat-param states directly,
        #    no FSDP wrapper so no collective ops
        optim_dir = staging_dir / f"optim-rank-{rank}"
        optim_dir.mkdir(exist_ok=True)
        torch.save(optimizer.state_dict(), optim_dir / "optim.pt")

        # 3. All ranks barrier — everyone has finished writing their shards
        self._barrier()

        # 4. Rank 0 writes scheduler + metadata and atomically promotes staging → final
        if rank == 0:
            torch.save({
                "scheduler": scheduler.state_dict(),
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics or {},
                "model_name": model_name,
                "world_size": self._world_size(),
                "sharded": True,
            }, staging_dir / "train_state.pt")

            with open(staging_dir / "meta.json", "w") as f:
                json.dump({
                    "step": step,
                    "sharded": True,
                    "world_size": self._world_size(),
                    "timestamp": datetime.now().isoformat(),
                }, f, indent=2)

            if final_dir.exists():
                shutil.rmtree(final_dir)
            staging_dir.rename(final_dir)
            self._cleanup_old_checkpoints()
            if self.s3_bucket:
                self._sync_to_s3(final_dir, step)
            print(f"[checkpoint] Saved sharded checkpoint step {step} → {final_dir}")

        self._barrier()

    def _restore_sharded(self, model, optimizer, scheduler, path):
        import torch.distributed.checkpoint as dist_cp
        rank = self._rank()
        ckpt_path = Path(path)

        # 1. Model: load shards via DCP — each rank reads its own shard
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            model_sd = model.state_dict()  # template with correct shapes
        dist_cp.load({"model": model_sd}, storage_reader=dist_cp.FileSystemReader(str(ckpt_path / "model")))
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            model.load_state_dict(model_sd)

        # 2. Optimizer: each rank loads its own flat-param state
        optim_path = ckpt_path / f"optim-rank-{rank}" / "optim.pt"
        if optim_path.exists():
            optimizer.load_state_dict(
                torch.load(optim_path, map_location="cpu", weights_only=False)
            )
        else:
            print(f"[checkpoint] WARNING: no optimizer shard found for rank {rank} at {optim_path}")

        # 3. Scheduler + step: rank 0's train_state is authoritative
        train_state = torch.load(ckpt_path / "train_state.pt", map_location="cpu", weights_only=False)
        scheduler.load_state_dict(train_state["scheduler"])
        return train_state["step"]

    # ------------------------------------------------------------------
    # Full (non-FSDP) save — original rank-0-only async write
    # ------------------------------------------------------------------

    def _save_full(self, model, optimizer, scheduler, step, metrics, rank, model_name):
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {},
            "fsdp": False,
            "sharded": False,
            "model_name": model_name,
        }

        if rank != 0:
            self._barrier()
            return

        with self._thread_lock:
            if self._write_thread and self._write_thread.is_alive():
                self._write_thread.join()
                self._raise_thread_error()
            self._write_thread = threading.Thread(
                target=self._async_write, args=(state, step), daemon=True
            )
            self._write_thread.start()

        self._barrier()

    def _async_write(self, state, step):
        staging_dir = self.base_dir / f".staging-{step}"
        final_dir = self.base_dir / f"checkpoint-{step}"
        try:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            staging_dir.mkdir(parents=True, exist_ok=True)

            state_path = staging_dir / "state.pt"
            torch.save(state, state_path)

            checksum = self._compute_checksum(state_path)
            meta = {
                "step": step,
                "timestamp": state["timestamp"],
                "checksum": checksum,
                "size_bytes": state_path.stat().st_size,
                "metrics": state.get("metrics", {}),
                "sharded": False,
            }
            with open(staging_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            if not self._validate_full(staging_dir, checksum):
                shutil.rmtree(staging_dir)
                print(f"[checkpoint] CORRUPTED checkpoint at step {step} — discarded")
                return

            if final_dir.exists():
                shutil.rmtree(final_dir)
            staging_dir.rename(final_dir)
            self._cleanup_old_checkpoints()
            if self.s3_bucket:
                self._sync_to_s3(final_dir, step)
        except Exception as exc:
            self._last_error = exc
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            print(f"[checkpoint] write failed for step {step}: {exc}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_full(self, checkpoint_dir, expected_checksum):
        state_path = checkpoint_dir / "state.pt"
        if not state_path.exists():
            return False
        if self._compute_checksum(state_path) != expected_checksum:
            return False
        try:
            torch.load(state_path, map_location="cpu", weights_only=False)
            return True
        except Exception:
            return False

    def _compute_checksum(self, path):
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    # ------------------------------------------------------------------
    # Checkpoint directory bookkeeping
    # ------------------------------------------------------------------

    def _checkpoint_step(self, checkpoint_dir):
        if not checkpoint_dir.name.startswith("checkpoint-"):
            return None
        try:
            return int(checkpoint_dir.name.split("-", 1)[1])
        except ValueError:
            return None

    def _checkpoint_dirs(self):
        checkpoints = []
        for path in self.base_dir.iterdir():
            if not path.is_dir():
                continue
            step = self._checkpoint_step(path)
            if step is not None:
                checkpoints.append((step, path))
        return sorted(checkpoints, key=lambda item: item[0])

    def _cleanup_old_checkpoints(self):
        checkpoints = self._checkpoint_dirs()
        while len(checkpoints) > self.max_kept:
            _, oldest = checkpoints.pop(0)
            shutil.rmtree(oldest)

    def latest_valid(self):
        for step, ckpt_dir in reversed(self._checkpoint_dirs()):
            meta_path = ckpt_dir / "meta.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)

            if meta.get("sharded"):
                # Valid if DCP model shards and train_state exist
                if (ckpt_dir / "model").exists() and (ckpt_dir / "train_state.pt").exists():
                    return str(ckpt_dir), meta.get("step", step)
            else:
                state_path = ckpt_dir / "state.pt"
                if not state_path.exists():
                    continue
                checksum = meta.get("checksum")
                if checksum and self._validate_full(ckpt_dir, checksum):
                    return str(ckpt_dir), meta.get("step", step)
        return None, 0

    # ------------------------------------------------------------------
    # S3
    # ------------------------------------------------------------------

    def _s3_prefix_value(self):
        return (self.s3_prefix or "checkpoints").strip("/")

    def _sync_to_s3(self, checkpoint_dir, step):
        try:
            import boto3
            s3 = boto3.client("s3")
            prefix = self._s3_prefix_value()
            for file_path in checkpoint_dir.rglob("*"):
                if file_path.is_file():
                    rel = file_path.relative_to(checkpoint_dir)
                    s3_key = f"{prefix}/checkpoint-{step}/{rel}"
                    s3.upload_file(str(file_path), self.s3_bucket, s3_key)
        except Exception as e:
            print(f"[checkpoint] S3 sync failed for step {step}: {e}")

    # ------------------------------------------------------------------
    # Distributed helpers
    # ------------------------------------------------------------------

    def _barrier(self):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _rank(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    def _world_size(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    def _raise_thread_error(self):
        if self._last_error is None:
            return
        error = self._last_error
        self._last_error = None
        raise RuntimeError(f"checkpoint write failed: {error}") from error

    def wait(self):
        with self._thread_lock:
            if self._write_thread and self._write_thread.is_alive():
                self._write_thread.join()
            self._raise_thread_error()

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def restore(self, model, optimizer, scheduler, device):
        rank = self._rank()
        path, step = self.latest_valid()

        if self._world_size() > 1:
            steps = [None for _ in range(self._world_size())]
            dist.all_gather_object(steps, step)
            if len(set(steps)) != 1:
                path, step = None, 0

        if path is None:
            return 0

        ckpt_path = Path(path)
        with open(ckpt_path / "meta.json") as f:
            meta = json.load(f)

        if meta.get("sharded") and isinstance(model, FSDP):
            step = self._restore_sharded(model, optimizer, scheduler, path)
        else:
            state = torch.load(ckpt_path / "state.pt", map_location="cpu", weights_only=False)
            if isinstance(model, FSDP) and state.get("fsdp", False):
                state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
                optim_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_config, optim_config):
                    model.load_state_dict(state["model"])
                    optimizer_state = FSDP.optim_state_dict_to_load(model, optimizer, state["optimizer"])
                    optimizer.load_state_dict(optimizer_state)
            else:
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            step = state.get("step", step)

        if rank == 0:
            print(f"[checkpoint] Restored from step {step} ({path})")
        return step
