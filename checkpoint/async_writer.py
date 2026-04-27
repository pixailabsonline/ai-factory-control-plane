"""
Async checkpoint writer — training doesn't pause while checkpoints are saved.
Validates integrity before promoting, syncs to S3 for durability.
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

    def save(self, model, optimizer, scheduler, step, metrics=None, rank=0):
        state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_config, optim_config):
            model_state = model.state_dict()
            optimizer_state = FSDP.optim_state_dict(model, optimizer)
            state = {
                "model": model_state,
                "optimizer": optimizer_state,
                "scheduler": scheduler.state_dict(),
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics or {},
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
            }
            with open(staging_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            if not self._validate(staging_dir, checksum):
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

    def _validate(self, checkpoint_dir, expected_checksum):
        state_path = checkpoint_dir / "state.pt"
        if not state_path.exists():
            return False

        actual_checksum = self._compute_checksum(state_path)
        if actual_checksum != expected_checksum:
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

    def _s3_prefix_value(self):
        return (self.s3_prefix or "checkpoints").strip("/")

    def _sync_to_s3(self, checkpoint_dir, step):
        try:
            import boto3
            s3 = boto3.client("s3")
            prefix = self._s3_prefix_value()
            for file_path in checkpoint_dir.iterdir():
                s3_key = f"{prefix}/checkpoint-{step}/{file_path.name}"
                s3.upload_file(str(file_path), self.s3_bucket, s3_key)
        except Exception as e:
            print(f"[checkpoint] S3 sync failed for step {step}: {e}")

    def _download_latest_from_s3(self, rank):
        if not self.s3_bucket:
            return None, 0

        try:
            import boto3
            s3 = boto3.client("s3")
            prefix = self._s3_prefix_value()
            paginator = s3.get_paginator("list_objects_v2")
            by_step = {}

            for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=f"{prefix}/checkpoint-"):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    rel = key[len(prefix) + 1:]
                    parts = rel.split("/", 1)
                    if len(parts) != 2 or not parts[0].startswith("checkpoint-"):
                        continue
                    try:
                        step = int(parts[0].split("-", 1)[1])
                    except ValueError:
                        continue
                    by_step.setdefault(step, {})[parts[1]] = key

            for step in sorted(by_step, reverse=True):
                keys = by_step[step]
                if "state.pt" not in keys or "meta.json" not in keys:
                    continue

                restore_root = self.base_dir / f".restore-rank-{rank}"
                staging_dir = restore_root / f".staging-checkpoint-{step}"
                final_dir = restore_root / f"checkpoint-{step}"
                if staging_dir.exists():
                    shutil.rmtree(staging_dir)
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                staging_dir.mkdir(parents=True, exist_ok=True)

                for filename in ("state.pt", "meta.json"):
                    s3.download_file(self.s3_bucket, keys[filename], str(staging_dir / filename))

                with open(staging_dir / "meta.json") as f:
                    meta = json.load(f)
                if self._validate(staging_dir, meta["checksum"]):
                    staging_dir.rename(final_dir)
                    return str(final_dir), step
                shutil.rmtree(staging_dir)
        except Exception as e:
            print(f"[checkpoint] S3 restore failed: {e}")

        return None, 0

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

    def latest_valid(self):
        for step, ckpt_dir in reversed(self._checkpoint_dirs()):
            meta_path = ckpt_dir / "meta.json"
            state_path = ckpt_dir / "state.pt"
            if not meta_path.exists() or not state_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            if self._validate(ckpt_dir, meta["checksum"]):
                return str(ckpt_dir), meta.get("step", step)
        return None, 0

    def restore(self, model, optimizer, scheduler, device):
        rank = self._rank()
        path, step = self.latest_valid()
        if self._world_size() > 1:
            steps = [None for _ in range(self._world_size())]
            dist.all_gather_object(steps, step)
            if len(set(steps)) != 1:
                path, step = None, 0

        if path is None and self.s3_bucket:
            path, step = self._download_latest_from_s3(rank)

        if path is None:
            return 0

        state = torch.load(Path(path) / "state.pt", map_location="cpu", weights_only=False)
        state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        optim_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_config, optim_config):
            model.load_state_dict(state["model"])
            optimizer_state = FSDP.optim_state_dict_to_load(model, optimizer, state["optimizer"])
            optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(state["scheduler"])
        if rank == 0:
            print(f"[checkpoint] Restored from step {step} ({path})")
        return step
