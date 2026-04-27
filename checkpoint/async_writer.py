"""
Async checkpoint writer — training doesn't pause while checkpoints are saved.
Validates integrity before promoting, syncs to S3 for durability.
"""

import os
import time
import json
import hashlib
import threading
from pathlib import Path
from datetime import datetime

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class AsyncCheckpointWriter:
    def __init__(self, base_dir, s3_bucket=None, s3_prefix=None, max_kept=5):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.max_kept = max_kept
        self._write_thread = None
        self._pending_writes = []

    def save(self, model, optimizer, scheduler, step, metrics=None, rank=0):
        with FSDP.state_dict_type(model, FSDP.StateDictType.FULL_STATE_DICT):
            if rank != 0:
                dist.barrier()
                return

            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics or {},
            }

            self._write_thread = threading.Thread(
                target=self._async_write, args=(state, step)
            )
            self._write_thread.start()

        dist.barrier()

    def _async_write(self, state, step):
        staging_dir = self.base_dir / f".staging-{step}"
        final_dir = self.base_dir / f"checkpoint-{step}"
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

        if self._validate(staging_dir, checksum):
            staging_dir.rename(final_dir)
            self._cleanup_old_checkpoints()
            if self.s3_bucket:
                self._sync_to_s3(final_dir, step)
        else:
            import shutil
            shutil.rmtree(staging_dir)
            print(f"[checkpoint] CORRUPTED checkpoint at step {step} — discarded")

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

    def _cleanup_old_checkpoints(self):
        checkpoints = sorted(
            [d for d in self.base_dir.iterdir() if d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[1]),
        )
        while len(checkpoints) > self.max_kept:
            oldest = checkpoints.pop(0)
            import shutil
            shutil.rmtree(oldest)

    def _sync_to_s3(self, checkpoint_dir, step):
        try:
            import boto3
            s3 = boto3.client("s3")
            for file_path in checkpoint_dir.iterdir():
                s3_key = f"{self.s3_prefix}/checkpoint-{step}/{file_path.name}"
                s3.upload_file(str(file_path), self.s3_bucket, s3_key)
        except Exception as e:
            print(f"[checkpoint] S3 sync failed for step {step}: {e}")

    def wait(self):
        if self._write_thread and self._write_thread.is_alive():
            self._write_thread.join()

    def latest_valid(self):
        checkpoints = sorted(
            [d for d in self.base_dir.iterdir() if d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[1]),
            reverse=True,
        )
        for ckpt_dir in checkpoints:
            meta_path = ckpt_dir / "meta.json"
            state_path = ckpt_dir / "state.pt"
            if not meta_path.exists() or not state_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            if self._validate(ckpt_dir, meta["checksum"]):
                return str(ckpt_dir), meta["step"]
        return None, 0

    def restore(self, model, optimizer, scheduler, device):
        path, step = self.latest_valid()
        if path is None:
            return 0

        state = torch.load(Path(path) / "state.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        print(f"[checkpoint] Restored from step {step} ({path})")
        return step
