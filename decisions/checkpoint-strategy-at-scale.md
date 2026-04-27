# Checkpoint strategy

Our p3.8xlarge costs $12.24/hr. If training crashes and we have no checkpoint, we restart from zero. If we checkpoint too often, GPUs sit idle writing to disk.

Checkpoint every 500 steps, async write, S3 sync, validate before restore.

At ~$12/hr, losing 500 steps costs maybe $6-12 in wasted compute. We're not going to burn GPU time checkpointing every 50 steps for a $6 risk.

We write async: a background thread saves to disk while the GPU keeps training. A 7B model checkpoint is ~14GB, takes 30-60 seconds to write. Without async, that's a full minute of idle GPUs every 500 steps. With async, the GPU never pauses.

We validate with SHA-256 checksum and load test before promoting a checkpoint. If the write was interrupted or the file is corrupt, we catch it before trying to restore from garbage. On restore, if the latest checkpoint is corrupt, we fall back to the previous one. Lose a few extra steps, but don't load a broken model.

S3 sync runs after validation. If the node dies, we don't lose everything. At our scale the S3 cost is pennies. At 1,000 H100s where a crash costs $50K+ in lost compute, S3 sync has to be there.

The sbatch scripts handle Slurm preemption. SIGTERM triggers an immediate checkpoint save before exit. Requeued job picks up where it left off.

At 1,000+ GPUs and $50K/hr, you'd checkpoint every 50-100 steps and use distributed per-shard writes to local NVMe instead of gathering to rank 0. Same pattern, different frequency and write strategy.
