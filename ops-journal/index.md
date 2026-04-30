# Ops Journal Index

This directory records real runs, failures, and fixes from the live cluster.

## Stage 3 Evidence

- [2026-04-30 — Stage 3 complete: both Slurm nodes idle](2026-04-30-stage3-slurm-worker.md)
  - Confirms both nodes were up, idle, and able to run `sbatch` jobs.
  - Shows the worker actually executed a Slurm job and returned clean output.

- [2026-04-30 — Stage 3 training proof complete](2026-04-30-stage3-training-proof.md)
  - Confirms single-node smoke training passed.
  - Confirms multi-node FSDP training passed.
  - Confirms checkpoint recovery passed.

## What This Proves

- The Slurm cluster is live.
- Distributed training is live.
- Checkpointing is live.
- Resume/recovery is live.
- The repo is not just a design doc. It contains run evidence.

## How To Read It

- Start with the dated entries.
- Look for job IDs, checkpoint paths, loss values, and exit states.
- Use the journal to trace the sequence from submission to training to eval.
