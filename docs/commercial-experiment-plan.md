# Commercial Experiment Plan

Goal: measure whether the platform reduces wasted GPU hours for the same model and hardware.

## Fixed Setup

- Hardware: current `g5` cluster shape
- Model: `gpt2-medium`
- Dataset: `wikitext/wikitext-2-raw-v1`
- Training path: Slurm + FSDP
- Reporting: `training/commercial_report.py`

## Run Order

1. Baseline run

```bash
make commercial-baseline
```

This runs the same model and hardware without the interruption/restart path.

2. Recovery run

```bash
make commercial-recovery
```

This runs the same model and hardware with an intentional interruption, then resumes from checkpoint.

3. Generate the commercial summary

```bash
make commercial-report \
  RUN_ROOT=/path/to/recovery-run-root \
  BASELINE_RUN_ROOT=/path/to/baseline-run-root \
  INSTANCE_TYPE=g5.xlarge \
  GPUS_PER_NODE=1 \
  OUTPUT=commercial-summary.md
```

## What to record

- GPU hours to passing checkpoint
- Tokens/sec
- Recovery step
- Time from checkpoint to serveable artifact
- Estimated cost per run
- Eval pass/fail

## What the story should say

- Same model
- Same hardware
- Same dataset
- Recovery resumes from checkpoint instead of restarting
- The platform reduces wasted GPU hours
