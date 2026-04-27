# AI Factory Control Plane

GPU training control plane for distributed model fine-tuning. Manages the full lifecycle: job scheduling, FSDP training, checkpoint recovery, evaluation, inference serving, and cost tracking.

## Architecture

- **Orchestrator** (Go) — Job scheduler, GPU health monitoring, automatic failure recovery
- **Training** (Python) — FSDP trainer, profiling, scaling benchmarks
- **Checkpoint** (Python) — Async writes, integrity validation, S3 sync, auto-restore
- **Cost** (Go) — Real-time cost tracking, MFU calculation, scaling projections
- **Ops Journal** — Real incident logs from actual training runs

## Hardware

Designed for and tested on:
- Single-node: p3.8xlarge (4x NVIDIA V100 16GB, NVLink)
- Multi-node: 2x p3.8xlarge (8x V100 across nodes, EFA)

## Quick Start

```bash
# Single-node FSDP training (4x V100)
torchrun --nproc_per_node=4 training/fsdp_trainer.py \
  --model mistralai/Mistral-7B-v0.1 \
  --dataset wikitext/wikitext-103-raw-v1 \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --max-steps 5000 \
  --checkpoint-every 500

# Multi-node (2x p3.8xlarge)
torchrun --nnodes=2 --nproc_per_node=4 \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
  training/fsdp_trainer.py --model mistralai/Mistral-7B-v0.1
```
