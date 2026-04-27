# AI Factory Control Plane

End-to-end GPU training infrastructure: FSDP fine-tuning, Slurm job management, checkpoint recovery, evaluation gates, vLLM inference serving.

## Stack

- **Slurm** — job scheduling, GPU allocation, preemption handling, multi-node coordination
- **PyTorch FSDP** — distributed training with full sharding, mixed precision
- **vLLM** — production inference with continuous batching and PagedAttention
- **Terraform** — instance provisioning, IAM, S3, CloudWatch, auto-provisioned with Deep Learning AMI
- **CloudWatch** — log shipping, GPU utilization metrics, idle alerts

## Hardware

Tested on:
- **Single-node:** p3.8xlarge (4x V100 16GB, NVLink 300 GB/s)
- **Multi-node:** 2x p3.8xlarge (8x V100 across nodes)

## Usage

```bash
# Provision infrastructure
make infra-init && make infra-up

# Submit training job via Slurm
make train                          # single-node, 4x V100
make train-multi                    # multi-node, 8x V100

# Monitor
make jobs                           # squeue
make gpu-status                     # sinfo
make logs                           # recent log files

# Profile a short run
make profile

# Evaluate checkpoint
make eval CHECKPOINT=./checkpoints/<job-id>/checkpoint-5000

# Serve trained model
make serve
make bench
```

## Ops Journal

`ops-journal/` contains real incident logs from training runs — NCCL hangs, GPU failures, checkpoint corruption, OOM debugging. Each entry follows:

```
Symptom → Diagnosis → Root cause → Fix → Time to resolve → Lesson
```
