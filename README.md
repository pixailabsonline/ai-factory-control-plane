# AI Factory Control Plane

End-to-end GPU training infrastructure: Kubernetes substrate, NVIDIA GPU enablement, Slurm researcher ergonomics, FSDP fine-tuning, checkpoint recovery, evaluation gates, and vLLM inference serving.

## Stack

- **Kubernetes substrate** - GPU node lifecycle, platform services, networking, monitoring primitives
- **NVIDIA GPU Operator** - GPU device/runtime enablement and health on the Kubernetes substrate
- **Slurm** - researcher-facing job scheduling, GPU allocation, preemption handling, multi-node coordination
- **PyTorch FSDP** - distributed training with full sharding, mixed precision
- **vLLM** - production inference with continuous batching and PagedAttention
- **Terraform** - instance provisioning, IAM, S3, CloudWatch, auto-provisioned with Deep Learning AMI
- **CloudWatch** - log shipping, GPU utilization metrics, idle alerts

## Architecture Direction

This repo follows a NVIDIA-style layered model: Kubernetes is the infrastructure substrate, while Slurm remains the batch scheduler users interact with. Researchers submit jobs with normal Slurm commands; operators validate the Kubernetes and GPU Operator layer underneath.

This is not a CoreWeave/SUNK-style unified scheduler. The default design does not allow general Kubernetes workloads to contend with Slurm jobs for the same GPUs. GPU capacity ownership must be explicit through node pools, partitions, labels, or reservations.

See [docs/nvidia-style-slurm-on-kubernetes.md](docs/nvidia-style-slurm-on-kubernetes.md) for the architecture contract.
See [decisions/kubernetes-substrate-vs-unified-scheduler.md](decisions/kubernetes-substrate-vs-unified-scheduler.md) for the scope decision and revisit criteria.

## Hardware

Tested on:
- **Single-node:** p3.8xlarge (4x V100 16GB, NVLink 300 GB/s)
- **Multi-node:** 2x p3.8xlarge (8x V100 across nodes)

## Usage

```bash
# One-time: create S3 state bucket and DynamoDB lock table
aws s3 mb s3://ai-factory-tfstate-af-ctrl-x7k2 --region us-east-1
aws dynamodb create-table \
  --table-name ai-factory-tf-locks-af-ctrl-x7k2 \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1

# Provision infrastructure
make infra-init
make infra-plan SSH_CIDR=<your-public-ip>/32   # dry-run
make infra-up SSH_CIDR=<your-public-ip>/32
make infra-down SSH_CIDR=<your-public-ip>/32   # spin down instances (keeps S3/state)

# Submit training job via Slurm
make train                          # single-node, 4x V100
make train-multi                    # multi-node, 8x V100

# Monitor
make jobs                           # squeue
make gpu-status                     # sinfo
make substrate-status               # Kubernetes nodes + GPU Operator pods
make slurm-status                   # Slurm nodes, queues, partitions
make platform-status                # substrate + Slurm proof
make logs                           # recent log files

# Profile a short run
make profile

# Evaluate checkpoint
make eval CHECKPOINT=./checkpoints/<job-id>/checkpoint-5000

# Serve trained model
make serve
make bench
```

`allowed_ssh_cidrs` is now required explicitly. The default-open `0.0.0.0/0` posture has been removed.

## Ops Journal

`ops-journal/` contains real incident logs from training runs — NCCL hangs, GPU failures, checkpoint corruption, OOM debugging. Each entry follows:

```
Symptom → Diagnosis → Root cause → Fix → Time to resolve → Lesson
```
