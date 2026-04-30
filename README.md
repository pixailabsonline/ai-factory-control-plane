# AI Factory Control Plane

End-to-end GPU training infrastructure: Kubernetes substrate, NVIDIA GPU enablement, Slurm researcher ergonomics, FSDP fine-tuning, checkpoint recovery, evaluation gates, model export/promotion, and vLLM inference serving.

## Stack

- **Kubernetes substrate** - GPU node lifecycle, platform services, networking, monitoring primitives
- **NVIDIA GPU Operator** - GPU device/runtime enablement and health on the Kubernetes substrate
- **Slurm** - researcher-facing job scheduling, GPU allocation, preemption handling, multi-node coordination
- **PyTorch FSDP** - distributed training with full sharding, mixed precision
- **vLLM** - production inference with continuous batching and PagedAttention
- **Terraform** - instance provisioning, IAM, S3, CloudWatch, auto-provisioned with Deep Learning AMI
- **CloudWatch** - infrastructure-level log shipping, GPU utilization metrics, idle alerts

## Architecture Direction

This repo follows a NVIDIA-style layered model: Kubernetes is the infrastructure substrate, while Slurm remains the batch scheduler users interact with. Researchers submit jobs with normal Slurm commands; operators validate the Kubernetes and GPU Operator layer underneath.

This is not a CoreWeave/SUNK-style unified scheduler. The default design does not allow general Kubernetes workloads to contend with Slurm jobs for the same GPUs. GPU capacity ownership must be explicit through node pools, partitions, labels, or reservations.

See [docs/nvidia-style-slurm-on-kubernetes.md](docs/nvidia-style-slurm-on-kubernetes.md) for the architecture contract.
See [decisions/kubernetes-substrate-vs-unified-scheduler.md](decisions/kubernetes-substrate-vs-unified-scheduler.md) for the scope decision and revisit criteria.
See [docs/project-overview.md](docs/project-overview.md) for the single-page map of the repo.
Read [docs/project-overview.md](docs/project-overview.md) first if you want the diagram and the end-to-end flow in one place.

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
make train-smoke                   # 1 GPU smoke test for the training loop
make train-recovery                # checkpoint write + resume test
make train                          # single-node training on the current GPU node
make train-multi                    # 2-node training on current GPU nodes, default gpt2-medium

# Monitor
make jobs                           # squeue
make gpu-status                     # sinfo
make substrate-status               # Kubernetes nodes + GPU Operator pods
make slurm-status                   # Slurm nodes, queues, partitions
make platform-status                # substrate + Slurm proof
make logs                           # recent log files

# Profile a short run
make profile

# Evaluate the latest checkpoint in the run root
make eval MODEL_NAME=<model-name>

# Export a trained checkpoint into a serveable model directory
make export-model CHECKPOINT=/tmp/checkpoints/<job-id>/checkpoint-1000 MODEL_DIR=/tmp/models/<run-name>

# Promote the exported model to S3
make publish-model CHECKPOINT=/tmp/checkpoints/<job-id>/checkpoint-1000 MODEL_DIR=/tmp/models/<run-name> MODEL_S3_ROOT=s3://<bucket>/runs/<run-name>

# Serve trained model
make serve MODEL_DIR=/tmp/models/<run-name>
make serve MODEL_S3_ROOT=s3://<bucket>/runs/<run-name>
make bench
```

`allowed_ssh_cidrs` is now required explicitly. The default-open `0.0.0.0/0` posture has been removed.

Stage 3 is now proven end to end on the current cluster shape:
- smoke training passes
- distributed multi-node training passes
- checkpoint recovery passes

Use [docs/model-artifact-manifest.md](docs/model-artifact-manifest.md) to record trained checkpoints or exported model artifacts without checking binaries into git.

`make serve` serves a base model unless you point it at an exported model directory or a promoted S3 artifact. Raw training checkpoints are for training and eval; export them first with `make export-model` or promote them with `make publish-model`. Each promoted artifact directory now includes `README.md` and `artifact_index.json` so the S3 prefix is browsable as evidence. The standardized S3 shape is `s3://<bucket>/runs/<run-name>/checkpoints/` for raw checkpoints and `s3://<bucket>/runs/<run-name>/models/latest` for the promoted artifact. If the checkpoint is node-local, pin the export/publish/serve job to that node with `NODELIST=...`.

Current cluster shape used for the proven training runs:
- **Single-node:** 1x A10G class GPU node
- **Multi-node:** 2x A10G class GPU nodes with FSDP FULL_SHARD

CloudWatch and vLLM are real repo components, but they live in the platform/inference layers:
- CloudWatch is implemented in `infra/`.
- vLLM is launched by `inference/server.py` and depends on `training/requirements.txt`.

## Ops Journal

`ops-journal/` contains real incident logs from training runs — NCCL hangs, GPU failures, checkpoint corruption, OOM debugging. Each entry follows:

```
Symptom → Diagnosis → Root cause → Fix → Time to resolve → Lesson
```
