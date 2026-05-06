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
- **Single-node:** 1x g5.xlarge (A10G 24GB)
- **Multi-node:** 2x g5.xlarge (2x A10G across nodes, FSDP FULL_SHARD)

## Live Results

### Commercial report — `recovery2` vs `baseline2` (2x g5.xlarge, gpt2-medium)

Run: `recovery2-vs-baseline2` · Instance: `g5.xlarge` · GPUs/node: 1

| Metric | Baseline | Recovery run | Delta |
|---|---:|---:|---:|
| GPU hours to passing checkpoint | 4.62 | 4.59 | **-0.04** |
| Tokens/sec | 123.30 | 124.20 | **+0.90** |
| Cost per trained model (USD) | $4.65 | $4.62 | **-$0.04** |
| Perplexity (↓ better) | 38.53 | 38.56 | +0.03 |
| Resumed from step | 0 | 0 | — |

Full report: [`commercial-summary.md`](commercial-summary.md) · JSON: [`commercial-summary.json`](commercial-summary.json)

S3 artifacts:
- `s3://ai-factory-checkpoints-737213639346/runs/gpt2-medium/recovery2/`
- `s3://ai-factory-checkpoints-737213639346/runs/gpt2-medium/baseline2/`
- `s3://ai-factory-checkpoints-737213639346/runs/gpt2-medium/commercial-summary.md`

### Data pipeline smoke test — Common Crawl CC-MAIN-2024-10 (5 WET files)

Run: `smoke-CC-MAIN-2024-10-20260502`

| Stage | Output |
|---|---|
| Raw ingest | 5 WET files, 538 MB |
| After filters | 161,786 documents (6.2% rejected) |
| After exact dedup (SHA-256) | 160,396 documents (0.86% duplicate rate) |
| Tokenized + packed (gpt2, 1024-ctx) | 537,022 sequences · 549M tokens |
| Packed dataset size | 2.1 GB |
| Pack efficiency | 1.0 (no padding waste) |

Dataset manifest: `s3://ai-factory-checkpoints-737213639346/runs/smoke-CC-MAIN-2024-10-20260502/datasets/v1/dataset_manifest.json`
Packed binary: `s3://ai-factory-checkpoints-737213639346/runs/smoke-CC-MAIN-2024-10-20260502/datasets/v1/packed.bin`

## Commercial Metrics

The commercial claim this repo is aiming at is:

> lower cost per useful GPU hour

| Metric | What it means | Status |
|---|---|---|
| GPU hours to passing checkpoint | How much GPU time it takes to produce a usable model | **Measured** — 4.59 h (recovery) vs 4.62 h (baseline) |
| Recovery time after failure | How long it takes to resume instead of restart | Proven in the recovery path |
| Tokens/sec | Training throughput on the same hardware | **Measured** — 124.2 tok/s |
| Cost per trained model | GPU spend required to get a passing model | **Measured** — $4.62 (recovery) vs $4.65 (baseline) |
| Runs salvaged without restart | How many interrupted runs resumed from checkpoint successfully | Proven in live runs |
| Time from checkpoint to serveable artifact | How long it takes to promote a trained checkpoint for inference | Implemented, not yet end-to-end benchmarked |

Proof runs completed:

- Smoke training on 1x g5.xlarge
- Multi-node FSDP training on 2x g5.xlarge with `gpt2-medium`
- Checkpoint recovery run with stop-and-resume
- Eval gate: loaded checkpoint, perplexity gate passed
- Baseline vs recovery commercial comparison — numbers above
- Data pipeline smoke test: 5 WET files → 537K packed sequences

What remains for the full commercial story:

- One clean train → eval → publish → serve run tied to the same artifact
- Scale data pipeline to full crawl (1 TB+)
- A customer or real workload using the platform

See [docs/commercial-experiment-plan.md](docs/commercial-experiment-plan.md) for the exact baseline/recovery sequence.
See [docs/data-pipeline-plan.md](docs/data-pipeline-plan.md) for the data pipeline design.
See [docs/data-pipeline-implementation-checklist.md](docs/data-pipeline-implementation-checklist.md) for build order and sign-off gates.

Regenerate the commercial report from S3 artifacts:

```bash
make commercial-report \
  RUN_ROOT=s3://ai-factory-checkpoints-737213639346/runs/gpt2-medium/recovery2 \
  BASELINE_RUN_ROOT=s3://ai-factory-checkpoints-737213639346/runs/gpt2-medium/baseline2 \
  INSTANCE_TYPE=g5.xlarge \
  GPUS_PER_NODE=1 \
  OUTPUT=commercial-summary.md
```

The interruption/restart path is automated by `make commercial-recovery`. `make commercial-baseline` gives you the plain comparison run on the same model class and cluster shape so you can compute the delta cleanly.

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
make commercial-baseline            # plain comparison run for commercial metrics
make commercial-recovery            # automated interruption + resume run for commercial metrics

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
- **Single-node:** 1x g5.xlarge (A10G 24GB)
- **Multi-node:** 2x g5.xlarge (A10G 24GB each, FSDP FULL_SHARD)

CloudWatch and vLLM are real repo components, but they live in the platform/inference layers:
- CloudWatch is implemented in `infra/`.
- vLLM is launched by `inference/server.py` and depends on `training/requirements.txt`.

## Ops Journal

`ops-journal/` contains real incident logs from training runs — NCCL hangs, GPU failures, checkpoint corruption, OOM debugging. Each entry follows:

```
Symptom → Diagnosis → Root cause → Fix → Time to resolve → Lesson
```
