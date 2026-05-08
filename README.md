# AI Factory Control Plane

A platform that takes raw web data and turns it into a trained, evaluated, production-ready model — automatically. You push data in one end, a serveable model comes out the other. The system handles GPU provisioning, distributed training, checkpoint recovery, quality gates, and model promotion without manual intervention.

**What it does in plain English:** you give it text data (e.g. Common Crawl), it cleans and tokenizes that data, trains a language model across multiple GPUs, automatically recovers if hardware fails mid-training, evaluates the result, and publishes the passing model ready to serve.

## How it works

1. **Data pipeline** — ingests raw text (Common Crawl), filters junk, deduplicates, tokenizes, and packs into fixed-length training sequences
2. **Distributed training** — runs PyTorch FSDP across multiple GPU nodes, with async checkpointing to S3
3. **Fault recovery** — if a node dies mid-training, the job resumes from the last checkpoint instead of starting over
4. **Eval gate** — loads the trained checkpoint, measures perplexity, passes or fails it
5. **Model promotion** — passing models get exported and served via vLLM

## Stack

| Layer | Tool | Role |
|---|---|---|
| Infrastructure | Terraform + AWS | Provisions GPU instances, networking, S3, IAM |
| Scheduling | Slurm | Job queuing, GPU allocation, multi-node coordination |
| Training | PyTorch FSDP | Distributed training with full sharding, mixed precision |
| Data | Custom pipeline | Common Crawl ingest → filter → dedup → tokenize → pack |
| Eval | Quality gate | Perplexity check on latest checkpoint |
| Serving | vLLM | Production inference with continuous batching |
| Monitoring | CloudWatch | GPU utilization metrics, idle alerts, log shipping |

## Architecture

Slurm is the user-facing layer — researchers submit jobs with `sbatch`. Underneath, Terraform manages the GPU nodes on AWS. This is not a Kubernetes-native ML platform; GPUs are owned by Slurm partitions, not shared with arbitrary workloads.

See [docs/project-overview.md](docs/project-overview.md) for the full architecture diagram.

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

## Why this matters

The platform's job is to reduce cost per useful GPU hour. Checkpoint recovery means interrupted training resumes instead of restarting from scratch. The automated eval gate means bad models are caught immediately rather than wasting inference budget.

The commercial report above proves this works on real hardware with real training runs.

See [docs/commercial-experiment-plan.md](docs/commercial-experiment-plan.md) for the baseline/recovery experiment design.
See [docs/data-pipeline-plan.md](docs/data-pipeline-plan.md) for the data pipeline design.

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

`allowed_ssh_cidrs` is required — no default-open posture.

## Ops Journal

`ops-journal/` contains real incident logs from training runs — NCCL hangs, GPU failures, checkpoint corruption, OOM debugging. Each entry follows:

```
Symptom → Diagnosis → Root cause → Fix → Time to resolve → Lesson
```
