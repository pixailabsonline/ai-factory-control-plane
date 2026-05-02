#!/bin/bash
#SBATCH --job-name=data-pipeline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/tmp/data-pipeline-%x-%j.out
#SBATCH --error=/tmp/data-pipeline-%x-%j.err

# Data pipeline: Common Crawl → normalize → dedup → tokenize → dataset manifest
#
# Required env vars:
#   S3_BUCKET   — S3 bucket for raw artifacts and dataset output
#   RUN_NAME    — unique run identifier, e.g. cc-2024-10-run-001
#
# Optional:
#   CC_CRAWL    — Common Crawl snapshot, default CC-MAIN-2024-10
#   CC_LIMIT    — max WET files to ingest (0 = all; use small number for smoke test)
#   TOKENIZER   — HuggingFace tokenizer, default gpt2
#   MAX_LENGTH  — sequence length, default 1024
#   DATASET_VER — dataset version tag, default v1

set -euo pipefail

export S3_BUCKET="${S3_BUCKET:-}"
export RUN_NAME="${RUN_NAME:-}"
export CC_CRAWL="${CC_CRAWL:-CC-MAIN-2024-10}"
export CC_LIMIT="${CC_LIMIT:-0}"
export TOKENIZER="${TOKENIZER:-gpt2}"
export MAX_LENGTH="${MAX_LENGTH:-1024}"
export DATASET_VER="${DATASET_VER:-v1}"

if [ -z "$S3_BUCKET" ]; then
    echo "[pipeline] ERROR: S3_BUCKET is required" >&2
    exit 1
fi
if [ -z "$RUN_NAME" ]; then
    echo "[pipeline] ERROR: RUN_NAME is required" >&2
    exit 1
fi

echo "=== Data pipeline job $SLURM_JOB_ID ==="
echo "Start:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Run:      $RUN_NAME"
echo "Crawl:    $CC_CRAWL (limit=$CC_LIMIT)"
echo "Bucket:   s3://$S3_BUCKET"
echo "Tokenizer: $TOKENIZER max_length=$MAX_LENGTH"
echo "Dataset:  $DATASET_VER"

REPO_DIR="$HOME/ai-factory-control-plane"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/pixailabsonline/ai-factory-control-plane.git "$REPO_DIR"
else
    git -C "$REPO_DIR" pull --ff-only
fi
cd "$REPO_DIR"

source /opt/pytorch/bin/activate
pip install -q boto3 transformers

# Scratch dirs — use Lustre if available, else /tmp
if mountpoint -q /mnt/lustre 2>/dev/null; then
    SCRATCH="/mnt/lustre/pipeline/$RUN_NAME"
else
    SCRATCH="/tmp/pipeline/$RUN_NAME"
fi
mkdir -p "$SCRATCH"

echo ""
echo "--- Phase 1: Raw ingestion ---"
python data_pipeline/ingest_commoncrawl.py \
    --crawl "$CC_CRAWL" \
    --limit "$CC_LIMIT" \
    --s3-bucket "$S3_BUCKET" \
    --run-name "$RUN_NAME" \
    --tmp-dir "$SCRATCH/raw"

echo ""
echo "--- Phase 2: Normalize and filter ---"
python data_pipeline/normalize.py \
    --input-manifest "runs/$RUN_NAME/raw/raw_input_manifest.json" \
    --output-dir "$SCRATCH/normalized" \
    --s3-bucket "$S3_BUCKET" \
    --run-name "$RUN_NAME"

echo ""
echo "--- Phase 3: Exact deduplication ---"
python data_pipeline/dedup_exact.py \
    --input "s3://$S3_BUCKET/runs/$RUN_NAME/normalized/normalized.jsonl" \
    --output-dir "$SCRATCH/deduped" \
    --s3-bucket "$S3_BUCKET" \
    --run-name "$RUN_NAME"

echo ""
echo "--- Phase 4: Tokenize and pack ---"
python data_pipeline/tokenize_pack.py \
    --input "s3://$S3_BUCKET/runs/$RUN_NAME/deduped/deduped.jsonl" \
    --output-dir "$SCRATCH/packed" \
    --s3-bucket "$S3_BUCKET" \
    --run-name "$RUN_NAME" \
    --tokenizer "$TOKENIZER" \
    --max-length "$MAX_LENGTH"

echo ""
echo "--- Phase 5: Write dataset manifest ---"
python data_pipeline/write_manifest.py \
    --run-name "$RUN_NAME" \
    --dataset-version "$DATASET_VER" \
    --s3-bucket "$S3_BUCKET" \
    --output-dir "$SCRATCH/manifest"

echo ""
echo "=== Pipeline complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Dataset artifact: s3://$S3_BUCKET/runs/$RUN_NAME/datasets/$DATASET_VER/"
echo "Manifest:         s3://$S3_BUCKET/runs/$RUN_NAME/datasets/$DATASET_VER/dataset_manifest.json"
