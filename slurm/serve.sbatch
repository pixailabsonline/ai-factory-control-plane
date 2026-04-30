#!/bin/bash
#SBATCH --job-name=vllm-serve
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

echo "=== vLLM inference server ==="

source /opt/training-env/bin/activate

MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
MODEL_DIR="${MODEL_DIR:-}"
MODEL_S3_URI="${MODEL_S3_URI:-}"
MODEL_S3_ROOT="${MODEL_S3_ROOT:-}"
CHECKPOINT="${CHECKPOINT:-}"
TOKENIZER="${TOKENIZER:-}"
EXPORT_DIR="${EXPORT_DIR:-/tmp/exported-models/$SLURM_JOB_ID}"
EXPORT_ARGS=()
PUBLISH_ARGS=()

if [ -n "$MODEL" ]; then
    EXPORT_ARGS+=(--model "$MODEL")
fi

if [ -n "$TOKENIZER" ]; then
    EXPORT_ARGS+=(--tokenizer "$TOKENIZER")
fi

if [ -n "$MODEL_S3_URI" ]; then
    PUBLISH_ARGS+=(--s3-uri "$MODEL_S3_URI")
elif [ -n "$MODEL_S3_ROOT" ]; then
    PUBLISH_ARGS+=(--s3-root "$MODEL_S3_ROOT")
fi

if [ -n "$CHECKPOINT" ] && [ -z "$MODEL_DIR" ]; then
    echo "Exporting checkpoint $CHECKPOINT to $EXPORT_DIR"
    python training/publish_model.py \
        --checkpoint "$CHECKPOINT" \
        --output-dir "$EXPORT_DIR" \
        "${PUBLISH_ARGS[@]}" \
        "${EXPORT_ARGS[@]}"
    MODEL_DIR="$EXPORT_DIR"
fi

if [ -n "$MODEL_S3_URI" ] && [ -z "$CHECKPOINT" ] && [ -z "$MODEL_DIR" ]; then
    MODEL_DIR="$EXPORT_DIR"
    echo "Downloading promoted model from $MODEL_S3_URI to $MODEL_DIR"
    mkdir -p "$MODEL_DIR"
    aws s3 sync "$MODEL_S3_URI" "$MODEL_DIR" --delete
elif [ -n "$MODEL_S3_ROOT" ] && [ -z "$CHECKPOINT" ] && [ -z "$MODEL_DIR" ]; then
    MODEL_DIR="$EXPORT_DIR"
    MODEL_URI="$MODEL_S3_ROOT/models/latest"
    echo "Downloading promoted model from $MODEL_URI to $MODEL_DIR"
    mkdir -p "$MODEL_DIR"
    aws s3 sync "$MODEL_URI" "$MODEL_DIR" --delete
fi

if [ -n "$MODEL_DIR" ]; then
    python inference/server.py \
        --model-dir "$MODEL_DIR" \
        --port "${PORT:-8080}" \
        --tensor-parallel "${TP:-1}"
else
    python inference/server.py \
        --model "$MODEL" \
        --port "${PORT:-8080}" \
        --tensor-parallel "${TP:-1}"
fi
