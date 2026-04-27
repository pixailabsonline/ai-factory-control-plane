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

python inference/server.py \
    --model "${MODEL:-mistralai/Mistral-7B-v0.1}" \
    --checkpoint "${CHECKPOINT:-}" \
    --port "${PORT:-8080}" \
    --tensor-parallel "${TP:-1}"
