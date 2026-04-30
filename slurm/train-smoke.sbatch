#!/bin/bash
#SBATCH --job-name=smoke-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --signal=B:SIGTERM@30

set -euo pipefail

echo "=== Smoke training job $SLURM_JOB_ID on $(hostname) ==="
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

source /opt/training-env/bin/activate

python training/fsdp_trainer.py \
    --smoke-test \
    --model smoke \
    --batch-size 4 \
    --gradient-accumulation 1 \
    --max-steps 20 \
    --checkpoint-every 5 \
    --checkpoint-dir "checkpoints/$SLURM_JOB_ID-smoke" \
    --max-length 32 \
    --epochs 1 \
    2>&1 | tee "logs/smoke-train-$SLURM_JOB_ID.log"

echo "=== Smoke training complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
