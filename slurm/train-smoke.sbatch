#!/bin/bash
#SBATCH --job-name=smoke-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/tmp/smoke-%x-%j.out
#SBATCH --error=/tmp/smoke-%x-%j.err
#SBATCH --signal=B:SIGTERM@30

set -euo pipefail

echo "=== Smoke training job $SLURM_JOB_ID on $(hostname) ==="
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

source /opt/pytorch/bin/activate

REPO_DIR="$HOME/ai-factory-control-plane"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/pixailabsonline/ai-factory-control-plane.git "$REPO_DIR"
else
    git -C "$REPO_DIR" pull --ff-only
fi
cd "$REPO_DIR"
mkdir -p logs

python training/fsdp_trainer.py \
    --smoke-test \
    --batch-size 4 \
    --gradient-accumulation 1 \
    --max-steps 20 \
    --checkpoint-every 5 \
    --checkpoint-dir "/tmp/checkpoints/$SLURM_JOB_ID-smoke" \
    --max-length 32 \
    --epochs 1 \
    2>&1 | tee "logs/smoke-train-$SLURM_JOB_ID.log"

echo "=== Smoke training complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
