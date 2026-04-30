#!/bin/bash
#SBATCH --job-name=fsdp-recovery
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/tmp/recovery-%x-%j.out
#SBATCH --error=/tmp/recovery-%x-%j.err

set -euo pipefail

echo "=== Recovery test job $SLURM_JOB_ID on $(hostname) ==="
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

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/tmp/checkpoints/$SLURM_JOB_ID-recovery}"
mkdir -p "$CHECKPOINT_DIR"

FIRST_LOG="logs/recovery-$SLURM_JOB_ID-first.log"
SECOND_LOG="logs/recovery-$SLURM_JOB_ID-second.log"

echo "Phase 1: write a checkpoint and stop cleanly"
python training/fsdp_trainer.py \
    --model distilgpt2 \
    --dataset wikitext/wikitext-2-raw-v1 \
    --batch-size 2 \
    --gradient-accumulation 1 \
    --lr 1e-5 \
    --warmup-steps 2 \
    --max-steps 6 \
    --checkpoint-every 2 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --stop-after-step 4 \
    --max-length 64 \
    --epochs 1 \
    2>&1 | tee "$FIRST_LOG"

echo "Phase 2: resume from the checkpoint and continue"
python training/fsdp_trainer.py \
    --model distilgpt2 \
    --dataset wikitext/wikitext-2-raw-v1 \
    --batch-size 2 \
    --gradient-accumulation 1 \
    --lr 1e-5 \
    --warmup-steps 2 \
    --max-steps 8 \
    --checkpoint-every 2 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --max-length 64 \
    --epochs 1 \
    2>&1 | tee "$SECOND_LOG"

grep -q "Resumed from step" "$SECOND_LOG"

echo "Recovery test complete"
