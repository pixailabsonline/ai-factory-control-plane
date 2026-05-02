#!/bin/bash
#SBATCH --job-name=eval-gate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/tmp/eval-%x-%j.out
#SBATCH --error=/tmp/eval-%x-%j.err

set -euo pipefail

S3_BUCKET="${S3_BUCKET:-ai-factory-checkpoints-737213639346}"
MODEL_NAME="${MODEL_NAME:-gpt2-medium}"
S3_RUN_KEY="${S3_RUN_KEY:-runs/$MODEL_NAME}"
S3_RUN_ROOT="${S3_RUN_ROOT:-s3://$S3_BUCKET/$S3_RUN_KEY}"
S3_CHECKPOINTS_URI="$S3_RUN_ROOT/checkpoints"

REPO_DIR="$HOME/ai-factory-control-plane"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/pixailabsonline/ai-factory-control-plane.git "$REPO_DIR"
else
    git -C "$REPO_DIR" pull --ff-only
fi
cd "$REPO_DIR"
mkdir -p eval/results

source /opt/pytorch/bin/activate

# On Lustre: read checkpoint directly — no download needed
# Without Lustre: pull latest from S3
if mountpoint -q /mnt/lustre 2>/dev/null; then
    LUSTRE_CKPT_DIR="/mnt/lustre/checkpoints/$MODEL_NAME"
    LATEST_STEP=$(ls "$LUSTRE_CKPT_DIR" 2>/dev/null \
        | grep '^checkpoint-' | sed 's/checkpoint-//' | sort -n | tail -1)

    if [ -z "$LATEST_STEP" ]; then
        echo "No checkpoints found at $LUSTRE_CKPT_DIR — aborting"
        exit 1
    fi

    LOCAL_CKPT="$LUSTRE_CKPT_DIR/checkpoint-$LATEST_STEP"
    echo "=== Eval gate (Lustre): $LOCAL_CKPT ==="
else
    echo "=== Eval gate (S3): $S3_CHECKPOINTS_URI ==="

    LATEST_STEP=$(aws s3 ls "$S3_CHECKPOINTS_URI/" \
        | awk '{print $2}' \
        | grep 'checkpoint-' \
        | sed 's|checkpoint-||;s|/||' \
        | sort -n \
        | tail -1)

    if [ -z "$LATEST_STEP" ]; then
        echo "No checkpoints found at $S3_CHECKPOINTS_URI/ — aborting"
        exit 1
    fi

    LOCAL_CKPT="/tmp/eval-ckpt-$SLURM_JOB_ID/checkpoint-$LATEST_STEP"
    mkdir -p "$LOCAL_CKPT"
    aws s3 sync "$S3_CHECKPOINTS_URI/checkpoint-$LATEST_STEP/" "$LOCAL_CKPT/"
fi

echo "Evaluating step $LATEST_STEP"

python eval/quality_gate.py \
    --checkpoint "$LOCAL_CKPT" \
    --max-perplexity "${MAX_PPL:-90.0}" \
    --output "eval/results/eval-$SLURM_JOB_ID.json"

EXIT=$?

# Only clean up if we downloaded from S3
if ! mountpoint -q /mnt/lustre 2>/dev/null; then
    rm -rf "/tmp/eval-ckpt-$SLURM_JOB_ID"
fi

if [ $EXIT -eq 0 ]; then
    echo "PASS — checkpoint step $LATEST_STEP promoted"
else
    echo "FAIL — checkpoint step $LATEST_STEP rejected"
fi
exit $EXIT
