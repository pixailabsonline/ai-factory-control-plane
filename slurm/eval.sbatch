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
S3_PREFIX="checkpoints/$MODEL_NAME"

echo "=== Eval gate: s3://$S3_BUCKET/$S3_PREFIX ==="

REPO_DIR="$HOME/ai-factory-control-plane"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/pixailabsonline/ai-factory-control-plane.git "$REPO_DIR"
else
    git -C "$REPO_DIR" pull --ff-only
fi
cd "$REPO_DIR"
mkdir -p eval/results

source /opt/pytorch/bin/activate

# Find the latest checkpoint-N in S3 and download it
LATEST_STEP=$(aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/" \
    | awk '{print $2}' \
    | grep 'checkpoint-' \
    | sed 's|checkpoint-||;s|/||' \
    | sort -n \
    | tail -1)

if [ -z "$LATEST_STEP" ]; then
    echo "No checkpoints found at s3://$S3_BUCKET/$S3_PREFIX/ — aborting"
    exit 1
fi

echo "Latest checkpoint: step $LATEST_STEP"
LOCAL_CKPT="/tmp/eval-ckpt-$SLURM_JOB_ID/checkpoint-$LATEST_STEP"
mkdir -p "$LOCAL_CKPT"
aws s3 sync "s3://$S3_BUCKET/$S3_PREFIX/checkpoint-$LATEST_STEP/" "$LOCAL_CKPT/"

python eval/quality_gate.py \
    --checkpoint "$LOCAL_CKPT" \
    --max-perplexity "${MAX_PPL:-90.0}" \
    --output "eval/results/eval-$SLURM_JOB_ID.json"

EXIT=$?
rm -rf "/tmp/eval-ckpt-$SLURM_JOB_ID"

if [ $EXIT -eq 0 ]; then
    echo "PASS — checkpoint step $LATEST_STEP promoted"
else
    echo "FAIL — checkpoint step $LATEST_STEP rejected"
fi
exit $EXIT
