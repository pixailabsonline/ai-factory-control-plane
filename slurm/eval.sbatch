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

CHECKPOINT="${CHECKPOINT:?CHECKPOINT env var required}"

echo "=== Eval gate: $CHECKPOINT ==="

REPO_DIR="$HOME/ai-factory-control-plane"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/pixailabsonline/ai-factory-control-plane.git "$REPO_DIR"
else
    git -C "$REPO_DIR" pull --ff-only
fi
cd "$REPO_DIR"
mkdir -p eval/results

source /opt/pytorch/bin/activate

python eval/quality_gate.py \
    --checkpoint "$CHECKPOINT" \
    --max-perplexity "${MAX_PPL:-50.0}" \
    --output "eval/results/eval-$SLURM_JOB_ID.json"

EXIT=$?
if [ $EXIT -eq 0 ]; then
    echo "PASS — checkpoint promoted"
else
    echo "FAIL — checkpoint rejected"
fi
exit $EXIT
