#!/bin/bash
#SBATCH --job-name=eval-gate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

CHECKPOINT="${1:?Usage: sbatch eval.sbatch <checkpoint-path>}"

echo "=== Eval gate: $CHECKPOINT ==="

source /opt/training-env/bin/activate

python eval/quality_gate.py \
    --checkpoint "$CHECKPOINT" \
    --max-perplexity "${MAX_PPL:-20.0}" \
    --output "eval/results/eval-$SLURM_JOB_ID.json"

EXIT=$?
if [ $EXIT -eq 0 ]; then
    echo "PASS — checkpoint promoted"
else
    echo "FAIL — checkpoint rejected"
fi
exit $EXIT
