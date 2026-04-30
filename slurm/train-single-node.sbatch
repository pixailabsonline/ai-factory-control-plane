#!/bin/bash
#SBATCH --job-name=fsdp-train-single
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=/tmp/train-%x-%j.out
#SBATCH --error=/tmp/train-%x-%j.err
#SBATCH --signal=B:SIGTERM@120
#SBATCH --requeue

set -euo pipefail

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Node: $SLURM_NODELIST"

nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

REPO_DIR="$HOME/ai-factory-control-plane"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/pixailabsonline/ai-factory-control-plane.git "$REPO_DIR"
else
    git -C "$REPO_DIR" pull --ff-only
fi
cd "$REPO_DIR"
mkdir -p logs

source /opt/pytorch/bin/activate

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/tmp/checkpoints/$SLURM_JOB_ID}"
mkdir -p "$CHECKPOINT_DIR"

trap 'echo "SIGTERM received, saving checkpoint..."; kill -TERM $TRAIN_PID 2>/dev/null; wait $TRAIN_PID; exit 0' SIGTERM

# DistilGPT2 is the default single-node model because it fits the current 1-GPU node reliably.
torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    training/fsdp_trainer.py \
    --model "${MODEL:-distilgpt2}" \
    --dataset "${DATASET:-wikitext/wikitext-2-raw-v1}" \
    --batch-size "${BATCH_SIZE:-2}" \
    --gradient-accumulation "${GRAD_ACCUM:-8}" \
    --lr "${LR:-2e-5}" \
    --max-steps "${MAX_STEPS:-5000}" \
    --checkpoint-every "${CKPT_EVERY:-500}" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --sharding-strategy "${SHARD_STRATEGY:-FULL_SHARD}" \
    2>&1 | tee "logs/training-$SLURM_JOB_ID.log" &

TRAIN_PID=$!
wait $TRAIN_PID

echo "=== Job complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
