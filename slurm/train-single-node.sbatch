#!/bin/bash
#SBATCH --job-name=fsdp-7b-single
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --signal=B:SIGTERM@120
#SBATCH --requeue

set -euo pipefail

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Node: $SLURM_NODELIST"

# Pre-flight: validate GPU hardware before burning compute
nvidia-smi --query-gpu=index,name,pcie.link.gen.current,memory.total --format=csv
echo ""

BAD_PCIE=$(nvidia-smi --query-gpu=index,pcie.link.gen.current --format=csv,noheader,nounits | awk -F', ' '$2 < 3 {print $1}')
if [ -n "$BAD_PCIE" ]; then
    echo "FATAL: GPU(s) $BAD_PCIE have degraded PCIe link. Aborting."
    exit 1
fi

# NCCL config
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=eth0

# Checkpoint dir on shared storage (or local + S3 sync)
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/root/ai-factory-control-plane/checkpoints/$SLURM_JOB_ID}"
mkdir -p "$CHECKPOINT_DIR" logs

# Activate training env
source /opt/training-env/bin/activate

# Handle SIGTERM from Slurm (preemption or timeout) — save checkpoint before dying
trap 'echo "SIGTERM received, saving checkpoint..."; kill -TERM $TRAIN_PID; wait $TRAIN_PID; exit 0' SIGTERM

# Launch training
torchrun \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --nnodes=1 \
    training/fsdp_trainer.py \
    --model "${MODEL:-mistralai/Mistral-7B-v0.1}" \
    --dataset "${DATASET:-wikitext/wikitext-103-raw-v1}" \
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
