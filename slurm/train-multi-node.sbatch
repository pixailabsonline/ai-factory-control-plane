#!/bin/bash
#SBATCH --job-name=fsdp-7b-multi
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --signal=B:SIGTERM@120
#SBATCH --requeue
#SBATCH --exclusive

set -euo pipefail

echo "=== Multi-node job $SLURM_JOB_ID ==="
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Nodes: $SLURM_NODELIST ($SLURM_NNODES)"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $(($SLURM_NNODES * $SLURM_GPUS_ON_NODE))"

# Resolve master addr from first node in allocation
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
MASTER_PORT=29500

echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Pre-flight on this node
nvidia-smi --query-gpu=index,name,pcie.link.gen.current,memory.total --format=csv

BAD_PCIE=$(nvidia-smi --query-gpu=index,pcie.link.gen.current --format=csv,noheader,nounits | awk -F', ' '$2 < 3 {print $1}')
if [ -n "$BAD_PCIE" ]; then
    echo "FATAL: GPU(s) $BAD_PCIE have degraded PCIe link on $(hostname). Aborting."
    exit 1
fi

# NCCL config for multi-node
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=NVL

# Checkpoint dir
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/root/ai-factory-control-plane/checkpoints/$SLURM_JOB_ID}"
mkdir -p "$CHECKPOINT_DIR" logs

source /opt/training-env/bin/activate

trap 'echo "SIGTERM received on $(hostname)"; kill -TERM $TRAIN_PID; wait $TRAIN_PID; exit 0' SIGTERM

# Launch with torchrun — each node runs this, torchrun handles rank assignment
torchrun \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    training/fsdp_trainer.py \
    --model "${MODEL:-mistralai/Mistral-7B-v0.1}" \
    --dataset "${DATASET:-wikitext/wikitext-103-raw-v1}" \
    --batch-size "${BATCH_SIZE:-2}" \
    --gradient-accumulation "${GRAD_ACCUM:-4}" \
    --lr "${LR:-2e-5}" \
    --max-steps "${MAX_STEPS:-5000}" \
    --checkpoint-every "${CKPT_EVERY:-500}" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --sharding-strategy "${SHARD_STRATEGY:-FULL_SHARD}" \
    2>&1 | tee "logs/training-$SLURM_JOB_ID-node$SLURM_NODEID.log" &

TRAIN_PID=$!
wait $TRAIN_PID

echo "=== Node $SLURM_NODEID complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
