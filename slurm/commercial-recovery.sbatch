#!/bin/bash
#SBATCH --job-name=commercial-recovery
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=/tmp/commercial-recovery-%j.out
#SBATCH --error=/tmp/commercial-recovery-%j.err
#SBATCH --signal=B:SIGTERM@120
#SBATCH --requeue
#SBATCH --exclusive

set -euo pipefail

export S3_BUCKET="${S3_BUCKET:-ai-factory-checkpoints-737213639346}"
export MODEL_NAME="${MODEL:-gpt2-medium}"
export S3_RUN_KEY="${S3_RUN_KEY:-runs/$MODEL_NAME/recovery}"
export S3_RUN_ROOT="${S3_RUN_ROOT:-s3://$S3_BUCKET/$S3_RUN_KEY}"
export MAX_STEPS="${MAX_STEPS:-1000}"
export STOP_AFTER_STEP="${STOP_AFTER_STEP:-200}"
export CKPT_EVERY="${CKPT_EVERY:-100}"
export DATASET="${DATASET:-wikitext/wikitext-2-raw-v1}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export LR="${LR:-1e-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-20}"
export SHARD_STRATEGY="${SHARD_STRATEGY:-FULL_SHARD}"

# Lustre = fast shared scratch during the run (all nodes see the same path, no cross-node copies)
# S3    = durable store — always written regardless of Lustre, survives instance termination
if mountpoint -q /mnt/lustre 2>/dev/null; then
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/lustre/checkpoints/$MODEL_NAME/commercial-recovery}"
    LOG_DIR="/mnt/lustre/logs"
    USE_LUSTRE=true
else
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/checkpoints/$MODEL_NAME/commercial-recovery}"
    LOG_DIR="$HOME/ai-factory-control-plane/logs"
    USE_LUSTRE=false
fi

echo "=== Commercial recovery job $SLURM_JOB_ID ==="
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Nodes: $SLURM_NODELIST ($SLURM_NNODES x 1 GPU = 2 GPUs total)"
echo "Checkpoints: $CHECKPOINT_DIR → S3: s3://$S3_BUCKET/$S3_RUN_KEY/checkpoints (Lustre: $USE_LUSTRE)"
echo "Recovery stop step: $STOP_AFTER_STEP"
echo "Max steps: $MAX_STEPS"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
MASTER_PORT=29500
echo "Master: $MASTER_ADDR:$MASTER_PORT"

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800

srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    bash -lc '
set -euo pipefail

REPO_DIR="$HOME/ai-factory-control-plane"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/pixailabsonline/ai-factory-control-plane.git "$REPO_DIR"
else
    git -C "$REPO_DIR" pull --ff-only
fi
cd "$REPO_DIR"
mkdir -p "'"$LOG_DIR"'" "'"$CHECKPOINT_DIR"'"

source /opt/pytorch/bin/activate

echo "Node $(hostname) rank=$SLURM_PROCID — GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

phase_one() {
    torchrun \
        --nproc_per_node=1 \
        --nnodes=2 \
        --node_rank=$SLURM_PROCID \
        --master_addr="'"$MASTER_ADDR"'" \
        --master_port="'"$MASTER_PORT"'" \
        training/fsdp_trainer.py \
        --model        "'"$MODEL_NAME"'" \
        --dataset      "'"$DATASET"'" \
        --batch-size   "'"$BATCH_SIZE"'" \
        --gradient-accumulation "'"$GRAD_ACCUM"'" \
        --lr           "'"$LR"'" \
        --warmup-steps "'"$WARMUP_STEPS"'" \
        --max-steps    "'"$MAX_STEPS"'" \
        --checkpoint-every "'"$CKPT_EVERY"'" \
        --checkpoint-dir   "'"$CHECKPOINT_DIR"'" \
        --s3-bucket        "'"$S3_BUCKET"'" \
        --s3-prefix        "'"$S3_RUN_KEY"'/checkpoints" \
        --sharding-strategy "'"$SHARD_STRATEGY"'" \
        --stop-after-step "'"$STOP_AFTER_STEP"'" \
        2>&1 | tee "'"$LOG_DIR"'/commercial-recovery-'"$SLURM_JOB_ID"'-phase1-node'"$SLURM_PROCID"'.log"
}

phase_two() {
    torchrun \
        --nproc_per_node=1 \
        --nnodes=2 \
        --node_rank=$SLURM_PROCID \
        --master_addr="'"$MASTER_ADDR"'" \
        --master_port="'"$MASTER_PORT"'" \
        training/fsdp_trainer.py \
        --model        "'"$MODEL_NAME"'" \
        --dataset      "'"$DATASET"'" \
        --batch-size   "'"$BATCH_SIZE"'" \
        --gradient-accumulation "'"$GRAD_ACCUM"'" \
        --lr           "'"$LR"'" \
        --warmup-steps "'"$WARMUP_STEPS"'" \
        --max-steps    "'"$MAX_STEPS"'" \
        --checkpoint-every "'"$CKPT_EVERY"'" \
        --checkpoint-dir   "'"$CHECKPOINT_DIR"'" \
        --s3-bucket        "'"$S3_BUCKET"'" \
        --s3-prefix        "'"$S3_RUN_KEY"'/checkpoints" \
        --sharding-strategy "'"$SHARD_STRATEGY"'" \
        2>&1 | tee "'"$LOG_DIR"'/commercial-recovery-'"$SLURM_JOB_ID"'-phase2-node'"$SLURM_PROCID"'.log"
}

echo "Phase 1: deliberate interruption at step '"$STOP_AFTER_STEP"'"
phase_one

echo "Phase 2: resume from checkpoint and continue to step '"$MAX_STEPS"'"
phase_two
'

echo "=== Commercial recovery complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

REPO_DIR="$HOME/ai-factory-control-plane"
EVAL_JOB=$(sbatch --parsable \
    --nodelist="$MASTER_ADDR" \
    --export=ALL,S3_BUCKET="$S3_BUCKET",MODEL_NAME="$MODEL_NAME",S3_RUN_ROOT="$S3_RUN_ROOT",MAX_PPL="${MAX_PPL:-90.0}" \
    "$REPO_DIR/slurm/eval.sbatch")
echo "Submitted eval job $EVAL_JOB"
