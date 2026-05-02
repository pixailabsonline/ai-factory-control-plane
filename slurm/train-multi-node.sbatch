#!/bin/bash
#SBATCH --job-name=fsdp-train-multi
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=/tmp/train-%x-%j.out
#SBATCH --error=/tmp/train-%x-%j.err
#SBATCH --signal=B:SIGTERM@120
#SBATCH --requeue
#SBATCH --exclusive

set -euo pipefail

S3_BUCKET="${S3_BUCKET:-ai-factory-checkpoints-737213639346}"
MODEL_NAME="${MODEL:-gpt2-medium}"
S3_RUN_KEY="${S3_RUN_KEY:-runs/$MODEL_NAME}"
S3_RUN_ROOT="${S3_RUN_ROOT:-s3://$S3_BUCKET/$S3_RUN_KEY}"

# Use Lustre if mounted, otherwise fall back to $HOME (persistent on EBS)
if mountpoint -q /mnt/lustre 2>/dev/null; then
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/lustre/checkpoints/$MODEL_NAME}"
    LOG_DIR="/mnt/lustre/logs"
    USE_LUSTRE=true
else
    CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/checkpoints/$MODEL_NAME}"
    LOG_DIR="$HOME/ai-factory-control-plane/logs"
    USE_LUSTRE=false
fi

echo "=== Multi-node job $SLURM_JOB_ID ==="
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Nodes: $SLURM_NODELIST ($SLURM_NNODES x 1 GPU = 2 GPUs total)"
echo "Checkpoints: $CHECKPOINT_DIR (Lustre: $USE_LUSTRE)"

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
mkdir -p '"$LOG_DIR"' '"$CHECKPOINT_DIR"'

source /opt/pytorch/bin/activate

echo "Node $(hostname) rank=$SLURM_PROCID — GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Only pass S3 args when not on Lustre — on Lustre the shared fs is the source of truth
S3_ARGS=""
if [ "'"$USE_LUSTRE"'" != "true" ]; then
    S3_ARGS="--s3-bucket '"$S3_BUCKET"' --s3-prefix '"$S3_RUN_KEY"'/checkpoints"
fi

torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=$SLURM_PROCID \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    training/fsdp_trainer.py \
    --model        "'"$MODEL_NAME"'" \
    --dataset      "'"${DATASET:-wikitext/wikitext-2-raw-v1}"'" \
    --batch-size   "'"${BATCH_SIZE:-2}"'" \
    --gradient-accumulation "'"${GRAD_ACCUM:-4}"'" \
    --lr           "'"${LR:-1e-5}"'" \
    --warmup-steps "'"${WARMUP_STEPS:-20}"'" \
    --max-steps    "'"${MAX_STEPS:-5000}"'" \
    --checkpoint-every "'"${CKPT_EVERY:-500}"'" \
    --checkpoint-dir   "'"$CHECKPOINT_DIR"'" \
    $S3_ARGS \
    --sharding-strategy "'"${SHARD_STRATEGY:-FULL_SHARD}"'" \
    2>&1 | tee "'"$LOG_DIR"'/training-$SLURM_JOB_ID-node$SLURM_PROCID.log"
'

echo "=== Training complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# Auto-submit eval against the latest checkpoint in S3
REPO_DIR="$HOME/ai-factory-control-plane"
EVAL_JOB=$(sbatch --parsable \
    --nodelist="$MASTER_ADDR" \
    --export=ALL,S3_BUCKET="$S3_BUCKET",MODEL_NAME="$MODEL_NAME",S3_RUN_ROOT="$S3_RUN_ROOT",MAX_PPL="${MAX_PPL:-90.0}" \
    "$REPO_DIR/slurm/eval.sbatch")
echo "Submitted eval job $EVAL_JOB"
