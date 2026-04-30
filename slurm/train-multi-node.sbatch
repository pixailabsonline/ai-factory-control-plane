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

echo "=== Multi-node job $SLURM_JOB_ID ==="
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Nodes: $SLURM_NODELIST ($SLURM_NNODES x 1 GPU = 2 GPUs total)"

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

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/tmp/checkpoints/$SLURM_JOB_ID}"
mkdir -p "$CHECKPOINT_DIR"

# Seed checkpoint from a previous run so all ranks resume at the same step
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_STEP=$(basename "$RESUME_FROM" | sed 's/checkpoint-//')
    RESUME_DEST="$CHECKPOINT_DIR/checkpoint-$RESUME_STEP"
    echo "Seeding resume checkpoint step=$RESUME_STEP from $RESUME_FROM"

    cp -r "$RESUME_FROM" "$RESUME_DEST"

    # Push to every other node so all ranks see the same checkpoint
    for NODE in $(scontrol show hostnames "$SLURM_NODELIST" | tail -n +2); do
        ssh "$NODE" "mkdir -p '$CHECKPOINT_DIR'"
        rsync -a "$RESUME_FROM/" "${NODE}:${RESUME_DEST}/"
    done
    echo "Checkpoint seeded on all nodes"
fi

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
mkdir -p logs '"$CHECKPOINT_DIR"'

source /opt/pytorch/bin/activate

echo "Node $(hostname) rank=$SLURM_PROCID — GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

    torchrun \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=$SLURM_PROCID \
    --master_addr='"$MASTER_ADDR"' \
    --master_port='"$MASTER_PORT"' \
    training/fsdp_trainer.py \
    --model "'"${MODEL:-gpt2-medium}"'" \
    --dataset "'"${DATASET:-wikitext/wikitext-2-raw-v1}"'" \
    --batch-size "'"${BATCH_SIZE:-2}"'" \
    --gradient-accumulation "'"${GRAD_ACCUM:-4}"'" \
    --lr "'"${LR:-1e-5}"'" \
    --warmup-steps "'"${WARMUP_STEPS:-20}"'" \
    --max-steps "'"${MAX_STEPS:-5000}"'" \
    --checkpoint-every "'"${CKPT_EVERY:-500}"'" \
    --checkpoint-dir "'"$CHECKPOINT_DIR"'" \
    --sharding-strategy "'"${SHARD_STRATEGY:-FULL_SHARD}"'" \
    2>&1 | tee "logs/training-$SLURM_JOB_ID-node$SLURM_PROCID.log"
'

echo "=== Training complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# Submit eval gate for the final checkpoint (runs on master where checkpoint lives)
FINAL_CKPT=$(ls -d "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -n1 || true)
if [ -n "$FINAL_CKPT" ]; then
    REPO_DIR="$HOME/ai-factory-control-plane"
    EVAL_JOB=$(sbatch --parsable \
        --nodelist="$MASTER_ADDR" \
        --export=ALL,CHECKPOINT="$FINAL_CKPT",MAX_PPL="${MAX_PPL:-90.0}" \
        "$REPO_DIR/slurm/eval.sbatch")
    echo "Submitted eval job $EVAL_JOB for checkpoint: $FINAL_CKPT"
else
    echo "No checkpoint found in $CHECKPOINT_DIR — skipping eval"
fi
