#!/bin/bash
#SBATCH --job-name=fsdp-profile
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

echo "=== Profiling job $SLURM_JOB_ID ==="
echo "Collecting torch.profiler traces + NCCL communication breakdown"

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

PROFILE_DIR="profiling/$SLURM_JOB_ID"
mkdir -p "$PROFILE_DIR" logs

source /opt/training-env/bin/activate

# Short run with profiling enabled
torchrun \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --nnodes=1 \
    training/fsdp_trainer.py \
    --model "${MODEL:-mistralai/Mistral-7B-v0.1}" \
    --batch-size "${BATCH_SIZE:-2}" \
    --gradient-accumulation 1 \
    --max-steps 50 \
    --checkpoint-every 999999 \
    --checkpoint-dir "$PROFILE_DIR/checkpoints" \
    2>&1 | tee "logs/profile-$SLURM_JOB_ID.log"

echo "=== Profile complete ==="
echo "Traces: $PROFILE_DIR"
