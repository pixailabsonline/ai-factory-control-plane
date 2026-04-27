#!/bin/bash
set -euo pipefail

# Setup script for p3.8xlarge (4x V100 16GB)
# Run once after launching the instance

echo "=== GPU Training Node Setup ==="

# Verify GPUs
echo "Checking GPUs..."
nvidia-smi
echo ""

# Verify NVLink
echo "Checking NVLink topology..."
nvidia-smi topo -m
echo ""

# Verify PCIe link speed (must be Gen3 for V100)
echo "Checking PCIe link speeds..."
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current --format=csv
echo ""

# Install system deps
echo "Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git htop nvtop

# Create venv
echo "Creating Python environment..."
python3 -m venv /opt/training-env
source /opt/training-env/bin/activate

# Install PyTorch with CUDA
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install training deps
pip install -r /root/ai-factory-control-plane/training/requirements.txt

# Install inference deps
pip install flask

# Verify torch sees GPUs
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1024**3:.0f}GB)')
print(f'NCCL available: {torch.distributed.is_nccl_available()}')
"

# Set NCCL env vars
cat >> /etc/environment <<'NCCL_EOF'
NCCL_DEBUG=WARN
NCCL_SOCKET_IFNAME=eth0
NCCL_IB_DISABLE=0
NCCL_P2P_LEVEL=NVL
NCCL_EOF

echo ""
echo "=== Setup complete ==="
echo "Activate with: source /opt/training-env/bin/activate"
echo "Run training with: torchrun --nproc_per_node=4 training/fsdp_trainer.py"
