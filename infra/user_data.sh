#!/bin/bash
set -euo pipefail

exec > /var/log/user-data.log 2>&1
echo "=== AI Factory Training Node Setup ==="
date

# NVIDIA drivers
echo "Installing NVIDIA drivers..."
apt-get update -qq
apt-get install -y -qq linux-headers-$(uname -r)
apt-get install -y -qq nvidia-driver-535 nvidia-utils-535

# Python + system deps
echo "Installing system dependencies..."
apt-get install -y -qq python3-pip python3-venv git htop awscli

# Create training venv
echo "Setting up Python environment..."
python3 -m venv /opt/training-env
source /opt/training-env/bin/activate

# PyTorch with CUDA
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Training dependencies
pip install transformers>=4.38.0 datasets>=2.17.0 accelerate>=0.27.0 \
    sentencepiece>=0.1.99 protobuf>=4.25.0 tensorboard>=2.15.0

# Inference dependencies
pip install flask

# Clone the repo
git clone https://github.com/pixailabsonline/ai-factory-control-plane.git /root/ai-factory-control-plane
cd /root/ai-factory-control-plane

# Install Go (for control plane)
wget -q https://go.dev/dl/go1.24.1.linux-amd64.tar.gz
tar -C /usr/local -xzf go1.24.1.linux-amd64.tar.gz
rm go1.24.1.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> /etc/profile.d/go.sh

# Build Go components
export PATH=$PATH:/usr/local/go/bin
go build ./...

# NCCL config
cat >> /etc/environment <<'EOF'
NCCL_DEBUG=WARN
NCCL_SOCKET_IFNAME=eth0
NCCL_IB_DISABLE=0
NCCL_P2P_LEVEL=NVL
EOF

# Signal ready
echo "=== Setup complete $(date) ==="
touch /root/.training-node-ready
