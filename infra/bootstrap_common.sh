#!/bin/bash
set -eo pipefail

echo "=== Common GPU node bootstrap ==="

echo "Validating GPU hardware..."
if ! nvidia-smi &>/dev/null; then
    echo "FATAL: nvidia-smi not available"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
export GPU_COUNT
echo "GPUs detected: $GPU_COUNT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

NET_IF=$(ip -o link show up | awk -F': ' '!/lo/{print $2; exit}')
export NET_IF
echo "Network interface: $NET_IF"

systemctl stop unattended-upgrades 2>/dev/null || true
while fuser /var/lib/dpkg/lock-frontend &>/dev/null 2>&1; do sleep 5; done

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    curl wget git jq htop apt-transport-https ca-certificates \
    gnupg lsb-release socat conntrack ipset slurm-wlm munge awscli

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /opt/pytorch/bin/activate

python3 << 'PYEOF'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPUs")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
PYEOF

pip install --quiet "transformers>=4.38.0" "datasets>=2.17.0" "accelerate>=0.27.0" \
    "sentencepiece>=0.1.99" "protobuf>=4.25.0" "tensorboard>=2.14.0" "boto3>=1.34.0"

cat > /etc/profile.d/training.sh <<PROF_EOF
export LD_LIBRARY_PATH=""
source /opt/pytorch/bin/activate
export NCCL_SOCKET_IFNAME=$NET_IF
PROF_EOF

cat >> /etc/environment <<ENV_EOF
NCCL_DEBUG=WARN
NCCL_SOCKET_IFNAME=$NET_IF
NCCL_P2P_LEVEL=SYS
ENV_EOF

git clone https://github.com/pixailabsonline/ai-factory-control-plane.git \
    /root/ai-factory-control-plane 2>/dev/null || true

# --- Mount FSx Lustre if DNS name is present in instance tags ---
LUSTRE_DNS=$(aws ec2 describe-tags \
  --filters "Name=resource-id,Values=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)" \
            "Name=key,Values=LustreDns" \
  --query "Tags[0].Value" --output text 2>/dev/null || echo "")

LUSTRE_MOUNT=$(aws ec2 describe-tags \
  --filters "Name=resource-id,Values=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)" \
            "Name=key,Values=LustreMountName" \
  --query "Tags[0].Value" --output text 2>/dev/null || echo "")

if [ -n "$LUSTRE_DNS" ] && [ "$LUSTRE_DNS" != "None" ]; then
  apt-get install -y lustre-client-modules-$(uname -r) lustre-utils 2>/dev/null || \
    apt-get install -y lustre-client-modules lustre-utils 2>/dev/null || true
  mkdir -p /mnt/lustre
  mount -t lustre "${LUSTRE_DNS}@tcp:/${LUSTRE_MOUNT}" /mnt/lustre
  echo "${LUSTRE_DNS}@tcp:/${LUSTRE_MOUNT} /mnt/lustre lustre defaults,_netdev 0 0" >> /etc/fstab
  mkdir -p /mnt/lustre/checkpoints /mnt/lustre/datasets /mnt/lustre/logs
  chmod 1777 /mnt/lustre/checkpoints /mnt/lustre/datasets /mnt/lustre/logs
  echo "Lustre mounted at /mnt/lustre"
fi
