#!/bin/bash
set -euo pipefail

LOGFILE="/var/log/ai-factory-setup.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=== AI Factory Node Provisioning ==="
echo "Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
echo "Instance Type: $(curl -s http://169.254.169.254/latest/meta-data/instance-type)"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# --- Validate GPU hardware before spending time on setup ---
echo "Validating GPU hardware..."
if ! nvidia-smi &>/dev/null; then
    echo "FATAL: nvidia-smi not available — wrong AMI or driver issue"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
echo "GPUs detected: $GPU_COUNT"

nvidia-smi --query-gpu=index,name,pcie.link.gen.current,memory.total --format=csv
echo ""
echo "NVLink topology:"
nvidia-smi topo -m
echo ""

# Validate PCIe link speed (V100 should be Gen3)
BAD_PCIE=$(nvidia-smi --query-gpu=index,pcie.link.gen.current --format=csv,noheader,nounits | awk -F', ' '$2 < 3 {print $1}')
if [ -n "$BAD_PCIE" ]; then
    echo "WARNING: GPUs with degraded PCIe link: $BAD_PCIE"
fi

# --- System packages ---
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv git htop jq

# --- CloudWatch agent for log shipping ---
wget -q https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i amazon-cloudwatch-agent.deb
rm amazon-cloudwatch-agent.deb

cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json <<'CW_EOF'
{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/ai-factory-setup.log",
            "log_group_name": "/ai-factory/training",
            "log_stream_name": "{instance_id}/setup"
          },
          {
            "file_path": "/root/ai-factory-control-plane/training.log",
            "log_group_name": "/ai-factory/training",
            "log_stream_name": "{instance_id}/training"
          },
          {
            "file_path": "/root/ai-factory-control-plane/nccl.log",
            "log_group_name": "/ai-factory/training",
            "log_stream_name": "{instance_id}/nccl"
          }
        ]
      }
    }
  }
}
CW_EOF

/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config -m ec2 -s \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json

# --- Python training environment ---
python3 -m venv /opt/training-env
source /opt/training-env/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.38.0 datasets>=2.17.0 accelerate>=0.27.0 \
    sentencepiece>=0.1.99 protobuf>=4.25.0 tensorboard>=2.15.0 vllm>=0.4.0 boto3

# --- Verify PyTorch sees GPUs ---
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
assert torch.cuda.device_count() >= 1, 'No GPUs found'
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPUs')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
assert torch.distributed.is_nccl_available(), 'NCCL not available'
print('NCCL: OK')
"

# --- Clone repo ---
git clone https://github.com/pixailabsonline/ai-factory-control-plane.git /root/ai-factory-control-plane || true

# --- Go (for control plane) ---
if ! command -v go &>/dev/null; then
    wget -q https://go.dev/dl/go1.24.1.linux-amd64.tar.gz
    tar -C /usr/local -xzf go1.24.1.linux-amd64.tar.gz
    rm go1.24.1.linux-amd64.tar.gz
    echo 'export PATH=$PATH:/usr/local/go/bin' >> /etc/profile.d/go.sh
fi

export PATH=$PATH:/usr/local/go/bin
cd /root/ai-factory-control-plane && go build ./...

# --- NCCL tuning ---
cat >> /etc/environment <<'EOF'
NCCL_DEBUG=WARN
NCCL_SOCKET_IFNAME=eth0
NCCL_P2P_LEVEL=NVL
EOF

# --- GPU health beacon (posts GPU util to CloudWatch every 60s) ---
cat > /usr/local/bin/gpu-health-beacon.sh <<'BEACON'
#!/bin/bash
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)

while true; do
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{s+=$1; n++} END {print s/n}')
    GPU_MEM=$(nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits | awk '{s+=$1; n++} END {print s/n}')
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | awk '{s+=$1; n++} END {print s/n}')

    aws cloudwatch put-metric-data --region "$REGION" --namespace AIFactory \
        --metric-data "[
            {\"MetricName\":\"GPUUtilization\",\"Value\":$GPU_UTIL,\"Unit\":\"Percent\",\"Dimensions\":[{\"Name\":\"InstanceId\",\"Value\":\"$INSTANCE_ID\"}]},
            {\"MetricName\":\"GPUMemoryUtilization\",\"Value\":$GPU_MEM,\"Unit\":\"Percent\",\"Dimensions\":[{\"Name\":\"InstanceId\",\"Value\":\"$INSTANCE_ID\"}]},
            {\"MetricName\":\"GPUTemperature\",\"Value\":$GPU_TEMP,\"Unit\":\"None\",\"Dimensions\":[{\"Name\":\"InstanceId\",\"Value\":\"$INSTANCE_ID\"}]}
        ]" 2>/dev/null

    sleep 60
done
BEACON

chmod +x /usr/local/bin/gpu-health-beacon.sh

cat > /etc/systemd/system/gpu-health-beacon.service <<'SVC'
[Unit]
Description=GPU Health Beacon — posts metrics to CloudWatch
After=network.target

[Service]
ExecStart=/usr/local/bin/gpu-health-beacon.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SVC

systemctl daemon-reload
systemctl enable gpu-health-beacon
systemctl start gpu-health-beacon

# --- Signal ready ---
echo "=== Provisioning complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
touch /root/.training-node-ready
