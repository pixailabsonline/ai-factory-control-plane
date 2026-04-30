#!/bin/bash
set -eo pipefail

LOGFILE="/var/log/ai-factory-setup.log"
exec > >(tee -a "$LOGFILE") 2>&1

TOKEN=$(curl -fsS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
md() {
  curl -fsS -H "X-aws-ec2-metadata-token: ${TOKEN}" "http://169.254.169.254/latest/meta-data/$1"
}

INSTANCE_ID=$(md instance-id)
REGION=$(md placement/region)
HOSTNAME=$(hostname -s)
NODE_IP=$(md local-ipv4)
NODE_ROLE=$(md tags/instance/Role 2>/dev/null || echo "worker")

echo "=== AI Factory Node Provisioning ==="
echo "Instance ID: $INSTANCE_ID"
echo "Instance Type: $(md instance-type)"
echo "Hostname: $HOSTNAME / IP: $NODE_IP / Role: $NODE_ROLE"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# --- Validate GPU hardware ---
echo "Validating GPU hardware..."
if ! nvidia-smi &>/dev/null; then
    echo "FATAL: nvidia-smi not available"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
echo "GPUs detected: $GPU_COUNT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# --- Detect network interface ---
NET_IF=$(ip -o link show up | awk -F': ' '!/lo/{print $2; exit}')
echo "Network interface: $NET_IF"

# --- System packages ---
# Wait for unattended-upgrades to release dpkg lock (race condition on first boot)
systemctl stop unattended-upgrades 2>/dev/null || true
while fuser /var/lib/dpkg/lock-frontend &>/dev/null 2>&1; do sleep 5; done

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    curl wget git jq htop apt-transport-https ca-certificates \
    gnupg lsb-release socat conntrack ipset slurm-wlm munge awscli

ACCOUNT_ID=$(curl -fsS -H "X-aws-ec2-metadata-token: ${TOKEN}" \
  http://169.254.169.254/latest/dynamic/instance-identity/document | jq -r .accountId)
BUCKET="ai-factory-checkpoints-$ACCOUNT_ID"

# --- Activate PyTorch env (pre-installed on Deep Learning AMI) ---
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /opt/pytorch/bin/activate

# Verify PyTorch + GPUs
python3 << 'PYEOF'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPUs")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
PYEOF

# Install HuggingFace stack
pip install --quiet "transformers>=4.38.0" "datasets>=2.17.0" "accelerate>=0.27.0" \
    "sentencepiece>=0.1.99" "protobuf>=4.25.0" "tensorboard>=2.14.0" "boto3>=1.34.0"

# --- Profile setup ---
cat > /etc/profile.d/training.sh <<PROF_EOF
export LD_LIBRARY_PATH=""
source /opt/pytorch/bin/activate
export NCCL_SOCKET_IFNAME=$NET_IF
PROF_EOF

# --- NCCL tuning ---
cat >> /etc/environment <<ENV_EOF
NCCL_DEBUG=WARN
NCCL_SOCKET_IFNAME=$NET_IF
NCCL_P2P_LEVEL=SYS
ENV_EOF

# --- Slurm ---
echo "Configuring Slurm..."

dd if=/dev/urandom bs=1 count=1024 > /etc/munge/munge.key 2>/dev/null
chown munge:munge /etc/munge/munge.key
chmod 400 /etc/munge/munge.key

CPUS=$(nproc)
SOCKETS=$(lscpu | awk '/^Socket\(s\):/{print $2}')
CORES=$(lscpu | awk '/^Core\(s\) per socket:/{print $4}')
THREADS=$(lscpu | awk '/^Thread\(s\) per core:/{print $4}')
MEM=$(free -m | awk '/Mem:/{print int($2*0.9)}')

GPU_FILES=""
for i in $(seq 0 $((GPU_COUNT - 1))); do
    if [ -n "$GPU_FILES" ]; then GPU_FILES="$GPU_FILES,"; fi
    GPU_FILES="$GPU_FILES/dev/nvidia$i"
done

cat > /etc/slurm/slurm.conf <<SLURM_EOF
ClusterName=ai-factory
SlurmctldHost=$HOSTNAME
MpiDefault=none
ProctrackType=proctrack/linuxproc
ReturnToService=2
SlurmctldPidFile=/run/slurmctld.pid
SlurmdPidFile=/run/slurmd.pid
SlurmdSpoolDir=/var/spool/slurmd
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
StateSaveLocation=/var/spool/slurmctld
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
GresTypes=gpu
NodeName=$HOSTNAME CPUs=$CPUS Sockets=$SOCKETS CoresPerSocket=$CORES ThreadsPerCore=$THREADS RealMemory=$MEM Gres=gpu:$GPU_COUNT State=UNKNOWN
PartitionName=gpu Nodes=$HOSTNAME Default=YES MaxTime=INFINITE State=UP
SLURM_EOF

cat > /etc/slurm/gres.conf <<GRES_EOF
NodeName=$HOSTNAME Name=gpu File=$GPU_FILES
GRES_EOF

mkdir -p /var/spool/slurmd /var/spool/slurmctld /var/log/slurm /run/slurm
chown slurm:slurm /var/spool/slurmctld /var/log/slurm

systemctl enable munge && systemctl start munge
systemctl enable slurmctld && systemctl start slurmctld
systemctl enable slurmd && systemctl start slurmd

sleep 2
scontrol update nodename=$HOSTNAME state=idle
echo "Slurm ready: $HOSTNAME with $GPU_COUNT GPUs"
sinfo

# --- Install k3s ---
echo "Installing k3s Kubernetes..."

if [ "$NODE_ROLE" = "master" ]; then
    curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server \
        --disable traefik \
        --disable servicelb \
        --node-ip $NODE_IP \
        --advertise-address $NODE_IP \
        --write-kubeconfig-mode 644" sh -

    # Wait for k3s ready
    until kubectl get nodes &>/dev/null; do sleep 2; done
    echo "k3s server ready"

    # Save join credentials to S3 for workers (aws s3 cp works on both CLI v1 and v2)
    K3S_TOKEN=$(cat /var/lib/rancher/k3s/server/node-token)
    echo "$K3S_TOKEN" | aws s3 cp - "s3://$BUCKET/k3s/node-token" --region "$REGION"
    echo "$NODE_IP" | aws s3 cp - "s3://$BUCKET/k3s/master-ip" --region "$REGION"
    echo "k3s credentials saved to s3://$BUCKET/k3s/"

    # --- Helm ---
    curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

    # --- NVIDIA GPU Operator ---
    echo "Installing NVIDIA GPU Operator..."
    helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
    helm repo update
    helm install gpu-operator nvidia/gpu-operator \
        --namespace gpu-operator \
        --create-namespace \
        --set driver.enabled=false \
        --set toolkit.enabled=true \
        --set devicePlugin.enabled=true \
        --set dcgmExporter.enabled=true \
        --wait --timeout 10m
    echo "NVIDIA GPU Operator installed"

    # --- Kubeflow Training Operator ---
    echo "Installing Kubeflow Training Operator..."
    kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"
    kubectl wait --for=condition=available --timeout=300s \
        deployment/training-operator -n kubeflow
    echo "Kubeflow Training Operator installed"

else
    # Worker: wait for master credentials then join
    echo "Waiting for k3s master credentials..."
    for i in $(seq 1 30); do
        MASTER_IP=$(aws s3 cp s3://$BUCKET/k3s/master-ip - --region "$REGION" 2>/dev/null || true)
        K3S_TOKEN=$(aws s3 cp s3://$BUCKET/k3s/node-token - --region "$REGION" 2>/dev/null || true)
        if [ -n "$MASTER_IP" ] && [ -n "$K3S_TOKEN" ]; then
            break
        fi
        echo "Attempt $i: master not ready, retrying in 10s..."
        sleep 10
    done

    if [ -z "$MASTER_IP" ] || [ -z "$K3S_TOKEN" ]; then
        echo "ERROR: Could not retrieve k3s master credentials after 5 minutes"
        exit 1
    fi

    curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="agent \
        --server https://$MASTER_IP:6443 \
        --token $K3S_TOKEN \
        --node-ip $NODE_IP" sh -
    echo "k3s worker joined cluster"
fi

# --- Clone repo ---
git clone https://github.com/pixailabsonline/ai-factory-control-plane.git \
    /root/ai-factory-control-plane 2>/dev/null || true

# --- CloudWatch agent ---
wget -q https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i amazon-cloudwatch-agent.deb
rm amazon-cloudwatch-agent.deb

cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json <<'CW_EOF'
{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {"file_path": "/var/log/ai-factory-setup.log", "log_group_name": "/ai-factory/training", "log_stream_name": "{instance_id}/setup"},
          {"file_path": "/root/ai-factory-control-plane/training.log", "log_group_name": "/ai-factory/training", "log_stream_name": "{instance_id}/training"}
        ]
      }
    }
  }
}
CW_EOF

/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config -m ec2 -s \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json

# --- GPU health beacon ---
cat > /usr/local/bin/gpu-health-beacon.sh <<'BEACON'
#!/bin/bash
TOKEN=$(curl -fsS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -fsS -H "X-aws-ec2-metadata-token: ${TOKEN}" \
  http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -fsS -H "X-aws-ec2-metadata-token: ${TOKEN}" \
  http://169.254.169.254/latest/dynamic/instance-identity/document | jq -r .region)

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
Description=GPU Health Beacon
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

echo "=== Provisioning complete: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
touch /root/.training-node-ready
