#!/bin/bash
set -eo pipefail

echo "=== Observability bootstrap ==="

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

cat > /usr/local/bin/gpu-health-beacon.sh <<'BEACON'
#!/bin/bash
TOKEN=$(curl -fsS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -fsS -H "X-aws-ec2-metadata-token: ${TOKEN}" \
  http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -fsS -H "X-aws-ec2-metadata-token: ${TOKEN}" \
  http://169.254.169.254/latest/meta-data/placement/region)

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
