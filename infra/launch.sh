#!/bin/bash
set -euo pipefail

# Launch a p3.8xlarge training instance
# Usage: ./infra/launch.sh [instance-type]

INSTANCE_TYPE="${1:-p3.8xlarge}"
AMI_ID="ami-0a0e5d9c7acc336f1"  # Ubuntu 22.04 LTS, us-east-1
KEY_NAME="ai-factory"
SECURITY_GROUP="ai-factory-training"
REGION="us-east-1"

echo "=== Launching $INSTANCE_TYPE ==="

# Check if key pair exists, create if not
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" &>/dev/null; then
    echo "Creating key pair: $KEY_NAME"
    aws ec2 create-key-pair --key-name "$KEY_NAME" --region "$REGION" \
        --query 'KeyMaterial' --output text > ~/.ssh/$KEY_NAME.pem
    chmod 600 ~/.ssh/$KEY_NAME.pem
fi

# Check if security group exists, create if not
SG_ID=$(aws ec2 describe-security-groups --group-names "$SECURITY_GROUP" --region "$REGION" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "")

if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
    echo "Creating security group: $SECURITY_GROUP"
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SECURITY_GROUP" \
        --description "AI Factory training instances" \
        --region "$REGION" \
        --query 'GroupId' --output text)

    # SSH access
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --region "$REGION" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0

    # NCCL inter-node communication
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --region "$REGION" \
        --protocol tcp --port 29500 --source-group "$SG_ID"

    # All traffic within security group (for NCCL)
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --region "$REGION" \
        --protocol -1 --source-group "$SG_ID"
fi

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --region "$REGION" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ai-factory-training},{Key=Project,Value=ai-factory-control-plane}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance launched: $INSTANCE_ID"
echo "Waiting for running state..."

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo ""
echo "=== Instance Ready ==="
echo "  ID:   $INSTANCE_ID"
echo "  IP:   $PUBLIC_IP"
echo "  Type: $INSTANCE_TYPE"
echo ""
echo "Connect: ssh -i ~/.ssh/$KEY_NAME.pem ubuntu@$PUBLIC_IP"
echo "Setup:   scp -i ~/.ssh/$KEY_NAME.pem -r . ubuntu@$PUBLIC_IP:~/ai-factory-control-plane"
echo "         ssh -i ~/.ssh/$KEY_NAME.pem ubuntu@$PUBLIC_IP 'cd ai-factory-control-plane && bash infra/p3-setup.sh'"
