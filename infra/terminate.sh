#!/bin/bash
set -euo pipefail

# Terminate training instances to stop burning credits
# Usage: ./infra/terminate.sh [instance-id]
#   No args = list running training instances
#   With arg = terminate that instance

REGION="us-east-1"

if [ $# -eq 0 ]; then
    echo "Running ai-factory instances:"
    aws ec2 describe-instances \
        --region "$REGION" \
        --filters "Name=tag:Project,Values=ai-factory-control-plane" "Name=instance-state-name,Values=running,stopped" \
        --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,PublicIpAddress,LaunchTime]' \
        --output table
    echo ""
    echo "To terminate: ./infra/terminate.sh <instance-id>"
    exit 0
fi

INSTANCE_ID="$1"

echo "Terminating $INSTANCE_ID..."
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
    --query 'TerminatingInstances[0].[InstanceId,CurrentState.Name]' --output text

echo "Done. Instance will be terminated shortly."
