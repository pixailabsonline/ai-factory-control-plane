#!/bin/bash
set -euo pipefail

# Quick cost check — what's running and how much it's burning
REGION="us-east-1"

echo "=== Running Instances ==="
aws ec2 describe-instances \
    --region "$REGION" \
    --filters "Name=instance-state-name,Values=running" \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,Tags[?Key==`Name`].Value|[0],LaunchTime]' \
    --output table

echo ""
echo "=== Cost Rates ==="
echo "  p3.8xlarge:  \$12.24/hr  (\$293.76/day)"
echo "  p3.16xlarge: \$24.48/hr  (\$587.52/day)"
echo "  g5.xlarge:   \$1.006/hr  (\$24.14/day)"
echo "  t3.micro:    \$0.0104/hr (\$0.25/day)"

echo ""
echo "=== GPU Quotas ==="
echo "P instances (training):"
aws service-quotas get-service-quota --service-code ec2 --quota-code L-417A185B --region "$REGION" \
    --query 'Quota.Value' --output text 2>/dev/null || echo "  (could not query)"

echo "G instances (inference/render):"
aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --region "$REGION" \
    --query 'Quota.Value' --output text 2>/dev/null || echo "  (could not query)"
