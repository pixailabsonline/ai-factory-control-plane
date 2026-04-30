#!/bin/bash
set -eo pipefail

: "${NODE_ROLE:?NODE_ROLE is required}"
: "${NODE_IP:?NODE_IP is required}"
: "${REGION:?REGION is required}"
: "${BUCKET:?BUCKET is required}"

echo "=== Kubernetes substrate bootstrap ==="

if [ "$NODE_ROLE" = "master" ]; then
    curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server \
        --disable traefik \
        --disable servicelb \
        --node-ip $NODE_IP \
        --advertise-address $NODE_IP \
        --write-kubeconfig-mode 644" sh -

    until kubectl get nodes &>/dev/null; do sleep 2; done
    echo "k3s server ready"

    K3S_TOKEN=$(cat /var/lib/rancher/k3s/server/node-token)
    echo "$K3S_TOKEN" | aws s3 cp - "s3://$BUCKET/k3s/node-token" --region "$REGION"
    echo "$NODE_IP" | aws s3 cp - "s3://$BUCKET/k3s/master-ip" --region "$REGION"
    echo "k3s credentials saved to s3://$BUCKET/k3s/"

    curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

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

else
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
