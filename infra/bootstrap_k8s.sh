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

    echo "Clearing stale Slurm bootstrap state for master $NODE_IP"
    aws s3 rm "s3://$BUCKET/slurm/$NODE_IP/" --recursive --region "$REGION" || true

    K3S_TOKEN=$(cat /var/lib/rancher/k3s/server/node-token)
    echo "$K3S_TOKEN" | aws s3 cp - "s3://$BUCKET/k3s/node-token" --region "$REGION"
    echo "$NODE_IP" | aws s3 cp - "s3://$BUCKET/k3s/master-ip" --region "$REGION"
    echo "k3s credentials saved to s3://$BUCKET/k3s/"

    EXPECTED_NODES="${CLUSTER_SIZE:-1}"
    echo "Waiting for $EXPECTED_NODES Kubernetes substrate node(s)..."
    for i in $(seq 1 60); do
        NODE_COUNT=$(kubectl get nodes --no-headers 2>/dev/null | awk 'END {print NR+0}')
        if [ "$NODE_COUNT" -ge "$EXPECTED_NODES" ]; then
            break
        fi
        echo "Kubernetes nodes ready: $NODE_COUNT/$EXPECTED_NODES"
        sleep 5
    done

    kubectl label nodes --all \
        ai-factory/capacity-owner=slurm-batch \
        ai-factory/slurm-pool=gpu \
        ai-factory/scheduler=slurm \
        --overwrite

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

    kubectl taint nodes --all ai-factory/gpu-owner=slurm-batch:NoSchedule --overwrite || true
    echo "GPU substrate nodes marked as Slurm-owned batch capacity"

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
