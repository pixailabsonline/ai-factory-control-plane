#!/bin/bash
set -eo pipefail

: "${HOSTNAME:?HOSTNAME is required}"
: "${NODE_IP:?NODE_IP is required}"
: "${NODE_ROLE:?NODE_ROLE is required}"
: "${NODE_INDEX:?NODE_INDEX is required}"
: "${CLUSTER_SIZE:?CLUSTER_SIZE is required}"
: "${GPU_COUNT:?GPU_COUNT is required}"
: "${BUCKET:?BUCKET is required}"
: "${REGION:?REGION is required}"

echo "=== Slurm batch scheduler bootstrap ==="

if [ "$NODE_ROLE" = "master" ]; then
    MASTER_NODE_IP="$NODE_IP"
else
    MASTER_NODE_IP=$(aws s3 cp "s3://$BUCKET/k3s/master-ip" - --region "$REGION" 2>/dev/null || true)
    if [ -z "$MASTER_NODE_IP" ]; then
        echo "ERROR: Could not resolve k3s master IP for Slurm coordination"
        exit 1
    fi
fi

SLURM_S3_PREFIX="s3://$BUCKET/slurm/$MASTER_NODE_IP"
SLURM_NODE_DIR="/tmp/ai-factory-slurm-nodes"
mkdir -p "$SLURM_NODE_DIR" /etc/slurm /var/spool/slurmd /var/spool/slurmctld /var/log/slurm /run/slurm

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

cat > "$SLURM_NODE_DIR/$HOSTNAME.env" <<NODE_EOF
SLURM_NODE_NAME="$HOSTNAME"
SLURM_NODE_IP="$NODE_IP"
SLURM_NODE_INDEX="$NODE_INDEX"
SLURM_NODE_CPUS="$CPUS"
SLURM_NODE_SOCKETS="$SOCKETS"
SLURM_NODE_CORES="$CORES"
SLURM_NODE_THREADS="$THREADS"
SLURM_NODE_MEM="$MEM"
SLURM_NODE_GPUS="$GPU_COUNT"
SLURM_NODE_GPU_FILES="$GPU_FILES"
NODE_EOF

aws s3 cp "$SLURM_NODE_DIR/$HOSTNAME.env" "$SLURM_S3_PREFIX/nodes/$HOSTNAME.env" --region "$REGION"
echo "Registered Slurm node $HOSTNAME in $SLURM_S3_PREFIX/nodes/"

if [ "$NODE_ROLE" = "master" ]; then
    echo "Creating shared Munge key..."
    dd if=/dev/urandom bs=1 count=1024 > /etc/munge/munge.key 2>/dev/null
    chown munge:munge /etc/munge/munge.key
    chmod 400 /etc/munge/munge.key
    aws s3 cp /etc/munge/munge.key "$SLURM_S3_PREFIX/munge/munge.key" --region "$REGION"

    echo "Waiting for $CLUSTER_SIZE Slurm node registration(s)..."
    for _ in $(seq 1 60); do
        REGISTERED_COUNT=$(aws s3 ls "$SLURM_S3_PREFIX/nodes/" --region "$REGION" 2>/dev/null | awk '/\.env$/ {count++} END {print count+0}')
        if [ "$REGISTERED_COUNT" -ge "$CLUSTER_SIZE" ]; then
            break
        fi
        echo "Registered nodes: $REGISTERED_COUNT/$CLUSTER_SIZE"
        sleep 5
    done

    REGISTERED_COUNT=$(aws s3 ls "$SLURM_S3_PREFIX/nodes/" --region "$REGION" 2>/dev/null | awk '/\.env$/ {count++} END {print count+0}')
    if [ "$REGISTERED_COUNT" -lt "$CLUSTER_SIZE" ]; then
        echo "ERROR: Expected $CLUSTER_SIZE Slurm nodes, saw $REGISTERED_COUNT"
        exit 1
    fi

    rm -rf "$SLURM_NODE_DIR"/registered
    mkdir -p "$SLURM_NODE_DIR"/registered
    aws s3 cp "$SLURM_S3_PREFIX/nodes/" "$SLURM_NODE_DIR/registered/" --recursive --region "$REGION"

    NODE_NAMES=""
    NODE_LINES=""
    for node_file in "$SLURM_NODE_DIR"/registered/*.env; do
        . "$node_file"
        if [ -n "$NODE_NAMES" ]; then NODE_NAMES="$NODE_NAMES,"; fi
        NODE_NAMES="$NODE_NAMES$SLURM_NODE_NAME"
        NODE_LINES="${NODE_LINES}NodeName=$SLURM_NODE_NAME NodeAddr=$SLURM_NODE_IP CPUs=$SLURM_NODE_CPUS Sockets=$SLURM_NODE_SOCKETS CoresPerSocket=$SLURM_NODE_CORES ThreadsPerCore=$SLURM_NODE_THREADS RealMemory=$SLURM_NODE_MEM Gres=gpu:$SLURM_NODE_GPUS State=UNKNOWN"$'\n'
    done

    cat > /etc/slurm/slurm.conf <<SLURM_EOF
ClusterName=ai-factory
SlurmctldHost=$HOSTNAME($NODE_IP)
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
${NODE_LINES}PartitionName=gpu Nodes=$NODE_NAMES Default=YES MaxTime=INFINITE State=UP
PartitionName=slurm-batch Nodes=$NODE_NAMES MaxTime=INFINITE State=UP
SLURM_EOF

    aws s3 cp /etc/slurm/slurm.conf "$SLURM_S3_PREFIX/config/slurm.conf" --region "$REGION"
    echo "Published shared Slurm config for nodes: $NODE_NAMES"
else
    echo "Waiting for shared Munge key and Slurm config..."
    for _ in $(seq 1 240); do
        if aws s3 cp "$SLURM_S3_PREFIX/munge/munge.key" /etc/munge/munge.key --region "$REGION" 2>/dev/null &&
           aws s3 cp "$SLURM_S3_PREFIX/config/slurm.conf" /etc/slurm/slurm.conf --region "$REGION" 2>/dev/null; then
            break
        fi
        sleep 5
    done

    if [ ! -s /etc/munge/munge.key ] || [ ! -s /etc/slurm/slurm.conf ]; then
        echo "ERROR: Could not retrieve shared Slurm config"
        exit 1
    fi

    chown munge:munge /etc/munge/munge.key
    chmod 400 /etc/munge/munge.key
fi

cat > /etc/slurm/gres.conf <<GRES_EOF
NodeName=$HOSTNAME Name=gpu File=$GPU_FILES
GRES_EOF

chown slurm:slurm /var/spool/slurmd /var/spool/slurmctld /var/log/slurm

systemctl enable munge && systemctl start munge

if [ "$NODE_ROLE" = "master" ]; then
    systemctl enable slurmctld && systemctl start slurmctld
fi

systemctl enable slurmd && systemctl start slurmd

sleep 2
if [ "$NODE_ROLE" = "master" ]; then
    for node_file in "$SLURM_NODE_DIR"/registered/*.env; do
        . "$node_file"
        scontrol update nodename="$SLURM_NODE_NAME" state=idle || true
    done
    sinfo
else
    systemctl status slurmd --no-pager || true
fi

echo "Slurm bootstrap complete for $HOSTNAME ($NODE_ROLE)"
