#!/bin/bash
set -eo pipefail

: "${HOSTNAME:?HOSTNAME is required}"
: "${GPU_COUNT:?GPU_COUNT is required}"

echo "=== Slurm batch scheduler bootstrap ==="

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
