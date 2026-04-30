# Stage 3 Slurm Substrate Proof

Stage 3 makes the architecture operationally explicit:

- Kubernetes is the substrate.
- NVIDIA GPU Operator enables and monitors GPU nodes.
- Slurm is the researcher-facing batch scheduler.
- The default GPU pool is marked as Slurm-owned batch capacity.
- General Kubernetes workloads are not allowed to silently consume the Slurm GPU pool.

## What Changed

The bootstrap now builds a real controller/worker Slurm shape:

- The instance tagged `Role=master` runs the k3s server and `slurmctld`.
- Worker instances join k3s as agents and run `slurmd`.
- Nodes register Slurm CPU, memory, GPU, and private IP metadata through S3.
- The master generates one shared `slurm.conf` and one shared Munge key.
- Workers download the shared Slurm config and Munge key before starting `slurmd`.
- Slurm exposes `gpu` and `slurm-batch` partitions over the same batch-owned nodes.
- The master clears stale Slurm bootstrap state before publishing k3s join credentials, so workers cannot race against cleanup.

The GPU ownership boundary is encoded in two layers:

- EC2 tags: `CapacityOwner=slurm-batch`, `SlurmPool=gpu`.
- Kubernetes labels: `ai-factory/capacity-owner=slurm-batch`, `ai-factory/slurm-pool=gpu`, `ai-factory/scheduler=slurm`.

After the NVIDIA GPU Operator is installed, the bootstrap taints substrate nodes with:

```bash
ai-factory/gpu-owner=slurm-batch:NoSchedule
```

That taint prevents arbitrary Kubernetes pods from landing on the Slurm GPU pool unless an operator explicitly grants a toleration. Slurm jobs are not Kubernetes pods, so normal `sbatch`/`srun` operation is unaffected.

## Validation Commands

Run these from the provisioned master node:

```bash
make train-smoke
make substrate-status
make slurm-status
make platform-status
```

Expected substrate evidence:

```bash
kubectl get nodes -L ai-factory/capacity-owner,ai-factory/slurm-pool,ai-factory/scheduler -o wide
kubectl get pods -n gpu-operator -o wide
```

Expected Slurm evidence:

```bash
sinfo -Nel
squeue -o "%.8i %.20j %.9P %.4t %.10M %.6D %R"
scontrol show partition gpu
scontrol show partition slurm-batch
```

## Acceptance Gate

Stage 3 is accepted when all of the following are true:

- `kubectl get nodes` shows the expected node count.
- All GPU substrate nodes carry the `ai-factory/*` Slurm ownership labels.
- GPU substrate nodes carry the `ai-factory/gpu-owner=slurm-batch:NoSchedule` taint after GPU Operator installation.
- GPU Operator pods are healthy in the `gpu-operator` namespace.
- `make train-smoke` completes a short model-training run and writes checkpoints.
- `sinfo -Nel` shows all expected Slurm nodes.
- `scontrol show partition gpu` and `scontrol show partition slurm-batch` both resolve.
- A researcher can submit with `sbatch` without knowing or using Kubernetes.

## Failure Signals

If workers do not appear in Slurm, check:

```bash
aws s3 ls s3://<checkpoint-bucket>/slurm/<master-private-ip>/nodes/
systemctl status slurmd --no-pager
journalctl -u slurmd --no-pager -n 100
```

If Kubernetes looks healthy but Slurm does not, the issue is in the batch layer, not the substrate.

If Slurm looks healthy but GPU Operator pods are failing, the issue is in the substrate GPU enablement layer, not the researcher-facing scheduler.
