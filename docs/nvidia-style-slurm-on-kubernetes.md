# NVIDIA-Style Slurm on Kubernetes Architecture

This project targets a layered GPU platform model:

- Kubernetes is the infrastructure substrate.
- NVIDIA GPU Operator owns GPU enablement on the substrate.
- Slurm is the researcher-facing batch scheduler.
- Training users keep normal Slurm ergonomics: `sbatch`, `srun`, `squeue`, `scancel`.

This is intentionally not a unified Slurm/Kubernetes scheduler. Kubernetes does not opportunistically schedule general Pods onto the same GPUs that Slurm believes it owns.

## Why This Model

Frontier training teams still use Slurm because large training runs are batch jobs. A run needs gang allocation, node lists, deterministic ranks, preemption behavior, requeue, fair-share policy, and accounting. Slurm handles that directly.

Kubernetes is valuable underneath because it gives the platform team a standard substrate for node lifecycle, GPU operator deployment, networking, service discovery, observability, and control-plane services. The researcher should not need to know that substrate exists to run a training job.

The product shape is therefore:

```text
Researcher workflow
  sbatch / srun / squeue / scancel

Batch scheduler
  Slurm controller, Slurm workers, Slurm partitions, GRES GPU accounting

Infrastructure substrate
  Kubernetes nodes, NVIDIA GPU Operator, monitoring, storage hooks, service plumbing

Hardware
  GPU instances / GPU nodes
```

## Ownership Boundary

The system must have a strict capacity model:

- Slurm owns the GPU pool assigned to batch training.
- Kubernetes owns the substrate and platform services.
- General Kubernetes workloads do not share the Slurm GPU pool by default.
- If Kubernetes services need GPUs, they must use a separate node pool or an explicitly reserved partition.

The goal is not maximum theoretical fungibility. The goal is operational clarity with familiar Slurm ergonomics.

## Default Workflows

Researchers submit training through Slurm:

```bash
make train
make train-multi
make jobs
make eval CHECKPOINT=./checkpoints/<job-id>/checkpoint-5000
```

Operators validate the substrate and Slurm layer separately:

```bash
kubectl get nodes
kubectl get pods -n gpu-operator
sinfo
squeue
scontrol show nodes
```

Both views matter. Kubernetes proves the substrate is healthy. Slurm proves the training scheduler is healthy.

## What Kubernetes Owns

Kubernetes owns:

- GPU node lifecycle and host-level substrate services.
- NVIDIA GPU Operator installation and health.
- Monitoring and log shipping agents.
- Platform APIs and non-training control-plane services.
- Optional service workloads on non-Slurm GPU pools or CPU pools.

Kubernetes should not be the default training job interface in this project.

## What Slurm Owns

Slurm owns:

- Batch queueing.
- GPU allocation for training jobs.
- Multi-node gang allocation.
- Node lists and rank environment used by distributed training.
- Preemption and requeue behavior.
- Fair-share and accounting policy.
- Researcher-facing command-line ergonomics.

The primary training path remains the Slurm scripts under `slurm/`.

## Non-Goals

This project does not currently aim to provide:

- A unified scheduler where Slurm and Kubernetes freely compete for the same GPUs.
- Kubernetes Pods scheduled through Slurm.
- A CoreWeave-style workload fungibility bridge.
- Kubernetes-native training as the default runtime.
- Opportunistic GPU sharing between Slurm jobs and arbitrary services.

Those are legitimate advanced platform directions, but they add operational complexity that is not needed for the current proof.

## Implementation Implications

The bootstrap should be split into clear layers:

- Kubernetes substrate setup first.
- NVIDIA GPU Operator setup on the substrate.
- Slurm controller and worker setup on top.
- Slurm partition/GRES config mapped to the Slurm-owned GPU pool.
- Kubernetes-native training operators are not installed in the default path.

The repo should make that layering visible in code and docs. A reader should not wonder whether Kubernetes and Slurm are peers, competitors, or separate islands. Kubernetes is the substrate. Slurm is the batch interface.

## Current Bootstrap Contract

The current bootstrap implements that split as follows:

- `infra/bootstrap_k8s.sh` starts k3s, installs NVIDIA GPU Operator, labels GPU substrate nodes as Slurm-owned capacity, and taints them against arbitrary Kubernetes scheduling.
- `infra/bootstrap_slurm.sh` starts `slurmctld` on the master node and `slurmd` on every training node.
- Slurm node metadata, shared Munge key, and shared Slurm config are coordinated through the checkpoint S3 bucket under a master-node-scoped prefix.
- `Makefile` exposes `make substrate-status`, `make slurm-status`, and `make platform-status` so demos show both substrate health and Slurm health.

See `docs/stage3-slurm-substrate-proof.md` for the validation checklist.

## Acceptance Criteria

The architecture is correctly expressed when:

- The README describes Kubernetes as substrate and Slurm as user-facing scheduler.
- Default training commands use Slurm.
- Kubernetes-native training operators are optional, not default.
- Slurm GPU nodes are labelled or tagged as batch-owned capacity.
- General Kubernetes workloads cannot silently contend with Slurm for the same GPUs.
- Demo evidence includes both `kubectl` substrate health and Slurm job execution.
