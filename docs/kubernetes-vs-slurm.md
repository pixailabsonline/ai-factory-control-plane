# Kubernetes vs Slurm for distributed training

This repo uses Slurm. This note covers why, and how the same workload runs on Kubernetes — which is what CoreWeave, and increasingly frontier labs, operate in production.

## What both solve

Both schedulers answer the same question: given a cluster of GPU nodes, allocate resources to training jobs, queue work when capacity is full, and recover when something fails.

Slurm owns bare metal. It knows about nodes, GPUs, and MPI natively. You submit a job with `sbatch`, Slurm finds nodes with free GPUs, runs your script, cleans up when it exits.

Kubernetes abstracts the metal behind pods. GPUs are exposed as extended resources via the NVIDIA device plugin — a DaemonSet that runs on every node, registers GPU capacity with the kubelet via the Device Plugin API, and mounts GPU devices into containers on scheduling.

## Why frontier labs use Slurm for training

Slurm has lower overhead and was designed for exactly this workload. A 1000-node training job is a batch job: it starts, runs for days, exits. Slurm handles that natively. Kubernetes was designed for long-running services, and distributed training support is layered on top via operators — more moving parts, more failure modes.

For large pre-training runs where stability over days or weeks matters, Slurm is the standard. Every major HPC center and most frontier labs (Anthropic, DeepMind, Meta) run Slurm for training.

## How it works on Kubernetes

The NVIDIA device plugin DaemonSet registers `nvidia.com/gpu` as a schedulable resource. A pod requests GPUs the same way it requests CPU:

```yaml
resources:
  limits:
    nvidia.com/gpu: 4
```

The scheduler places the pod on a node with 4 free GPUs. The device plugin mounts `/dev/nvidia*` into the container and injects the CUDA libraries.

For distributed training across multiple nodes, you need all ranks to start simultaneously and coordinate. Standard Kubernetes Jobs don't guarantee this — you could have 7 of 8 pods running while the 8th waits for a node. NCCL hangs waiting for all ranks to rendezvous.

This is solved with the **PyTorch Operator** (Kubeflow) — a CRD and controller that manages the full distributed job lifecycle:

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: fsdp-training
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
            - name: trainer
              image: training:latest
              resources:
                limits:
                  nvidia.com/gpu: 4
    Worker:
      replicas: 7
      template:
        spec:
          containers:
            - name: trainer
              image: training:latest
              resources:
                limits:
                  nvidia.com/gpu: 4
```

The operator's controller watches PyTorchJob resources, creates the pods, injects `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, and `RANK` environment variables, and coordinates startup so all ranks come up together. If a worker pod dies, the controller handles restart and re-rendezvous.

## Gang scheduling

The default kube-scheduler places pods one at a time. For a 512-GPU training job this means pods trickle onto nodes over minutes, holding GPU resources while waiting for the rest of the gang. This is wasteful and can deadlock if the cluster is full.

Gang scheduling fixes this: either all pods in a job are scheduled simultaneously or none are. Implemented via the **Volcano scheduler** or kube-scheduler plugins. CoreWeave runs gang scheduling in production — it's a hard requirement for large multi-node jobs.

Slurm handles this natively. The equivalent is `--nodes=16 --ntasks-per-node=4` in an sbatch script — Slurm won't start the job until all 16 nodes are available.

## Topology-aware scheduling

For InfiniBand clusters, not all nodes are equal. Nodes on the same InfiniBand switch have lower latency than nodes across switches. Placing a training job's pods on nodes that share a switch reduces all-reduce communication time.

Kubernetes handles this with topology spread constraints and node labels. CoreWeave labels nodes with their switch affinity and the scheduler respects it. Slurm handles it via partition configuration and `--constraint` flags.

## Where each fits

| Workload | Slurm | Kubernetes |
|---|---|---|
| Large pre-training runs | Preferred | Possible with operators |
| Fine-tuning experiments | Good | Good |
| Inference serving | Not designed for it | Native |
| Mixed training + serving cluster | Awkward | Natural |
| Bare metal HPC | Native | Overhead |

Labs increasingly run both: Slurm for large training jobs, Kubernetes for inference serving, fine-tuning pipelines, and everything else. CoreWeave supports both. The operator pattern on Kubernetes and the sbatch pattern on Slurm are solving the same distributed coordination problem with different primitives.
