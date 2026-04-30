# Kubernetes scope decision

We are using Kubernetes as a substrate, not as a competing job scheduler.

That means:

- Slurm owns researcher-facing batch jobs.
- Kubernetes owns node/bootstrap/platform plumbing.
- Kubernetes does not opportunistically schedule general workloads onto the same GPU pool that Slurm owns.

Why this choice:

- It keeps the product honest for multi-node GPU training.
- It avoids double-scheduling the same GPUs.
- It preserves Slurm ergonomics, which match how frontier training teams operate.
- It still lets us demonstrate platform-level credibility for GPU operator setup, node lifecycle, labels, taints, and observability.

Tradeoffs:

- Extra control-plane pieces mean more bootstrap steps, more failure modes, and more time spent debugging platform state before training even starts.
- Because the same machines carry both substrate and batch roles, we still need strict ownership rules so service workloads do not steal training capacity.
- Kubernetes stays only where it gives a concrete product benefit: GPU enablement, node lifecycle, labels/taints, observability, or platform services we can actually use.
- If Kubernetes does not move one of those outcomes, it is not a platform asset, it is overhead.

When to revisit:

- If we need multi-tenant platform services that truly benefit from Kubernetes-native orchestration.
- If we decide to reserve separate node pools for platform services and GPU training.
- If a future product requires a unified scheduler or a tighter workload-fungibility story.

Current scope rule:

- Do not let Kubernetes become a second product.
- Do not make it compete with Slurm for the same GPUs.
- Keep it as infrastructure support unless we explicitly raise the architecture bar later.
