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

- Shared substrate adds operational complexity.
- Some host resources are still shared.
- Kubernetes is useful here only if it is proving a real platform capability.
- If Kubernetes does not visibly help the training/product story, it becomes noise.

When to revisit:

- If we need multi-tenant platform services that truly benefit from Kubernetes-native orchestration.
- If we decide to reserve separate node pools for platform services and GPU training.
- If a future product requires a unified scheduler or a tighter workload-fungibility story.

Current scope rule:

- Do not let Kubernetes become a second product.
- Do not make it compete with Slurm for the same GPUs.
- Keep it as infrastructure support unless we explicitly raise the architecture bar later.
