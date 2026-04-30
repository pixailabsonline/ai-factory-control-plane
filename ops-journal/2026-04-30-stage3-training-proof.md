# 2026-04-30 — Stage 3 training proof complete

Single-node smoke path validated on the current 1-GPU node with `distilgpt2`.

- Loss moved from `9.72` to `7.25`.
- No `nan` occurred.
- Throughput was `2132 tokens/sec`.

Multi-node path validated on 2x A10G with FSDP.

- Model: `gpt2-medium`
- Loss moved from `11.75` to `9.12`.
- Throughput was `223 tokens/sec`.
- Checkpoints were written.

Lesson:

- Distributed training works end to end through Slurm, `torchrun`, FSDP, and checkpointing.
- Plain Ethernet on g5-class nodes is a real bottleneck for multi-node all-reduce.
- The current defaults should stay conservative on single-node hardware and only use larger models when the cluster shape supports them.
