# FP8 training

V100s don't support FP8 in hardware. This decision is about what we'd recommend for an H100 cluster, not what we're running.

Not applicable to our cluster. For H100+, adopt FP8 from day one.

FP8 doubles tensor core throughput and halves activation memory on H100s. DeepSeek proved it works at 671B parameters with no quality loss. On a 3-month pre-training run at $50K/hr, FP8 cuts runtime to roughly 2 months, saving ~$36M. If the hardware supports it, there's no reason to run BF16.

The catch is precision. FP8 has 3 bits of mantissa, so numbers like 1.001 and 1.002 round to the same value. Gradients can round to zero, loss can spike, training can diverge. DeepSeek solved this with per-block scaling factors instead of per-tensor, and by keeping optimizer states, gradient accumulation, and loss computation in FP32. The heavy matrix multiplications (60-70% of step time) go to FP8. Everything else stays in higher precision.

We're running BF16 on V100s. When we profile, we'll calculate what percentage of step time is GEMM operations. That number is the theoretical FP8 speedup ceiling on H100s and goes in the scaling analysis.

Recommendation if someone handed us an H100 cluster tomorrow: use NVIDIA Transformer Engine for FP8 GEMM, keep master weights in FP32, validate every 500 steps with a BF16 eval run to catch quality regressions early.
