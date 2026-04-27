# Dense vs MoE for this cluster

We're training on 4-8 V100s with 16GB each and slow Ethernet between nodes. Which architecture?

Dense.

V100s have 16GB memory. An MoE model loads ALL experts into memory even though only a few are active per token. Mixtral 8x7B has 47B total parameters, 94GB in FP16. We'd need 6+ V100s just to hold the weights before training even starts. A dense 7B model fits in ~14GB and leaves room for gradients and activations.

MoE also needs all-to-all communication to route tokens to the right expert GPUs. Our inter-node bandwidth is ~10 Gbps Ethernet. All-to-all on slow networking kills throughput. Dense models use all-reduce which is simpler and better optimized in NCCL for slow networks.

If we had H100s with 80GB and InfiniBand, MoE would be worth it. Mistral's inference costs are ~5x lower because Mixtral only activates 13B of 47B parameters per token. On our hardware, the memory and networking constraints rule it out.

We're going dense 7B (Mistral-7B-v0.1 or Llama 3 8B), FSDP full-shard across 4 GPUs.
