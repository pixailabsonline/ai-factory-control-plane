# How DeepSeek Trained Frontier Models for 10-20x Less

And what it means for anyone building GPU training infrastructure.

---

## The Numbers

| Model | Parameters | Active per token | Training cost | GPUs | Tokens trained |
|---|---|---|---|---|---|
| DeepSeek-V3 | 671B | 37B (5.5%) | $5.6M | 2,048 H800s | 14.8T |
| Meta Llama 3 405B | 405B | 405B (100%) | ~$60-100M | 16,384 H100s | 15T |
| GPT-4 (estimated) | ~1.8T | unknown | ~$78-100M | thousands of A100s | unknown |

Similar training data. 10-20x cheaper. 8x fewer GPUs. And DeepSeek's GPUs were slower (H800 is the export-controlled H100 with reduced NVLink bandwidth).

---

## The Four Things They Did Differently

### 1. Only use 5% of the model per token (fine-grained MoE)

A normal 671B parameter model uses all 671B parameters for every single token. That's an enormous amount of computation per token.

DeepSeek-V3 has 256 small "expert" sub-networks plus 1 shared expert. For each token, a router picks the best 8 experts to handle it. The other 248 sit idle. Only 37B out of 671B parameters are active per token.

**How this is different from Mistral's approach:**

Mistral's Mixtral has 8 experts, picks 2. That's 8 possible experts in combinations of 2 — about 28 possible combinations.

DeepSeek has 256 experts, picks 8. That's over 4 billion possible combinations. Way more specialization. Each token gets routed to the exact combination of experts that handles it best.

Plus DeepSeek has a "shared expert" that's always active — it handles the common knowledge that every token needs, so the routed experts can focus on specialized knowledge.

**What this means for infrastructure:**

- Total model is 671B parameters — all of them must be loaded into GPU memory even though only 37B are used per token
- Communication pattern changes: instead of all-reduce (every GPU sends gradients to every GPU), you need all-to-all (tokens get routed to whichever GPU holds the right expert)
- Load balancing becomes critical — if all tokens route to the same few experts, some GPUs are overloaded while others idle
- DeepSeek solved the load balancing problem without the traditional "auxiliary loss" penalty, which slightly hurts model quality at other labs

### 2. Compress the memory bottleneck (Multi-head Latent Attention)

**The problem:** When a model generates text, it needs to remember what it's already seen. It stores this as Key and Value vectors (the "KV cache") for every token in the conversation. For a long conversation, this cache grows to tens of GB. Reading it from GPU memory is slow because GPU memory bandwidth is limited.

**Normal attention:** Store full K and V vectors per token per attention head. For a model with 128 heads and 128-dimensional vectors, that's a lot of data per token.

**DeepSeek's MLA:** Compress K and V into a tiny shared "latent" vector (dimension 512 instead of 12,288+). When the model needs the full K/V vectors, it reconstructs them on the fly from the compressed version using a learned projection.

**The result:** 93.3% reduction in KV cache size.

**Why this matters for infrastructure:**

On current GPUs, inference is usually "memory-bandwidth bound" — the GPU can compute faster than it can read data from its own memory. MLA shifts the bottleneck from memory bandwidth to compute. And compute is the thing GPUs are best at.

This means:
- More concurrent users per GPU (smaller cache = more fit in memory)
- Higher tokens per second (less time waiting for memory reads)
- Lower cost per token served

An infrastructure engineer needs to understand this because it changes the hardware requirements. A memory-bound workload needs GPUs with high HBM bandwidth. A compute-bound workload needs GPUs with high TFLOPS. Different purchasing decisions.

### 3. Train in FP8 instead of FP16

**What are floating point formats:**

Numbers in a computer are stored with a fixed number of bits. More bits = more precision but more memory and slower computation.

| Format | Bits | Memory per parameter | Use case |
|---|---|---|---|
| FP32 | 32 | 4 bytes | Full precision, slow |
| BF16/FP16 | 16 | 2 bytes | Standard training precision |
| FP8 | 8 | 1 byte | New — DeepSeek proved it works at scale |

**What FP8 gives you:**

Compared to FP16:
- **2x throughput** — GPU tensor cores process twice as many FP8 operations per second
- **Half the memory** for weights and activations during matrix multiplication
- Same GPU, same power, same cost — just twice the useful work

**The risk:**

FP8 has much less precision. Numbers can overflow or underflow. Tiny gradients might round to zero. Training could become unstable and produce garbage.

**How DeepSeek handled it:**

- Fine-grained scaling: instead of one scale factor for a whole matrix, they use per-block scaling. Each small block of the matrix gets its own scale factor to keep values in the representable range.
- Critical operations stay in FP32: gradient accumulation, loss computation, optimizer state updates.
- Master weights kept in FP32: the "source of truth" model weights are full precision. FP8 is only used for the heavy matrix multiplications.

**They were the first to prove this works at 671B scale with no measurable quality loss.**

**What this means for infrastructure:**

Any lab still training in FP16 is using half the potential throughput of their H100/H800 GPUs. The factory builder should be:
- Benchmarking FP8 training on their cluster
- Measuring quality impact (if any)
- Calculating the cost savings (roughly 2x for compute-bound operations)
- Understanding which operations can go to FP8 and which must stay in higher precision

### 4. Hide communication behind computation (DualPipe)

**The problem:** DeepSeek's H800 GPUs have 400 GB/s NVLink — less than half the H100's 900 GB/s. In MoE training, tokens need to be routed to experts on other GPUs (all-to-all communication). With slower interconnect, GPUs spend more time waiting for data.

**Normal pipeline parallelism has "bubbles":**

```
GPU 0: [compute][wait   ][compute][wait   ]
GPU 1: [wait   ][compute][wait   ][compute]
```

The "wait" periods are wasted GPU time. At 1,000+ GPUs, this waste adds up to millions of dollars.

**DeepSeek's DualPipe:**

They designed a pipeline schedule where communication and computation happen simultaneously:

```
GPU 0: [compute + send][compute + receive][compute + send]
GPU 1: [receive + compute][send + compute][receive + compute]
```

The GPU is never idle. While it's computing on one microbatch, it's simultaneously sending/receiving data for the next one. They achieved near-zero pipeline bubbles.

**This is the most important lesson for infrastructure engineers:**

They didn't solve the bandwidth problem by buying faster hardware. They solved it with better scheduling. The GPUs were SLOWER than what Meta used, but the infrastructure was more efficient.

**What this means at your scale:**

When you profile your multi-node training and see 60%+ of step time in NCCL communication, the naive fix is "buy faster networking." The DeepSeek fix is "overlap communication with computation." Specifically:

- Start the all-gather for the next layer while computing the current layer
- Pipeline microbatches so communication and computation overlap
- Profile to find which specific NCCL operations can be hidden

You won't implement DualPipe on 2x p3.8xlarge. But measuring the communication overhead, understanding why it exists, and articulating the solutions — that's what gets you the job.

---

## The Misleading Part

**The $5.6M figure is the compute cost only.** It does not include:

- R&D costs (architecture search, failed experiments, researcher salaries)
- The years of prior work that led to MLA and DeepSeekMoE designs
- Hardware procurement and data center costs (DeepSeek likely owns their GPUs)
- H800 pricing in China is different from H100 cloud pricing in the US
- Data acquisition and curation costs

The real total cost is likely 3-5x higher. But even at $20M, it's still 3-5x cheaper than Meta's approach. The efficiency gains are real, not just accounting tricks.

---

## What Other Labs Are Doing About It

### Meta

After Llama 3, Meta invested heavily in their own MoE research. Expect future Llama models to use MoE architecture, reducing compute requirements.

### Mistral

Already uses MoE (Mixtral). Their architecture is closer to DeepSeek's approach than Meta's or OpenAI's dense models. But their MoE is coarser-grained (8 experts vs 256).

### Google/DeepMind

Gemini uses MoE internally. Google has the most experience with MoE at scale (their Switch Transformer paper from 2021 pioneered many of these ideas).

### Anthropic

Claude's architecture is not publicly disclosed. But as a competitor, they're certainly studying DeepSeek's efficiency results.

### The industry direction

The era of "just throw more GPUs at it" is ending. Efficiency now matters as much as scale. The factory builder of the future needs to understand:

1. How model architecture (MoE, MLA) changes infrastructure requirements
2. How numerical precision (FP8) changes throughput economics
3. How communication scheduling (DualPipe) changes utilization at scale
4. How all of these interact

---

## How This Connects to Your Project

### What you can demonstrate

When you profile your FSDP training and see the communication overhead:

> "My profiling shows 42% communication overhead at 4 GPUs on NVLink. At 2 nodes over Ethernet, it rises to 65%. This is the fundamental scaling bottleneck. DeepSeek's DualPipe addresses this by overlapping all-to-all expert routing with pipeline computation. For dense models like ours, the equivalent technique is overlapping FSDP all-gather with the forward pass of the next microbatch. I measured a 12% throughput improvement by enabling FSDP's `limit_all_gathers=True` parameter, which does a simpler version of this overlap."

That paragraph proves you:
- Measured the real bottleneck
- Understand why it exists
- Know what the frontier solution is
- Applied a practical version to your own setup
- Can translate between your small-scale results and frontier-scale implications

### What you should put in decisions/

Write a short decision doc: "Dense vs MoE: Infrastructure Implications"

Cover:
- Memory requirements: MoE loads all parameters but only activates a subset
- Communication pattern: all-reduce (dense) vs all-to-all (MoE)
- Hardware implications: MoE benefits more from memory bandwidth, dense benefits more from compute
- Inference economics: MoE is faster per token (fewer active params) but needs more total memory
- When each makes sense for a training infrastructure investment

This is the kind of thinking that separates an operator from a factory builder.

### The interview question this prepares you for

> "DeepSeek trained a 671B model for $5.6M. What infrastructure decisions made that possible, and how would you apply similar thinking to our training cluster?"

Your answer should cover architecture (MoE), precision (FP8), communication scheduling (DualPipe), and what you'd measure first on their specific hardware to identify the biggest efficiency win.
