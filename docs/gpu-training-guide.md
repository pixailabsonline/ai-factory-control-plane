# GPU Training Infrastructure: The Complete Guide

Everything you need to understand to build and operate a GPU training factory.
Written for someone who can code but hasn't worked with GPU clusters before.

---

## Part 1: Why GPUs and Not Regular Computers

### What a CPU does

Your laptop has a CPU (like an Intel i7 or Apple M3). A CPU is good at doing one complicated thing at a time, very fast. It can run your browser, your code editor, and your music app all at once by switching between them quickly.

### What a GPU does

A GPU has thousands of tiny processors instead of a few powerful ones. It can't do one complicated thing fast — but it can do thousands of simple things at the same time. This is called parallel processing.

### Why AI training needs GPUs

Training an AI model is mostly matrix multiplication — giant grids of numbers being multiplied together. This is exactly the kind of simple, repetitive work that GPUs are built for. A task that takes a CPU 100 hours might take a GPU 1 hour, because the GPU does thousands of multiplications at the same time.

A single GPU can be 10-100x faster than a CPU for training. That's why every AI lab buys thousands of them.

---

## Part 2: How AI Training Actually Works

### The basic idea

You have a model (a big pile of numbers called "parameters") and training data (text, images, whatever). Training means adjusting those numbers until the model gets good at predicting the next word, or classifying an image, or whatever task you're training for.

### One training step

Each training step has three parts:

1. **Forward pass** — You feed a batch of data through the model. The model makes predictions. You compare those predictions to the correct answers and calculate how wrong the model was. This "wrongness score" is called the **loss**. Lower loss = better model.

2. **Backward pass** — You work backwards through the model, calculating how much each parameter contributed to the error. These calculations are called **gradients**. Each gradient says "this parameter should go up a little" or "this parameter should go down a lot."

3. **Weight update** — You adjust every parameter based on its gradient. The optimizer (usually AdamW) decides exactly how much to adjust each one. This is one "step" complete.

Repeat this thousands or millions of times. The loss goes down. The model gets better.

### What "loss going down" looks like

```
Step 100:   Loss: 8.42    (model is basically guessing randomly)
Step 500:   Loss: 5.13    (model is learning basic patterns)
Step 1000:  Loss: 3.67    (model understands common words and grammar)
Step 3000:  Loss: 2.41    (model can write coherent sentences)
Step 5000:  Loss: 2.18    (diminishing returns — each step improves less)
```

If loss suddenly spikes UP, something went wrong — bad data batch, learning rate too high, numerical instability. This is the kind of thing you'd write in the ops journal.

### Batch size

You don't feed one example at a time. You feed a "batch" — say 32 examples at once. The model processes all 32, averages the gradients, and does one weight update. Larger batches are more stable but use more GPU memory.

### Learning rate

How much you adjust the weights each step. Too high = the model overshoots and loss spikes. Too low = training takes forever. Getting this right matters enormously. Most runs use a "warmup" (start low, ramp up) then a "cosine decay" (gradually reduce over time).

### Epochs

One epoch = the model has seen every example in the dataset once. If your dataset has 1 million examples and your batch size is 32, one epoch is ~31,250 steps. Most fine-tuning runs do 1-3 epochs. Pre-training from scratch might do many more.

---

## Part 3: Why One GPU Isn't Enough

### The memory problem

A model's parameters are stored as numbers. Each parameter in FP16 (half precision) takes 2 bytes. A 7 billion parameter model takes:

```
7,000,000,000 × 2 bytes = 14 GB just for the model weights
```

But during training you also need:
- **Gradients** — same size as the model: 14 GB
- **Optimizer states** — AdamW stores 2 extra values per parameter: 28 GB
- **Activations** — intermediate calculations from the forward pass: varies, can be 10-30+ GB

Total: roughly 60-80 GB for a 7B model.

A V100 GPU has 16 GB of memory. An A100 has 40 or 80 GB. An H100 has 80 GB.

**A 7B model doesn't fit on a single V100.** You need to split it across multiple GPUs or use tricks to reduce memory usage. This is why distributed training exists.

### The speed problem

Even if a model fits on one GPU, training on billions of tokens takes too long. If one GPU processes 1,000 tokens per second, training on 1 trillion tokens takes:

```
1,000,000,000,000 / 1,000 = 1,000,000,000 seconds = 31.7 years
```

With 1,000 GPUs working together, that's ~11.6 days. This is why frontier labs use thousands of GPUs.

---

## Part 4: How Multiple GPUs Work Together

### The fundamental problem

You have a model that's too big for one GPU and training that's too slow for one GPU. You need multiple GPUs working as a team. But GPUs are separate devices — they have their own memory, their own processors. They need to communicate to coordinate.

### Data Parallel (DDP — Distributed Data Parallel)

The simplest approach. Every GPU has a complete copy of the model. Each GPU gets a different batch of data, does its own forward and backward pass, then they all share their gradients and average them. Every GPU ends up with the same weight update.

**Problem:** Every GPU needs a full copy of the model + optimizer states. For a 7B model, that's 60-80 GB per GPU. Doesn't fit on a V100 (16 GB).

**When to use:** Small models that fit on one GPU but you want faster training.

### FSDP (Fully Sharded Data Parallel)

This is what we use. Instead of every GPU having a full copy, the model is split into pieces (shards). Each GPU holds only its shard — about 1/4 of the model on 4 GPUs.

When a GPU needs to do a forward pass through a layer it doesn't have, it asks the other GPUs to send their pieces temporarily. It does the computation, then throws away the borrowed pieces to free memory.

**How it works step by step on 4 GPUs:**

```
1. Model is split into 4 shards: GPU 0 has shard A, GPU 1 has shard B, etc.
2. Forward pass starts. GPU 0 needs the full model for layer 1.
3. GPU 0 broadcasts "send me your shards" (all-gather operation).
4. All GPUs send their shards to GPU 0. Now GPU 0 has the full layer temporarily.
5. GPU 0 computes the forward pass for layer 1.
6. GPU 0 throws away the borrowed shards (frees memory).
7. Repeat for each layer.
8. Backward pass: same thing in reverse.
9. After backward pass, each GPU has gradients for its shard only.
10. Weight update: each GPU updates only its shard.
```

**Memory per GPU:** ~20 GB for a 7B model on 4 GPUs (vs 60-80 GB with DDP).

**Cost:** All that sending shards back and forth takes time. This is the communication overhead. On 4 GPUs with NVLink, it's manageable (30-40% of step time). Across nodes over a network, it gets worse.

**When to use:** Models that don't fit on one GPU. This is the standard approach for fine-tuning 7B+ models.

### Tensor Parallelism (TP)

Splits individual layers across GPUs. Instead of each GPU having different layers, each GPU has a piece of every layer. One matrix multiplication gets split so GPU 0 does the left half and GPU 1 does the right half.

**Requires extremely fast communication** because GPUs need to exchange data within each layer, not just between layers. Only works well with NVLink (within a single machine). You'd never do this across nodes on a network.

**When to use:** Very large models where even FSDP isn't enough, and the GPUs are connected by NVLink.

### Pipeline Parallelism (PP)

Splits the model by layers. GPU 0 has layers 1-10, GPU 1 has layers 11-20, etc. Data flows through GPU 0, then GPU 1, then GPU 2 like a factory assembly line.

**Problem:** While GPU 0 is processing batch 1, GPU 1 is idle waiting. This is called a "pipeline bubble." You waste GPU time. Microbatching (splitting batches into smaller pieces and overlapping them) reduces this but doesn't eliminate it.

**When to use:** Very large models across many nodes. Often combined with TP within a node and PP across nodes.

### Expert Parallelism (EP)

Only for Mixture of Experts (MoE) models like Mistral's Mixtral. An MoE model has multiple "expert" sub-networks. For each input, a router picks which 2 experts (out of 8, say) to use. Expert parallelism puts different experts on different GPUs.

**Why Mistral cares about this:** Mixtral 8x7B has 8 experts, each 7B parameters. Only 2 are active per input, so inference is fast. But training requires all 8 to be loaded. Expert parallelism distributes them across GPUs.

**When to use:** MoE models. If you understand this, you understand Mistral's architecture at the infrastructure level.

### How labs combine these

At frontier scale (1000+ GPUs), labs use all of these simultaneously:

```
Within one machine (8 GPUs, NVLink):
  - Tensor Parallelism across 4 GPUs (fast, needs NVLink)
  - FSDP within the TP groups

Across machines (many nodes, network):
  - Pipeline Parallelism across nodes (less communication needed)
  - Data Parallelism across pipeline stages
```

This is called a "3D parallelism" or "4D parallelism" strategy. Choosing how to combine them is one of the hardest decisions in training infrastructure. It depends on model size, GPU memory, network bandwidth, and number of GPUs.

---

## Part 5: How GPUs Talk to Each Other

### Why communication matters

Every training step, GPUs need to exchange data. In FSDP, they send model shards (all-gather) and combine gradients (reduce-scatter). If this communication is slow, GPUs spend most of their time waiting instead of computing. This is the communication overhead.

### NVLink — GPUs within one machine

NVLink is a direct high-speed connection between GPUs on the same machine. On a p3.8xlarge (4x V100), NVLink provides 300 GB/s between GPUs. This is fast enough that communication overhead is manageable (30-40% of step time).

Think of it as GPUs sitting next to each other and handing papers back and forth.

### PCIe — the slow alternative

Without NVLink, GPUs communicate over PCIe (the same bus your SSD and network card use). PCIe Gen3 is about 16 GB/s — roughly 20x slower than NVLink. This is why g5 instances (A10G GPUs, no NVLink) are bad for distributed training. The GPUs spend most of their time waiting for data instead of computing.

### Network — GPUs across machines

When you have GPUs on different machines, they communicate over the network. Options:

- **InfiniBand** — dedicated high-speed network for HPC. 200-400 Gbps. What most bare-metal GPU clusters use. Very low latency.
- **EFA (Elastic Fabric Adapter)** — AWS's version of high-speed networking. 100 Gbps. What you get on p4d and p5 instances. Decent but not InfiniBand.
- **Regular Ethernet** — 10-25 Gbps. Way too slow for distributed training at scale. This is what p3 instances use between nodes.

**This is why multi-node training on p3 instances is hard.** Within a node, you have NVLink at 300 GB/s. Between nodes, you have regular Ethernet at maybe 10 Gbps. That's a 240x bandwidth drop. Communication-heavy operations (like FSDP all-gather) become the bottleneck the moment you cross the node boundary.

This is also the exact finding that goes in your scaling analysis — you'll measure the throughput drop when going from 4 GPUs (1 node) to 8 GPUs (2 nodes) and explain why.

### NCCL — the software that manages it all

NCCL (NVIDIA Collective Communication Library, pronounced "nickel") is the software layer that handles all GPU-to-GPU communication. When FSDP does an all-gather, it calls NCCL. NCCL figures out the fastest way to move data between GPUs — using NVLink within a node and the network across nodes.

**Why NCCL matters operationally:**

- When NCCL hangs, training freezes with no error message. You set `NCCL_DEBUG=INFO` to see what's happening.
- NCCL picks communication paths automatically, but sometimes picks wrong. You tune it with environment variables (`NCCL_P2P_LEVEL`, `NCCL_SOCKET_IFNAME`, etc.).
- NCCL bugs or misconfigurations are the #1 cause of "training just stopped and nothing is in the logs."

**The key NCCL operations:**

| Operation | What it does | When it happens |
|---|---|---|
| All-gather | Every GPU sends its piece, every GPU gets the full thing | FSDP forward pass (reassemble model shards) |
| Reduce-scatter | Every GPU sends gradients, each GPU gets its reduced piece | FSDP backward pass (combine gradients) |
| All-reduce | All GPUs send values, all GPUs get the average | DDP gradient sync |
| Broadcast | One GPU sends, all GPUs receive | Loading model weights |

When you profile training and see "42% of step time in NCCL ops," that means GPUs are spending 42% of their time talking to each other instead of computing. Reducing this is one of the main jobs of a training infrastructure engineer.

---

## Part 6: Checkpointing — Saving Your Work

### Why checkpointing is critical

Training a model on 1,000 H100s costs about $50,000 per hour. If the training crashes at step 10,000 and you have no checkpoint, you start over from step 0. That's potentially hundreds of thousands of dollars wasted.

A checkpoint is a snapshot of everything needed to resume training: the model weights, optimizer states, scheduler state, and the current step number.

### How our checkpoint system works

```
1. Every 500 steps, save a checkpoint
2. Save happens in a background thread (training doesn't pause)
3. Write to a staging directory first
4. Compute SHA-256 checksum of the saved file
5. Verify the file can be loaded (not corrupted)
6. If valid, rename staging dir to final checkpoint dir
7. If invalid, delete it (corrupted write)
8. Sync to S3 (survives even if the machine is terminated)
9. Keep only the 5 most recent checkpoints (disk space)
```

### Why async writes matter

Saving a 7B model checkpoint can take 30-60 seconds. If training pauses every 500 steps for 60 seconds, that's a lot of wasted GPU time. Async checkpointing writes the checkpoint in a background thread while training continues on the GPU. The GPU never stops working.

### Why validation matters

Checkpoint files can be corrupted — disk errors, process killed mid-write, network issues during S3 upload. If you restore from a corrupted checkpoint, training produces garbage. Our system computes a checksum when writing and verifies it when restoring. If the latest checkpoint is corrupt, it falls back to the previous one.

### What happens when training crashes

```
1. Training was at step 2,347 when the process was killed
2. Latest checkpoint is at step 2,000 (saved every 500 steps)
3. Training restarts, finds checkpoint-2000
4. Validates checksum — it's good
5. Loads model weights, optimizer state, scheduler state
6. Resumes from step 2,000
7. Steps 2,000-2,347 are recomputed (347 steps lost, not 2,347)
```

The cost of losing 347 steps on a p3.8xlarge at $12.24/hr is maybe $2-5. Without checkpointing, losing 2,347 steps might cost $15-30. At frontier scale on 1,000 H100s, that difference is $50K vs $200K+.

### Checkpoint frequency tradeoff

More frequent = less work lost on crash, but more I/O overhead.
Less frequent = less I/O overhead, but more work lost on crash.

The right answer depends on how much each hour of training costs. At $50K/hr, you checkpoint every 5-10 minutes. At $12/hr, every 500 steps (maybe 30-60 minutes) is fine.

---

## Part 7: Slurm — The Job Scheduler

### What problem Slurm solves

You have a cluster of GPU machines. Multiple people (or automated systems) want to run training jobs. Someone needs to decide which job gets which GPUs, when, and for how long. Slurm is that someone.

### How Slurm works

Slurm has two parts:

1. **Head node** — the manager. Doesn't have GPUs. Runs the scheduler. Accepts job submissions, tracks which nodes are free, assigns jobs to nodes.

2. **Compute nodes** — the workers. These are your GPU machines (p3.8xlarge instances). They register with the head node and say "I have 4 V100 GPUs available."

### Submitting a job

You write an sbatch script — a bash script with special `#SBATCH` headers that tell Slurm what resources you need:

```bash
#SBATCH --nodes=2              # I need 2 machines
#SBATCH --gpus-per-node=4      # with 4 GPUs each
#SBATCH --time=24:00:00        # for up to 24 hours
#SBATCH --partition=gpu        # from the GPU partition
```

Then you run `sbatch train.sbatch`. Slurm puts your job in the queue. When the resources are available, Slurm allocates the nodes, sets up environment variables (which node is master, what's the node list, etc.), and runs your script on all the allocated nodes.

### Why not just SSH in and run torchrun?

For 1-2 machines that you own, SSH works. But:

- **At scale, you can't manually manage 100+ nodes.** Slurm handles allocation, scheduling, and cleanup.
- **Multiple users/jobs.** If two training runs want the same GPUs, Slurm queues them.
- **Preemption.** A high-priority job can kick a low-priority job off the GPUs. The low-priority job gets a SIGTERM, saves a checkpoint, and re-queues.
- **Accounting.** Slurm tracks who used how many GPU-hours. This is how labs track cost.
- **Fault handling.** If a node dies, Slurm detects it, marks it as down, and can re-queue the job on healthy nodes.

### AWS ParallelCluster

On AWS, you don't install Slurm manually. AWS ParallelCluster is a tool that creates a complete Slurm cluster for you — head node, compute nodes, shared filesystem, networking. You give it a config file describing what you want, and it creates the whole thing.

This is the standard way to run Slurm on AWS. It's what most GPU training teams on AWS use.

### Key Slurm commands

| Command | What it does |
|---|---|
| `sbatch job.sbatch` | Submit a job to the queue |
| `squeue` | See what jobs are running and queued |
| `scancel <job-id>` | Cancel a job |
| `sinfo` | See what nodes are available and their state |
| `scontrol show job <id>` | Detailed info about a specific job |
| `sacct` | Historical job accounting (who used what, when) |

---

## Part 8: Hardware — What You're Actually Buying

### The GPU generations

| GPU | Year | Memory | FP16 TFLOPS | NVLink BW | Used by |
|---|---|---|---|---|---|
| V100 | 2017 | 16 or 32 GB | 125 | 300 GB/s | Your p3 instances. Older but still capable. |
| A100 | 2020 | 40 or 80 GB | 312 | 600 GB/s | Workhorse of most current clusters. |
| H100 | 2023 | 80 GB | 989 | 900 GB/s | Frontier training. What labs are buying now. |
| B200 | 2025 | 192 GB | ~2,500 | 1,800 GB/s | Next generation. Starting to ship. |

Each generation roughly doubles compute speed and network bandwidth. Memory grows slower but it's what determines how big a model you can train.

### TFLOPS — what it means

TFLOPS = trillion floating point operations per second. A V100 does 125 trillion multiplications per second in FP16. An H100 does 989. This is the raw compute power.

**But you never get 100% of this.** Real training achieves 30-60% of theoretical TFLOPS. The rest is lost to communication overhead, memory loading, data pipeline stalls, etc. The percentage you actually achieve is called MFU (Model FLOPs Utilization).

### MFU — the number that matters most

```
MFU = actual TFLOPS used / theoretical maximum TFLOPS
```

If you're running on 4x V100 (500 theoretical TFLOPS total) and your training uses 200 TFLOPS:

```
MFU = 200 / 500 = 40%
```

**40% MFU is typical for fine-tuning on V100s.** Frontier labs like Meta aim for 40-50% on H100s for pre-training. Getting above 50% is excellent. Below 30% means something is wrong — probably communication overhead or data loading bottleneck.

**When someone asks "what was your MFU?", they're asking: how efficiently did you use the hardware?** A high MFU means you're not wasting GPUs. This is the single most important efficiency metric in GPU training.

### How to calculate MFU from your training run

```
1. Measure tokens per second from your training run (the trainer logs this)
2. Calculate FLOPS per token: roughly 6 × model_parameters
   For a 7B model: 6 × 7,000,000,000 = 42 TFLOPS per token
3. Multiply: tokens_per_sec × 42 TFLOPS = achieved TFLOPS
4. Divide by theoretical: achieved / (4 × 125) = MFU
```

Example: if you train at 1,500 tokens/sec on 4x V100:
```
1,500 × 42,000,000,000 / 1e12 = 63 TFLOPS achieved
63 / 500 = 12.6% MFU
```

That would be low — it means something's wrong. You'd profile to find out what's eating the remaining 87.4%.

### Memory types that matter

- **HBM (High Bandwidth Memory)** — the GPU's main memory. V100 has 16 GB HBM2 at 900 GB/s. H100 has 80 GB HBM3 at 3.35 TB/s. This is where model weights, gradients, and activations live.

- **System RAM** — the machine's regular memory (CPU memory). Used for data loading, preprocessing. If the data pipeline can't feed data to the GPU fast enough, GPUs idle.

- **NVMe/SSD** — where checkpoints are saved, datasets are stored. Checkpoint write speed depends on this. A slow disk means longer checkpoint writes means more I/O overhead.

### Network types that matter

| Type | Bandwidth | Latency | Used for |
|---|---|---|---|
| NVLink | 300-900 GB/s | ~microseconds | GPU-to-GPU within one machine |
| PCIe Gen4 | 32 GB/s | ~microseconds | GPU to CPU, slow GPU-to-GPU fallback |
| InfiniBand | 200-400 Gbps (~25-50 GB/s) | ~1-5 microseconds | GPU cluster networking (bare metal) |
| EFA (AWS) | 100 Gbps (~12.5 GB/s) | ~5-20 microseconds | GPU networking on AWS (p4d/p5) |
| Ethernet | 10-25 Gbps (~1-3 GB/s) | ~50-200 microseconds | Regular networking (p3 instances) |

Notice the massive gap between NVLink (300 GB/s) and Ethernet (1-3 GB/s). This is why going from 1 node to 2 nodes causes a huge throughput drop. Your scaling analysis will measure exactly this.

---

## Part 9: Profiling — Finding Bottlenecks

### Why profile

Your training is running at 12% MFU. Where is the other 88% going? You need to measure, not guess. Profiling tells you exactly where time is spent.

### What to measure

1. **Compute time** — time the GPU spends doing actual matrix multiplication. This is the useful work.

2. **Communication time** — time spent in NCCL operations (all-gather, reduce-scatter). This is overhead from distributing across GPUs.

3. **Data loading time** — time the GPU is idle waiting for the next batch of data from CPU/disk.

4. **Memory operations** — time moving data between CPU memory and GPU memory.

### Using torch.profiler

```python
with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # run one training step
    outputs = model(input_ids, labels=input_ids)
    outputs.loss.backward()
```

This captures every operation — every matrix multiplication, every NCCL call, every memory copy. It produces a trace file you can view in Chrome's trace viewer (chrome://tracing) or TensorBoard.

### What a good profile looks like

```
Total CUDA time: 1,200 ms per step
  Matrix multiply (compute): 680 ms (57%)
  NCCL all-gather:           350 ms (29%)
  NCCL reduce-scatter:       120 ms (10%)
  Memory operations:          50 ms (4%)
```

57% compute is decent for FSDP on 4x V100 with NVLink. The 39% in NCCL is the price of distributing the model across GPUs.

### What a bad profile looks like

```
Total CUDA time: 3,400 ms per step
  Matrix multiply (compute): 680 ms (20%)
  NCCL all-gather:         2,100 ms (62%)
  NCCL reduce-scatter:       450 ms (13%)
  Memory operations:         170 ms (5%)
```

62% in NCCL all-gather means GPUs are spending most of their time waiting for data from other GPUs. This could mean:
- Bad NCCL configuration
- Running across nodes with slow networking
- Using PCIe instead of NVLink
- Batch size too small (communication cost dominates)

### What to do with profiling results

This is your scaling-analysis.md content. You measure on 1 GPU, 2 GPUs, 4 GPUs (1 node), 8 GPUs (2 nodes) and show how the communication overhead changes. Then you project: "At 64 GPUs across 16 nodes, communication would be X% based on the trend. To mitigate this, you'd switch from FSDP to a hybrid TP+PP strategy."

---

## Part 10: Common Failures and How to Debug Them

These are the incidents that go in your ops journal. Every one of these WILL happen when you run on real hardware.

### NCCL hang

**Symptom:** Training freezes. No error message. GPU utilization drops to 0%. Nothing in the logs.

**Diagnosis:**
1. Set `NCCL_DEBUG=INFO` and rerun
2. Look for which rank (GPU) is stuck and at which operation
3. Usually: one GPU finished its work and is waiting for another GPU that's stuck

**Common causes:**
- One GPU has degraded PCIe link speed (Gen1 instead of Gen3)
- Network issue between nodes (for multi-node)
- NCCL version mismatch between nodes
- Firewall blocking NCCL ports

**Fix:** Identify the stuck rank, check its hardware, exclude bad GPUs if needed, restart from checkpoint.

### OOM (Out of Memory)

**Symptom:** `CUDA out of memory. Tried to allocate X MiB` error.

**Diagnosis:**
1. Check `nvidia-smi` for current memory usage
2. Check batch size and sequence length — longer sequences use more memory
3. Check if gradient accumulation is configured correctly

**Common fixes:**
- Reduce batch size
- Reduce sequence length (max_length)
- Enable gradient checkpointing (recomputes activations instead of storing them — slower but uses less memory)
- Switch FSDP sharding strategy from SHARD_GRAD_OP to FULL_SHARD

### Loss spike

**Symptom:** Loss was decreasing nicely, then suddenly jumps up.

**Diagnosis:**
1. Check the data at that step — was it a bad batch?
2. Check learning rate — is it too high at this point?
3. Check for NaN gradients — numerical instability

**Common fixes:**
- Skip the bad batch and continue
- Reduce learning rate
- Add gradient clipping (we already do this: `clip_grad_norm_(model.parameters(), 1.0)`)
- Restore from the checkpoint before the spike

### GPU ECC error

**Symptom:** `Xid error` in `dmesg`, or GPU marked as "unhealthy" by nvidia-smi.

**Diagnosis:**
1. `nvidia-smi -q -d ECC` — check ECC error counts
2. If volatile ECC errors are accumulating, the GPU memory is failing

**Fix:** Exclude that GPU from training. In a real cluster, you'd drain the node (move all jobs off it) and request a hardware replacement. On AWS, you terminate the instance and launch a new one.

### Checkpoint corruption

**Symptom:** Training restarts but the model produces garbage, or `torch.load()` fails with an error.

**Diagnosis:**
1. Check meta.json for the checkpoint — does the checksum match?
2. Try loading the checkpoint manually: `torch.load("state.pt")`

**Fix:** Our system handles this automatically. If the latest checkpoint fails validation, it falls back to the previous one. If all checkpoints are corrupt, you need to re-download from S3 (assuming S3 sync was working).

### Data pipeline stall

**Symptom:** GPU utilization drops periodically to 0%, then comes back. Training is slow but not stuck.

**Diagnosis:**
1. Profile — is there a gap between training steps where the GPU is idle?
2. Check CPU utilization — data loading happens on CPU
3. Check disk I/O — is data loading from a slow disk?

**Fix:**
- Increase `num_workers` in DataLoader (more CPU threads loading data)
- Move dataset to faster storage (NVMe SSD instead of network storage)
- Preprocess and cache tokenized data instead of tokenizing on the fly

---

## Part 11: Inference Serving — The Other Half

### Training vs inference

**Training:** Model is learning. Processes data in batches, computes gradients, updates weights. Needs lots of GPUs, runs for hours/days. Cost measured in GPU-hours.

**Inference:** Model is serving predictions. Takes one request at a time (or batches of requests), generates output. Needs fewer GPUs, runs continuously. Cost measured per token or per request.

### Why vLLM and not just loading the model

When you load a model with plain PyTorch and generate text, each request gets processed one at a time. If you're serving 100 users, 99 of them are waiting.

vLLM solves this with:

1. **Continuous batching** — While generating tokens for user A, vLLM can start processing user B's prompt. Multiple requests are in flight simultaneously.

2. **PagedAttention** — The key-value cache (memory used during text generation) is managed like virtual memory in an OS. Instead of pre-allocating a huge block per request, it allocates small pages as needed. This means more requests fit in GPU memory simultaneously.

3. **Tensor parallelism** — For large models, vLLM can split inference across multiple GPUs.

The result: much higher throughput (tokens/sec) per GPU dollar.

### The economics question

Self-hosted inference makes sense when:

```
Cost per token (self-hosted) < Cost per token (API)

Where:
  Self-hosted cost per token = Instance cost per hour / (tokens per second × 3,600)
  API cost per token = the provider's published price
```

Example: if you serve a 7B model on a g5.xlarge ($1/hr) at 50 tokens/sec:

```
Self-hosted: $1 / (50 × 3,600) = $0.0000056 per token = $5.56 per 1M tokens
Claude Sonnet 4.6 output: $15 per 1M tokens
```

Self-hosting is 2.7x cheaper — IF the GPU is being used. An idle GPU still costs $1/hr. The breakeven utilization is the point where self-hosting becomes cheaper.

This is the analysis your inference benchmark produces. It answers: "Should this lab self-host or use an API?" for any given model and hardware combination.

---

## Part 12: The Economics of GPU Training

### Cost per GPU-hour

This is the base unit. Everything else derives from it.

| GPU | Instance | On-demand $/hr | GPU-hours per $1,000 |
|---|---|---|---|
| V100 (×4) | p3.8xlarge | $12.24 | 326 GPU-hours |
| A100 (×8) | p4d.24xlarge | $32.77 | 244 GPU-hours |
| H100 (×8) | p5.48xlarge | $98.32 | 81 GPU-hours |

H100 is 3x more expensive per GPU-hour but does 8x the FLOPS. So for compute-bound workloads, H100 is actually cheaper per FLOP.

### Cost per token trained

```
Cost per token = GPU-hours used × cost per GPU-hour / tokens processed
```

This is the metric that matters for training budgets. "We can train on 1 trillion tokens for $X" is how labs plan.

### Where money gets wasted

1. **GPU idle time** — GPUs sitting at 0% utilization because training crashed, data pipeline stalled, or the job hasn't started yet. This is the #1 source of waste.

2. **Communication overhead** — GPUs waiting for NCCL operations instead of computing. Every percent of MFU you don't achieve is wasted money.

3. **Failed runs** — Training crashes at step 10,000, no checkpoint. All that compute is gone.

4. **Bad hyperparameters** — Training runs for 3 days and the model doesn't converge because the learning rate was wrong. Prevention: small experiment first, then scale up.

5. **Over-provisioning** — Using 8 GPUs when 4 would train at 90% the speed. The scaling efficiency might not justify the extra GPUs.

### How a $1B training budget breaks down

A lab with $1B for a frontier model run:

```
Total budget:              $1,000,000,000
GPU compute (70%):         $700,000,000
  → ~7.1M H100 GPU-hours
  → ~890K H100-hours (8 GPUs per instance)
  → ~37,000 H100-days
  → Equivalent to 10,000 H100s running for ~3.7 months

Infrastructure (15%):      $150,000,000
  Networking, storage, cooling, data center space

Engineering (10%):         $100,000,000
  Salaries for the team building and operating the factory

Data (5%):                 $50,000,000
  Acquiring, cleaning, curating training data
```

The person managing this budget needs to know:
- What MFU are we achieving? (Every 1% improvement at this scale saves $7M)
- What's our checkpoint overhead? (1% overhead = $7M wasted on I/O)
- How much compute do we lose to failures? (If 5% of GPU-hours are wasted on crashed runs, that's $35M)
- Is our data pipeline keeping up? (If GPUs idle 2% of the time waiting for data, that's $14M)

This is why the "factory builder" role exists. These numbers are too big to get wrong.

---

## Part 13: Scaling — What Changes When You Add More GPUs

### 4 GPUs (1 node) — where we start

Everything is connected by NVLink. FSDP communication is fast. The bottleneck is compute — you're limited by how fast the V100s can multiply matrices.

This is Phase 1 of your project. You'll measure baseline MFU, profile communication vs compute, establish your reference numbers.

### 8 GPUs (2 nodes) — the first hard jump

Half the GPUs are on a different machine. Communication between nodes goes over the network (Ethernet on p3, ~10 Gbps) instead of NVLink (300 GB/s). This is a 30x bandwidth drop.

FSDP all-gather operations that took 50ms within a node might take 500ms across nodes. Your MFU will drop significantly. The profiling data from this jump is your most valuable artifact — it shows you understand the real scaling challenge.

This is Phase 2. The difference between Phase 1 and Phase 2 numbers tells the whole story of why training infrastructure is hard.

### 64 GPUs (8 nodes) — where architecture decisions matter

You can't just use FSDP across 64 GPUs. The all-gather operations would be enormous. This is where you'd switch to hybrid strategies:

- Tensor Parallelism within each node (8 GPUs, NVLink)
- FSDP or Pipeline Parallelism across nodes (network)

You won't run this (too expensive), but you can credibly project from your 4→8 GPU data. Your scaling analysis should include this projection.

### 1,000+ GPUs — frontier territory

At this scale, everything breaks:
- **Network bandwidth** becomes the dominant bottleneck
- **Stragglers** — one slow GPU out of 1,000 slows everyone down
- **Failure rate** — with 1,000 GPUs, something fails every few hours
- **Checkpoint size** — model checkpoints are hundreds of GB, writing them takes minutes
- **Scheduling** — keeping 1,000 GPUs all busy, all the time, is a logistics problem

This is where the factory builder earns their salary. Not by writing code, but by designing systems that keep 1,000+ GPUs running efficiently.

### The scaling efficiency curve

Ideal (linear) scaling: 2x GPUs = 2x speed. Real scaling is always worse.

```
GPUs    Ideal speedup    Real speedup    Efficiency
1       1.0x             1.0x            100%
2       2.0x             1.8x            90%
4       4.0x             3.2x            80%
8       8.0x             5.6x            70%    ← node boundary hit
16      16.0x            9.6x            60%
64      64.0x            32.0x           50%
256     256.0x           102.0x          40%
1024    1024.0x          307.0x          30%
```

These numbers are made up — your actual measurements will differ. The point is: scaling efficiency always drops. The question is how fast. Your project measures this on real hardware and explains why.

---

## Part 14: What Your Project Proves

Everything in the repo is designed to demonstrate specific skills:

| Component | Proves you can... |
|---|---|
| FSDP trainer running on real hardware | Operate distributed training, not just read about it |
| Profiling with real numbers and MFU | Identify and diagnose performance bottlenecks |
| Scaling benchmark (4 GPU → 8 GPU) | Understand the communication overhead problem |
| Checkpoint async writer + S3 sync | Build reliability into the training pipeline |
| Fault injection + recovery demo | Handle failures without losing compute |
| Slurm job scripts | Use the tool that real labs use |
| Terraform infrastructure | Provision and manage GPU infrastructure as code |
| Ops journal with real incidents | Debug GPU cluster problems under pressure |
| Cost projection | Translate GPU metrics into business decisions |
| Inference serving + benchmark | Close the train-to-serve loop with economics |

### What this project does NOT prove (and what comes next)

- **You haven't operated at 100+ GPU scale** — this is 4-8 GPUs. The Mistral SRE role fills this gap.
- **You haven't trained a frontier model** — fine-tuning a 7B model is not pre-training a 70B model. But the infrastructure challenges are the same, just multiplied.
- **You haven't done multi-month training runs** — your runs are hours, not weeks. But the checkpoint/recovery architecture scales.

The project is a proof-of-concept that demonstrates you understand every layer. The next step (the job) is operating it at real scale.

---

## Part 15: Glossary

| Term | What it means |
|---|---|
| **Activation** | Intermediate values computed during forward pass. Stored in GPU memory for the backward pass. |
| **All-gather** | NCCL operation: every GPU sends its piece, every GPU gets the full collection. |
| **All-reduce** | NCCL operation: every GPU sends a value, every GPU gets the average/sum. |
| **Batch size** | Number of training examples processed in one step. |
| **Checkpoint** | Saved snapshot of model weights, optimizer state, and training step. |
| **DDP** | Distributed Data Parallel. Each GPU has a full copy of the model. |
| **EFA** | Elastic Fabric Adapter. AWS high-speed networking for GPU instances. |
| **Epoch** | One complete pass through the entire training dataset. |
| **Expert parallelism** | Distributing MoE expert networks across GPUs. |
| **FP16** | 16-bit floating point. Half the memory of FP32. Standard for training. |
| **FSDP** | Fully Sharded Data Parallel. Model split across GPUs to save memory. |
| **Gradient** | How much each parameter should change, calculated during backward pass. |
| **HBM** | High Bandwidth Memory. The GPU's fast memory. |
| **InfiniBand** | High-speed networking for GPU clusters. Faster than Ethernet. |
| **Loss** | How wrong the model's predictions are. Training minimizes this. |
| **MFU** | Model FLOPs Utilization. Percentage of theoretical GPU compute actually used. |
| **Mixed precision** | Using FP16 for most operations but FP32 for sensitive ones. Saves memory. |
| **MoE** | Mixture of Experts. Model with multiple sub-networks, only some active per input. |
| **NCCL** | NVIDIA Collective Communication Library. Software for GPU-to-GPU communication. |
| **NVLink** | High-speed direct connection between GPUs within one machine. |
| **OOM** | Out of Memory. GPU ran out of HBM. |
| **Optimizer** | Algorithm that decides how to update weights (e.g., AdamW). |
| **PCIe** | PCI Express. Standard connection between GPU and CPU. Slower than NVLink. |
| **Pipeline parallelism** | Different model layers on different GPUs, data flows through like assembly line. |
| **Preemption** | Slurm killing a low-priority job to make room for a high-priority one. |
| **Reduce-scatter** | NCCL operation: gradients are combined and distributed to each GPU's shard. |
| **Rank** | A GPU's ID in a distributed training run. Rank 0 is usually the "master." |
| **Sharding** | Splitting the model across GPUs so each GPU holds only a piece. |
| **SIGTERM** | Signal sent to a process to ask it to shut down gracefully. |
| **Slurm** | Job scheduler for GPU clusters. Manages who gets which GPUs when. |
| **Step** | One forward pass + backward pass + weight update. |
| **Tensor parallelism** | Splitting individual layers across GPUs. Needs NVLink. |
| **torchrun** | PyTorch's launcher for distributed training. Sets up ranks and communication. |
| **Throughput** | Tokens processed per second. Higher is better. |
| **vLLM** | Fast inference engine with continuous batching and PagedAttention. |
| **World size** | Total number of GPUs in a distributed training run. |
