"""
FSDP fine-tuning trainer for 7B+ parameter models on multi-GPU setups.
Designed for p3.8xlarge (4x V100 16GB) and multi-node p3 clusters.
"""

import os
import sys
import time
import json
import signal
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from checkpoint.async_writer import AsyncCheckpointWriter


_SIGTERM_RECEIVED = False


def _handle_sigterm(signum, frame):
    global _SIGTERM_RECEIVED
    _SIGTERM_RECEIVED = True


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size == 1 and "MASTER_ADDR" not in os.environ:
        return rank, world_size, local_rank, False

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, True


def get_fsdp_config(sharding_strategy="FULL_SHARD"):
    strategies = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }

    if torch.cuda.is_available():
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    else:
        mp_policy = None

    config = {
        "sharding_strategy": strategies.get(sharding_strategy, ShardingStrategy.FULL_SHARD),
        "mixed_precision": mp_policy,
        "limit_all_gathers": True,
    }
    if torch.cuda.is_available():
        config["device_id"] = torch.cuda.current_device()
    return config


class SyntheticCausalLMDataset(Dataset):
    def __init__(self, size=64, max_length=32, vocab_size=128):
        self.size = size
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        tokens = torch.arange(self.max_length, dtype=torch.long) % self.vocab_size
        tokens = (tokens + idx) % self.vocab_size
        attention_mask = torch.ones(self.max_length, dtype=torch.long)
        return {"input_ids": tokens, "attention_mask": attention_mask}


def build_smoke_model(max_length):
    config = GPT2Config(
        vocab_size=128,
        n_positions=max_length,
        n_ctx=max_length,
        n_embd=64,
        n_layer=2,
        n_head=4,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )
    return GPT2LMHeadModel(config)


def load_model_and_tokenizer(model_name, rank, smoke_test=False, max_length=32):
    if rank == 0:
        print(f"Loading model: {model_name}")

    if smoke_test:
        return build_smoke_model(max_length), None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_cache=False,
    )

    return model, tokenizer


def prepare_dataset(tokenizer, dataset_name, max_length=512, split="train", smoke_test=False):
    if smoke_test:
        return SyntheticCausalLMDataset(size=64, max_length=max_length)

    dataset = load_dataset(dataset_name, split=split)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    dataset.set_format("torch")
    return dataset


def train(args):
    signal.signal(signal.SIGTERM, _handle_sigterm)

    rank, world_size, local_rank, distributed = setup_distributed()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        rank,
        smoke_test=args.smoke_test,
        max_length=args.max_length,
    )
    model = model.to(device)

    use_fsdp = distributed and torch.cuda.is_available() and not args.smoke_test
    if use_fsdp:
        fsdp_config = get_fsdp_config(args.sharding_strategy)
        model = FSDP(model, **fsdp_config)

    dataset = prepare_dataset(
        tokenizer,
        args.dataset,
        max_length=args.max_length,
        smoke_test=args.smoke_test,
    )
    sampler = None
    if distributed and world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    ckpt_writer = AsyncCheckpointWriter(
        base_dir=args.checkpoint_dir,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        max_kept=args.max_checkpoints,
    )

    # Restore from latest checkpoint if available
    global_step = ckpt_writer.restore(model, optimizer, scheduler, local_rank)
    if rank == 0 and global_step > 0:
        print(f"Resumed from step {global_step}")

    metrics_log = []
    start_time = time.time()
    tokens_processed = 0

    if rank == 0:
        print(f"Training config:")
        print(f"  Model: {args.model}")
        print(f"  World size: {world_size}")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Effective batch size: {args.batch_size * world_size * args.gradient_accumulation}")
        print(f"  Sharding: {args.sharding_strategy}")
        print(f"  Max steps: {args.max_steps}")
        print(f"  Checkpoint every: {args.checkpoint_every} steps")
        print(f"  Resuming from step: {global_step}")

    model.train()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(dataloader):
            if _SIGTERM_RECEIVED:
                if rank == 0:
                    print(f"SIGTERM — saving checkpoint at step {global_step}")
                ckpt_writer.save(model, optimizer, scheduler, global_step, rank=rank)
                ckpt_writer.wait()
                if distributed:
                    dist.destroy_process_group()
                return

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss / args.gradient_accumulation
            loss.backward()

            tokens_in_batch = int(attention_mask.sum().item()) * world_size
            tokens_processed += tokens_in_batch

            if (batch_idx + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = tokens_processed / elapsed
                    step_metrics = {
                        "step": global_step,
                        "loss": loss.item() * args.gradient_accumulation,
                        "lr": scheduler.get_last_lr()[0],
                        "tokens_per_sec": tokens_per_sec,
                        "tokens_processed": tokens_processed,
                        "elapsed_sec": elapsed,
                        "gpu_mem_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                        "gpu_mem_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                    }
                    metrics_log.append(step_metrics)

                    if global_step % 10 == 0:
                        print(
                            f"Step {global_step}/{args.max_steps} | "
                            f"Loss: {step_metrics['loss']:.4f} | "
                            f"LR: {step_metrics['lr']:.2e} | "
                            f"Tokens/sec: {tokens_per_sec:.0f} | "
                            f"GPU Mem: {step_metrics['gpu_mem_allocated_mb']:.0f}MB"
                        )

                if global_step % args.checkpoint_every == 0:
                    ckpt_writer.save(
                        model, optimizer, scheduler, global_step,
                        metrics={"loss": loss.item() * args.gradient_accumulation},
                        rank=rank,
                    )

                if global_step >= args.max_steps:
                    break

        if global_step >= args.max_steps:
            break

    # Wait for any pending async checkpoint write
    ckpt_writer.wait()

    if rank == 0:
        elapsed = time.time() - start_time
        metrics_path = Path(args.checkpoint_dir) / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "config": vars(args),
                "world_size": world_size,
                "total_steps": global_step,
                "total_tokens": tokens_processed,
                "total_time_sec": elapsed,
                "final_tokens_per_sec": tokens_processed / elapsed,
                "steps": metrics_log,
            }, f, indent=2)
        print(f"\nTraining complete. {global_step} steps, {tokens_processed:,} tokens in {elapsed:.0f}s")
        print(f"Throughput: {tokens_processed / elapsed:.0f} tokens/sec")

    # Final checkpoint
    ckpt_writer.save(model, optimizer, scheduler, global_step, rank=rank)
    ckpt_writer.wait()

    if distributed:
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--dataset", default="wikitext/wikitext-103-raw-v1")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--max-checkpoints", type=int, default=5)
    parser.add_argument("--s3-bucket", default=None)
    parser.add_argument("--s3-prefix", default=None)
    parser.add_argument("--sharding-strategy", default="FULL_SHARD",
                        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"])
    parser.add_argument("--smoke-test", action="store_true",
                        help="Use a tiny synthetic dataset and a tiny local model for a fast local validation run.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
