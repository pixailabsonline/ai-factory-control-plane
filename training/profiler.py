"""
Profiling wrapper for FSDP training.
Captures torch.profiler traces, NCCL communication breakdown, and MFU calculation.
"""

import os
import json
import time
from pathlib import Path

import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler


V100_FP16_TFLOPS = 125.0
A100_FP16_TFLOPS = 312.0
H100_FP16_TFLOPS = 989.8

GPU_TFLOPS = {
    "V100": V100_FP16_TFLOPS,
    "A100": A100_FP16_TFLOPS,
    "H100": H100_FP16_TFLOPS,
}


def detect_gpu_model():
    name = torch.cuda.get_device_name(0)
    for key in GPU_TFLOPS:
        if key in name:
            return key
    return "V100"


def calculate_mfu(tokens_per_sec, model_params, seq_length, gpu_count, gpu_model=None):
    if gpu_model is None:
        gpu_model = detect_gpu_model()

    theoretical_tflops = GPU_TFLOPS[gpu_model] * gpu_count
    flops_per_token = 6 * model_params
    achieved_tflops = (tokens_per_sec * flops_per_token) / 1e12

    mfu = achieved_tflops / theoretical_tflops
    return {
        "mfu_percent": round(mfu * 100, 1),
        "achieved_tflops": round(achieved_tflops, 1),
        "theoretical_tflops": theoretical_tflops,
        "gpu_model": gpu_model,
        "gpu_count": gpu_count,
        "tokens_per_sec": round(tokens_per_sec, 1),
        "model_params": model_params,
    }


def profile_training_step(model, batch, local_rank, output_dir="./profiling"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("forward"):
            outputs = model(
                input_ids=batch["input_ids"].to(local_rank),
                attention_mask=batch["attention_mask"].to(local_rank),
                labels=batch["input_ids"].to(local_rank),
            )
        with record_function("backward"):
            outputs.loss.backward()

    trace_path = output_path / "trace.json"
    prof.export_chrome_trace(str(trace_path))

    key_averages = prof.key_averages()

    total_cuda_time = sum(e.cuda_time_total for e in key_averages)
    nccl_time = sum(e.cuda_time_total for e in key_averages if "nccl" in e.key.lower())
    compute_time = total_cuda_time - nccl_time

    breakdown = {
        "total_cuda_time_ms": total_cuda_time / 1000,
        "nccl_time_ms": nccl_time / 1000,
        "compute_time_ms": compute_time / 1000,
        "nccl_percent": round(nccl_time / total_cuda_time * 100, 1) if total_cuda_time > 0 else 0,
        "compute_percent": round(compute_time / total_cuda_time * 100, 1) if total_cuda_time > 0 else 0,
        "top_ops": [],
    }

    for event in sorted(key_averages, key=lambda e: e.cuda_time_total, reverse=True)[:10]:
        breakdown["top_ops"].append({
            "name": event.key,
            "cuda_time_ms": event.cuda_time_total / 1000,
            "cpu_time_ms": event.cpu_time_total / 1000,
            "calls": event.count,
        })

    breakdown_path = output_path / "communication_breakdown.json"
    with open(breakdown_path, "w") as f:
        json.dump(breakdown, f, indent=2)

    return breakdown


def profile_nccl_ops(model, batch, local_rank, num_steps=5, output_dir="./profiling"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    nccl_timings = []

    for step in range(num_steps):
        start = time.time()
        outputs = model(
            input_ids=batch["input_ids"].to(local_rank),
            attention_mask=batch["attention_mask"].to(local_rank),
            labels=batch["input_ids"].to(local_rank),
        )
        torch.cuda.synchronize()
        forward_time = time.time() - start

        start = time.time()
        outputs.loss.backward()
        torch.cuda.synchronize()
        backward_time = time.time() - start

        nccl_timings.append({
            "step": step,
            "forward_ms": forward_time * 1000,
            "backward_ms": backward_time * 1000,
            "total_ms": (forward_time + backward_time) * 1000,
        })

    timing_path = output_path / "step_timings.json"
    with open(timing_path, "w") as f:
        json.dump(nccl_timings, f, indent=2)

    return nccl_timings
