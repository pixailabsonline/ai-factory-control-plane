"""
Inference benchmark against vLLM's OpenAI-compatible API.
Measures latency, throughput, time to first token, and cost per token.
"""

import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


PROMPTS = [
    "Explain the difference between FSDP and DDP in distributed training.",
    "What is the purpose of gradient accumulation in large batch training?",
    "Describe how NVLink improves multi-GPU training performance.",
    "What are the tradeoffs between model parallelism and data parallelism?",
    "How does mixed precision training reduce memory usage?",
    "Explain the concept of Model FLOPs Utilization (MFU).",
    "What happens during an all-reduce operation in distributed training?",
    "Why is checkpoint management critical for large-scale training runs?",
    "Describe the memory savings from ZeRO Stage 3 optimization.",
    "What is pipeline parallelism and when should you use it?",
]

INSTANCE_COST_PER_HOUR = {
    "p3.8xlarge": 12.24,
    "p3.2xlarge": 3.06,
    "g5.xlarge": 1.006,
    "g5.2xlarge": 1.212,
}


def bench_single(url, prompt, max_tokens=128):
    start = time.time()
    resp = requests.post(
        f"{url}/v1/completions",
        json={
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        },
    )
    elapsed = time.time() - start

    if resp.status_code != 200:
        return {"error": resp.text, "latency_sec": elapsed}

    data = resp.json()
    choice = data["choices"][0]
    tokens_generated = data["usage"]["completion_tokens"]

    return {
        "tokens_generated": tokens_generated,
        "latency_sec": elapsed,
        "tokens_per_sec": tokens_generated / elapsed if elapsed > 0 else 0,
    }


def bench_throughput(url, num_requests=50, max_tokens=128, concurrency=8):
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_requests)]

    # Warmup
    for i in range(3):
        bench_single(url, PROMPTS[i])

    results = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(bench_single, url, p, max_tokens) for p in prompts]
        for f in as_completed(futures):
            results.append(f.result())
    total_time = time.time() - start

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    total_tokens = sum(r["tokens_generated"] for r in successful)

    latencies = sorted([r["latency_sec"] for r in successful])

    return {
        "total_requests": num_requests,
        "successful": len(successful),
        "failed": len(failed),
        "concurrency": concurrency,
        "total_time_sec": round(total_time, 2),
        "total_tokens_generated": total_tokens,
        "aggregate_tokens_per_sec": round(total_tokens / total_time, 1),
        "requests_per_sec": round(len(successful) / total_time, 2),
        "latency_p50_sec": round(latencies[len(latencies) // 2], 3) if latencies else 0,
        "latency_p95_sec": round(latencies[int(len(latencies) * 0.95)], 3) if latencies else 0,
        "latency_p99_sec": round(latencies[int(len(latencies) * 0.99)], 3) if latencies else 0,
    }


def cost_analysis(bench_result, instance_type):
    cost_per_hour = INSTANCE_COST_PER_HOUR.get(instance_type, 0)
    if cost_per_hour == 0 or bench_result["aggregate_tokens_per_sec"] == 0:
        return {}

    cost_per_sec = cost_per_hour / 3600
    tokens_per_sec = bench_result["aggregate_tokens_per_sec"]
    cost_per_1m_tokens = (cost_per_sec / tokens_per_sec) * 1_000_000

    # Compare against API pricing
    api_prices = {
        "Claude Sonnet 4.6 output": 15.00,
        "Claude Haiku 4.5 output": 5.00,
        "GPT-4o output": 15.00,
    }

    comparisons = {}
    for name, api_price in api_prices.items():
        breakeven_util = (cost_per_1m_tokens / api_price) * 100
        comparisons[name] = {
            "api_cost_per_1m": api_price,
            "breakeven_utilization_pct": round(breakeven_util, 1),
            "cheaper_self_hosted": cost_per_1m_tokens < api_price,
        }

    return {
        "instance_type": instance_type,
        "cost_per_hour": cost_per_hour,
        "self_hosted_cost_per_1m_tokens": round(cost_per_1m_tokens, 4),
        "vs_api": comparisons,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--instance-type", default="p3.8xlarge")
    parser.add_argument("--output", default="inference_bench.json")
    args = parser.parse_args()

    print(f"Benchmarking vLLM at {args.url}")
    print(f"  Requests: {args.requests}, Concurrency: {args.concurrency}, Max tokens: {args.max_tokens}")

    result = bench_throughput(args.url, args.requests, args.max_tokens, args.concurrency)
    cost = cost_analysis(result, args.instance_type)

    output = {"benchmark": result, "cost_analysis": cost}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nThroughput: {result['aggregate_tokens_per_sec']} tokens/sec")
    print(f"Latency P50: {result['latency_p50_sec']}s  P95: {result['latency_p95_sec']}s  P99: {result['latency_p99_sec']}s")
    print(f"Requests/sec: {result['requests_per_sec']}")

    if cost:
        print(f"\nSelf-hosted cost: ${cost['self_hosted_cost_per_1m_tokens']:.4f} per 1M tokens")
        for name, c in cost.get("vs_api", {}).items():
            status = "CHEAPER" if c["cheaper_self_hosted"] else "MORE EXPENSIVE"
            print(f"  vs {name}: {status} (breakeven at {c['breakeven_utilization_pct']}% utilization)")


if __name__ == "__main__":
    main()
