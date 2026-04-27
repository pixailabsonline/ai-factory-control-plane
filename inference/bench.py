"""
Inference benchmark — measures latency, throughput, and cost per token
for comparing self-hosted vs API-based serving.
"""

import json
import time
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


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


def benchmark_single(url, prompt, max_tokens=128):
    start = time.time()
    resp = requests.post(
        f"{url}/v1/completions",
        json={"prompt": prompt, "max_tokens": max_tokens},
    )
    elapsed = time.time() - start

    if resp.status_code != 200:
        return {"error": resp.text, "latency_sec": elapsed}

    result = resp.json()
    result["e2e_latency_sec"] = elapsed
    return result


def benchmark_throughput(url, num_requests=50, max_tokens=128, concurrency=4):
    results = []
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(num_requests)]

    start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(benchmark_single, url, p, max_tokens) for p in prompts]
        for f in as_completed(futures):
            results.append(f.result())
    total_time = time.time() - start

    successful = [r for r in results if "error" not in r]
    total_tokens = sum(r.get("tokens_generated", 0) for r in successful)
    avg_latency = sum(r.get("e2e_latency_sec", 0) for r in successful) / len(successful) if successful else 0

    return {
        "total_requests": num_requests,
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "total_time_sec": round(total_time, 2),
        "total_tokens_generated": total_tokens,
        "requests_per_sec": round(num_requests / total_time, 2),
        "tokens_per_sec": round(total_tokens / total_time, 2),
        "avg_latency_sec": round(avg_latency, 3),
        "concurrency": concurrency,
    }


def cost_comparison(self_hosted_tps, instance_type, api_cost_per_1m_output):
    from cost.tracker import KnownPricing
    pricing = KnownPricing.get(instance_type)
    if not pricing:
        return None

    self_hosted_cost_per_1m = (pricing["on_demand_per_hr"] / 3600 / self_hosted_tps) * 1_000_000

    return {
        "self_hosted_cost_per_1m_tokens": round(self_hosted_cost_per_1m, 4),
        "api_cost_per_1m_tokens": api_cost_per_1m_output,
        "ratio": round(api_cost_per_1m_output / self_hosted_cost_per_1m, 2) if self_hosted_cost_per_1m > 0 else 0,
        "self_hosted_breakeven_utilization": round(self_hosted_cost_per_1m / api_cost_per_1m_output * 100, 1) if api_cost_per_1m_output > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", default="inference_bench.json")
    args = parser.parse_args()

    print(f"Benchmarking {args.url} — {args.requests} requests, concurrency {args.concurrency}")

    print("\nWarmup (5 requests)...")
    for i in range(5):
        benchmark_single(args.url, PROMPTS[i % len(PROMPTS)])

    print("Running benchmark...")
    results = benchmark_throughput(args.url, args.requests, args.max_tokens, args.concurrency)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults:")
    print(f"  Requests/sec: {results['requests_per_sec']}")
    print(f"  Tokens/sec: {results['tokens_per_sec']}")
    print(f"  Avg latency: {results['avg_latency_sec']}s")
    print(f"  Success rate: {results['successful']}/{results['total_requests']}")


if __name__ == "__main__":
    main()
