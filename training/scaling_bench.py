"""
Scaling benchmark — systematic measurement across GPU configs.
Runs the same workload at 1/2/4/8 GPUs and measures throughput,
communication overhead, and scaling efficiency.
"""

import os
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def run_config(num_gpus, num_nodes, model, batch_size, max_steps, output_dir):
    config_name = f"{num_nodes}node_{num_gpus}gpu_bs{batch_size}"
    config_dir = Path(output_dir) / config_name
    config_dir.mkdir(parents=True, exist_ok=True)

    nproc_per_node = num_gpus // num_nodes

    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc_per_node}",
    ]

    if num_nodes > 1:
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        cmd.extend([
            f"--nnodes={num_nodes}",
            "--rdzv_backend=c10d",
            f"--rdzv_endpoint={master_addr}:29500",
        ])

    cmd.extend([
        "training/fsdp_trainer.py",
        f"--model={model}",
        f"--batch-size={batch_size}",
        "--gradient-accumulation=1",
        f"--max-steps={max_steps}",
        f"--checkpoint-dir={config_dir}/checkpoints",
        "--checkpoint-every=999999",
    ])

    env = os.environ.copy()
    env["NCCL_DEBUG"] = "INFO"
    env["NCCL_DEBUG_SUBSYS"] = "ALL"

    nccl_log = config_dir / "nccl.log"

    start = time.time()
    with open(nccl_log, "w") as log_f:
        result = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=log_f,
            text=True,
        )
    elapsed = time.time() - start

    metrics_path = config_dir / "checkpoints" / "training_metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return {
        "config": config_name,
        "num_gpus": num_gpus,
        "num_nodes": num_nodes,
        "batch_size": batch_size,
        "wall_time_sec": round(elapsed, 1),
        "exit_code": result.returncode,
        "final_tokens_per_sec": metrics.get("final_tokens_per_sec", 0),
        "total_tokens": metrics.get("total_tokens", 0),
        "total_steps": metrics.get("total_steps", 0),
    }


def compute_scaling_efficiency(results):
    if not results:
        return results

    baseline = None
    for r in results:
        if r["num_gpus"] == 1 and r["exit_code"] == 0:
            baseline = r
            break

    if baseline is None:
        baseline = next((r for r in results if r["exit_code"] == 0), None)

    if baseline is None:
        return results

    baseline_tps = baseline["final_tokens_per_sec"]
    baseline_gpus = baseline["num_gpus"]

    for r in results:
        if r["exit_code"] != 0 or r["final_tokens_per_sec"] == 0:
            r["scaling_efficiency"] = 0
            r["speedup"] = 0
            continue

        ideal_speedup = r["num_gpus"] / baseline_gpus
        actual_speedup = r["final_tokens_per_sec"] / baseline_tps
        r["speedup"] = round(actual_speedup, 2)
        r["scaling_efficiency"] = round(actual_speedup / ideal_speedup * 100, 1)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--output-dir", default="./scaling_results")
    parser.add_argument("--configs", default="1:1:4,2:1:4,4:1:4",
                        help="Comma-separated gpu:nodes:batch_size configs")
    args = parser.parse_args()

    configs = []
    for c in args.configs.split(","):
        parts = c.split(":")
        configs.append({
            "num_gpus": int(parts[0]),
            "num_nodes": int(parts[1]),
            "batch_size": int(parts[2]),
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Running: {cfg['num_gpus']} GPUs, {cfg['num_nodes']} nodes, batch {cfg['batch_size']}")
        print(f"{'='*60}")

        result = run_config(
            cfg["num_gpus"], cfg["num_nodes"], args.model,
            cfg["batch_size"], args.max_steps, output_dir,
        )
        results.append(result)
        print(f"  Tokens/sec: {result['final_tokens_per_sec']}")
        print(f"  Exit code: {result['exit_code']}")

    results = compute_scaling_efficiency(results)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "max_steps": args.max_steps,
        "results": results,
    }

    summary_path = output_dir / "scaling_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("SCALING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'Tokens/s':<12} {'Speedup':<10} {'Efficiency':<10}")
    print(f"{'-'*60}")
    for r in results:
        eff = f"{r.get('scaling_efficiency', 0)}%" if r['exit_code'] == 0 else "FAILED"
        print(f"{r['config']:<25} {r['final_tokens_per_sec']:<12.0f} {r.get('speedup', 0):<10.2f} {eff:<10}")


if __name__ == "__main__":
    main()
