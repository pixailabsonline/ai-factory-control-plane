"""
Inference server using vLLM — production-grade model serving with:
- Continuous batching (serves multiple requests concurrently)
- PagedAttention (efficient KV cache, higher throughput)
- Tensor parallelism (split model across GPUs)
- OpenAI-compatible API out of the box
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to fine-tuned checkpoint (merged weights)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    model_path = args.checkpoint if args.checkpoint else args.model

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--disable-log-requests",
    ]

    print(f"Starting vLLM server:")
    print(f"  Model: {model_path}")
    print(f"  Tensor parallel: {args.tensor_parallel} GPU(s)")
    print(f"  Max context: {args.max_model_len} tokens")
    print(f"  Endpoint: http://{args.host}:{args.port}/v1/completions")
    print(f"  Health: http://{args.host}:{args.port}/health")

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
