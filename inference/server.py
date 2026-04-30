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
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1",
                        help="Base model name or Hugging Face repo ID")
    parser.add_argument("--model-dir", default=None,
                        help="Path to an exported, serveable model directory")
    parser.add_argument("--checkpoint", default=None,
                        help="Deprecated alias for --model-dir")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    model_path = args.model_dir or args.checkpoint or args.model
    model_source = "exported model directory" if args.model_dir or args.checkpoint else "base model"

    model_path_obj = Path(model_path)
    if model_path_obj.exists() and model_path_obj.is_dir():
        if (model_path_obj / "state.pt").exists() and not (model_path_obj / "config.json").exists():
            raise SystemExit(
                f"{model_path} looks like a raw training checkpoint, not a serveable model directory. "
                "Export it first with training/export_model.py, then point --model-dir at the exported artifact."
            )

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
    print(f"  Source: {model_source}")
    print(f"  Tensor parallel: {args.tensor_parallel} GPU(s)")
    print(f"  Max context: {args.max_model_len} tokens")
    print(f"  Endpoint: http://{args.host}:{args.port}/v1/completions")
    print(f"  Health: http://{args.host}:{args.port}/health")

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
