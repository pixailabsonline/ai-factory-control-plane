"""
Inference server for serving fine-tuned models.
Measures latency, throughput, and cost per token for the model you trained.
"""

import json
import time
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify


app = Flask(__name__)
model = None
tokenizer = None
stats = {"requests": 0, "tokens_generated": 0, "total_latency_ms": 0}


def load_model(model_name, checkpoint_path=None, device="cuda"):
    global model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)

    if checkpoint_path:
        state = torch.load(
            Path(checkpoint_path) / "state.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(state["model"])
        print(f"Loaded checkpoint from step {state.get('step', '?')}")

    model.eval()
    print(f"Model loaded: {model_name} on {device}")


@app.route("/v1/completions", methods=["POST"])
def completions():
    body = request.json
    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", 128)
    temperature = body.get("temperature", 0.7)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    start = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.9,
        )
    elapsed_ms = (time.time() - start) * 1000

    generated_ids = output[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens_generated = len(generated_ids)

    stats["requests"] += 1
    stats["tokens_generated"] += tokens_generated
    stats["total_latency_ms"] += elapsed_ms

    return jsonify({
        "text": generated_text,
        "tokens_generated": tokens_generated,
        "latency_ms": round(elapsed_ms, 1),
        "tokens_per_sec": round(tokens_generated / (elapsed_ms / 1000), 1) if elapsed_ms > 0 else 0,
    })


@app.route("/v1/stats", methods=["GET"])
def get_stats():
    avg_latency = stats["total_latency_ms"] / stats["requests"] if stats["requests"] > 0 else 0
    avg_tokens_per_req = stats["tokens_generated"] / stats["requests"] if stats["requests"] > 0 else 0

    return jsonify({
        "total_requests": stats["requests"],
        "total_tokens_generated": stats["tokens_generated"],
        "avg_latency_ms": round(avg_latency, 1),
        "avg_tokens_per_request": round(avg_tokens_per_req, 1),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    load_model(args.model, args.checkpoint)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
