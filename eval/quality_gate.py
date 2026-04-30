"""
Quality gate — automated evaluation after training.
Pass/fail threshold determines whether a model is promoted to serving.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def evaluate_perplexity(model, tokenizer, dataset, max_samples=500, max_length=512, device="cuda"):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break

            inputs = tokenizer(
                sample["text"],
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            if inputs["input_ids"].shape[1] < 2:
                continue

            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return {"perplexity": perplexity, "avg_loss": avg_loss, "tokens_evaluated": total_tokens}


def evaluate_generation(model, tokenizer, prompts, max_new_tokens=128, device="cuda"):
    model.eval()
    results = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append({"prompt": prompt, "generated": generated})

    return results


def run_quality_gate(checkpoint_path, baseline_perplexity=None, max_perplexity=None,
                     eval_dataset="wikitext/wikitext-2-raw-v1", device="cuda", model_name=None):
    checkpoint = torch.load(checkpoint_path / "state.pt", map_location=device, weights_only=False)

    model_name = model_name or checkpoint.get("model_name", "distilgpt2")
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(f"Model: {model_name}, Step: {checkpoint.get('step', 0)}")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.load_state_dict(checkpoint["model"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(eval_dataset, split="test")

    perplexity_result = evaluate_perplexity(model, tokenizer, dataset, device=device)

    test_prompts = [
        "The key difference between supervised and unsupervised learning is",
        "To debug a CUDA out of memory error, you should first",
        "The transformer architecture consists of",
    ]
    generation_result = evaluate_generation(model, tokenizer, test_prompts, device=device)

    passed = True
    reasons = []

    if max_perplexity and perplexity_result["perplexity"] > max_perplexity:
        passed = False
        reasons.append(f"perplexity {perplexity_result['perplexity']:.2f} exceeds threshold {max_perplexity}")

    if baseline_perplexity and perplexity_result["perplexity"] > baseline_perplexity * 1.05:
        passed = False
        reasons.append(f"perplexity {perplexity_result['perplexity']:.2f} regressed vs baseline {baseline_perplexity:.2f}")

    result = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": str(checkpoint_path),
        "step": checkpoint.get("step", 0),
        "passed": passed,
        "reasons": reasons,
        "perplexity": perplexity_result,
        "generations": generation_result,
    }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None, help="Override model name (defaults to value saved in checkpoint)")
    parser.add_argument("--baseline-perplexity", type=float, default=None)
    parser.add_argument("--max-perplexity", type=float, default=None)
    parser.add_argument("--eval-dataset", default="wikitext/wikitext-2-raw-v1")
    parser.add_argument("--output", default="eval_result.json")
    args = parser.parse_args()

    result = run_quality_gate(
        Path(args.checkpoint),
        baseline_perplexity=args.baseline_perplexity,
        max_perplexity=args.max_perplexity,
        eval_dataset=args.eval_dataset,
        model_name=args.model,
    )

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    status = "PASS" if result["passed"] else "FAIL"
    print(f"Quality gate: {status}")
    print(f"  Perplexity: {result['perplexity']['perplexity']:.2f}")
    if result["reasons"]:
        for r in result["reasons"]:
            print(f"  FAIL: {r}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    exit(main())
