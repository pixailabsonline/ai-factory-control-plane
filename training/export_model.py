"""
Export a trained checkpoint into a Hugging Face-style model directory that vLLM can load.

This does not commit artifacts to git. It converts a saved training checkpoint into a
servable model folder with weights, config, and tokenizer files.
"""

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_dir_checksum(path: Path) -> str:
    sha256 = hashlib.sha256()
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        sha256.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
    return sha256.hexdigest()


def export_checkpoint(checkpoint_dir: Path, output_dir: Path, model_name: str | None = None, tokenizer_name: str | None = None):
    checkpoint = torch.load(checkpoint_dir / "state.pt", map_location="cpu", weights_only=False)

    model_name = model_name or checkpoint.get("model_name") or "distilgpt2"
    tokenizer_name = tokenizer_name or model_name

    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_dir)

    files = [p for p in output_dir.rglob("*") if p.is_file()]
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "source_checkpoint": str(checkpoint_dir),
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "source_step": checkpoint.get("step", 0),
        "artifact_format": "huggingface_model_dir",
        "artifact_file_count": len(files),
        "artifact_size_bytes": sum(p.stat().st_size for p in files),
        "artifact_checksum": f"sha256:{compute_dir_checksum(output_dir)}",
    }
    with open(output_dir / "export_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory containing state.pt")
    parser.add_argument("--output-dir", required=True, help="Destination model directory for vLLM")
    parser.add_argument("--model", default=None, help="Override base model name if not stored in the checkpoint")
    parser.add_argument("--tokenizer", default=None, help="Override tokenizer name")
    args = parser.parse_args()

    manifest = export_checkpoint(
        Path(args.checkpoint),
        Path(args.output_dir),
        model_name=args.model,
        tokenizer_name=args.tokenizer,
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
