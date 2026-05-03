"""
Phase 4: Tokenization and packing of deduplicated JSONL into fixed-length training sequences.

Pins the tokenizer version, records max length and packing strategy,
and emits token counts and packed sequence counts.

The same input + same tokenizer version + same max_length = same output (deterministic).

Usage:
    python data_pipeline/tokenize_pack.py \
        --input s3://my-bucket/runs/my-run/deduped/deduped.jsonl \
        --output-dir /tmp/packed \
        --s3-bucket my-bucket \
        --run-name my-run-001 \
        --tokenizer gpt2 \
        --max-length 1024
"""

import argparse
import hashlib
import json
import os
import struct
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_from_s3(bucket: str, key: str, dest: Path) -> None:
    import boto3
    dest.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, str(dest))


def upload_to_s3(local_path: Path, bucket: str, key: str) -> None:
    import boto3
    boto3.client("s3").upload_file(str(local_path), bucket, key)


def resolve_input(input_path: str, s3_bucket: str, tmp_dir: Path) -> Path:
    if input_path.startswith("s3://"):
        parts = input_path[5:].split("/", 1)
        bucket, key = parts[0], parts[1]
        local = tmp_dir / Path(key).name
        print(f"[tokenize] downloading {input_path}")
        download_from_s3(bucket, key, local)
        return local
    return Path(input_path)


def pack_sequences(token_ids: list[int], max_length: int) -> list[list[int]]:
    """
    Greedy packing: fill each sequence to max_length, then start a new one.
    This is the simplest deterministic packing strategy.
    """
    sequences = []
    current: list[int] = []
    for tok in token_ids:
        current.append(tok)
        if len(current) == max_length:
            sequences.append(current)
            current = []
    # drop the final incomplete sequence — keeps output deterministic
    return sequences


def tokenize_and_pack(
    input_path: Path,
    output_dir: Path,
    tokenizer_name: str,
    max_length: int,
) -> dict:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer_version = getattr(tokenizer, "vocab_size", "unknown")

    output_bin = output_dir / "packed.bin"
    output_jsonl = output_dir / "packed.jsonl"

    record_count = 0
    total_tokens = 0
    sequence_count = 0
    current: list[int] = []

    # Stream through JSONL — never accumulate all tokens in memory
    with open(input_path) as fin, \
         open(output_bin, "wb") as fbin, \
         open(output_jsonl, "w") as fjsonl:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("text", "")
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(ids)
            record_count += 1
            current.extend(ids)
            # flush complete sequences as we go
            while len(current) >= max_length:
                seq = current[:max_length]
                current = current[max_length:]
                fbin.write(struct.pack(f"<{max_length}i", *seq))
                fjsonl.write(json.dumps({"tokens": seq}) + "\n")
                sequence_count += 1
        # drop the final incomplete sequence — keeps output deterministic

    return {
        "tokenizer_name": tokenizer_name,
        "tokenizer_vocab_size": tokenizer_version,
        "max_length": max_length,
        "packing_strategy": "greedy_fill",
        "input_records": record_count,
        "total_tokens": total_tokens,
        "sequence_count": sequence_count,
        "tokens_packed": sequence_count * max_length,
        "pack_efficiency": round(sequence_count * max_length / total_tokens, 4) if total_tokens else 0.0,
        "output_bin": str(output_bin),
        "output_jsonl": str(output_jsonl),
        "bin_checksum_sha256": sha256_file(output_bin),
        "bin_size_bytes": output_bin.stat().st_size,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path or S3 URI to deduped.jsonl")
    ap.add_argument("--output-dir", default="/tmp/packed")
    ap.add_argument("--s3-bucket", default=os.environ.get("S3_BUCKET", ""))
    ap.add_argument("--run-name", default=os.environ.get("RUN_NAME", ""), required=False)
    ap.add_argument("--tokenizer", default="gpt2", help="HuggingFace tokenizer name or path")
    ap.add_argument("--max-length", type=int, default=1024)
    args = ap.parse_args()

    if not args.run_name:
        ap.error("--run-name is required (or set RUN_NAME env var)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    local_input = resolve_input(args.input, args.s3_bucket, tmp_dir)

    print(f"[tokenize] tokenizing with {args.tokenizer}, max_length={args.max_length}")
    stats = tokenize_and_pack(local_input, output_dir, args.tokenizer, args.max_length)

    tokenize_manifest = {
        "manifest_type": "tokenize",
        "run_name": args.run_name,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "input": args.input,
        "tokenize_stats": stats,
    }

    manifest_path = output_dir / "tokenize_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(tokenize_manifest, f, indent=2)

    if args.s3_bucket:
        run_root = f"runs/{args.run_name}/packed"
        upload_to_s3(output_dir / "packed.bin", args.s3_bucket, f"{run_root}/packed.bin")
        upload_to_s3(output_dir / "packed.jsonl", args.s3_bucket, f"{run_root}/packed.jsonl")
        upload_to_s3(manifest_path, args.s3_bucket, f"{run_root}/tokenize_manifest.json")
        print(f"[tokenize] → s3://{args.s3_bucket}/{run_root}/")

    print(f"[tokenize] {stats['input_records']} records → {stats['total_tokens']:,} tokens "
          f"→ {stats['sequence_count']:,} sequences of {args.max_length} "
          f"(efficiency {stats['pack_efficiency']})")


if __name__ == "__main__":
    main()
