"""
Phase 3: Exact deduplication of normalized JSONL records.

Two records are duplicates if their content_hash is identical.
The first occurrence is kept; subsequent ones are dropped.

Records the dedup ratio and surviving corpus size in a manifest.
Near-deduplication (MinHash/SimHash) is explicitly out of scope for v1.

Usage:
    python data_pipeline/dedup_exact.py \
        --input s3://my-bucket/runs/my-run/normalized/normalized.jsonl \
        --output-dir /tmp/deduped \
        --s3-bucket my-bucket \
        --run-name my-run-001
"""

import argparse
import gzip
import hashlib
import json
import os
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
        print(f"[dedup] downloading {input_path}")
        download_from_s3(bucket, key, local)
        return local
    return Path(input_path)


def dedup(input_path: Path) -> tuple[list[dict], dict]:
    seen: set[str] = set()
    kept = []
    total = 0
    duplicates = 0

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            h = rec.get("content_hash")
            if h is None:
                # compute on the fly if not present
                h = hashlib.sha256(rec.get("text", "").encode()).hexdigest()
                rec["content_hash"] = h
            if h in seen:
                duplicates += 1
                continue
            seen.add(h)
            kept.append(rec)

    return kept, {
        "total_input": total,
        "duplicates_removed": duplicates,
        "surviving": len(kept),
        "dedup_ratio": round(duplicates / total, 4) if total else 0.0,
        "method": "exact_sha256",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path or S3 URI to normalized.jsonl")
    ap.add_argument("--output-dir", default="/tmp/deduped")
    ap.add_argument("--s3-bucket", default=os.environ.get("S3_BUCKET", ""))
    ap.add_argument("--run-name", default=os.environ.get("RUN_NAME", ""), required=False)
    args = ap.parse_args()

    if not args.run_name:
        ap.error("--run-name is required (or set RUN_NAME env var)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    local_input = resolve_input(args.input, args.s3_bucket, tmp_dir)

    print(f"[dedup] running exact dedup on {local_input}")
    records, stats = dedup(local_input)

    output_path = output_dir / "deduped.jsonl"
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    checksum = sha256_file(output_path)

    dedup_manifest = {
        "manifest_type": "dedup",
        "run_name": args.run_name,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "input": args.input,
        "dedup_stats": stats,
        "output_record_count": len(records),
        "output_checksum_sha256": checksum,
        "output_size_bytes": output_path.stat().st_size,
        "near_dedup": False,
    }

    manifest_path = output_dir / "dedup_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(dedup_manifest, f, indent=2)

    if args.s3_bucket:
        run_root = f"runs/{args.run_name}/deduped"
        upload_to_s3(output_path, args.s3_bucket, f"{run_root}/deduped.jsonl")
        upload_to_s3(manifest_path, args.s3_bucket, f"{run_root}/dedup_manifest.json")
        print(f"[dedup] → s3://{args.s3_bucket}/{run_root}/")

    print(f"[dedup] {stats['total_input']} in → {stats['surviving']} out "
          f"({stats['duplicates_removed']} removed, dedup_ratio={stats['dedup_ratio']})")


if __name__ == "__main__":
    main()
