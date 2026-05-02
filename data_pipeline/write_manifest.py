"""
Phase 5: Assemble the final dataset_manifest.json from all phase manifests.

Pulls together raw_input_manifest, normalize_manifest, dedup_manifest,
and tokenize_manifest into a single versioned artifact.

Publishes to s3://<bucket>/runs/<run-name>/datasets/<dataset-version>/

Usage:
    python data_pipeline/write_manifest.py \
        --run-name my-run-001 \
        --dataset-version v1 \
        --s3-bucket my-bucket
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def download_manifest(bucket: str, key: str) -> dict:
    import boto3
    obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())


def upload_to_s3(local_path: Path, bucket: str, key: str) -> None:
    import boto3
    boto3.client("s3").upload_file(str(local_path), bucket, key)


def load_phase_manifest(run_name: str, phase_path: str, s3_bucket: str, local_dir: Path) -> dict:
    if s3_bucket:
        key = f"runs/{run_name}/{phase_path}"
        try:
            return download_manifest(s3_bucket, key)
        except Exception as e:
            print(f"[manifest] warning: could not load s3://{s3_bucket}/{key}: {e}")
            return {}
    local = local_dir / phase_path
    if local.exists():
        with open(local) as f:
            return json.load(f)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default=os.environ.get("RUN_NAME", ""), required=False)
    ap.add_argument("--dataset-version", default="v1")
    ap.add_argument("--s3-bucket", default=os.environ.get("S3_BUCKET", ""))
    ap.add_argument("--output-dir", default="/tmp/dataset_manifest")
    # Optional forward links — filled in after training
    ap.add_argument("--training-run-id", default="")
    ap.add_argument("--checkpoint-path", default="")
    ap.add_argument("--eval-result-path", default="")
    ap.add_argument("--published-model-path", default="")
    args = ap.parse_args()

    if not args.run_name:
        ap.error("--run-name is required (or set RUN_NAME env var)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_phase_manifest(args.run_name, "raw/raw_input_manifest.json", args.s3_bucket, output_dir)
    norm = load_phase_manifest(args.run_name, "normalized/normalize_manifest.json", args.s3_bucket, output_dir)
    dedup = load_phase_manifest(args.run_name, "deduped/dedup_manifest.json", args.s3_bucket, output_dir)
    tok = load_phase_manifest(args.run_name, "packed/tokenize_manifest.json", args.s3_bucket, output_dir)

    tok_stats = tok.get("tokenize_stats", {})
    dedup_stats = dedup.get("dedup_stats", {})

    dataset_manifest = {
        "manifest_type": "dataset",
        "dataset_version": args.dataset_version,
        "run_name": args.run_name,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),

        # Source provenance
        "source_name": raw.get("source_name", "common_crawl"),
        "crawl_id": raw.get("crawl_id", ""),
        "source_file_count": raw.get("file_count", 0),
        "source_total_bytes": raw.get("total_bytes", 0),
        "source_uri": f"https://commoncrawl.org/connect/blog/crawl-data/{raw.get('crawl_id', '')}",
        "raw_checksum": _pluck(raw, "files", 0, "raw_checksum_sha256"),

        # Pipeline steps applied
        "filters_applied": norm.get("filters_applied", []),
        "filter_stats": norm.get("filter_stats", {}),
        "dedup_method": dedup_stats.get("method", "exact_sha256"),
        "near_dedup": dedup.get("near_dedup", False),
        "dedup_stats": dedup_stats,

        # Tokenization
        "tokenizer_name": tok_stats.get("tokenizer_name", ""),
        "tokenizer_vocab_size": tok_stats.get("tokenizer_vocab_size", ""),
        "max_length": tok_stats.get("max_length", 0),
        "packing_strategy": tok_stats.get("packing_strategy", ""),
        "token_count": tok_stats.get("total_tokens", 0),
        "sequence_count": tok_stats.get("sequence_count", 0),
        "pack_efficiency": tok_stats.get("pack_efficiency", 0.0),

        # Output artifact
        "output_checksum_sha256": tok_stats.get("bin_checksum_sha256", ""),
        "output_size_bytes": tok_stats.get("bin_size_bytes", 0),
        "artifact_location": (
            f"s3://{args.s3_bucket}/runs/{args.run_name}/datasets/{args.dataset_version}/packed.bin"
            if args.s3_bucket else ""
        ),

        # Forward links (filled in after training and eval)
        "training_run_id": args.training_run_id,
        "checkpoint_path": args.checkpoint_path,
        "eval_result_path": args.eval_result_path,
        "published_model_path": args.published_model_path,
    }

    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(dataset_manifest, f, indent=2)

    print(f"[manifest] dataset_manifest.json written to {manifest_path}")

    if args.s3_bucket:
        dataset_root = f"runs/{args.run_name}/datasets/{args.dataset_version}"
        upload_to_s3(manifest_path, args.s3_bucket, f"{dataset_root}/dataset_manifest.json")

        # Copy packed artifact into versioned dataset location
        try:
            import boto3
            s3 = boto3.client("s3")
            src_key = f"runs/{args.run_name}/packed/packed.bin"
            dst_key = f"{dataset_root}/packed.bin"
            s3.copy_object(
                Bucket=args.s3_bucket,
                CopySource={"Bucket": args.s3_bucket, "Key": src_key},
                Key=dst_key,
            )
            print(f"[manifest] packed.bin → s3://{args.s3_bucket}/{dst_key}")
        except Exception as e:
            print(f"[manifest] warning: could not copy packed.bin: {e}")

        print(f"[manifest] dataset → s3://{args.s3_bucket}/{dataset_root}/")

    print(f"[manifest] dataset_version={args.dataset_version} "
          f"tokens={dataset_manifest['token_count']:,} "
          f"sequences={dataset_manifest['sequence_count']:,}")


def _pluck(d: dict, *keys):
    cur = d
    for k in keys:
        if isinstance(cur, list):
            try:
                cur = cur[k]
            except (IndexError, TypeError):
                return ""
        elif isinstance(cur, dict):
            cur = cur.get(k, "")
        else:
            return ""
    return cur


if __name__ == "__main__":
    main()
