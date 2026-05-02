"""
Phase 2: Normalization and filtering of Common Crawl WET files.

Reads raw WET files (gzipped WARC-formatted text records) from S3 or local.
Filters out spam, boilerplate, malformed records, and low-signal text.
Emits cleaned JSONL and records filter stats for the manifest.

The same input always produces the same output (deterministic).

Usage:
    python data_pipeline/normalize.py \
        --input-manifest runs/my-run/raw/raw_input_manifest.json \
        --output-dir /tmp/normalized \
        --s3-bucket my-bucket \
        --run-name my-run-001
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path


# Minimum content length after stripping whitespace
MIN_CONTENT_BYTES = 200
# Maximum ratio of non-alphabetic characters (catches garbled/binary content)
MAX_NON_ALPHA_RATIO = 0.4
# Lines that appear verbatim in many boilerplate footers/headers
BOILERPLATE_LINES = frozenset([
    "all rights reserved",
    "privacy policy",
    "terms of service",
    "terms and conditions",
    "cookie policy",
    "subscribe to our newsletter",
    "follow us on",
    "share this",
    "click here",
    "read more",
    "skip to content",
    "skip to main content",
])


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def is_low_signal(text: str) -> tuple[bool, str]:
    stripped = text.strip()

    if len(stripped.encode()) < MIN_CONTENT_BYTES:
        return True, "too_short"

    alpha_chars = sum(1 for c in stripped if c.isalpha())
    if len(stripped) > 0 and (1 - alpha_chars / len(stripped)) > MAX_NON_ALPHA_RATIO:
        return True, "non_alpha_ratio"

    lines = [l.strip().lower() for l in stripped.splitlines() if l.strip()]
    if not lines:
        return True, "empty_after_strip"

    boilerplate_count = sum(1 for l in lines if l in BOILERPLATE_LINES)
    if len(lines) > 0 and boilerplate_count / len(lines) > 0.5:
        return True, "boilerplate"

    return False, ""


def normalize_text(text: str) -> str:
    # NFC unicode normalization — deterministic across platforms
    text = unicodedata.normalize("NFC", text)
    # Collapse runs of whitespace to single space, preserve paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_wet_records(fileobj) -> list[dict]:
    """
    Parse a WET (WARC text) file into individual records.
    Each record has: uri, content, content_length.
    """
    records = []
    content = fileobj.read().decode("utf-8", errors="replace")

    # Split on WARC record boundaries
    blocks = re.split(r"WARC/1\.0\r?\n", content)
    for block in blocks:
        if not block.strip():
            continue

        lines = block.split("\n")
        headers = {}
        body_start = 0
        for i, line in enumerate(lines):
            line = line.rstrip("\r")
            if line == "":
                body_start = i + 1
                break
            if ":" in line:
                k, _, v = line.partition(":")
                headers[k.strip().lower()] = v.strip()

        warc_type = headers.get("warc-type", "")
        if warc_type != "conversion":
            continue

        uri = headers.get("warc-target-uri", "")
        body = "\n".join(lines[body_start:]).strip()
        if body:
            records.append({"uri": uri, "content": body})

    return records


def process_file(local_path: Path) -> tuple[list[dict], dict]:
    stats = {
        "total": 0,
        "passed": 0,
        "rejected_too_short": 0,
        "rejected_non_alpha_ratio": 0,
        "rejected_boilerplate": 0,
        "rejected_empty_after_strip": 0,
        "rejected_malformed": 0,
    }

    try:
        with gzip.open(local_path, "rb") as f:
            raw_records = parse_wet_records(f)
    except Exception as e:
        stats["rejected_malformed"] += 1
        return [], stats

    results = []
    for rec in raw_records:
        stats["total"] += 1
        low, reason = is_low_signal(rec["content"])
        if low:
            key = f"rejected_{reason}"
            stats[key] = stats.get(key, 0) + 1
            continue
        normalized = normalize_text(rec["content"])
        low2, reason2 = is_low_signal(normalized)
        if low2:
            key = f"rejected_{reason2}"
            stats[key] = stats.get(key, 0) + 1
            continue
        stats["passed"] += 1
        results.append({
            "uri": rec["uri"],
            "text": normalized,
            "content_hash": sha256_str(normalized),
        })

    return results, stats


def load_manifest(path: str, s3_bucket: str) -> dict:
    if path.startswith("s3://") or (s3_bucket and not path.startswith("/")):
        import boto3
        s3 = boto3.client("s3")
        key = path.replace(f"s3://{s3_bucket}/", "") if path.startswith("s3://") else path
        obj = s3.get_object(Bucket=s3_bucket, Key=key)
        return json.loads(obj["Body"].read())
    with open(path) as f:
        return json.load(f)


def download_from_s3(bucket: str, key: str, dest: Path) -> None:
    import boto3
    dest.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, str(dest))


def upload_to_s3(local_path: Path, bucket: str, key: str) -> None:
    import boto3
    boto3.client("s3").upload_file(str(local_path), bucket, key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-manifest", required=True, help="Path or S3 key of raw_input_manifest.json")
    ap.add_argument("--output-dir", default="/tmp/normalized", help="Local output directory")
    ap.add_argument("--s3-bucket", default=os.environ.get("S3_BUCKET", ""))
    ap.add_argument("--run-name", default=os.environ.get("RUN_NAME", ""), required=False)
    args = ap.parse_args()

    if not args.run_name:
        ap.error("--run-name is required (or set RUN_NAME env var)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    input_manifest = load_manifest(args.input_manifest, args.s3_bucket)
    crawl = input_manifest.get("crawl_id", "unknown")

    all_records = []
    aggregate_stats: dict = {}
    file_count = 0

    for file_rec in input_manifest["files"]:
        loc = file_rec["artifact_location"]
        filename = Path(loc).name
        local_path = tmp_dir / filename

        print(f"[normalize] processing {filename}")

        if loc.startswith("s3://"):
            parts = loc[5:].split("/", 1)
            bucket, key = parts[0], parts[1]
            download_from_s3(bucket, key, local_path)
        else:
            local_path = Path(loc)

        records, stats = process_file(local_path)
        all_records.extend(records)
        file_count += 1

        for k, v in stats.items():
            aggregate_stats[k] = aggregate_stats.get(k, 0) + v

        if loc.startswith("s3://"):
            local_path.unlink(missing_ok=True)

    # Write JSONL output
    output_path = output_dir / "normalized.jsonl"
    with open(output_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    checksum = ""
    h = hashlib.sha256()
    with open(output_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    checksum = h.hexdigest()

    normalize_manifest = {
        "manifest_type": "normalize",
        "run_name": args.run_name,
        "crawl_id": crawl,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_manifest": args.input_manifest,
        "files_processed": file_count,
        "filter_stats": aggregate_stats,
        "output_record_count": len(all_records),
        "output_checksum_sha256": checksum,
        "output_size_bytes": output_path.stat().st_size,
        "filters_applied": [
            "min_content_bytes",
            "max_non_alpha_ratio",
            "boilerplate_line_ratio",
            "nfc_normalization",
        ],
    }

    manifest_path = output_dir / "normalize_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(normalize_manifest, f, indent=2)

    if args.s3_bucket:
        run_root = f"runs/{args.run_name}/normalized"
        upload_to_s3(output_path, args.s3_bucket, f"{run_root}/normalized.jsonl")
        upload_to_s3(manifest_path, args.s3_bucket, f"{run_root}/normalize_manifest.json")
        print(f"[normalize] → s3://{args.s3_bucket}/{run_root}/")

    print(f"[normalize] done — {len(all_records)} records passed from {aggregate_stats.get('total', 0)} total")
    print(f"[normalize] filter stats: {json.dumps(aggregate_stats)}")


if __name__ == "__main__":
    main()
