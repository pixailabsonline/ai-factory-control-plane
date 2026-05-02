"""
Phase 1: Raw ingestion of Common Crawl WARC/WET files into S3.

Records source URI, retrieval time, and raw checksum for each file.
Emits a raw input manifest that can be used to replay the batch exactly.

Usage:
    python data_pipeline/ingest_commoncrawl.py \
        --crawl CC-MAIN-2024-10 \
        --limit 5 \
        --s3-bucket my-bucket \
        --run-name my-run-001
"""

import argparse
import gzip
import hashlib
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


CC_PATHS_URL = "https://data.commoncrawl.org/crawl-data/{crawl}/wet.paths.gz"
CC_BASE_URL = "https://data.commoncrawl.org/"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_wet_paths(crawl: str) -> list[str]:
    url = CC_PATHS_URL.format(crawl=crawl)
    print(f"[ingest] fetching path list: {url}")
    with urllib.request.urlopen(url) as resp:
        with gzip.GzipFile(fileobj=resp) as gz:
            lines = gz.read().decode().splitlines()
    return [l.strip() for l in lines if l.strip()]


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)


def upload_to_s3(local_path: Path, bucket: str, key: str) -> None:
    import boto3
    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)


def ingest(
    crawl: str,
    limit: int,
    s3_bucket: str,
    run_name: str,
    tmp_dir: Path,
) -> dict:
    paths = fetch_wet_paths(crawl)
    if limit:
        paths = paths[:limit]

    print(f"[ingest] ingesting {len(paths)} WET files from {crawl}")
    run_root = f"runs/{run_name}/raw/{crawl}"
    records = []

    for i, rel_path in enumerate(paths):
        source_url = CC_BASE_URL + rel_path
        filename = Path(rel_path).name
        local_path = tmp_dir / filename

        print(f"[ingest] [{i+1}/{len(paths)}] {filename}")
        retrieved_at = datetime.now(timezone.utc).isoformat()

        download_file(source_url, local_path)
        checksum = sha256_file(local_path)

        s3_key = f"{run_root}/{filename}"
        if s3_bucket:
            upload_to_s3(local_path, s3_bucket, s3_key)
            artifact_location = f"s3://{s3_bucket}/{s3_key}"
        else:
            artifact_location = str(local_path)

        records.append({
            "source_name": "common_crawl",
            "crawl_id": crawl,
            "source_url": source_url,
            "retrieved_at": retrieved_at,
            "license": "https://commoncrawl.org/terms-of-use/",
            "raw_checksum_sha256": checksum,
            "artifact_location": artifact_location,
            "size_bytes": local_path.stat().st_size,
        })

        local_path.unlink()

    manifest = {
        "manifest_type": "raw_input",
        "run_name": run_name,
        "crawl_id": crawl,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "file_count": len(records),
        "total_bytes": sum(r["size_bytes"] for r in records),
        "files": records,
    }
    return manifest


def write_manifest(manifest: dict, s3_bucket: str, run_name: str, tmp_dir: Path) -> None:
    local_path = tmp_dir / "raw_input_manifest.json"
    with open(local_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[ingest] raw input manifest: {local_path}")

    if s3_bucket:
        key = f"runs/{run_name}/raw/raw_input_manifest.json"
        upload_to_s3(local_path, s3_bucket, key)
        print(f"[ingest] manifest → s3://{s3_bucket}/{key}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crawl", default="CC-MAIN-2024-10", help="Common Crawl snapshot ID")
    ap.add_argument("--limit", type=int, default=0, help="Max WET files to fetch (0 = all)")
    ap.add_argument("--s3-bucket", default=os.environ.get("S3_BUCKET", ""), help="S3 bucket for raw artifacts")
    ap.add_argument("--run-name", default=os.environ.get("RUN_NAME", ""), required=False, help="Run name (e.g. run-001)")
    ap.add_argument("--tmp-dir", default="/tmp/cc_ingest", help="Local scratch directory")
    args = ap.parse_args()

    if not args.run_name:
        ap.error("--run-name is required (or set RUN_NAME env var)")

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    manifest = ingest(
        crawl=args.crawl,
        limit=args.limit,
        s3_bucket=args.s3_bucket,
        run_name=args.run_name,
        tmp_dir=tmp_dir,
    )
    write_manifest(manifest, args.s3_bucket, args.run_name, tmp_dir)

    total_gb = manifest["total_bytes"] / (1 << 30)
    print(f"[ingest] done — {manifest['file_count']} files, {total_gb:.2f} GB")


if __name__ == "__main__":
    main()
