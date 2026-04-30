"""
Export a trained checkpoint into a serveable model directory and optionally publish it to S3.

This is the promotion step between training/eval and serving. It keeps raw checkpoints
for recovery while giving inference a stable artifact path.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.export_model import export_checkpoint


def sync_to_s3(local_dir: Path, s3_uri: str) -> None:
    subprocess.run(
        ["aws", "s3", "sync", str(local_dir), s3_uri, "--delete"],
        check=True,
    )


def s3_join(root: str, *parts: str) -> str:
    return "/".join([root.rstrip("/")] + [part.strip("/") for part in parts if part])


def write_browseable_index(output_dir: Path, manifest: dict) -> None:
    files = sorted(
        (p for p in output_dir.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(output_dir).as_posix(),
    )
    rel_files = [p.relative_to(output_dir).as_posix() for p in files]

    index_json = {
        "artifact": manifest,
        "files": rel_files,
    }
    (output_dir / "artifact_index.json").write_text(json.dumps(index_json, indent=2) + "\n")

    lines = [
        "# Model Artifact Evidence",
        "",
        f"- Source checkpoint: `{manifest.get('source_checkpoint', '')}`",
        f"- Base model: `{manifest.get('model_name', '')}`",
        f"- Step: `{manifest.get('source_step', '')}`",
        f"- Artifact format: `{manifest.get('artifact_format', '')}`",
        f"- Artifact checksum: `{manifest.get('artifact_checksum', '')}`",
        f"- Storage location: `{manifest.get('storage_location', '')}`",
        "",
        "## Files",
    ]
    for rel in rel_files:
        lines.append(f"- `{rel}`")
    lines.append("")
    (output_dir / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory containing state.pt")
    parser.add_argument("--output-dir", required=True, help="Local serveable model directory")
    parser.add_argument("--s3-uri", default=None, help="Optional s3://bucket/prefix destination for the exported artifact")
    parser.add_argument("--s3-root", default=None, help="Optional s3://bucket/runs/<run> root for checkpoint/model evidence")
    parser.add_argument("--model", default=None, help="Override base model name if not stored in the checkpoint")
    parser.add_argument("--tokenizer", default=None, help="Override tokenizer name")
    args = parser.parse_args()

    manifest = export_checkpoint(
        Path(args.checkpoint),
        Path(args.output_dir),
        model_name=args.model,
        tokenizer_name=args.tokenizer,
    )

    if args.s3_uri:
        manifest["storage_location"] = args.s3_uri
    elif args.s3_root:
        versioned_uri = s3_join(args.s3_root, "models", f"checkpoint-{manifest['source_step']}")
        latest_uri = s3_join(args.s3_root, "models", "latest")
        manifest["storage_location"] = versioned_uri
        manifest["latest_location"] = latest_uri

    write_browseable_index(Path(args.output_dir), manifest)
    (Path(args.output_dir) / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if args.s3_uri:
        sync_to_s3(Path(args.output_dir), args.s3_uri)
    elif args.s3_root:
        sync_to_s3(Path(args.output_dir), versioned_uri)
        sync_to_s3(Path(args.output_dir), latest_uri)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    sys.exit(main())
