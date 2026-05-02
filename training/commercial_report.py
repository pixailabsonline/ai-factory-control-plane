"""
Generate a commercial summary from run artifacts.

This is a thin aggregation layer over:
- training_metrics.json
- eval result JSON
- export manifest JSON

It can emit a markdown summary and a machine-readable JSON summary.
"""

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


INSTANCE_HOURLY_PRICES = {
    "p3.2xlarge": 3.06,
    "p3.8xlarge": 12.24,
    "p3.16xlarge": 24.48,
    "p4d.24xlarge": 32.77,
    "p4de.24xlarge": 40.96,
    "p5.48xlarge": 98.32,
    "g5.xlarge": 1.006,
    "g5.2xlarge": 1.212,
    "g5.12xlarge": 5.672,
    "g4dn.xlarge": 0.526,
    "g6.xlarge": 0.805,
    "g6.12xlarge": 6.196,
}


@dataclass
class RunMetrics:
    run_name: str
    instance_type: str
    gpus_per_node: int
    world_size: int
    total_time_sec: float
    gpu_hours: float
    estimated_cost_usd: Optional[float]
    final_tokens_per_sec: Optional[float]
    total_tokens: Optional[int]
    resumed_from_step: Optional[int]
    perplexity: Optional[float]
    passed_eval: Optional[bool]
    checkpoint_path: Optional[str]
    export_path: Optional[str]
    artifact_checksum: Optional[str]
    artifact_size_bytes: Optional[int]
    storage_location: Optional[str]


def load_json(path: Optional[Path]) -> dict:
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


def find_latest(root: Path, pattern: str) -> Optional[Path]:
    matches = [p for p in root.rglob(pattern) if p.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def find_eval_result(root: Path) -> Optional[Path]:
    candidates = []
    for directory in (root / "eval" / "results", root / "eval", root):
        if directory.exists():
            if directory.is_dir():
                candidates.extend([p for p in directory.rglob("*.json") if p.is_file()])
            elif directory.is_file() and directory.suffix == ".json":
                candidates.append(directory)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_paths(args):
    run_root = Path(args.run_root) if args.run_root else None
    baseline_run_root = Path(args.baseline_run_root) if args.baseline_run_root else None

    training_metrics = Path(args.training_metrics) if args.training_metrics else None
    eval_result = Path(args.eval_result) if args.eval_result else None
    export_manifest = Path(args.export_manifest) if args.export_manifest else None

    if run_root:
        training_metrics = training_metrics or find_latest(run_root, "training_metrics.json")
        eval_result = eval_result or find_eval_result(run_root)
        export_manifest = export_manifest or find_latest(run_root, "export_manifest.json")

    baseline_training_metrics = Path(args.baseline_training_metrics) if args.baseline_training_metrics else None
    baseline_eval_result = Path(args.baseline_eval_result) if args.baseline_eval_result else None
    baseline_export_manifest = Path(args.baseline_export_manifest) if args.baseline_export_manifest else None

    if baseline_run_root:
        baseline_training_metrics = baseline_training_metrics or find_latest(baseline_run_root, "training_metrics.json")
        baseline_eval_result = baseline_eval_result or find_eval_result(baseline_run_root)
        baseline_export_manifest = baseline_export_manifest or find_latest(baseline_run_root, "export_manifest.json")

    return training_metrics, eval_result, export_manifest, baseline_training_metrics, baseline_eval_result, baseline_export_manifest


def build_summary(training_metrics_path: Path, eval_result_path: Optional[Path], export_manifest_path: Optional[Path], instance_type: str, gpus_per_node: int, run_name: Optional[str]) -> RunMetrics:
    training = load_json(training_metrics_path)
    eval_result = load_json(eval_result_path)
    export_manifest = load_json(export_manifest_path)

    world_size = int(training.get("world_size", 1))
    total_time_sec = float(training.get("total_time_sec", 0.0))
    gpu_hours = (total_time_sec / 3600.0) * world_size
    hourly_cost = INSTANCE_HOURLY_PRICES.get(instance_type)
    per_gpu_cost = hourly_cost / gpus_per_node if hourly_cost is not None else None
    estimated_cost = gpu_hours * per_gpu_cost if per_gpu_cost is not None else None

    perplexity = None
    passed_eval = None
    if eval_result:
        perpl = eval_result.get("perplexity", {})
        perplexity = perpl.get("perplexity")
        passed_eval = eval_result.get("passed")

    artifact_checksum = export_manifest.get("artifact_checksum")
    artifact_size_bytes = export_manifest.get("artifact_size_bytes")
    storage_location = export_manifest.get("storage_location")

    if not run_name:
        if export_manifest.get("source_checkpoint"):
            run_name = Path(export_manifest["source_checkpoint"]).parent.name
        elif training_metrics_path:
            run_name = training_metrics_path.parent.name
        else:
            run_name = "unknown-run"

    return RunMetrics(
        run_name=run_name,
        instance_type=instance_type,
        gpus_per_node=gpus_per_node,
        world_size=world_size,
        total_time_sec=total_time_sec,
        gpu_hours=gpu_hours,
        estimated_cost_usd=estimated_cost,
        final_tokens_per_sec=training.get("final_tokens_per_sec"),
        total_tokens=training.get("total_tokens"),
        resumed_from_step=training.get("resumed_from_step"),
        perplexity=perplexity,
        passed_eval=passed_eval,
        checkpoint_path=training_metrics_path.parent.as_posix() if training_metrics_path else None,
        export_path=export_manifest_path.parent.as_posix() if export_manifest_path else None,
        artifact_checksum=artifact_checksum,
        artifact_size_bytes=artifact_size_bytes,
        storage_location=storage_location,
    )


def fmt_num(value, digits=2):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_int(value):
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def write_markdown(summary: RunMetrics, out: Path, baseline: Optional[RunMetrics] = None):
    lines = [
        "# Commercial Summary",
        "",
        f"Run: `{summary.run_name}`",
        f"Instance type: `{summary.instance_type}`",
        f"GPUs per node: `{summary.gpus_per_node}`",
        "",
        "| Metric | Baseline | This run | Delta |",
        "|---|---:|---:|---:|",
    ]

    rows = [
        ("GPU hours to passing checkpoint", baseline.gpu_hours if baseline else None, summary.gpu_hours),
        ("Tokens/sec", baseline.final_tokens_per_sec if baseline else None, summary.final_tokens_per_sec),
        ("Cost per trained model (USD)", baseline.estimated_cost_usd if baseline else None, summary.estimated_cost_usd),
        ("Perplexity", baseline.perplexity if baseline else None, summary.perplexity),
        ("Resumed from step", baseline.resumed_from_step if baseline else None, summary.resumed_from_step),
    ]

    for label, base, current in rows:
        delta = None
        if base is not None and current is not None:
            delta = current - base
        lines.append(
            f"| {label} | {fmt_num(base)} | {fmt_num(current)} | {fmt_num(delta)} |"
        )

    lines += [
        "",
        "## Evidence",
        f"- Checkpoint metrics: `{summary.checkpoint_path or 'n/a'}`",
        f"- Export manifest: `{summary.export_path or 'n/a'}`",
        f"- Artifact checksum: `{summary.artifact_checksum or 'n/a'}`",
        f"- Artifact size bytes: `{fmt_int(summary.artifact_size_bytes)}`",
        f"- Storage location: `{summary.storage_location or 'n/a'}`",
        f"- Resumed from step: `{summary.resumed_from_step if summary.resumed_from_step is not None else 'n/a'}`",
        f"- Eval passed: `{summary.passed_eval}`",
        "",
        "## Notes",
        "- Lower GPU hours and lower cost mean less wasted compute per useful model.",
        "- Fill in the baseline with the same model and hardware for a credible before/after comparison.",
    ]

    out.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=None, help="Root directory containing run artifacts")
    parser.add_argument("--training-metrics", default=None, help="Path to training_metrics.json")
    parser.add_argument("--eval-result", default=None, help="Path to eval result JSON")
    parser.add_argument("--export-manifest", default=None, help="Path to export_manifest.json")
    parser.add_argument("--instance-type", default="g5.xlarge")
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--baseline-training-metrics", default=None)
    parser.add_argument("--baseline-eval-result", default=None)
    parser.add_argument("--baseline-export-manifest", default=None)
    parser.add_argument("--baseline-run-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    training_metrics, eval_result, export_manifest, baseline_training, baseline_eval, baseline_export = resolve_paths(args)
    if not training_metrics:
        raise SystemExit("training_metrics.json not found. Pass --training-metrics or --run-root.")

    summary = build_summary(
        training_metrics,
        eval_result,
        export_manifest,
        instance_type=args.instance_type,
        gpus_per_node=args.gpus_per_node,
        run_name=args.run_name,
    )

    baseline = None
    if baseline_training:
        baseline = build_summary(
            baseline_training,
            baseline_eval,
            baseline_export,
            instance_type=args.instance_type,
            gpus_per_node=args.gpus_per_node,
            run_name=f"{summary.run_name}-baseline",
        )

    write_markdown(summary, Path(args.output), baseline=baseline)

    if args.json_output:
        payload = {"summary": asdict(summary)}
        if baseline:
            payload["baseline"] = asdict(baseline)
        Path(args.json_output).write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote commercial summary to {args.output}")


if __name__ == "__main__":
    main()
