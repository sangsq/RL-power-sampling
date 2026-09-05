"""Inspect one saved training run without loading its model or evaluating answers."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    folder = ROOT / "results" / args.name
    job = ROOT / "results/jobs" / args.name
    output = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "run": args.name}
    summary = folder / "training_summary.json"
    if summary.exists():
        output["training"] = json.loads(summary.read_text())
    else:
        output["status"] = "No training termination summary found"
    if (job / "exit_code.txt").exists():
        output["exit_code"] = int((job / "exit_code.txt").read_text())
    log = folder / "metrics.jsonl"
    metrics = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    output["logged_updates"] = len(metrics)
    if metrics:
        keys = ["objective", "kl_estimate", "mean_length", "eos_rate", "truncated_rate", "short_rate", "gradient_norm"]
        output["finite_metrics"] = all(math.isfinite(row[key]) for row in metrics for key in keys)
        output["first_16_updates"] = {key: sum(r[key] for r in metrics[:16]) / len(metrics[:16]) for key in keys}
        output["last_16_updates"] = {key: sum(r[key] for r in metrics[-16:]) / len(metrics[-16:]) for key in keys}
        output["sampled_tokens"] = sum(r["sampled_tokens"] for r in metrics)
        output["peak_memory_gb"] = max(r["peak_memory_gb"] for r in metrics)
        output["clip_fraction"] = sum(r["gradient_clipped"] for r in metrics) / len(metrics)
    adapter = ROOT / "checkpoints" / args.name / "final/adapter_model.safetensors"
    output["final_adapter_exists"] = adapter.exists()
    if adapter.exists():
        output["final_adapter_sha256"] = hashlib.sha256(adapter.read_bytes()).hexdigest()
    if (folder / "source_hashes.json").exists():
        hashes = json.loads((folder / "source_hashes.json").read_text())
        output["source_snapshots_verified"] = all(
            hashlib.sha256((folder / "source" / Path(path).name).read_bytes()).hexdigest() == digest
            for path, digest in hashes.items())
    gpu_log = job / "gpu.csv"
    if gpu_log.exists():
        with gpu_log.open() as handle:
            gpu_rows = [{key.strip(): value.strip() for key, value in row.items()}
                        for row in csv.DictReader(handle)]
        utilization = [float(row["utilization.gpu [%]"].split()[0]) for row in gpu_rows]
        if utilization:
            output["gpu_utilization_mean_percent"] = sum(utilization) / len(utilization)
            output["gpu_monitor_samples"] = len(utilization)
    job.mkdir(parents=True, exist_ok=True)
    (job / "inspection.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
