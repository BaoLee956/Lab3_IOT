import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--out", default="evidence/latency-summary.json")
    args = parser.parse_args()
    with Path(args.csv).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    errors = [r for r in rows if r["status"] != "ok"]
    steady = [r for r in rows if r["status"] == "ok" and r["phase"] == "steady"]
    report = {"attempts_in_csv": len(rows), "errors": len(errors),
              "error_rate": len(errors) / max(len(rows), 1),
              "warmup_successes": sum(r["status"] == "ok" and r["phase"] == "warmup" for r in rows),
              "steady_successes": len(steady), "error_messages": dict(Counter(r["error"] for r in errors)),
              "model_sha256": sorted(set(r["model_sha256"] for r in steady)),
              "metrics_ms": {}, "by_scene": {}}
    for name in ["inference_ms", "rtt_ms", "e2e_ms", "cloud_pipeline_ms", "queue_ms"]:
        values = np.array([float(r[name]) for r in steady if r.get(name)], dtype=float)
        if len(values):
            report["metrics_ms"][name] = {"n": len(values), "mean": float(values.mean()),
                "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                "min": float(values.min()), "max": float(values.max())}
    for scene in sorted(set(r["scene"] for r in steady)):
        subset = [r for r in steady if r["scene"] == scene]
        result = {"n": len(subset)}
        if scene in ["with_mask", "without_mask", "mask_weared_incorrect"]:
            result["exact_single_label_matches"] = sum(r["detected_labels"] == scene for r in subset)
        if scene == "mixed_three":
            target = {"with_mask", "without_mask", "mask_weared_incorrect"}
            result["frames_with_all_three_labels"] = sum(target <= set(r["detected_labels"].split("|")) for r in subset)
        report["by_scene"][scene] = result
    report["note"] = "Scene checks are demo checks, not object-level test-set precision/recall. e2e ends at a browser paint opportunity."
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
