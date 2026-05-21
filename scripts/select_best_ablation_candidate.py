#!/usr/bin/env python3
"""Select the best ablation candidate using all-metric improvement gates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def norm_key(key: str) -> str:
    return key.strip().lower().replace(" ", "")


def pick(row: dict, names: list[str]) -> float:
    normalized = {norm_key(k): v for k, v in row.items()}
    for name in names:
        key = norm_key(name)
        if key in normalized and normalized[key] != "":
            return float(normalized[key])
    raise KeyError(f"None of the columns found: {names}")


def read_best(run_dir: Path) -> dict:
    csv_path = run_dir / "results.csv"
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for row in rows:
        p = pick(row, ["metrics/precision(B)", "precision", "P"])
        r = pick(row, ["metrics/recall(B)", "recall", "R"])
        map50 = pick(row, ["metrics/mAP50(B)", "mAP50"])
        map5095 = pick(row, ["metrics/mAP50-95(B)", "mAP50-95"])
        f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        parsed.append({"precision": p, "recall": r, "map50": map50, "map5095": map5095, "f1": f1})
    return max(parsed, key=lambda x: x["map5095"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    args = parser.parse_args()

    ref = read_best(Path(args.reference))
    rows = []
    for item in args.candidate:
        path = Path(item)
        metrics = read_best(path)
        diff = {k: metrics[k] - ref[k] for k in ref}
        passed = all(v > 0 for v in diff.values())
        rows.append((passed, metrics["map5095"], path, metrics, diff))

    rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = rows[0]
    print(f"selected: {best[2]}")
    print(f"all_metrics_improved: {best[0]}")
    print("| candidate | passed | mAP50-95 | ΔP | ΔR | ΔmAP50 | ΔmAP50-95 | ΔF1 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for passed, _, path, metrics, diff in rows:
        print(
            f"| {path} | {'yes' if passed else 'no'} | {metrics['map5095']:.5f} | "
            f"{diff['precision']:.5f} | {diff['recall']:.5f} | {diff['map50']:.5f} | "
            f"{diff['map5095']:.5f} | {diff['f1']:.5f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
