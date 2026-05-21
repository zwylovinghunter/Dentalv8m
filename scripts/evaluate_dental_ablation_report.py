#!/usr/bin/env python3
"""Evaluate dental ablation results and generate Markdown reports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
RUNS = {
    "baseline": {
        "label": "YOLOv8m baseline",
        "dir": REPO / "exp00_baseline_yolov8m_none_standard_e100_i640_b16_s42",
    },
    "stage1": {
        "label": "YOLOv8m + SPD-Conv",
        "dir": REPO / "exp01_stage1_yolov8m_spdconv2_from_baseline",
    },
    "stage2": {
        "label": "YOLOv8m + SPD-Conv + CoordAtt",
        "dir": REPO / "exp02_stage2_yolov8m_spdconv2_coordatt_p3p4_from_stage1",
    },
    "stage3": {
        "label": "YOLOv8m + SPD-Conv + CoordAtt + Dynamic Head",
        "dir": REPO / "exp03_stage3_yolov8m_spdconv2_coordatt_p3p4_dyhead_from_stage2",
    },
}
FAIR_FIELDS = [
    "data",
    "epochs",
    "imgsz",
    "batch",
    "device",
    "workers",
    "seed",
    "patience",
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "box",
    "cls",
    "dfl",
    "mosaic",
    "mixup",
    "copy_paste",
    "close_mosaic",
    "cos_lr",
    "amp",
    "cache",
    "rect",
]


def norm_key(key: str) -> str:
    return key.strip().lower().replace(" ", "")


def pick(row: dict, names: list[str]) -> float:
    normalized = {norm_key(k): v for k, v in row.items()}
    for name in names:
        key = norm_key(name)
        if key in normalized and normalized[key] != "":
            return float(normalized[key])
    raise KeyError(f"None of the columns found: {names}")


def read_metrics(run_dir: Path) -> dict | None:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print(f"skipped: missing {csv_path}")
        return None
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"skipped: empty {csv_path}")
        return None

    parsed = []
    for row in rows:
        try:
            p = pick(row, ["metrics/precision(B)", "precision", "P"])
            r = pick(row, ["metrics/recall(B)", "recall", "R"])
            map50 = pick(row, ["metrics/mAP50(B)", "mAP50"])
            map5095 = pick(row, ["metrics/mAP50-95(B)", "mAP50-95"])
            epoch = int(float(pick(row, ["epoch"])))
        except Exception as exc:
            raise RuntimeError(f"Unable to parse {csv_path}: {exc}") from exc
        f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        parsed.append(
            {"epoch": epoch, "precision": p, "recall": r, "map50": map50, "map5095": map5095, "f1": f1}
        )
    return max(parsed, key=lambda x: x["map5095"])


def read_args(run_dir: Path) -> dict:
    path = run_dir / "args.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def fair_warning(base_dir: Path, cmp_dir: Path) -> tuple[bool, list[str], tuple[str, str] | None]:
    base_args, cmp_args = read_args(base_dir), read_args(cmp_dir)
    mismatches = []
    for field in FAIR_FIELDS:
        if str(base_args.get(field)) != str(cmp_args.get(field)):
            mismatches.append(field)
    model_pair = (str(base_args.get("model")), str(cmp_args.get("model")))
    fair = not mismatches
    return fair, mismatches, model_pair


def fmt(value: float) -> str:
    return f"{value:.5f}"


def metric_row(label: str, m: dict) -> str:
    return (
        f"| {label} | {m['epoch']} | {fmt(m['precision'])} | {fmt(m['recall'])} | "
        f"{fmt(m['map50'])} | {fmt(m['map5095'])} | {fmt(m['f1'])} |"
    )


def diff_metrics(a: dict, b: dict) -> dict:
    return {k: a[k] - b[k] for k in ("precision", "recall", "map50", "map5095", "f1")}


def render_stage1(metrics: dict[str, dict]) -> str:
    base = metrics["baseline"]
    stage = metrics["stage1"]
    diff = diff_metrics(stage, base)
    improved = {k: diff[k] > 0 for k in diff}
    fair, mismatches, model_pair = fair_warning(RUNS["baseline"]["dir"], RUNS["stage1"]["dir"])
    lines = [
        "# Stage1 SPD-Conv vs Baseline",
        "",
        "当前阶段：Stage1 SPD-Conv",
        "",
        f"baseline 结果目录：`{RUNS['baseline']['dir']}`",
        f"Stage1 结果目录：`{RUNS['stage1']['dir']}`",
        "",
        "主表口径：best mAP50-95 epoch",
        "",
        "| 模型 | Epoch | Precision | Recall | mAP50 | mAP50-95 | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        metric_row(RUNS["baseline"]["label"], base),
        metric_row(RUNS["stage1"]["label"], stage),
        "",
        "| 对比 | ΔPrecision | ΔRecall | ΔmAP50 | ΔmAP50-95 | ΔF1 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Stage1 - Baseline | {fmt(diff['precision'])} | {fmt(diff['recall'])} | "
            f"{fmt(diff['map50'])} | {fmt(diff['map5095'])} | {fmt(diff['f1'])} |"
        ),
        "",
        "| 指标 | 是否提升 |",
        "|---|---|",
        f"| Precision | {'是' if improved['precision'] else '否'} |",
        f"| Recall | {'是' if improved['recall'] else '否'} |",
        f"| mAP50 | {'是' if improved['map50'] else '否'} |",
        f"| mAP50-95 | {'是' if improved['map5095'] else '否'} |",
        f"| F1 | {'是' if improved['f1'] else '否'} |",
        "",
    ]
    if not fair:
        lines.extend(
            [
                "WARNING: not a fair comparison.",
                f"不一致字段：{', '.join(mismatches)}",
                "该结果不能作为正式消融结论。",
                "",
            ]
        )
    lines.append(f"模型配置差异：`{model_pair[0]}` vs `{model_pair[1]}`，这是本阶段的结构变量。")
    lines.append("")
    all_up = all(improved.values())
    missing = [name for key, name in [("precision", "Precision"), ("recall", "Recall"), ("map50", "mAP50"), ("map5095", "mAP50-95"), ("f1", "F1")] if not improved[key]]
    lines.append(f"Stage1 是否五项指标全部提升：{'是' if all_up else '否'}")
    lines.append(f"未提升指标：{', '.join(missing) if missing else '无'}")
    lines.append(f"是否建议保留 SPD-Conv：{'是' if all_up and fair else '否'}")
    lines.append(f"是否建议进入 Stage2：{'是' if all_up and fair else '否，建议先复核 Stage1 结果或等待用户决策'}")
    lines.append("")
    lines.append("已完成 baseline 和 Stage1 训练及对比。当前未启动 Stage2。请用户确认是否继续 Stage2。")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stage1", "stage2", "stage3", "all"], default="all")
    args = parser.parse_args()

    metrics = {key: read_metrics(info["dir"]) for key, info in RUNS.items()}
    metrics = {k: v for k, v in metrics.items() if v is not None}
    if args.stage == "stage1":
        required = {"baseline", "stage1"}
        if not required.issubset(metrics):
            missing = ", ".join(sorted(required - set(metrics)))
            raise SystemExit(f"Cannot evaluate stage1; missing results for: {missing}")
        report = render_stage1(metrics)
        out = REPO / "reports/exp01_stage1_spdconv_vs_baseline_report.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(report)
        print(f"report: {out}")
        return 0

    print("| 模型 | Epoch | Precision | Recall | mAP50 | mAP50-95 | F1 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for key in ("baseline", "stage1", "stage2", "stage3"):
        if key in metrics:
            print(metric_row(RUNS[key]["label"], metrics[key]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
