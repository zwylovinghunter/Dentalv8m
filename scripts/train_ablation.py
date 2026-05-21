#!/usr/bin/env python3
"""Unified training entrypoint for dental ablation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_cache(value):
    if isinstance(value, bool):
        return value
    raw = str(value).strip()
    low = raw.lower()
    if low in {"true", "1", "yes", "y", "on"}:
        return True
    if low in {"false", "0", "no", "n", "off"}:
        return False
    return raw


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--cos-lr", type=str2bool, required=True)
    parser.add_argument("--amp", type=str2bool, required=True)
    parser.add_argument("--cache", type=parse_cache, required=True)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_dir = Path(args.project).expanduser().resolve() / args.name
    if result_dir.exists() and not args.exist_ok:
        raise SystemExit(f"Result directory already exists, refusing to overwrite: {result_dir}")

    model = YOLO(args.model)
    weights = Path(args.weights).expanduser()
    load_target = str(weights if weights.exists() else weights.name)
    model.load(load_target)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        optimizer=args.optimizer,
        cos_lr=args.cos_lr,
        amp=args.amp,
        cache=args.cache,
        close_mosaic=args.close_mosaic,
        exist_ok=args.exist_ok,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
