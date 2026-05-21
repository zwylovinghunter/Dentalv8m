#!/usr/bin/env python3
"""Validate the dental lesion large dataset layout and labels."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ALLOWED_CLASSES = {"0", "1", "2"}
CLASS_NAMES = {"0": "Caries", "1": "Periapical Lesion", "2": "Impacted"}


def label_dir_for(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    if "images" not in parts:
        raise ValueError(f"Cannot derive label directory from image path: {image_dir}")
    parts[parts.index("images")] = "labels"
    return Path(*parts)


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_split(root: Path, split: str, rel: str) -> tuple[int, int, Counter]:
    image_dir = (root / rel).resolve()
    label_dir = label_dir_for(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"{split} image directory not found: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"{split} label directory not found: {label_dir}")

    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    labels = sorted(p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
    if not images:
        raise RuntimeError(f"{split} image directory is empty: {image_dir}")
    if not labels:
        raise RuntimeError(f"{split} label directory is empty: {label_dir}")

    counts: Counter = Counter()
    bad_classes: set[str] = set()
    for txt in labels:
        with txt.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                fields = line.strip().split()
                if not fields:
                    continue
                cls = fields[0]
                if cls not in ALLOWED_CLASSES:
                    bad_classes.add(f"{txt}:{line_no}:{cls}")
                else:
                    counts[cls] += 1
    if bad_classes:
        sample = "\n".join(sorted(bad_classes)[:20])
        raise RuntimeError(f"Invalid class id found; only 0, 1, 2 are allowed:\n{sample}")

    return len(images), len(labels), counts


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(repo / "data/dental_lesion_3cls_large.yaml"))
    args = parser.parse_args()

    cfg_path = Path(args.data).resolve()
    cfg = load_yaml(cfg_path)
    root = Path(cfg["path"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    expected_names = {0: "Caries", 1: "Periapical Lesion", 2: "Impacted"}
    if int(cfg.get("nc", -1)) != 3 or cfg.get("names") != expected_names:
        raise RuntimeError(f"Unexpected dataset classes in {cfg_path}: nc={cfg.get('nc')} names={cfg.get('names')}")

    total = Counter()
    print(f"data: {cfg_path}")
    print(f"path: {root}")
    print("| split | images | labels | Caries | Periapical Lesion | Impacted |")
    print("|---|---:|---:|---:|---:|---:|")
    for split in ("train", "val", "test"):
        images, labels, counts = validate_split(root, split, cfg[split])
        total.update(counts)
        print(
            f"| {split} | {images} | {labels} | {counts['0']} | {counts['1']} | {counts['2']} |"
        )
    print(
        f"| total | - | - | {total['0']} | {total['1']} | {total['2']} |"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
