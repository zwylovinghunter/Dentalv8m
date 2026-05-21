#!/usr/bin/env python3
"""Build and forward-check dental ablation model YAML files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--expect-spd", type=int, required=True)
    parser.add_argument("--expect-ca", type=int, required=True)
    parser.add_argument("--expect-dyhead", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    yolo = YOLO(args.model)
    model = yolo.model
    model.eval()

    modules = list(model.modules())
    counts = {
        "SPDConv": sum(m.__class__.__name__ == "SPDConv" for m in modules),
        "CoordAtt": sum(m.__class__.__name__ == "CoordAtt" for m in modules),
        "DyHeadDetect": sum(m.__class__.__name__ == "DyHeadDetect" for m in modules),
    }
    expected = {"SPDConv": args.expect_spd, "CoordAtt": args.expect_ca, "DyHeadDetect": args.expect_dyhead}
    for name, want in expected.items():
        got = counts[name]
        if got != want:
            raise RuntimeError(f"{args.model}: expected {want} {name}, found {got}")

    captured = []

    def pre_hook(module, inputs):
        x = inputs[0]
        if not isinstance(x, (list, tuple)):
            raise RuntimeError(f"Detect input is not a list/tuple: {type(x)}")
        captured.append([tuple(t.shape[-2:]) for t in x])

    handles = [
        m.register_forward_pre_hook(pre_hook)
        for m in modules
        if m.__class__.__name__ in {"Detect", "DyHeadDetect"}
    ]
    if len(handles) != 1:
        raise RuntimeError(f"{args.model}: expected one Detect/DyHeadDetect module, found {len(handles)}")

    with torch.no_grad():
        _ = model(torch.zeros(1, 3, 640, 640))
    for h in handles:
        h.remove()

    if not captured:
        raise RuntimeError(f"{args.model}: Detect input was not captured")
    shapes = captured[-1]
    if len(shapes) != 3:
        raise RuntimeError(f"{args.model}: expected 3 Detect inputs, got {len(shapes)}: {shapes}")
    if shapes != [(80, 80), (40, 40), (20, 20)]:
        raise RuntimeError(f"{args.model}: expected Detect feature sizes 80/40/20, got {shapes}")
    if any(s == (160, 160) for s in shapes):
        raise RuntimeError(f"{args.model}: unexpected P2 160x160 detection head found")

    print(
        f"OK {args.model}: SPDConv={counts['SPDConv']} CoordAtt={counts['CoordAtt']} "
        f"DyHeadDetect={counts['DyHeadDetect']} DetectShapes={shapes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
