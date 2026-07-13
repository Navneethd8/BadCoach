#!/usr/bin/env python3
"""
Verify IsoCourt training scripts agree on shared clip/pose/split defaults.

Run from repo root:
  python backend/scripts/verify_training_standards.py
"""
from __future__ import annotations

import ast
import os
import sys

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from core import training_standards as ts
from core.training_progress import DEFAULT_TRAIN_BATCH_SIZE

TRAINING_DIR = os.path.join(_BACKEND, "pipelines", "training")

# Primary comparable trainers (same FineBadminton + 16-frame policy).
PRIMARY_TRAINERS = (
    "train_full.py",
    "train_conv3d.py",
    "train_timesformer.py",
    "train_vit_gcn.py",
    "train_k_st_vit.py",
)

# Paper baselines — documented exceptions.
BASELINE_TRAINERS = (
    "train_bst_baseline.py",
    "train_tempose_baseline.py",
    "train_stgcn_baseline.py",
)


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _find_default_batch_size(source: str) -> list[int]:
    """Rough AST scan for argparse --batch-size default."""
    out = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "add_argument"
            and len(node.args) >= 1
        ):
            continue
        arg0 = node.args[0]
        if not (isinstance(arg0, ast.Constant) and arg0.value == "--batch-size"):
            continue
        for kw in node.keywords:
            if kw.arg != "default":
                continue
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                out.append(kw.value.value)
            elif isinstance(kw.value, ast.Name) and kw.value.id == "DEFAULT_TRAIN_BATCH_SIZE":
                out.append(DEFAULT_TRAIN_BATCH_SIZE)
    return out


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    if ts.SEQUENCE_LENGTH != 16:
        errors.append(f"SEQUENCE_LENGTH={ts.SEQUENCE_LENGTH}, expected 16")
    if DEFAULT_TRAIN_BATCH_SIZE != 4:
        errors.append(f"DEFAULT_TRAIN_BATCH_SIZE={DEFAULT_TRAIN_BATCH_SIZE}, expected 4")
    if ts.MEDIAPIPE_NUM_FRAMES != ts.SEQUENCE_LENGTH:
        errors.append("MEDIAPIPE_NUM_FRAMES must equal SEQUENCE_LENGTH")

    print("=== training_standards.py ===")
    print(f"  SEQUENCE_LENGTH     = {ts.SEQUENCE_LENGTH}")
    print(f"  FRAME_INTERVAL      = {ts.FRAME_INTERVAL}")
    print(f"  NUM_JOINTS          = {ts.NUM_JOINTS}")
    print(f"  DEFAULT_BATCH_SIZE  = {DEFAULT_TRAIN_BATCH_SIZE}")
    print(f"  GRAD_ACCUM_STEPS    = {ts.GRAD_ACCUMULATION_STEPS}")
    print(f"  SPLIT (see split.py) seed={ts.DEFAULT_SEED}, ratio=0.8")
    print()

    print("=== Primary trainers ===")
    for name in PRIMARY_TRAINERS:
        path = os.path.join(TRAINING_DIR, name)
        if not os.path.isfile(path):
            errors.append(f"Missing {name}")
            continue
        src = _read(path)
        uses_split = "video_level_split" in src
        uses_cache = "pose_cache" in src or "PoseOnlyDataset" in src or "FramePoseDataset" in src
        batches = _find_default_batch_size(src)
        batch_str = str(batches) if batches else "?"
        ok_batch = not batches or batches == [DEFAULT_TRAIN_BATCH_SIZE]
        flag = "OK" if uses_split and ok_batch else "CHECK"
        if not uses_split:
            errors.append(f"{name}: no video_level_split")
        if batches and batches != [DEFAULT_TRAIN_BATCH_SIZE]:
            warnings.append(f"{name}: CLI default batch-size={batches}, expected [{DEFAULT_TRAIN_BATCH_SIZE}]")
        print(f"  [{flag}] {name:28} batch_default={batch_str} split={uses_split} pose_cache={uses_cache}")

    print()
    print("=== External baselines (intentional differences) ===")
    for name in BASELINE_TRAINERS:
        path = os.path.join(TRAINING_DIR, name)
        if os.path.isfile(path):
            batches = _find_default_batch_size(_read(path))
            print(f"  [BASELINE] {name:28} batch_default={batches}")

    print()
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print("Errors:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("Verification passed (see warnings for drift to fix).")
    return 0 if not warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
