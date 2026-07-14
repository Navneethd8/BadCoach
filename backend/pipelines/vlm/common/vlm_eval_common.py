"""Shared split loading and stroke_type scoring for VLM eval."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

_COMMON = Path(__file__).resolve().parent
_VLM = _COMMON.parent
_BACKEND = _VLM.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from core.label_maps import (
    STROKE_TYPE_CLASSES,
    map_hit_type_to_stroke_index,
    stroke_class_to_index,
)
from core.split import SPLIT_SEED, vlm_jsonl_video_level_split
from load_dataset_jsonl import _image_paths_from_row, _load_image
from vlm_pose_cache import load_pose_cache_tensor, pose_text_for_dataset_index
from vlm_stroke_protocol import (
    FRAME_SIZE,
    build_stroke_classify_instruction,
    build_user_instruction,
)
from vlm_train_metrics import extract_stroke_label, parse_stroke_type


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_split_rows(
    jsonl_path: str,
    *,
    split: str = "test",
    split_seed: int = SPLIT_SEED,
) -> tuple[list[dict[str, Any]], Path]:
    """Load JSONL rows for ``train``, ``val``, or ``test`` (default: benchmark test)."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train|val|test, got {split!r}")
    path = Path(jsonl_path).expanduser().resolve()
    rows = read_jsonl_rows(path)
    if not rows:
        raise ValueError(f"Empty JSONL: {path}")
    split_key = "images" if "images" in rows[0] else "image"
    train_idx, val_idx, test_idx = vlm_jsonl_video_level_split(
        rows, image_key=split_key, seed=split_seed
    )
    indices = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
    return [rows[i] for i in indices], path.parent


def load_test_rows(
    jsonl_path: str,
    *,
    split_seed: int = SPLIT_SEED,
) -> tuple[list[dict[str, Any]], Path]:
    return load_split_rows(jsonl_path, split="test", split_seed=split_seed)


def load_val_rows(
    jsonl_path: str,
    *,
    split_seed: int = SPLIT_SEED,
) -> tuple[list[dict[str, Any]], Path]:
    return load_split_rows(jsonl_path, split="val", split_seed=split_seed)


def ground_truth_stroke_index(row: dict[str, Any]) -> int:
    """From JSONL response caption (training label format)."""
    gt_text = row.get("response", "")
    raw = parse_stroke_type(gt_text)
    if raw is None:
        return stroke_class_to_index("Other")
    return map_hit_type_to_stroke_index(raw)


def build_instruction_with_pose(
    row: dict[str, Any],
    task_instruction: str,
    pose_cache: Any | None,
    *,
    num_frames: int | None = None,
    include_format_hint: bool = True,
    prompt_mode: str = "classify",
) -> str:
    pose_text = None
    if pose_cache is not None and "dataset_index" in row:
        pose_text = pose_text_for_dataset_index(
            pose_cache, int(row["dataset_index"]), num_frames=num_frames
        )
    if prompt_mode == "classify":
        return build_stroke_classify_instruction(
            pose_text=pose_text,
            include_format_hint=include_format_hint,
        )
    return build_user_instruction(
        task_instruction,
        pose_text=pose_text,
        include_format_hint=include_format_hint,
    )


def load_row_images(
    row: dict[str, Any],
    base_dir: Path,
    *,
    frame_size: int = FRAME_SIZE,
) -> list[Image.Image]:
    paths = _image_paths_from_row(row, "image")
    out: list[Image.Image] = []
    for rel in paths:
        p = Path(rel).expanduser()
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        out.append(_load_image(p, frame_size))
    return out


def prediction_stroke_index(prediction: str) -> int | None:
    raw = extract_stroke_label(prediction)
    if raw is None:
        return None
    return map_hit_type_to_stroke_index(raw)


def score_predictions(
    rows: list[dict[str, Any]],
    predictions: list[str],
) -> dict[str, Any]:
    if len(rows) != len(predictions):
        raise ValueError(f"rows ({len(rows)}) != predictions ({len(predictions)})")
    correct = 0
    total = 0
    per_class_correct: dict[int, int] = {i: 0 for i in range(len(STROKE_TYPE_CLASSES))}
    per_class_total: dict[int, int] = {i: 0 for i in range(len(STROKE_TYPE_CLASSES))}
    unparsed = 0

    for row, pred in zip(rows, predictions, strict=True):
        gt = ground_truth_stroke_index(row)
        per_class_total[gt] = per_class_total.get(gt, 0) + 1
        total += 1
        pr = prediction_stroke_index(pred)
        if pr is None:
            unparsed += 1
            continue
        if pr == gt:
            correct += 1
            per_class_correct[gt] = per_class_correct.get(gt, 0) + 1

    acc = correct / total if total else 0.0
    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "unparsed": unparsed,
        "per_class_correct": per_class_correct,
        "per_class_total": per_class_total,
    }


def print_eval_report(metrics: dict[str, Any], *, title: str = "VLM stroke_type eval") -> None:
    print(f"\n=== {title} ===")
    print(
        f"stroke_type accuracy: {metrics['correct']}/{metrics['total']} "
        f"({100.0 * metrics['accuracy']:.2f}%)"
    )
    if metrics["unparsed"]:
        print(
            f"Unparsed predictions: {metrics['unparsed']} "
            f"({100.0 * metrics['unparsed'] / metrics['total']:.1f}%)"
        )
    print("\nPer-class (correct/total):")
    for i, name in enumerate(STROKE_TYPE_CLASSES):
        c = metrics["per_class_correct"].get(i, 0)
        t = metrics["per_class_total"].get(i, 0)
        if t:
            print(f"  {name:18s} {c:4d}/{t:4d} ({100.0 * c / t:.1f}%)")
