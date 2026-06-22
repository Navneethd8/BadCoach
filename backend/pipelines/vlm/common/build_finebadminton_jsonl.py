#!/usr/bin/env python3
"""
Build VLM JSONL from FineBadminton dataset labels + dataset/image/*.jpg.

Modes:
  contact  — one JPEG at hit_frame (legacy default)
  16frame  — 16 linspace JPEGs per hit + dataset_index for pose_cache alignment

Example (FineBadminton-20K, cluster):
  python backend/pipelines/vlm/common/prepare_finebadminton_20k.py --skip-download --extract-training-frames
  python backend/pipelines/vlm/common/build_finebadminton_jsonl.py \\
    --mode 16frame \\
    --data-root backend/data \\
    --output backend/data/FineBadminton-20K/dataset/finebadminton_vlm_16frame.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.dataset import FineBadmintonDataset, default_training_jpeg_dir
from core.training_standards import SEQUENCE_LENGTH
from vlm_stroke_protocol import DEFAULT_16FRAME_INSTRUCTION, DEFAULT_CONTACT_INSTRUCTION


def _stem(video: str) -> str:
    return Path(video).stem


def _format_response(hit: dict) -> str:
    parts: list[str] = []
    parts.append(f"Stroke: {hit.get('hit_type', '')}")
    sub = hit.get("subtype") or []
    if sub:
        parts.append(f"Subtype: {', '.join(str(s) for s in sub)}")
    parts.append(f"Player: {hit.get('player', '')}")
    parts.append(f"Hitter side: {hit.get('hitter', '')}")
    acts = hit.get("player_actions") or []
    if acts:
        parts.append(f"Player actions: {', '.join(str(a) for a in acts)}")
    parts.append(f"Ball area: {hit.get('ball_area', '')}")
    parts.append(f"Quality: {hit.get('quality', '')}/5")
    ch = hit.get("shot_characteristics") or []
    if ch:
        parts.append(f"Shot characteristics: {', '.join(str(c) for c in ch)}")
    st = hit.get("strategies") or []
    if st:
        parts.append(f"Strategy: {', '.join(str(s) for s in st)}")
    cmt = (hit.get("comment") or "").strip()
    if cmt:
        parts.append(f"Comment: {cmt}")
    return " ".join(parts)


def _hit_from_sample(sample: dict) -> dict:
    return {
        "hit_type": sample.get("hit_type", ""),
        "subtype": sample.get("subtype", []),
        "player": sample.get("player", ""),
        "hitter": sample.get("hitter", ""),
        "player_actions": sample.get("player_actions", []),
        "ball_area": sample.get("ball_area", ""),
        "quality": sample.get("quality", 1),
        "shot_characteristics": sample.get("shot_characteristics", []),
        "strategies": sample.get("strategies", []),
        "comment": sample.get("comment", ""),
    }


def _frame_indices(start: int, end: int, sequence_length: int) -> np.ndarray:
    if end - start <= 0:
        return np.zeros(sequence_length, dtype=int)
    return np.linspace(start, end - 1, sequence_length).astype(int)


def _jpeg_rel(stem: str, frame_idx: int, image_prefix: str = "image") -> str:
    return f"{image_prefix}/{stem}_{int(frame_idx)}.jpg"


def build_16frame_via_dataset(
    data_root: str,
    list_file: str,
    dataset_dir: Path,
    *,
    sequence_length: int,
    instruction: str,
    skip_missing: bool,
) -> tuple[list[dict], int]:
    ds = FineBadmintonDataset(
        data_root,
        list_file,
        transform=None,
        sequence_length=sequence_length,
    )
    rows: list[dict] = []
    skipped_missing = 0
    for dataset_index in range(len(ds.samples)):
        sample = ds.samples[dataset_index]
        video_path = sample["video_path"]
        stem = Path(video_path).stem
        sf, ef = int(sample["start_frame"]), int(sample["end_frame"])
        indices = _frame_indices(sf, ef, sequence_length)
        rels: list[str] = []
        ok = True
        for idx in indices:
            rel = _jpeg_rel(stem, int(idx))
            if not (dataset_dir / rel).is_file():
                if skip_missing:
                    ok = False
                    break
                raise FileNotFoundError(f"Missing image: {dataset_dir / rel}")
            rels.append(rel)
        if not ok:
            skipped_missing += 1
            continue
        hit = _hit_from_sample(sample)
        rows.append(
            {
                "images": rels,
                "dataset_index": dataset_index,
                "instruction": instruction,
                "response": _format_response(hit),
                "video_stem": stem,
                "start_frame": sf,
                "end_frame": ef,
            }
        )
    return rows, skipped_missing


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("contact", "16frame"), default="contact")
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=None, help="FineBadminton data root (for 16frame dataset index).")
    p.add_argument("--dataset_dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--instruction", type=str, default=None)
    p.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
        help="Frames per clip for --mode 16frame.",
    )
    p.add_argument(
        "--skip_missing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--max_videos", type=int, default=None, metavar="N")
    args = p.parse_args()

    backend = _BACKEND
    default_data_root = backend / "data"
    default_dataset = backend / "data" / "FineBadminton-master" / "dataset"
    default_list = default_data_root / "transformed_combined_rounds_output_en_evals_translated.json"

    data_root = (args.data_root or default_data_root).resolve()
    dataset_dir = (args.dataset_dir or default_dataset).resolve()
    labels_path = (args.labels or default_list).resolve()

    if args.mode == "16frame":
        out_name = "finebadminton_vlm_16frame.jsonl"
        default_instruction = DEFAULT_16FRAME_INSTRUCTION
    else:
        out_name = "finebadminton_vlm_train.jsonl"
        default_instruction = DEFAULT_CONTACT_INSTRUCTION

    instruction = args.instruction if args.instruction is not None else default_instruction
    out_path = (args.output or dataset_dir / out_name).resolve()

    # Prefer 20K image tree when present
    jpeg_dir = default_training_jpeg_dir(str(data_root))
    if jpeg_dir and Path(jpeg_dir).is_dir():
        image_root = Path(jpeg_dir)
        if args.dataset_dir is None:
            dataset_dir = image_root.parent
    else:
        image_root = dataset_dir / "image"

    if not labels_path.is_file():
        raise SystemExit(f"Labels not found: {labels_path}")
    if not image_root.is_dir():
        raise SystemExit(f"Image folder not found: {image_root}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "16frame":
        list_file = str(labels_path)
        rows, skipped_missing = build_16frame_via_dataset(
            str(data_root),
            list_file,
            dataset_dir,
            sequence_length=args.sequence_length,
            instruction=instruction,
            skip_missing=args.skip_missing,
        )
        if args.max_videos is not None:
            allowed = set(
                sorted({r["video_stem"] for r in rows})[: args.max_videos]
            )
            before = len(rows)
            rows = [r for r in rows if r["video_stem"] in allowed]
            print(f"max_videos={args.max_videos}: kept {len(rows)}/{before} hits")
        with out_path.open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} lines -> {out_path}")
        print(f"Skipped (missing images): {skipped_missing}")
        return

    with labels_path.open(encoding="utf-8") as f:
        rounds = json.load(f)

    if args.max_videos is not None:
        stems_sorted = sorted(
            {_stem(c.get("video") or "") for c in rounds if (c.get("video") or "").strip()}
        )
        allowed = set(stems_sorted[: args.max_videos])
        rounds = [c for c in rounds if _stem(c.get("video") or "") in allowed]

    written = 0
    skipped_no_frame = 0
    skipped_missing_file = 0

    with out_path.open("w", encoding="utf-8") as out:
        for clip in rounds:
            video = clip.get("video") or ""
            stem = _stem(video)
            for hit in clip.get("hitting") or []:
                hf = hit.get("hit_frame")
                if hf is None:
                    skipped_no_frame += 1
                    continue
                rel = _jpeg_rel(stem, int(hf))
                img_path = dataset_dir / rel
                if not img_path.is_file():
                    if args.skip_missing:
                        skipped_missing_file += 1
                        continue
                    raise FileNotFoundError(f"Missing image: {img_path}")
                row = {
                    "image": rel,
                    "instruction": instruction,
                    "response": _format_response(hit),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} lines -> {out_path}")
    print(f"Skipped (no hit_frame): {skipped_no_frame}")
    print(f"Skipped (missing image): {skipped_missing_file}")


if __name__ == "__main__":
    main()
