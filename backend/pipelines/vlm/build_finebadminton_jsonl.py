#!/usr/bin/env python3
"""
Build VLM JSONL from FineBadminton dataset labels + dataset/image/*.jpg.

Default labels: backend/data/FineBadminton-master/dataset/transformed_combined_rounds_output_en_evals_translated.json
Output next to images so paths stay short: .../dataset/finebadminton_vlm_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INSTRUCTION = (
    "You are a badminton analyst. This frame is captured at shuttle contact. "
    "Describe the stroke type, technique, which player (name or top/bottom), "
    "court area, quality score out of 5, and any tactical intent. Answer in concise English."
)


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="FineBadminton combined rounds JSON (English).",
    )
    p.add_argument(
        "--dataset_dir",
        type=Path,
        default=None,
        help="Folder containing image/ and where output JSONL is written.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .jsonl path (default: dataset_dir/finebadminton_vlm_train.jsonl).",
    )
    p.add_argument(
        "--instruction",
        type=str,
        default=DEFAULT_INSTRUCTION,
    )
    p.add_argument(
        "--skip_missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rows when image file is missing.",
    )
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    backend = here.parent.parent
    default_dataset = (
        backend / "data" / "FineBadminton-master" / "dataset"
    )
    dataset_dir = (args.dataset_dir or default_dataset).resolve()
    labels_path = (
        args.labels
        or dataset_dir
        / "transformed_combined_rounds_output_en_evals_translated.json"
    ).resolve()
    out_path = (args.output or dataset_dir / "finebadminton_vlm_train.jsonl").resolve()
    image_root = dataset_dir / "image"

    if not labels_path.is_file():
        raise SystemExit(f"Labels not found: {labels_path}")
    if not image_root.is_dir():
        raise SystemExit(f"Image folder not found: {image_root}")

    with labels_path.open(encoding="utf-8") as f:
        rounds = json.load(f)

    written = 0
    skipped_no_frame = 0
    skipped_missing_file = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for clip in rounds:
            video = clip.get("video") or ""
            stem = _stem(video)
            for hit in clip.get("hitting") or []:
                hf = hit.get("hit_frame")
                if hf is None:
                    skipped_no_frame += 1
                    continue
                rel = f"image/{stem}_{int(hf)}.jpg"
                img_path = dataset_dir / rel
                if not img_path.is_file():
                    if args.skip_missing:
                        skipped_missing_file += 1
                        continue
                    raise FileNotFoundError(f"Missing image: {img_path}")
                row = {
                    "image": rel,
                    "instruction": args.instruction,
                    "response": _format_response(hit),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} lines -> {out_path}")
    print(f"Skipped (no hit_frame): {skipped_no_frame}")
    print(f"Skipped (missing image): {skipped_missing_file}")


if __name__ == "__main__":
    main()
