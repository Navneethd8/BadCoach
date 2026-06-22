#!/usr/bin/env python3
"""
Zero-shot GPT stroke_type eval on the val split (16-frame JSONL).

  export OPENAI_API_KEY=...
  python backend/pipelines/vlm/openai/eval_openai_stroke.py \\
    --jsonl backend/data/FineBadminton-20K/dataset/finebadminton_vlm_16frame.jsonl \\
    --cache_path outputs/openai_gpt55_val_cache.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent
_COMMON = _SCRIPT.parent / "common"
_BACKEND = _SCRIPT.parent.parent.parent
for p in (_BACKEND, _COMMON, _SCRIPT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from openai import OpenAI
from openai_vlm_config import DEFAULT_MODEL
from openai_stroke_client import generate_stroke_caption
from vlm_eval_common import (
    build_instruction_with_pose,
    load_row_images,
    load_val_rows,
    print_eval_report,
    score_predictions,
)
from vlm_pose_cache import load_pose_cache_tensor, resolve_pose_cache_path
from vlm_stroke_protocol import SEQUENCE_LENGTH


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI VLM stroke_type eval")
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--pose_mode", choices=("none", "cache_text"), default="cache_text")
    p.add_argument("--pose_cache_path", type=str, default=None)
    p.add_argument("--num_frames", type=int, default=SEQUENCE_LENGTH)
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--cache_path", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--prompt_mode",
        choices=("classify", "jsonl"),
        default="classify",
        help="classify = 9-class benchmark prompt; jsonl = legacy open caption in JSONL.",
    )
    return p.parse_args()


def _row_id(row: dict, idx: int) -> str:
    if "dataset_index" in row:
        return f"idx_{row['dataset_index']}"
    return f"row_{idx}"


def _load_cache(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec.get("row_id")
            if rid and rec.get("response"):
                out[str(rid)] = rec["response"]
    return out


def main() -> None:
    args = _parse_args()
    client = OpenAI()

    val_rows, base_dir = load_val_rows(args.jsonl)
    if args.max_samples is not None:
        val_rows = val_rows[: args.max_samples]

    pose_cache = None
    if args.pose_mode == "cache_text":
        pose_cache = load_pose_cache_tensor(resolve_pose_cache_path(args.pose_cache_path))

    cache_path = Path(args.cache_path) if args.cache_path else None
    cached = _load_cache(cache_path) if cache_path and args.resume else {}

    predictions: list[str] = []
    cache_f = cache_path.open("a", encoding="utf-8") if cache_path else None

    try:
        for i, row in enumerate(val_rows):
            rid = _row_id(row, i)
            if rid in cached:
                predictions.append(cached[rid])
                continue
            images = load_row_images(row, base_dir, frame_size=args.frame_size)
            instruction = build_instruction_with_pose(
                row,
                row["instruction"],
                pose_cache if args.pose_mode == "cache_text" else None,
                num_frames=args.num_frames,
                prompt_mode=args.prompt_mode,
            )
            pred = generate_stroke_caption(client, instruction, images, model=args.model)
            predictions.append(pred)
            if cache_f:
                cache_f.write(
                    json.dumps({"row_id": rid, "response": pred}, ensure_ascii=False) + "\n"
                )
                cache_f.flush()
            if (i + 1) % 10 == 0:
                print(f"OpenAI eval {i + 1}/{len(val_rows)}", flush=True)
    finally:
        if cache_f:
            cache_f.close()

    metrics = score_predictions(val_rows, predictions)
    print_eval_report(metrics, title=f"OpenAI {args.model} zero-shot stroke_type")


if __name__ == "__main__":
    main()
