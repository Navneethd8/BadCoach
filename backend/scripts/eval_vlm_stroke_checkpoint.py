#!/usr/bin/env python3
"""
Evaluate Qwen3-VL-8B stroke_type accuracy on the val split (16-frame JSONL).

  cd backend/pipelines/vlm/qwen-8b
  pip install -r ../common/requirements-unsloth-vlm.txt
  python backend/scripts/eval_vlm_stroke_checkpoint.py \\
    --jsonl ../../../data/FineBadminton-20K/dataset/finebadminton_vlm_16frame.jsonl \\
    --pose_mode cache_text \\
    --max_samples 200
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_backend = Path(__file__).resolve().parent.parent.parent
_vlm_common = _backend / "pipelines" / "vlm" / "common"
_qwen8b = _backend / "pipelines" / "vlm" / "qwen-8b"
for p in (_backend, _vlm_common, _qwen8b):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.split import SPLIT_RATIO, SPLIT_SEED
from qwen3_vl_config import DEFAULT_MODEL_ID_8B
from vlm_eval_common import (
    build_instruction_with_pose,
    load_row_images,
    load_val_rows,
    print_eval_report,
    score_predictions,
)
from vlm_pose_cache import load_pose_cache_tensor, resolve_pose_cache_path
from vlm_processor_utils import apply_vision_processor_limits
from vlm_qwen3_defaults import DEFAULT_TRAIN_MAX_PIXELS_PER_IMAGE
from vlm_stroke_protocol import SEQUENCE_LENGTH


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval Qwen3-VL stroke_type on val JSONL")
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--model_name", type=str, default=DEFAULT_MODEL_ID_8B)
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num_frames", type=int, default=SEQUENCE_LENGTH)
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument(
        "--pose_mode",
        choices=("none", "cache_text"),
        default="cache_text",
    )
    p.add_argument("--pose_cache_path", type=str, default=None)
    p.add_argument("--split_seed", type=int, default=SPLIT_SEED)
    p.add_argument("--split_ratio", type=float, default=SPLIT_RATIO)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_pixels_per_image", type=int, default=DEFAULT_TRAIN_MAX_PIXELS_PER_IMAGE)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    import torch
    from unsloth import FastVisionModel

    if not torch.cuda.is_available():
        print("CUDA recommended for eval.", file=sys.stderr)

    val_rows, base_dir = load_val_rows(
        args.jsonl, split_seed=args.split_seed, split_ratio=args.split_ratio
    )
    if args.max_samples is not None:
        val_rows = val_rows[: args.max_samples]

    pose_cache = None
    if args.pose_mode == "cache_text":
        pose_cache = load_pose_cache_tensor(resolve_pose_cache_path(args.pose_cache_path))

    if args.lora_path:
        model, tokenizer = FastVisionModel.from_pretrained(
            args.lora_path, load_in_4bit=args.load_in_4bit
        )
    else:
        model, tokenizer = FastVisionModel.from_pretrained(
            args.model_name, load_in_4bit=args.load_in_4bit
        )
    apply_vision_processor_limits(tokenizer, max_pixels=args.max_pixels_per_image)
    FastVisionModel.for_inference(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictions: list[str] = []
    for i, row in enumerate(val_rows):
        images = load_row_images(row, base_dir, frame_size=args.frame_size)
        if len(images) != args.num_frames:
            raise ValueError(f"Row {i}: expected {args.num_frames} images, got {len(images)}")
        instruction = build_instruction_with_pose(
            row,
            row["instruction"],
            pose_cache if args.pose_mode == "cache_text" else None,
            num_frames=args.num_frames,
        )
        user_content: list[dict] = [{"type": "text", "text": instruction}]
        for im in images:
            user_content.append({"type": "image", "image": im})
        messages = [{"role": "user", "content": user_content}]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(
            images,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        predictions.append(text)
        if (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{len(val_rows)}", flush=True)

    metrics = score_predictions(val_rows, predictions)
    tag = "LoRA" if args.lora_path else "zero-shot"
    print_eval_report(metrics, title=f"Qwen3-VL-8B {tag} stroke_type")


if __name__ == "__main__":
    main()
