#!/usr/bin/env python3
"""
Run inference with Unsloth Qwen3-VL-8B-Instruct (base or saved LoRA folder).

Single image:
  python infer_qwen3_vl_8b.py --image img.jpg --prompt "..."

16-frame clip:
  python infer_qwen3_vl_8b.py --images f0.jpg f1.jpg ... --prompt "..."
  python infer_qwen3_vl_8b.py --image_dir /path/to/frames/ --prompt "..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _SCRIPT_DIR.parent / "common"
for _p in (_SCRIPT_DIR, _COMMON_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qwen3_vl_config import DEFAULT_MODEL_ID_8B
from vlm_pose import apply_pose_to_pil, create_pose_estimator


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference: Qwen3-VL-8B via Unsloth")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=str, help="Single image path.")
    g.add_argument("--images", type=str, nargs="+", help="Multiple frame paths (16-frame clip).")
    g.add_argument("--image_dir", type=str, help="Directory of frames (sorted by name).")
    p.add_argument("--prompt", type=str, default="Describe this image in detail.")
    p.add_argument("--model_name", type=str, default=DEFAULT_MODEL_ID_8B)
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.5)
    p.add_argument("--min_p", type=float, default=0.1)
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument(
        "--pose_mode",
        type=str,
        choices=("none", "overlay", "text", "both"),
        default="none",
    )
    p.add_argument("--pose_model_path", type=str, default=None)
    p.add_argument("--pose_min_short_edge", type=int, default=960)
    return p.parse_args()


def _collect_paths(args: argparse.Namespace) -> list[Path]:
    if args.image:
        return [Path(args.image).expanduser().resolve()]
    if args.images:
        return [Path(p).expanduser().resolve() for p in args.images]
    d = Path(args.image_dir).expanduser().resolve()
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise SystemExit(f"No images in {d}")
    return paths


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    print("Warning: CUDA not available.", file=sys.stderr)
    return "cpu"


def main() -> None:
    args = _parse_args()
    from PIL import Image
    from transformers import TextStreamer
    from unsloth import FastVisionModel

    paths = _collect_paths(args)
    images = []
    for p in paths:
        if not p.is_file():
            print(f"Image not found: {p}", file=sys.stderr)
            sys.exit(1)
        im = Image.open(p).convert("RGB")
        if args.frame_size > 0:
            im = im.resize((args.frame_size, args.frame_size), Image.Resampling.BILINEAR)
        images.append(im)

    prompt = args.prompt
    if len(images) == 1 and args.pose_mode != "none":
        pose_estimator = create_pose_estimator(args.pose_model_path)
        pose_min = None if args.pose_min_short_edge == 0 else args.pose_min_short_edge
        images[0], prompt = apply_pose_to_pil(
            images[0],
            pose_estimator,
            mode=args.pose_mode,
            instruction=prompt,
            min_short_edge_for_pose=pose_min,
        )

    device = _device()

    if args.lora_path:
        model, tokenizer = FastVisionModel.from_pretrained(
            args.lora_path, load_in_4bit=args.load_in_4bit
        )
    else:
        model, tokenizer = FastVisionModel.from_pretrained(
            args.model_name, load_in_4bit=args.load_in_4bit
        )

    FastVisionModel.for_inference(model)

    user_content: list[dict] = [{"type": "text", "text": prompt}]
    for _ in images:
        user_content.append({"type": "image"})
    messages = [{"role": "user", "content": user_content}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    vision_input = images[0] if len(images) == 1 else images
    inputs = tokenizer(
        vision_input,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        temperature=args.temperature,
        min_p=args.min_p,
    )


if __name__ == "__main__":
    main()
