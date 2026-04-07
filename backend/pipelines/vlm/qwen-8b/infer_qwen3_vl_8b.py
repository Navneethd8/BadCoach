#!/usr/bin/env python3
"""
Run inference with Unsloth Qwen3-VL-8B-Instruct (base or saved LoRA folder).

Example (base 4-bit model):
  python infer_qwen3_vl_8b.py --image /path/to/img.jpg --prompt "Describe this image."

Example (after training):
  python infer_qwen3_vl_8b.py --lora_path outputs/qwen3_vl_8b_lora/lora_adapter --image img.jpg --prompt "..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_VLM_ROOT = _SCRIPT_DIR.parent
_COMMON = _VLM_ROOT / "common"
_BACKEND_ROOT = _VLM_ROOT.parent.parent
for p in (_BACKEND_ROOT, _COMMON, _SCRIPT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from qwen3_vl_config import DEFAULT_MODEL_ID
from vlm_pose import apply_pose_to_pil, create_pose_estimator


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference: Qwen3-VL-8B via Unsloth")
    p.add_argument("--image", type=str, required=True, help="Path to an image file.")
    p.add_argument("--prompt", type=str, default="Describe this image in detail.")
    p.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Base model id when not loading from --lora_path.",
    )
    p.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Directory with saved LoRA adapter (from train_qwen3_vl_8b.py).",
    )
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.5)
    p.add_argument("--min_p", type=float, default=0.1)
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--pose_mode",
        type=str,
        choices=("none", "overlay", "text", "both"),
        default="none",
        help="MediaPipe pose: overlay / text / both (see train_qwen3_vl_8b.py).",
    )
    p.add_argument(
        "--pose_model_path",
        type=str,
        default=None,
        help="Optional path to pose_landmarker *.task file.",
    )
    p.add_argument(
        "--pose_min_short_edge",
        type=int,
        default=960,
        help="Upscale before pose if min(h,w) below this; 0 disables.",
    )
    return p.parse_args()


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    print(
        "Warning: CUDA not available. Unsloth Qwen3-VL is intended for GPU; "
        "CPU/MPS may fail or be extremely slow.",
        file=sys.stderr,
    )
    return "cpu"


def main() -> None:
    args = _parse_args()
    from PIL import Image
    from transformers import TextStreamer
    from unsloth import FastVisionModel

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        print(f"Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")
    prompt = args.prompt
    if args.pose_mode != "none":
        pose_estimator = create_pose_estimator(args.pose_model_path)
        pose_min = None if args.pose_min_short_edge == 0 else args.pose_min_short_edge
        image, prompt = apply_pose_to_pil(
            image,
            pose_estimator,
            mode=args.pose_mode,
            instruction=prompt,
            min_short_edge_for_pose=pose_min,
        )
    device = _device()

    if args.lora_path:
        model, tokenizer = FastVisionModel.from_pretrained(
            args.lora_path,
            load_in_4bit=args.load_in_4bit,
        )
    else:
        model, tokenizer = FastVisionModel.from_pretrained(
            args.model_name,
            load_in_4bit=args.load_in_4bit,
        )

    FastVisionModel.for_inference(model)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        image,
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
