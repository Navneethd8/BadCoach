#!/usr/bin/env python3
"""
Fine-tune Qwen3-VL-8B-Instruct with Unsloth (LoRA + 4-bit base).

16-frame FineBadminton-20K example (H100):

  cd backend/pipelines/vlm/qwen-8b
  pip install -r ../common/requirements-unsloth-vlm.txt
  python train_qwen3_vl_8b.py \\
    --jsonl ../../../data/FineBadminton-20K/dataset/finebadminton_vlm_16frame.jsonl \\
    --num_frames 16 --frame_size 224 \\
    --pose_mode cache_text \\
    --num_train_epochs 5 \\
    --per_device_train_batch_size 2 \\
    --output_dir ./outputs/qwen3_vl_8b_16frame_lora
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

_BACKEND = _SCRIPT_DIR.parent.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from load_dataset_jsonl import (
    load_jsonl_conversations,
    load_jsonl_conversations_train_val,
    trainer_vision_kwargs,
)
from core.split import SPLIT_RATIO, SPLIT_SEED
from qwen3_vl_config import DEFAULT_MAX_SEQ_LENGTH, DEFAULT_MODEL_ID_8B
from vlm_processor_utils import apply_vision_processor_limits
from vlm_qwen3_defaults import DEFAULT_TRAIN_MAX_PIXELS_PER_IMAGE
from vlm_stroke_protocol import SEQUENCE_LENGTH
from vlm_train_metrics import build_sft_eval_compute_metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unsloth SFT for Qwen3-VL-8B-Instruct")
    p.add_argument("--jsonl", type=str, required=True)
    p.add_argument("--model_name", type=str, default=DEFAULT_MODEL_ID_8B)
    p.add_argument("--output_dir", type=str, default="outputs/qwen3_vl_8b_lora")
    p.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gradient_checkpointing", type=str, default="unsloth")
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Use 2+ on H100 for 16-frame; default 1 for <=24GB.",
    )
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--num_train_epochs", type=float, default=5.0)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--report_to", type=str, default="none")
    p.add_argument("--finetune_vision", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--finetune_language", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--pose_mode",
        type=str,
        choices=("none", "overlay", "text", "both", "cache_text"),
        default="cache_text",
    )
    p.add_argument("--pose_model_path", type=str, default=None)
    p.add_argument("--pose_cache_path", type=str, default=None)
    p.add_argument("--pose_min_short_edge", type=int, default=960)
    p.add_argument("--num_frames", type=int, default=SEQUENCE_LENGTH)
    p.add_argument("--frame_size", type=int, default=224)
    p.add_argument("--max_pixels_per_image", type=int, default=DEFAULT_TRAIN_MAX_PIXELS_PER_IMAGE)
    p.add_argument("--no_val_split", action="store_true")
    p.add_argument("--split_seed", type=int, default=SPLIT_SEED)
    p.add_argument("--split_ratio", type=float, default=SPLIT_RATIO)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--max_eval_samples", type=int, default=500)
    p.add_argument("--dataloader_num_workers", type=int, default=0)
    p.add_argument("--save_total_limit", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    import torch

    if not torch.cuda.is_available():
        print("CUDA is not available. Unsloth Qwen3-VL training expects an NVIDIA GPU.", file=sys.stderr)
        sys.exit(1)

    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    print(f"Loading model: {args.model_name}")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )
    apply_vision_processor_limits(tokenizer, max_pixels=args.max_pixels_per_image)

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision,
        finetune_language_layers=args.finetune_language,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    print(f"Loading dataset: {args.jsonl} (num_frames={args.num_frames}, pose_mode={args.pose_mode})")
    pose_min = None if args.pose_min_short_edge == 0 else args.pose_min_short_edge
    loader_kw = dict(
        pose_mode=args.pose_mode,
        pose_model_path=args.pose_model_path,
        pose_min_short_edge=pose_min,
        frame_size=args.frame_size,
        num_frames=args.num_frames,
        pose_cache_path=args.pose_cache_path,
    )
    if args.no_val_split:
        train_dataset = load_jsonl_conversations(args.jsonl, **loader_kw)
        eval_dataset = None
    else:
        train_dataset, eval_dataset = load_jsonl_conversations_train_val(
            args.jsonl,
            split_seed=args.split_seed,
            split_ratio=args.split_ratio,
            **loader_kw,
        )
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            print(
                f"Capping eval_dataset: {len(eval_dataset)} -> {args.max_eval_samples}",
                file=sys.stderr,
            )
            eval_dataset = eval_dataset[: args.max_eval_samples]

    FastVisionModel.for_training(model)

    tkwargs = trainer_vision_kwargs(max_length=args.max_seq_length)
    train_kwargs: dict = {
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "dataloader_num_workers": args.dataloader_num_workers,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "optim": "adamw_8bit",
        "weight_decay": 0.001,
        "lr_scheduler_type": "linear",
        "seed": args.seed,
        "output_dir": args.output_dir,
        "report_to": args.report_to,
        **tkwargs,
    }
    if args.max_steps is not None:
        train_kwargs["max_steps"] = args.max_steps
    else:
        train_kwargs["num_train_epochs"] = args.num_train_epochs

    if eval_dataset is not None:
        train_kwargs.update(
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_stroke_accuracy",
            greater_is_better=True,
        )
    else:
        train_kwargs["save_strategy"] = "epoch"

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=(
            build_sft_eval_compute_metrics(tokenizer) if eval_dataset is not None else None
        ),
        args=SFTConfig(**train_kwargs),
    )

    trainer.train()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    adapter_dir = out / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
