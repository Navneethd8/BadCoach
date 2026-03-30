#!/usr/bin/env python3
"""
Fine-tune Qwen3-VL-4B-Instruct with Unsloth (LoRA + 4-bit base).

Requires a CUDA GPU and the dependencies in requirements-unsloth-vlm.txt.

Example:
  cd backend/pipelines/vlm
  pip install -r requirements-unsloth-vlm.txt
  python train_qwen3_vl_4b.py --jsonl example_data/sample.jsonl --output_dir ./outputs/qwen3vl4b_lora
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from torch.utils.data import Subset

from load_dataset_jsonl import (
    load_jsonl_conversations,
    load_jsonl_conversations_train_val,
    trainer_vision_kwargs,
)
from core.split import SPLIT_RATIO, SPLIT_SEED
from qwen3_vl_config import (
    DEFAULT_MODEL_ID,
    DEFAULT_TRAIN_MAX_PIXELS,
    DEFAULT_TRAIN_MAX_SEQ_LENGTH,
)
from vlm_processor_utils import apply_vision_processor_limits
from vlm_train_metrics import build_sft_eval_compute_metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unsloth SFT for Qwen3-VL-4B-Instruct")
    p.add_argument(
        "--jsonl",
        type=str,
        required=True,
        help="Path to JSONL (image path, instruction, response per line).",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id (default: Unsloth 4-bit Qwen3-VL-4B-Instruct).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/qwen3_vl_4b_lora",
        help="Directory for checkpoints and final adapter.",
    )
    p.add_argument(
        "--max_seq_length",
        type=int,
        default=DEFAULT_TRAIN_MAX_SEQ_LENGTH,
        help="Training max sequence length (default 1536; use 2048 for longer captions).",
    )
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--gradient_checkpointing",
        type=str,
        default="unsloth",
        help='Use "unsloth", True, or False (see Unsloth docs).',
    )
    p.add_argument("--r", type=int, default=16, help="LoRA rank.")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Default 1 for T4/16GB VRAM + vision; increase if you have headroom.",
    )
    p.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Effective batch ≈ batch_size × this (default 8 matches old 2×4).",
    )
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument(
        "--num_train_epochs",
        type=float,
        default=25.0,
        help="Upper bound; use with early stopping to avoid pointless late-epoch overfitting.",
    )
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--report_to",
        type=str,
        default="none",
        help='Set to "wandb" if configured.',
    )
    p.add_argument(
        "--finetune_vision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--finetune_language",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--pose_mode",
        type=str,
        choices=("none", "overlay", "text", "both"),
        default="none",
        help="MediaPipe pose: overlay draws skeleton on images; text prepends landmark "
        "summary; both. Requires backend/models/pose_landmarker_lite.task by default.",
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
        help="Upscale frame before MediaPipe if min(h,w) is below this (helps wide shots). 0 disables.",
    )
    p.add_argument(
        "--max_image_long_edge",
        type=int,
        default=None,
        help="If set, resize each frame so max(w,h) is at most this (saves RAM on CPU decode).",
    )
    p.add_argument(
        "--max_pixels",
        type=int,
        default=DEFAULT_TRAIN_MAX_PIXELS,
        help="Vision preprocessor max_pixels (aligns with PIL resize; 0 = leave processor defaults).",
    )
    p.add_argument(
        "--min_pixels",
        type=int,
        default=None,
        help="Optional vision preprocessor min_pixels (Qwen smart_resize).",
    )
    p.add_argument(
        "--no_val_split",
        action="store_true",
        help="Train on full JSONL with no validation (no eval_loss / eval_accuracy).",
    )
    p.add_argument("--split_seed", type=int, default=SPLIT_SEED)
    p.add_argument(
        "--split_ratio",
        type=float,
        default=SPLIT_RATIO,
        help="Video-level train fraction (same policy as core.split.video_level_split).",
    )
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Cap validation size (first N after split). Strongly recommended on low-RAM "
        "hosts: eval with stroke metrics materializes logits on CPU.",
    )
    p.add_argument(
        "--eval_stroke_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If disabled, validation only reports eval_loss (no logits on CPU). "
        "Much lower host RAM at end of each epoch; use --no-eval_stroke_metrics on Colab.",
    )
    p.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="BF16 mixed precision (Ampere+). Default off; use --fp16 for T4.",
    )
    p.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="FP16 mixed precision (default on for T4). Disabled if --bf16 is set.",
    )
    p.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Use 0 on Colab to avoid extra RAM from worker processes.",
    )
    p.add_argument(
        "--save_total_limit",
        type=int,
        default=5,
        help="Max checkpoints to keep on disk.",
    )
    p.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="If set, also save every N steps (save_strategy=steps) in addition to epoch eval.",
    )
    p.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Stop after this many evaluations without improvement (0 = disabled).",
    )
    p.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.0,
        help="Minimum change to qualify as improvement (for metric_for_best_model).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    import random

    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not torch.cuda.is_available():
        print(
            "CUDA is not available. Unsloth Qwen3-VL training expects an NVIDIA GPU.",
            file=sys.stderr,
        )
        sys.exit(1)

    from transformers import EarlyStoppingCallback
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

    max_pixels = None if args.max_pixels == 0 else args.max_pixels
    apply_vision_processor_limits(
        tokenizer,
        max_pixels=max_pixels,
        min_pixels=args.min_pixels,
    )
    if max_pixels is not None:
        print(f"Vision preprocessor: max_pixels={max_pixels}", file=sys.stderr)

    print(f"Loading dataset: {args.jsonl}")
    pose_min = None if args.pose_min_short_edge == 0 else args.pose_min_short_edge
    if args.no_val_split:
        train_dataset = load_jsonl_conversations(
            args.jsonl,
            pose_mode=args.pose_mode,
            pose_model_path=args.pose_model_path,
            pose_min_short_edge=pose_min,
            max_image_long_edge=args.max_image_long_edge,
        )
        eval_dataset = None
    else:
        train_dataset, eval_dataset = load_jsonl_conversations_train_val(
            args.jsonl,
            pose_mode=args.pose_mode,
            pose_model_path=args.pose_model_path,
            pose_min_short_edge=pose_min,
            split_seed=args.split_seed,
            split_ratio=args.split_ratio,
            max_image_long_edge=args.max_image_long_edge,
        )
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            print(
                f"Capping eval_dataset: {len(eval_dataset)} -> {args.max_eval_samples} "
                f"(RAM-friendly eval; metrics are on this subset only).",
                file=sys.stderr,
            )
            eval_dataset = Subset(
                eval_dataset, range(min(args.max_eval_samples, len(eval_dataset)))
            )

    FastVisionModel.for_training(model)

    tkwargs = trainer_vision_kwargs(max_length=args.max_seq_length)
    train_kwargs: dict = {
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": False,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "optim": "adamw_8bit",
        "weight_decay": 0.001,
        "lr_scheduler_type": "linear",
        "seed": args.seed,
        "output_dir": args.output_dir,
        "report_to": args.report_to,
        "bf16": args.bf16,
        "fp16": args.fp16 and not args.bf16,
        **tkwargs,
    }
    if args.max_steps is not None:
        train_kwargs["max_steps"] = args.max_steps
    else:
        train_kwargs["num_train_epochs"] = args.num_train_epochs

    callbacks: list = []
    if eval_dataset is not None:
        if args.eval_stroke_metrics:
            train_kwargs.update(
                eval_strategy="epoch",
                save_total_limit=args.save_total_limit,
                load_best_model_at_end=True,
                metric_for_best_model="eval_stroke_accuracy",
                greater_is_better=True,
            )
        else:
            train_kwargs.update(
                eval_strategy="epoch",
                save_total_limit=args.save_total_limit,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                prediction_loss_only=True,
            )
        if args.save_steps is not None:
            train_kwargs["save_strategy"] = "steps"
            train_kwargs["save_steps"] = args.save_steps
        else:
            train_kwargs["save_strategy"] = "epoch"
        if args.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=args.early_stopping_patience,
                    early_stopping_threshold=args.early_stopping_threshold,
                )
            )
    else:
        train_kwargs["save_strategy"] = (
            "steps" if args.save_steps is not None else "epoch"
        )
        if args.save_steps is not None:
            train_kwargs["save_steps"] = args.save_steps
        train_kwargs["save_total_limit"] = args.save_total_limit

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=(
            build_sft_eval_compute_metrics(tokenizer)
            if eval_dataset is not None and args.eval_stroke_metrics
            else None
        ),
        callbacks=callbacks,
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
