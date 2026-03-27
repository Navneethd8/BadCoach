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

from load_dataset_jsonl import load_jsonl_conversations, trainer_vision_kwargs
from qwen3_vl_config import DEFAULT_MAX_SEQ_LENGTH, DEFAULT_MODEL_ID


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
    p.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--gradient_checkpointing",
        type=str,
        default="unsloth",
        help='Use "unsloth", True, or False (see Unsloth docs).',
    )
    p.add_argument("--r", type=int, default=16, help="LoRA rank.")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=3407)
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
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    import torch

    if not torch.cuda.is_available():
        print(
            "CUDA is not available. Unsloth Qwen3-VL training expects an NVIDIA GPU.",
            file=sys.stderr,
        )
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

    print(f"Loading dataset: {args.jsonl}")
    train_dataset = load_jsonl_conversations(
        args.jsonl,
        pose_mode=args.pose_mode,
        pose_model_path=args.pose_model_path,
    )

    FastVisionModel.for_training(model)

    tkwargs = trainer_vision_kwargs(max_length=args.max_seq_length)
    train_kwargs: dict = {
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
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

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
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
