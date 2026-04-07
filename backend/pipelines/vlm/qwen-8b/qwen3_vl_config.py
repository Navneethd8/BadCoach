"""Defaults for Qwen3-VL-8B Instruct via Unsloth (4-bit)."""

from vlm_qwen3_defaults import (
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_TRAIN_MAX_PIXELS,
    DEFAULT_TRAIN_MAX_SEQ_LENGTH,
)

# Unsloth-hosted 4-bit weights (same pattern as 4B; fits many server GPUs).
DEFAULT_MODEL_ID = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"

BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

__all__ = [
    "BASE_MODEL_ID",
    "DEFAULT_MAX_SEQ_LENGTH",
    "DEFAULT_MODEL_ID",
    "DEFAULT_TRAIN_MAX_PIXELS",
    "DEFAULT_TRAIN_MAX_SEQ_LENGTH",
]
