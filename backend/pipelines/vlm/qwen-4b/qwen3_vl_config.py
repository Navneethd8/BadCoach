"""Defaults for Qwen3-VL-4B Instruct via Unsloth (4-bit)."""

from vlm_qwen3_defaults import (
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_TRAIN_MAX_PIXELS,
    DEFAULT_TRAIN_MAX_SEQ_LENGTH,
)

# Unsloth-hosted 4-bit weights (fast download, fits consumer GPUs).
DEFAULT_MODEL_ID = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"

# Base model (full precision / alternate entry); use with load_in_4bit=False if desired.
BASE_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

__all__ = [
    "BASE_MODEL_ID",
    "DEFAULT_MAX_SEQ_LENGTH",
    "DEFAULT_MODEL_ID",
    "DEFAULT_TRAIN_MAX_PIXELS",
    "DEFAULT_TRAIN_MAX_SEQ_LENGTH",
]
