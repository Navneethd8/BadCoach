"""Defaults for Qwen3-VL 4B Instruct via Unsloth (4-bit)."""

# Unsloth-hosted 4-bit weights (fast download, fits consumer GPUs).
DEFAULT_MODEL_ID = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"

# Base model (full precision / alternate entry); use with load_in_4bit=False if desired.
BASE_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

# Max context for packing; use full 2048 when captions are long.
DEFAULT_MAX_SEQ_LENGTH = 2048

# Training default: lower VRAM and often less overfitting per step (raise if needed).
DEFAULT_TRAIN_MAX_SEQ_LENGTH = 1536

# Cap total pixels for vision preprocessing (T4-friendly; matches ~1024²). Increase for quality.
DEFAULT_TRAIN_MAX_PIXELS = 1_048_576
