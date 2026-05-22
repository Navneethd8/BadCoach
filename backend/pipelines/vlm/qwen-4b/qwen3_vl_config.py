"""Defaults for Qwen3-VL 4B / 8B Instruct via Unsloth (4-bit)."""

# --- 4B ---
# Unsloth-hosted 4-bit weights (fast download, fits consumer GPUs).
DEFAULT_MODEL_ID = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"

# Base model (full precision / alternate entry); use with load_in_4bit=False if desired.
BASE_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

# --- 8B (use with train_qwen3_vl_8b.py; default data: FineBadminton-master 40-video JSONL) ---
DEFAULT_MODEL_ID_8B = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
BASE_MODEL_ID_8B = "Qwen/Qwen3-VL-8B-Instruct"

DEFAULT_MAX_SEQ_LENGTH = 2048
