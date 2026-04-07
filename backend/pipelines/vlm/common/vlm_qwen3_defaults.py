"""Shared sequence / vision defaults for Qwen3-VL (4B and 8B) SFT."""

# Max context for packing; use full 2048 when captions are long.
DEFAULT_MAX_SEQ_LENGTH = 2048

# Training default: lower VRAM and often less overfitting per step (raise if needed).
DEFAULT_TRAIN_MAX_SEQ_LENGTH = 1536

# Cap total pixels for vision preprocessing (~1024²). Raise on A100 if you want sharper crops.
DEFAULT_TRAIN_MAX_PIXELS = 1_048_576
