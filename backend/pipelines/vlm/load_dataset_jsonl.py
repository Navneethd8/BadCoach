"""Build conversation-style datasets for Unsloth vision SFT from JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from qwen3_vl_config import DEFAULT_MAX_SEQ_LENGTH

from vlm_pose import PoseMode, apply_pose_to_pil, create_pose_estimator


def _load_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def _row_to_messages(
    row: dict[str, Any],
    image_key: str,
    instruction_key: str,
    response_key: str,
    base_dir: Path,
    pose_mode: PoseMode,
    pose_estimator: Any | None,
) -> list[dict[str, Any]]:
    raw = row[image_key]
    image_path = Path(raw).expanduser()
    if not image_path.is_absolute():
        image_path = (base_dir / image_path).resolve()
    instruction = row[instruction_key]
    response = row[response_key]
    pil = _load_image(image_path)
    if pose_mode != "none" and pose_estimator is not None:
        pil, instruction = apply_pose_to_pil(
            pil,
            pose_estimator,
            mode=pose_mode,
            instruction=instruction,
        )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": pil},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": response}],
        },
    ]


def load_jsonl_conversations(
    jsonl_path: str,
    *,
    image_key: str = "image",
    instruction_key: str = "instruction",
    response_key: str = "response",
    pose_mode: PoseMode = "none",
    pose_model_path: str | None = None,
) -> list[dict[str, Any]]:
    """Each line: JSON object with image path, instruction, response."""
    path = Path(jsonl_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JSONL not found: {path}")

    base_dir = path.parent
    pose_estimator = (
        create_pose_estimator(pose_model_path) if pose_mode != "none" else None
    )
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from e
            conv = _row_to_messages(
                row,
                image_key=image_key,
                instruction_key=instruction_key,
                response_key=response_key,
                base_dir=base_dir,
                pose_mode=pose_mode,
                pose_estimator=pose_estimator,
            )
            rows.append({"messages": conv})
    if not rows:
        raise ValueError(f"No samples in {path}")
    return rows


def trainer_vision_kwargs(max_length: int | None = None) -> dict[str, Any]:
    ml = max_length if max_length is not None else DEFAULT_MAX_SEQ_LENGTH
    return {
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "max_length": ml,
    }
