"""Build conversation-style datasets for Unsloth vision SFT from JSONL."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

_VLM_DIR = Path(__file__).resolve().parent
_BACKEND_ROOT = _VLM_DIR.parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from PIL import Image

from qwen3_vl_config import DEFAULT_MAX_SEQ_LENGTH

from vlm_pose import (
    DEFAULT_POSE_MIN_SHORT_EDGE,
    PoseMode,
    apply_pose_to_pil,
    create_pose_estimator,
)

from core.split import SPLIT_RATIO, SPLIT_SEED, vlm_jsonl_video_level_split


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from e
    return rows


def _load_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def _resize_pil_max_long_edge(pil: Image.Image, max_long: int) -> Image.Image:
    """Downscale so max(width, height) <= max_long; no-op if already smaller."""
    w, h = pil.size
    long_edge = max(w, h)
    if long_edge <= max_long:
        return pil
    scale = max_long / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return pil.resize((new_w, new_h), Image.LANCZOS)


def _row_to_messages(
    row: dict[str, Any],
    image_key: str,
    instruction_key: str,
    response_key: str,
    base_dir: Path,
    pose_mode: PoseMode,
    pose_estimator: Any | None,
    pose_min_short_edge: int | None,
    max_image_long_edge: int | None,
) -> list[dict[str, Any]]:
    raw = row[image_key]
    image_path = Path(raw).expanduser()
    if not image_path.is_absolute():
        image_path = (base_dir / image_path).resolve()
    instruction = row[instruction_key]
    response = row[response_key]
    pil = _load_image(image_path)
    if max_image_long_edge is not None and max_image_long_edge > 0:
        pil = _resize_pil_max_long_edge(pil, max_image_long_edge)
    if pose_mode != "none" and pose_estimator is not None:
        pil, instruction = apply_pose_to_pil(
            pil,
            pose_estimator,
            mode=pose_mode,
            instruction=instruction,
            min_short_edge_for_pose=pose_min_short_edge,
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


class JsonlConversationDataset(Dataset):
    """
    Lazy-loads images on ``__getitem__`` so the full train set is not held in RAM.

    Eagerly materializing hundreds of HD PIL images (plus optional MediaPipe upscales)
    commonly exhausts Colab CPU RAM before the first step.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        indices: list[int],
        *,
        base_dir: Path,
        image_key: str,
        instruction_key: str,
        response_key: str,
        pose_mode: PoseMode,
        pose_estimator: Any | None,
        pose_min_short_edge: int | None,
        max_image_long_edge: int | None,
    ) -> None:
        self._rows = rows
        self._indices = indices
        self._base_dir = base_dir
        self._image_key = image_key
        self._instruction_key = instruction_key
        self._response_key = response_key
        self._pose_mode = pose_mode
        self._pose_estimator = pose_estimator
        self._pose_min_short_edge = pose_min_short_edge
        self._max_image_long_edge = max_image_long_edge

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self._rows[self._indices[i]]
        conv = _row_to_messages(
            row,
            image_key=self._image_key,
            instruction_key=self._instruction_key,
            response_key=self._response_key,
            base_dir=self._base_dir,
            pose_mode=self._pose_mode,
            pose_estimator=self._pose_estimator,
            pose_min_short_edge=self._pose_min_short_edge,
            max_image_long_edge=self._max_image_long_edge,
        )
        return {"messages": conv}


def load_jsonl_conversations(
    jsonl_path: str,
    *,
    image_key: str = "image",
    instruction_key: str = "instruction",
    response_key: str = "response",
    pose_mode: PoseMode = "none",
    pose_model_path: str | None = None,
    pose_min_short_edge: int | None = DEFAULT_POSE_MIN_SHORT_EDGE,
    max_image_long_edge: int | None = None,
) -> JsonlConversationDataset:
    """Each line: JSON object with image path, instruction, response."""
    path = Path(jsonl_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JSONL not found: {path}")

    rows = _read_jsonl_rows(path)
    if not rows:
        raise ValueError(f"No samples in {path}")

    pose_estimator = (
        create_pose_estimator(pose_model_path) if pose_mode != "none" else None
    )
    return JsonlConversationDataset(
        rows,
        list(range(len(rows))),
        base_dir=path.parent,
        image_key=image_key,
        instruction_key=instruction_key,
        response_key=response_key,
        pose_mode=pose_mode,
        pose_estimator=pose_estimator,
        pose_min_short_edge=pose_min_short_edge,
        max_image_long_edge=max_image_long_edge,
    )


def load_jsonl_conversations_train_val(
    jsonl_path: str,
    *,
    image_key: str = "image",
    instruction_key: str = "instruction",
    response_key: str = "response",
    pose_mode: PoseMode = "none",
    pose_model_path: str | None = None,
    pose_min_short_edge: int | None = DEFAULT_POSE_MIN_SHORT_EDGE,
    split_seed: int = SPLIT_SEED,
    split_ratio: float = SPLIT_RATIO,
    max_image_long_edge: int | None = None,
) -> tuple[JsonlConversationDataset, JsonlConversationDataset]:
    """
    Same video-level 80/20 policy as ``core.split.video_level_split`` (see
    ``vlm_jsonl_video_level_split``).
    """
    path = Path(jsonl_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JSONL not found: {path}")

    rows = _read_jsonl_rows(path)
    if not rows:
        raise ValueError(f"No samples in {path}")

    train_idx, val_idx = vlm_jsonl_video_level_split(
        rows,
        image_key=image_key,
        seed=split_seed,
        ratio=split_ratio,
    )
    pose_estimator = (
        create_pose_estimator(pose_model_path) if pose_mode != "none" else None
    )
    base_dir = path.parent
    kw = dict(
        base_dir=base_dir,
        image_key=image_key,
        instruction_key=instruction_key,
        response_key=response_key,
        pose_mode=pose_mode,
        pose_estimator=pose_estimator,
        pose_min_short_edge=pose_min_short_edge,
        max_image_long_edge=max_image_long_edge,
    )
    train_ds = JsonlConversationDataset(rows, train_idx, **kw)
    val_ds = JsonlConversationDataset(rows, val_idx, **kw)
    return train_ds, val_ds


def trainer_vision_kwargs(max_length: int | None = None) -> dict[str, Any]:
    ml = max_length if max_length is not None else DEFAULT_MAX_SEQ_LENGTH
    return {
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "max_length": ml,
    }
