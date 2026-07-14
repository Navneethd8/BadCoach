"""Build conversation-style datasets for Unsloth vision SFT from JSONL."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

from PIL import Image

_COMMON = Path(__file__).resolve().parent
_VLM_DIR = _COMMON.parent
_BACKEND_ROOT = _VLM_DIR.parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from vlm_pose import (
    DEFAULT_POSE_MIN_SHORT_EDGE,
    PoseMode,
    apply_pose_to_pil,
    create_pose_estimator,
)
from vlm_pose_cache import load_pose_cache_tensor, pose_text_for_dataset_index
from vlm_qwen3_defaults import DEFAULT_MAX_SEQ_LENGTH
from vlm_stroke_protocol import FRAME_SIZE, SEQUENCE_LENGTH, build_user_instruction

from core.split import SPLIT_SEED, vlm_jsonl_video_level_split

PoseModeExtended = Literal["none", "overlay", "text", "both", "cache_text"]


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


def _load_image(path: Path, frame_size: int | None) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    if frame_size is not None and frame_size > 0:
        img = img.resize((frame_size, frame_size), Image.Resampling.BILINEAR)
    return img


def _image_paths_from_row(row: dict[str, Any], image_key: str) -> list[str]:
    if image_key in row and row[image_key]:
        raw = row[image_key]
        if isinstance(raw, list):
            return [str(p) for p in raw]
        return [str(raw)]
    if "images" in row and row["images"]:
        return [str(p) for p in row["images"]]
    if "image" in row and row["image"]:
        return [str(row["image"])]
    raise KeyError("JSONL row needs 'image' or 'images'")


def _resolve_pose_text(
    row: dict[str, Any],
    pose_mode: PoseModeExtended,
    pose_cache: Any | None,
    num_frames: int | None,
) -> str | None:
    if pose_mode != "cache_text":
        return None
    if pose_cache is None:
        raise ValueError("pose_mode=cache_text requires pose_cache tensor")
    if "dataset_index" not in row:
        raise KeyError("pose_mode=cache_text requires 'dataset_index' in JSONL row")
    return pose_text_for_dataset_index(
        pose_cache,
        int(row["dataset_index"]),
        num_frames=num_frames,
    )


def _row_to_messages(
    row: dict[str, Any],
    image_key: str,
    instruction_key: str,
    response_key: str,
    base_dir: Path,
    pose_mode: PoseModeExtended,
    pose_estimator: Any | None,
    pose_min_short_edge: int | None,
    *,
    frame_size: int | None = FRAME_SIZE,
    num_frames: int | None = SEQUENCE_LENGTH,
    pose_cache: Any | None = None,
    include_format_hint: bool = True,
) -> list[dict[str, Any]]:
    rel_paths = _image_paths_from_row(row, image_key)
    if num_frames is not None and len(rel_paths) != num_frames:
        raise ValueError(
            f"Expected {num_frames} images, got {len(rel_paths)} for row keys {list(row.keys())[:6]}"
        )

    instruction = row[instruction_key]
    response = row[response_key]

    images: list[Image.Image] = []
    for rel in rel_paths:
        image_path = Path(rel).expanduser()
        if not image_path.is_absolute():
            image_path = (base_dir / image_path).resolve()
        images.append(_load_image(image_path, frame_size))

    pose_text = _resolve_pose_text(row, pose_mode, pose_cache, num_frames)
    instruction = build_user_instruction(
        instruction,
        pose_text=pose_text,
        include_format_hint=include_format_hint,
    )

    if pose_mode in ("overlay", "text", "both") and pose_estimator is not None:
        if len(images) != 1:
            raise ValueError(
                f"pose_mode={pose_mode} supports a single image per row, got {len(images)}"
            )
        images[0], instruction = apply_pose_to_pil(
            images[0],
            pose_estimator,
            mode=pose_mode,  # type: ignore[arg-type]
            instruction=instruction,
            min_short_edge_for_pose=pose_min_short_edge,
        )

    user_content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
    for pil in images:
        user_content.append({"type": "image", "image": pil})

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": response}]},
    ]


def _rows_to_conversations(
    rows: list[dict[str, Any]],
    indices: list[int],
    *,
    base_dir: Path,
    image_key: str,
    instruction_key: str,
    response_key: str,
    pose_mode: PoseModeExtended,
    pose_estimator: Any | None,
    pose_min_short_edge: int | None,
    frame_size: int | None,
    num_frames: int | None,
    pose_cache: Any | None,
    include_format_hint: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in indices:
        conv = _row_to_messages(
            rows[i],
            image_key=image_key,
            instruction_key=instruction_key,
            response_key=response_key,
            base_dir=base_dir,
            pose_mode=pose_mode,
            pose_estimator=pose_estimator,
            pose_min_short_edge=pose_min_short_edge,
            frame_size=frame_size,
            num_frames=num_frames,
            pose_cache=pose_cache,
            include_format_hint=include_format_hint,
        )
        out.append({"messages": conv})
    return out


def _loader_kwargs(
    *,
    image_key: str,
    instruction_key: str,
    response_key: str,
    pose_mode: PoseModeExtended,
    pose_model_path: str | None,
    pose_min_short_edge: int | None,
    frame_size: int | None,
    num_frames: int | None,
    pose_cache_path: str | None,
    include_format_hint: bool,
) -> dict[str, Any]:
    pose_estimator = (
        create_pose_estimator(pose_model_path)
        if pose_mode in ("overlay", "text", "both")
        else None
    )
    pose_cache = (
        load_pose_cache_tensor(pose_cache_path) if pose_mode == "cache_text" else None
    )
    return {
        "image_key": image_key,
        "instruction_key": instruction_key,
        "response_key": response_key,
        "pose_mode": pose_mode,
        "pose_estimator": pose_estimator,
        "pose_min_short_edge": pose_min_short_edge,
        "frame_size": frame_size,
        "num_frames": num_frames,
        "pose_cache": pose_cache,
        "include_format_hint": include_format_hint,
    }


def load_jsonl_conversations(
    jsonl_path: str,
    *,
    image_key: str = "image",
    instruction_key: str = "instruction",
    response_key: str = "response",
    pose_mode: PoseModeExtended = "none",
    pose_model_path: str | None = None,
    pose_min_short_edge: int | None = DEFAULT_POSE_MIN_SHORT_EDGE,
    frame_size: int | None = FRAME_SIZE,
    num_frames: int | None = None,
    pose_cache_path: str | None = None,
    include_format_hint: bool = True,
) -> list[dict[str, Any]]:
    """Each line: JSON object with image path(s), instruction, response."""
    path = Path(jsonl_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JSONL not found: {path}")

    rows = _read_jsonl_rows(path)
    if not rows:
        raise ValueError(f"No samples in {path}")

    kw = _loader_kwargs(
        image_key=image_key,
        instruction_key=instruction_key,
        response_key=response_key,
        pose_mode=pose_mode,
        pose_model_path=pose_model_path,
        pose_min_short_edge=pose_min_short_edge,
        frame_size=frame_size,
        num_frames=num_frames,
        pose_cache_path=pose_cache_path,
        include_format_hint=include_format_hint,
    )
    return _rows_to_conversations(
        rows,
        list(range(len(rows))),
        base_dir=path.parent,
        **kw,
    )


def load_jsonl_conversations_train_val(
    jsonl_path: str,
    *,
    image_key: str = "image",
    instruction_key: str = "instruction",
    response_key: str = "response",
    pose_mode: PoseModeExtended = "none",
    pose_model_path: str | None = None,
    pose_min_short_edge: int | None = DEFAULT_POSE_MIN_SHORT_EDGE,
    frame_size: int | None = FRAME_SIZE,
    num_frames: int | None = None,
    pose_cache_path: str | None = None,
    include_format_hint: bool = True,
    split_seed: int = SPLIT_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Same video-level 70/10/20 split as ``core.split.video_level_split``.
    Test rows are held out and not returned.
    """
    path = Path(jsonl_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JSONL not found: {path}")

    rows = _read_jsonl_rows(path)
    if not rows:
        raise ValueError(f"No samples in {path}")

    split_key = "images" if rows and "images" in rows[0] else image_key
    train_idx, val_idx, _test_idx = vlm_jsonl_video_level_split(
        rows,
        image_key=split_key,
        seed=split_seed,
    )
    kw = _loader_kwargs(
        image_key=image_key,
        instruction_key=instruction_key,
        response_key=response_key,
        pose_mode=pose_mode,
        pose_model_path=pose_model_path,
        pose_min_short_edge=pose_min_short_edge,
        frame_size=frame_size,
        num_frames=num_frames,
        pose_cache_path=pose_cache_path,
        include_format_hint=include_format_hint,
    )
    base_dir = path.parent
    train_ds = _rows_to_conversations(rows, train_idx, base_dir=base_dir, **kw)
    val_ds = _rows_to_conversations(rows, val_idx, base_dir=base_dir, **kw)
    return train_ds, val_ds


def trainer_vision_kwargs(max_length: int | None = None) -> dict[str, Any]:
    ml = max_length if max_length is not None else DEFAULT_MAX_SEQ_LENGTH
    return {
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "max_length": ml,
    }
