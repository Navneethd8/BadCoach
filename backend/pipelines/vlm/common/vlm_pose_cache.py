"""Pose cache → text for VLM prompts (same tensor as native trainers)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

_COMMON = Path(__file__).resolve().parent
_BACKEND = _COMMON.parent.parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.pose_cache_build import load_pose_cache_bundle

# VLM 16-frame protocol uses span_linspace clips (same as native classifiers).
DEFAULT_VLM_POSE_CACHE_FILENAME = "pose_cache_span_linspace.pt"

# Compact subset (matches vlm_pose._LM_NAMES).
_LM_IDS = (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
_LM_NAMES = (
    "nose",
    "L_shoulder",
    "R_shoulder",
    "L_elbow",
    "R_elbow",
    "L_wrist",
    "R_wrist",
    "L_hip",
    "R_hip",
    "L_knee",
    "R_knee",
    "L_ankle",
    "R_ankle",
)


def default_vlm_pose_cache_path() -> str:
    return str(_BACKEND / "models" / DEFAULT_VLM_POSE_CACHE_FILENAME)


def load_pose_cache_tensor(cache_path: str | None = None) -> torch.Tensor:
    path = cache_path or default_vlm_pose_cache_path()
    bundle = load_pose_cache_bundle(path)
    if bundle is None:
        raise FileNotFoundError(
            f"Pose cache not found: {path}. Build with:\n"
            f"  python backend/pipelines/training/build_full_pose_cache.py \\\n"
            f"    --data-root backend/data --list-file <annotations.json> \\\n"
            f"    --output {path} --sampling span_linspace"
        )
    cache = bundle["pose_cache"]
    if not isinstance(cache, torch.Tensor):
        cache = torch.as_tensor(cache)
    return cache


def format_frame_pose_text(frame_pose: torch.Tensor) -> str:
    """One frame (33, 3) or (J, 3) → compact landmark string."""
    parts: list[str] = []
    for idx, name in zip(_LM_IDS, _LM_NAMES, strict=True):
        if idx >= frame_pose.shape[0]:
            break
        x, y, z = frame_pose[idx].tolist()
        parts.append(f"{name}=({x:.3f},{y:.3f},{z:.3f})")
    return " ".join(parts) if parts else "no_landmarks"


def format_sequence_pose_text(pose_row: torch.Tensor, *, num_frames: int | None = None) -> str:
    """
    ``pose_row``: (T, 33, 3) aligned with training clip.
    Returns a block starting with ``[Pose sequence]``.
    """
    if pose_row.dim() != 3:
        raise ValueError(f"Expected pose_row (T, J, 3), got shape {tuple(pose_row.shape)}")
    t = int(pose_row.shape[0])
    if num_frames is not None and t != num_frames:
        pose_row = pose_row[:num_frames]
        t = int(pose_row.shape[0])
    lines = ["[Pose sequence]"]
    for fi in range(t):
        lines.append(f"t={fi}: {format_frame_pose_text(pose_row[fi])}")
    return "\n".join(lines)


def pose_text_for_dataset_index(
    pose_cache: torch.Tensor,
    dataset_index: int,
    *,
    num_frames: int | None = None,
) -> str:
    row = pose_cache[int(dataset_index)]
    return format_sequence_pose_text(row, num_frames=num_frames)


def resolve_pose_cache_path(path: str | None) -> str:
    return path or default_vlm_pose_cache_path()
