"""
MediaPipe pose integration for the Qwen3-VL pipeline.

Reuses backend ``PoseEstimator`` (pose landmarker task). Requires
``backend/models/pose_landmarker_lite.task`` unless a custom path is passed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image

_COMMON_DIR = Path(__file__).resolve().parent
_VLM_ROOT = _COMMON_DIR.parent
_BACKEND_ROOT = _VLM_ROOT.parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.pose_utils import PoseEstimator  # noqa: E402

PoseMode = Literal["none", "overlay", "text", "both"]

# Upscale before pose when min(h, w) is below this (broadcast badminton frames).
DEFAULT_POSE_MIN_SHORT_EDGE = 960

# Subset of landmarks for a short text cue (MediaPipe pose, 33 landmarks).
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


def _pil_rgb_to_bgr(arr: Image.Image) -> np.ndarray:
    rgb = np.asarray(arr.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_pil_rgb(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _bgr_upscale_min_short_edge(bgr: np.ndarray, min_short: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Return (possibly upscaled BGR, (orig_w, orig_h)) for resizing overlays back."""
    h, w = bgr.shape[:2]
    orig_wh = (w, h)
    short = min(h, w)
    if short >= min_short:
        return bgr, orig_wh
    scale = min_short / float(short)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    up = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return up, orig_wh


def format_pose_context_text(detection_result: Any) -> str:
    """Compact pose summary for prepending to the user instruction."""
    if not detection_result.pose_landmarks:
        return "[Pose: no person detected.]"

    person = detection_result.pose_landmarks[0]
    parts: list[str] = []
    for idx, name in zip(_LM_IDS, _LM_NAMES, strict=True):
        if idx >= len(person):
            break
        lm = person[idx]
        parts.append(
            f"{name}=({lm.x:.3f},{lm.y:.3f},vis={lm.visibility:.2f})"
        )
    return "[Pose summary] " + " ".join(parts)


def apply_pose_to_pil(
    pil_rgb: Image.Image,
    estimator: PoseEstimator,
    *,
    mode: PoseMode,
    instruction: str,
    min_short_edge_for_pose: int | None = DEFAULT_POSE_MIN_SHORT_EDGE,
) -> tuple[Image.Image, str]:
    """
    Optionally draw skeleton on the image and/or prepend pose text to ``instruction``.

    ``min_short_edge_for_pose``: if set, run MediaPipe on an upscaled copy when the frame
    is smaller on the short edge (helps wide shots). ``None`` disables upscaling.

    Returns:
        (image_for_vlm, instruction_for_vlm)
    """
    if mode == "none":
        return pil_rgb, instruction

    bgr = _pil_rgb_to_bgr(pil_rgb)
    if min_short_edge_for_pose is not None and min_short_edge_for_pose > 0:
        bgr_pose, orig_wh = _bgr_upscale_min_short_edge(bgr, min_short_edge_for_pose)
    else:
        bgr_pose, orig_wh = bgr, (bgr.shape[1], bgr.shape[0])

    detection_result = estimator.process_frame(bgr_pose)

    out_img = pil_rgb
    if mode in ("overlay", "both"):
        drawn = estimator.draw_landmarks(bgr_pose, detection_result)
        if drawn.shape[1] != orig_wh[0] or drawn.shape[0] != orig_wh[1]:
            drawn = cv2.resize(drawn, orig_wh, interpolation=cv2.INTER_LINEAR)
        out_img = _bgr_to_pil_rgb(drawn)

    out_text = instruction
    if mode in ("text", "both"):
        prefix = format_pose_context_text(detection_result)
        out_text = f"{prefix}\n\n{instruction}"

    return out_img, out_text


def create_pose_estimator(model_path: str | None = None) -> PoseEstimator:
    """
    Use more lenient thresholds than MediaPipe defaults (0.5): broadcast badminton
    frames often show small figures, so the lite model frequently misses at 0.5.
    ``num_poses=2`` helps doubles / two players in frame (we still read the first pose).
    For harder scenes, try the full ``pose_landmarker.task`` model from Google.
    """
    return PoseEstimator(
        model_path=model_path,
        num_poses=2,
        min_pose_detection_confidence=0.25,
        min_pose_presence_confidence=0.25,
        min_tracking_confidence=0.25,
    )
