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

_VLM_DIR = Path(__file__).resolve().parent
_BACKEND_ROOT = _VLM_DIR.parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.pose_utils import PoseEstimator  # noqa: E402

PoseMode = Literal["none", "overlay", "text", "both"]

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
) -> tuple[Image.Image, str]:
    """
    Optionally draw skeleton on the image and/or prepend pose text to ``instruction``.

    Returns:
        (image_for_vlm, instruction_for_vlm)
    """
    if mode == "none":
        return pil_rgb, instruction

    bgr = _pil_rgb_to_bgr(pil_rgb)
    detection_result = estimator.process_frame(bgr)

    out_img = pil_rgb
    if mode in ("overlay", "both"):
        drawn = estimator.draw_landmarks(bgr, detection_result)
        out_img = _bgr_to_pil_rgb(drawn)

    out_text = instruction
    if mode in ("text", "both"):
        prefix = format_pose_context_text(detection_result)
        out_text = f"{prefix}\n\n{instruction}"

    return out_img, out_text


def create_pose_estimator(model_path: str | None = None) -> PoseEstimator:
    return PoseEstimator(model_path=model_path)
