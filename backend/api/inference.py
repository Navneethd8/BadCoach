"""Architecture-aware stroke model forward (CNN-LSTM vs pose+transformer stacks)."""
from __future__ import annotations

from typing import List

import cv2
import numpy as np
import torch

from api import state
from api.model_loader import ARCH_CNN_LSTM


def imagenet_normalize_btc_hw(x01: torch.Tensor) -> torch.Tensor:
    """Normalize clip tensor from [0,1] to ImageNet stats. Shape (B, T, 3, H, W)."""
    mean = torch.tensor([0.485, 0.456, 0.406], device=x01.device, dtype=x01.dtype).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x01.device, dtype=x01.dtype).view(1, 1, 3, 1, 1)
    return (x01 - mean) / std


def joint_seq_btj3_from_rgb_frames(
    segment_frames_rgb: List[np.ndarray],
    pose_estimator,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build (1, T, 33, 3) pose tensor from RGB frames (uint8 or float)."""
    T = len(segment_frames_rgb)
    out = torch.zeros(1, T, 33, 3, device=device, dtype=dtype)
    for t, fr in enumerate(segment_frames_rgb):
        if fr.dtype != np.uint8:
            if float(fr.max()) <= 1.0:
                fr = (np.clip(fr, 0, 1) * 255.0).astype(np.uint8)
            else:
                fr = fr.astype(np.uint8)
        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        res = pose_estimator.process_frame(bgr)
        lm_list = pose_estimator.get_landmarks_as_list(res)
        if not lm_list or len(lm_list[0]) < 33:
            continue
        person = lm_list[0]
        for j in range(33):
            d = person[j]
            out[0, t, j, 0] = float(d["x"])
            out[0, t, j, 1] = float(d["y"])
            out[0, t, j, 2] = float(d["z"])
    return out


def run_stroke_model(
    segment_tensor_01: torch.Tensor,
    segment_frames_rgb: List[np.ndarray],
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Run loaded stroke model on a 16-frame window.

    ``segment_tensor_01``: (1, 16, 3, 224, 224) float in [0, 1].
    ``segment_frames_rgb``: same frames as numpy RGB (for per-frame MediaPipe).
    """
    dev = device or state.device
    arch = getattr(state, "model_architecture", ARCH_CNN_LSTM)
    model = state.model
    pe = state.pose_estimator

    if arch == ARCH_CNN_LSTM:
        return model(segment_tensor_01.to(dev))

    joint = joint_seq_btj3_from_rgb_frames(segment_frames_rgb, pe, dev, dtype=segment_tensor_01.dtype)
    x = imagenet_normalize_btc_hw(segment_tensor_01.to(dev))
    return model(x, joint)
