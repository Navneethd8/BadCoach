"""
Joint / bone / motion streams for skeleton models (MediaPipe 33).

Derived from cached ``(B, T, J, 3)`` tensors — no extra pose extraction.
Used by SkateFormer-B (4×3 = 12 input channels).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

# BlazePose body limbs (same connectivity as ``pose_utils.PoseEstimator.POSE_CONNECTIONS``).
MEDIAPIPE_BONE_PAIRS: Tuple[Tuple[int, int], ...] = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)

NUM_SKELETON_STREAMS = 4
STREAMS_PER_COORD = 3


def root_center_pose(pose: torch.Tensor) -> torch.Tensor:
    """Hip-centered, scale-normalized joints ``(B, T, J, 3)``."""
    B, T = pose.shape[0], pose.shape[1]
    root = pose[:, :, 23:25, :].mean(dim=2, keepdim=True)
    centered = pose - root
    scale = centered.reshape(B, T, -1).norm(dim=-1).mean(dim=1, keepdim=True).clamp(min=1e-6)
    return centered / scale.view(B, 1, 1, 1)


def joints_to_bones(joints: torch.Tensor, pairs: Sequence[Tuple[int, int]] = MEDIAPIPE_BONE_PAIRS) -> torch.Tensor:
    """Bone vectors at child joint indices ``(B, T, J, 3)``."""
    bone = torch.zeros_like(joints)
    for parent, child in pairs:
        bone[:, :, child, :] = joints[:, :, child, :] - joints[:, :, parent, :]
    return bone


def temporal_delta(x: torch.Tensor) -> torch.Tensor:
    """First-order temporal difference; first frame zero ``(B, T, J, 3)``."""
    out = torch.zeros_like(x)
    out[:, 1:] = x[:, 1:] - x[:, :-1]
    return out


def build_four_stream_pose(pose: torch.Tensor) -> torch.Tensor:
    """
    Stack joint, bone, joint-motion, bone-motion along the channel dim.

    Args:
        pose: ``(B, T, J, 3)`` raw MediaPipe cache layout.

    Returns:
        ``(B, T, J, 12)``
    """
    joint = root_center_pose(pose)
    bone = joints_to_bones(joint)
    jmotion = temporal_delta(joint)
    bmotion = temporal_delta(bone)
    return torch.cat([joint, bone, jmotion, bmotion], dim=-1)


def four_stream_to_skateformer_input(
    pose: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert pose to SkateFormer layout with 12 input channels.

    Returns:
        data: ``(B, 12, T, V, 1)``
        index_t: ``(B, T)`` in ``[-1, 1]``
    """
    B, T, J, _ = pose.shape
    streams = build_four_stream_pose(pose)
    data = streams.permute(0, 3, 1, 2).unsqueeze(-1).contiguous()
    index_t = torch.linspace(-1.0, 1.0, T, device=pose.device, dtype=pose.dtype)
    index_t = index_t.unsqueeze(0).expand(B, -1)
    return data, index_t
