"""
Build (N, T, 33, 3) MediaPipe pose caches without ``list`` + ``torch.stack``.

Appending one tensor per sample duplicates memory (list holds N tensors, then stack
allocates again). On Colab that often forces swap; iteration time creeps up and tqdm
ETAs look like they are "getting longer". Preallocating a single buffer keeps RAM flat.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch

from core.training_progress import tqdm_pose_cache_build

# Default on-disk name (shared across trainers). Legacy: ``pose_cache_staeformer.pt``.
DEFAULT_POSE_CACHE_FILENAME = "pose_cache_mediapipe.pt"
LEGACY_POSE_CACHE_FILENAME = "pose_cache_staeformer.pt"


def default_pose_cache_path(backend_root: str) -> str:
    return os.path.join(os.path.abspath(backend_root), "models", DEFAULT_POSE_CACHE_FILENAME)


def _pose_cache_load_candidates(cache_path: str) -> List[str]:
    d = os.path.dirname(os.path.abspath(cache_path)) or "."
    base = os.path.basename(cache_path)
    if base == DEFAULT_POSE_CACHE_FILENAME:
        new_p = os.path.join(d, DEFAULT_POSE_CACHE_FILENAME)
        leg_p = os.path.join(d, LEGACY_POSE_CACHE_FILENAME)
        return [new_p, leg_p] if new_p != leg_p else [new_p]
    return [cache_path]


def load_pose_cache_bundle(cache_path: str) -> Optional[Dict[str, Any]]:
    """
    Load ``{"pose_cache": Tensor, ...}`` from ``cache_path``, or from the legacy
    ``pose_cache_staeformer.pt`` in the same directory when the requested basename is
    the default mediapipe name.
    """
    for p in _pose_cache_load_candidates(cache_path):
        if not os.path.isfile(p):
            continue
        if p != os.path.abspath(cache_path):
            print(
                f"Loading pose cache from {p} (legacy filename; "
                f"prefer renaming to {DEFAULT_POSE_CACHE_FILENAME})"
            )
        else:
            print(f"Loading pose cache from {p}...")
        return torch.load(p, map_location="cpu", weights_only=False)
    return None


def media_pipe_fill_pose_cache(dataset_raw, pose_estimator) -> torch.Tensor:
    """
    Fill a single float32 tensor of shape (len(dataset_raw), T, 33, 3) in index order.

    ``dataset_raw`` must match ``FineBadmintonDataset`` layout (``sequence_length``, ``__getitem__``).
    """
    n = len(dataset_raw)
    T = int(dataset_raw.sequence_length)
    out = torch.empty((n, T, 33, 3), dtype=torch.float32)
    for i in tqdm_pose_cache_build(n):
        frames, _ = dataset_raw[i]
        with torch.no_grad():
            p = pose_estimator.extract_tensor_poses(frames)
        if p.dim() == 2:
            row = p.detach().cpu().view(T, 33, 3).to(torch.float32)
        else:
            row = p.detach().cpu().reshape(T, 33, 3).to(torch.float32)
        if row.shape != (T, 33, 3):
            raise RuntimeError(
                f"pose row shape {tuple(row.shape)} != expected ({T}, 33, 3) at index {i}"
            )
        out[i].copy_(row)
    return out.contiguous()
