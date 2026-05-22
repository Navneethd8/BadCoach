"""
Shared training constants for comparable IsoCourt experiments.

All primary trainers (CNN-LSTM, Conv3D, TimeSformer, ST-TR, GCN-ST-TR, SkateFormer, …)
should import from here so clip length, pose layout, split, and multitask heads stay aligned.

External paper baselines (BST, TemPose collate, ST-GCN) may use different T/batch by design.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch

from core.dataset import FineBadmintonDataset
from core.training_progress import DEFAULT_TRAIN_BATCH_SIZE

# --- Clip / pose layout (MediaPipe) ---
SEQUENCE_LENGTH = 16
FRAME_INTERVAL = 2
NUM_JOINTS = 33
POSE_DIM = 3
EXPECTED_POSE_CACHE_SHAPE_SUFFIX = (SEQUENCE_LENGTH, NUM_JOINTS, POSE_DIM)

# --- Training defaults ---
DEFAULT_EPOCHS = 60
DEFAULT_LR = 5e-4
DEFAULT_SEED = 42
GRAD_ACCUMULATION_STEPS = 4
GRAD_CLIP_NORM = 1.0

# Aliases used by SkateFormer wrapper (same values).
MEDIAPIPE_NUM_FRAMES = SEQUENCE_LENGTH
MEDIAPIPE_NUM_JOINTS = NUM_JOINTS

# Fixed stroke_type CE weights (9 classes) — same vector across pose/video trainers.
STROKE_TYPE_CLASS_WEIGHTS: Tuple[float, ...] = (
    1.0, 1.5, 1.3, 2.0, 1.5, 1.5, 1.5, 2.0, 5.0,
)

DEFAULT_MULTITASK_LOSS_WEIGHTS: Dict[str, float] = {
    "stroke_type": 2.0,
    "position": 1.0,
    "technique": 0.5,
    "placement": 0.5,
    "intent": 0.5,
    "quality": 0.5,
}


def default_list_file(backend_root: str) -> str:
    return os.path.join(
        backend_root,
        "data",
        "transformed_combined_rounds_output_en_evals_translated.json",
    )


def configure_mlflow(backend_root: str) -> str:
    """Use ``backend/mlruns`` unless ``MLFLOW_TRACKING_URI`` is set."""
    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        mlruns_dir = os.path.join(backend_root, "mlruns")
        os.makedirs(mlruns_dir, exist_ok=True)
        uri = f"file:{mlruns_dir}"
        os.environ["MLFLOW_TRACKING_URI"] = uri
    mlflow.set_tracking_uri(uri)
    return uri


def build_task_classes(dataset: FineBadmintonDataset) -> Dict[str, int]:
    """Multitask head sizes; drops ``stroke_subtype``; fixes ``quality`` bands."""
    task_classes = {k: len(v) for k, v in dataset.classes.items()}
    task_classes["quality"] = 7
    if "stroke_subtype" in task_classes:
        del task_classes["stroke_subtype"]
    return task_classes


def load_training_dataset(
    data_root: str,
    list_file: str,
    *,
    transform=None,
    sequence_length: int = SEQUENCE_LENGTH,
    frame_interval: int = FRAME_INTERVAL,
    **kwargs,
) -> FineBadmintonDataset:
    """FineBadminton with standardized clip sampling."""
    return FineBadmintonDataset(
        data_root,
        list_file,
        transform=transform,
        sequence_length=sequence_length,
        frame_interval=frame_interval,
        **kwargs,
    )


def validate_pose_cache(
    pose_cache: torch.Tensor,
    *,
    num_frames: int = SEQUENCE_LENGTH,
    num_joints: int = NUM_JOINTS,
    pose_dim: int = POSE_DIM,
) -> None:
    if pose_cache.ndim != 4:
        raise ValueError(f"pose_cache must be (N,T,J,C), got shape {tuple(pose_cache.shape)}")
    _, t, j, c = pose_cache.shape
    if (t, j, c) != (num_frames, num_joints, pose_dim):
        raise ValueError(
            f"Pose cache shape {tuple(pose_cache.shape)} expected "
            f"(N, {num_frames}, {num_joints}, {pose_dim}). "
            "Rebuild pose_cache_mediapipe.pt or match --sequence-length."
        )


def stroke_type_criterion(device: torch.device, label_smoothing: float = 0.1) -> torch.nn.Module:
    import torch.nn as nn

    weights = torch.tensor(STROKE_TYPE_CLASS_WEIGHTS, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)


def default_multitask_criteria(device: torch.device, label_smoothing: float = 0.1):
    """Returns (criterion_stroke_type, criterion_other)."""
    import torch.nn as nn

    return (
        stroke_type_criterion(device, label_smoothing=label_smoothing),
        nn.CrossEntropyLoss(label_smoothing=label_smoothing),
    )


def common_mlflow_clip_params(
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    frame_interval: int = FRAME_INTERVAL,
    batch_size: int = DEFAULT_TRAIN_BATCH_SIZE,
    **extra,
) -> Dict:
    base = {
        "sequence_length": sequence_length,
        "frame_interval": frame_interval,
        "input_frames": sequence_length,
        "num_joints": NUM_JOINTS,
        "batch_size": batch_size,
    }
    base.update(extra)
    return base
