"""Teaser figures: fixed clip picks + MediaPipe on native broadcast frames.

Default workflow:
  1. Lock one curated (sample_index, timestep) per stroke.
  2. Load the native video frame (not the 224×224 training resize).
  3. Run MediaPipe on a near-court crop (per-stroke ``y_start``), draw on full frame.
  4. Optionally nudge landmarks via ``adjust_pose`` (dx/dy/scale).
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch

from core.dataset import FineBadmintonDataset
from core.pose_cache_build import _pick_primary_pose_index
from core.pose_utils import PoseEstimator
from core.skeleton_streams import MEDIAPIPE_BONE_PAIRS

PoseTune = Mapping[str, float]

POSE_UPSCALE_MIN_SHORT = 960
NEAR_COURT_Y_START = 0.38  # default; per-panel overrides often work better
Y_START_FALLBACKS = (0.30, 0.35, 0.38, 0.42, 0.45, 0.50, 0.0)


def create_teaser_pose_estimator(model_path: str) -> PoseEstimator:
    return PoseEstimator(
        model_path=model_path,
        num_poses=3,
        min_pose_detection_confidence=0.25,
        min_pose_presence_confidence=0.25,
        min_tracking_confidence=0.25,
    )


def frame_index_for_timestep(sample: Dict[str, Any], timestep: int, num_frames: int = 16) -> int:
    start = int(sample["start_frame"])
    end = int(sample["end_frame"])
    indices = np.linspace(start, end - 1, num_frames).astype(int)
    return int(indices[timestep])


def contact_timestep(sample: Dict[str, Any], num_frames: int = 16) -> int:
    start = int(sample["start_frame"])
    end = int(sample["end_frame"])
    hit = int(sample.get("hit_frame") or (start + end) // 2)
    indices = np.linspace(start, end - 1, num_frames).astype(int)
    return int(np.argmin(np.abs(indices - hit)))


def load_native_rgb(sample: Dict[str, Any], frame_idx: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(sample["video_path"])
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _upscale_rgb(rgb: np.ndarray, min_short: int = POSE_UPSCALE_MIN_SHORT) -> np.ndarray:
    h, w = rgb.shape[:2]
    short = min(h, w)
    if short >= min_short:
        return rgb
    scale = min_short / float(short)
    return cv2.resize(
        rgb,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_LINEAR,
    )


def _landmarks_to_array(person) -> np.ndarray:
    out = np.zeros((33, 3), dtype=np.float32)
    for j, lm in enumerate(person):
        out[j] = [lm.x, lm.y, lm.z]
    return out


def _remap_crop_pose_to_full(pose: np.ndarray, y_start: float, crop_h: int, full_h: int) -> np.ndarray:
    out = pose.copy()
    for j in range(33):
        if out[j, 1] > 1e-4:
            out[j, 1] = y_start + out[j, 1] * (crop_h / full_h)
    return out


def _score_striker_pose_crop(pose_crop: np.ndarray) -> Optional[float]:
    """Score a pose inside the near-court crop (before remap to full frame)."""
    if not is_plausible_human_pose(pose_crop) or not is_single_person_pose(pose_crop):
        return None
    bbox = _pose_bbox(pose_crop)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    h, w = y1 - y0, x1 - x0
    hip_y = float((pose_crop[23, 1] + pose_crop[24, 1]) / 2.0)
    cx = float((x0 + x1) / 2.0)
    if hip_y < 0.48 or cx < 0.30 or cx > 0.70:
        return None
    center_penalty = 1.0 - min(abs(cx - 0.5) / 0.20, 1.0)
    return h * w * hip_y * (0.5 + center_penalty)


def _score_striker_pose_fullframe(pose: np.ndarray) -> Optional[float]:
    """Prefer large, low, centered poses (near-court striker on broadcast)."""
    if not is_plausible_human_pose(pose) or not is_single_person_pose(pose):
        return None
    bbox = _pose_bbox(pose)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    h, w = y1 - y0, x1 - x0
    hip_y = float((pose[23, 1] + pose[24, 1]) / 2.0)
    cx = float((x0 + x1) / 2.0)
    if hip_y < 0.52 or cx < 0.25 or cx > 0.75:
        return None
    center_penalty = 1.0 - min(abs(cx - 0.5) / 0.25, 1.0)
    return h * w * hip_y * (0.5 + center_penalty)


def _score_striker_pose_crop_relaxed(pose_crop: np.ndarray) -> Optional[float]:
    if not is_plausible_human_pose(pose_crop) or not is_single_person_pose(pose_crop):
        return None
    bbox = _pose_bbox(pose_crop)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    h, w = y1 - y0, x1 - x0
    hip_y = float((pose_crop[23, 1] + pose_crop[24, 1]) / 2.0)
    cx = float((x0 + x1) / 2.0)
    if hip_y < 0.40 or cx < 0.18 or cx > 0.82:
        return None
    edge_penalty = 1.0 - min(max(0.0, 0.30 - cx, cx - 0.70) / 0.30, 1.0)
    return h * w * hip_y * (0.3 + edge_penalty)


def infer_striker_pose_rgb(
    rgb: np.ndarray,
    estimator: PoseEstimator,
    y_start: float = NEAR_COURT_Y_START,
) -> Optional[np.ndarray]:
    """MediaPipe on near-court crop; return landmarks normalized to full frame."""
    full_h, full_w = rgb.shape[:2]
    crop = rgb[int(y_start * full_h) :, :]
    if crop.size == 0:
        return None
    crop_h, crop_w = crop.shape[:2]
    up = _upscale_rgb(crop)
    result = estimator.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=up))
    if not result.pose_landmarks:
        return None

    for scorer in (_score_striker_pose_crop, _score_striker_pose_crop_relaxed):
        best_pose: Optional[np.ndarray] = None
        best_score = -1.0
        for person in result.pose_landmarks:
            pose_crop = _landmarks_to_array(person)
            score = scorer(pose_crop)
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best_pose = _remap_crop_pose_to_full(pose_crop, y_start, crop_h, full_h)
        if best_pose is not None:
            return best_pose

    return None


def is_plausible_human_pose(pose: np.ndarray, *, min_h: float = 0.14) -> bool:
    visible = (pose[:, 0] > 1e-4) | (pose[:, 1] > 1e-4)
    if visible.sum() < 12:
        return False
    bbox = _pose_bbox(pose)
    if bbox is None:
        return False
    x0, y0, x1, y1 = bbox
    h, w = y1 - y0, x1 - x0
    return h >= min_h and w >= 0.05 and h <= 0.90


def _score_striker_pose_fullframe_native(pose: np.ndarray) -> Optional[float]:
    """Full broadcast frame: players are small, so relax min bbox height."""
    if not is_plausible_human_pose(pose, min_h=0.07) or not is_single_person_pose(pose):
        return None
    bbox = _pose_bbox(pose)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    h, w = y1 - y0, x1 - x0
    hip_y = float((pose[23, 1] + pose[24, 1]) / 2.0)
    cx = float((x0 + x1) / 2.0)
    if hip_y < 0.50 or cx < 0.22 or cx > 0.78:
        return None
    center_penalty = 1.0 - min(abs(cx - 0.5) / 0.28, 1.0)
    return h * w * hip_y * (0.5 + center_penalty)


def infer_striker_pose_native_fullframe(
    rgb: np.ndarray,
    estimator: PoseEstimator,
) -> Optional[np.ndarray]:
    """MediaPipe on the upscaled full native frame (no crop)."""
    up = _upscale_rgb(rgb)
    result = estimator.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=up))
    if not result.pose_landmarks:
        return None
    best_pose: Optional[np.ndarray] = None
    best_score = -1.0
    for person in result.pose_landmarks:
        pose = _landmarks_to_array(person)
        score = _score_striker_pose_fullframe_native(pose)
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_pose = pose
    return best_pose


def _pose_bbox(pose: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    mask = (pose[:, 0] > 1e-4) | (pose[:, 1] > 1e-4)
    if mask.sum() < 12:
        return None
    xs, ys = pose[mask, 0], pose[mask, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def is_single_person_pose(pose: np.ndarray) -> bool:
    """Reject tall skinny poses that span two vertically stacked players."""
    bbox = _pose_bbox(pose)
    if bbox is None:
        return False
    x0, y0, x1, y1 = bbox
    h, w = y1 - y0, x1 - x0
    if w < 1e-6 or h > 0.38:
        return False
    aspect = h / w
    if aspect > 2.8 or aspect < 1.15:
        return False
    nose_y = float(pose[0, 1])
    hip_y = float((pose[23, 1] + pose[24, 1]) / 2.0)
    if nose_y <= 1e-4 or hip_y <= 1e-4 or nose_y >= hip_y - 0.04:
        return False
    if hip_y - nose_y > 0.32:
        return False
    return True


def score_pose_for_teaser(pose: np.ndarray) -> Optional[float]:
    return _score_striker_pose_fullframe(pose) or _score_striker_pose_crop(pose)


def crop_frame_and_pose(
    rgb: np.ndarray,
    pose: np.ndarray,
    margin: float = 0.18,
) -> Tuple[np.ndarray, np.ndarray]:
    bbox = _pose_bbox(pose)
    if bbox is None:
        return rgb, pose
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, x0 - margin)
    y0 = max(0.0, y0 - margin)
    x1 = min(1.0, x1 + margin)
    y1 = min(1.0, y1 + margin)
    if x1 - x0 < 0.08 or y1 - y0 < 0.08:
        return rgb, pose
    h, w = rgb.shape[:2]
    crop = rgb[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)].copy()
    pose_c = pose.copy()
    pose_c[:, 0] = (pose[:, 0] - x0) / max(x1 - x0, 1e-6)
    pose_c[:, 1] = (pose[:, 1] - y0) / max(y1 - y0, 1e-6)
    return crop, pose_c


def adjust_pose(
    pose: np.ndarray,
    dx: float = 0.0,
    dy: float = 0.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Shift/scale normalized landmarks around frame center (manual teaser tuning)."""
    out = pose.copy()
    for j in range(33):
        x, y = float(out[j, 0]), float(out[j, 1])
        if x <= 1e-4 and y <= 1e-4:
            continue
        out[j, 0] = (x - 0.5) * scale + 0.5 + dx
        out[j, 1] = (y - 0.5) * scale + 0.5 + dy
    return out


def infer_pose_on_224_frame(frame_chw: torch.Tensor, estimator: PoseEstimator) -> Optional[np.ndarray]:
    arr = (frame_chw.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    result = estimator.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
    if not result.pose_landmarks:
        return None
    idx = _pick_primary_pose_index(result.pose_landmarks)
    if idx is None:
        return None
    return _landmarks_to_array(result.pose_landmarks[idx])


def infer_striker_pose_rgb_with_fallback(
    rgb: np.ndarray,
    estimator: PoseEstimator,
    y_start: float = NEAR_COURT_Y_START,
) -> Optional[np.ndarray]:
    """Near-court crop first, then full native frame."""
    candidates = [y_start] + [v for v in Y_START_FALLBACKS if v != y_start]
    for ys in candidates:
        pose = infer_striker_pose_rgb(rgb, estimator, y_start=ys)
        if pose is not None:
            return pose
    return infer_striker_pose_native_fullframe(rgb, estimator)


def render_curated_panel(
    ds: FineBadmintonDataset,
    estimator: PoseEstimator,
    sample_index: int,
    timestep: int,
    display_size: int = 384,
    y_start: float = NEAR_COURT_Y_START,
    pose_tune: Optional[PoseTune] = None,
    pose_cache: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Render one teaser panel: native frame + live MediaPipe, cache224 fallback."""
    sample = ds.samples[sample_index]
    rgb = load_native_rgb(sample, frame_index_for_timestep(sample, timestep))
    if rgb is None:
        raise RuntimeError(f"Cannot read frame from {sample['video_path']}")

    pose = infer_striker_pose_rgb_with_fallback(rgb, estimator, y_start=y_start)
    if pose is not None:
        if pose_tune:
            pose = adjust_pose(pose, **dict(pose_tune))
        bgr = draw_mediapipe_overlay(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), pose)
        rgb_out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb_out, (display_size, display_size), interpolation=cv2.INTER_CUBIC)

    if pose_cache is not None:
        print(f"  panel {sample_index}: native pose failed, using 224 cache fallback", flush=True)
        return render_curated_panel_cache224(
            ds, pose_cache, estimator, sample_index, timestep,
            display_size=display_size, pose_tune=pose_tune,
        )
    raise RuntimeError(f"No native pose for sample {sample_index} t={timestep}")


def render_curated_panel_cache224(
    ds: FineBadmintonDataset,
    pose_cache: torch.Tensor,
    estimator: PoseEstimator,
    sample_index: int,
    timestep: int,
    display_size: int = 384,
    crop: bool = True,
    pose_tune: Optional[PoseTune] = None,
) -> np.ndarray:
    """Legacy: cache landmarks on the 224×224 training frame."""
    clip, _ = ds[sample_index]
    frame = clip[timestep]
    rgb = (frame.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    pose = pose_cache[sample_index, timestep].numpy()
    if not is_plausible_human_pose(pose):
        pose = infer_pose_on_224_frame(frame, estimator)
    if pose is None:
        raise RuntimeError(f"No pose for sample {sample_index} t={timestep}")
    if pose_tune:
        pose = adjust_pose(pose, **dict(pose_tune))
    if crop:
        rgb, pose = crop_frame_and_pose(rgb, pose, margin=0.15)
    bgr = draw_mediapipe_overlay(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), pose)
    rgb_out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb_out, (display_size, display_size), interpolation=cv2.INTER_CUBIC)


def draw_mediapipe_overlay(bgr: np.ndarray, pose: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = max(1.0, min(h, w) / 400.0)
    joint_r = max(5, int(round(5 * scale)))
    bone_w = max(3, int(round(4 * scale)))
    out = bgr.copy()
    pts: List[Any] = [None] * 33
    for j in range(33):
        x, y = float(pose[j, 0]), float(pose[j, 1])
        if x <= 1e-4 and y <= 1e-4:
            continue
        cx, cy = int(x * w), int(y * h)
        pts[j] = (cx, cy)
        cv2.circle(out, (cx, cy), joint_r + 1, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (cx, cy), joint_r, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    for a, b in MEDIAPIPE_BONE_PAIRS:
        if pts[a] is None or pts[b] is None:
            continue
        cv2.line(out, pts[a], pts[b], (0, 0, 0), bone_w + 2, lineType=cv2.LINE_AA)
        cv2.line(out, pts[a], pts[b], (0, 255, 255), bone_w, lineType=cv2.LINE_AA)
    return out


def pick_all_stroke_panels(
    ds: FineBadmintonDataset,
    estimator: PoseEstimator,
    stroke_names: Tuple[str, ...],
) -> Dict[str, Tuple[int, int]]:
    best: Dict[str, Optional[Tuple[float, int, int]]] = {s: None for s in stroke_names}
    for i, sample in enumerate(ds.samples):
        labels = ds._map_labels(sample)
        stroke = ds.classes["stroke_type"][labels["stroke_type"]]
        if stroke not in stroke_names:
            continue
        t = contact_timestep(sample)
        rgb = load_native_rgb(sample, frame_index_for_timestep(sample, t))
        if rgb is None:
            continue
        pose = infer_striker_pose_rgb(rgb, estimator)
        if pose is None:
            continue
        score = score_pose_for_teaser(pose)
        if score is None:
            continue
        prev = best[stroke]
        if prev is None or score > prev[0]:
            best[stroke] = (score, i, t)
        if i and i % 2000 == 0:
            print(f"  scanned {i}/{len(ds.samples)}...", flush=True)

    out: Dict[str, Tuple[int, int]] = {}
    for stroke in stroke_names:
        if best[stroke] is None:
            raise RuntimeError(f"No aligned native pose for {stroke!r}")
        out[stroke] = (best[stroke][1], best[stroke][2])
    return out


def render_panel(
    ds: FineBadmintonDataset,
    estimator: PoseEstimator,
    sample_index: int,
    timestep: int,
    display_size: int = 384,
    crop: bool = False,
) -> np.ndarray:
    sample = ds.samples[sample_index]
    rgb = load_native_rgb(sample, frame_index_for_timestep(sample, timestep))
    if rgb is None:
        raise RuntimeError(f"Cannot read frame from {sample['video_path']}")
    pose = infer_striker_pose_rgb(rgb, estimator)
    if pose is None:
        raise RuntimeError(f"No pose at sample {sample_index} t={timestep}")
    if crop:
        rgb, pose = crop_frame_and_pose(rgb, pose)
    bgr = draw_mediapipe_overlay(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), pose)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (display_size, display_size), interpolation=cv2.INTER_CUBIC)
