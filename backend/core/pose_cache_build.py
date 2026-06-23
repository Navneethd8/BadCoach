"""
Build (N, T, 33, 3) MediaPipe pose caches without ``list`` + ``torch.stack``.

Appending one tensor per sample duplicates memory (list holds N tensors, then stack
allocates again). On memory-constrained hosts that often forces swap; iteration time creeps up and tqdm
ETAs look like they are "getting longer". Preallocating a single buffer keeps RAM flat.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from core.training_progress import tqdm_pose_cache_build

# Default on-disk name (shared across trainers).
DEFAULT_POSE_CACHE_FILENAME = "pose_cache_mediapipe.pt"
ST_TR_POSE_CACHE_FILENAME = "pose_cache_st_tr_collated.pt"
# Older repos used this filename; still loaded if mediapipe cache is missing.
LEGACY_POSE_CACHE_FILENAME = "pose_cache_staeformer.pt"


def default_pose_cache_path(backend_root: str) -> str:
    return os.path.join(os.path.abspath(backend_root), "models", DEFAULT_POSE_CACHE_FILENAME)


def default_st_tr_pose_cache_path(backend_root: str) -> str:
    """Pose cache built for upstream ST-TR (native-res MediaPipe, primary-player pick)."""
    return os.path.join(os.path.abspath(backend_root), "models", ST_TR_POSE_CACHE_FILENAME)


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
    Load ``{"pose_cache": Tensor, ...}`` from ``cache_path``, or from
    ``LEGACY_POSE_CACHE_FILENAME`` in the same directory when the default file is absent.
    """
    requested = os.path.abspath(cache_path)
    for p in _pose_cache_load_candidates(cache_path):
        resolved = os.path.abspath(p)
        if not os.path.isfile(resolved):
            continue
        if resolved != requested:
            print(
                f"Loading pose cache from {resolved} (legacy filename; "
                f"prefer renaming to {DEFAULT_POSE_CACHE_FILENAME})"
            )
        else:
            print(f"Loading pose cache from {resolved}...")
        return torch.load(resolved, map_location="cpu", weights_only=False)
    return None


def _load_native_res_frames(dataset_raw, i: int, T: int) -> torch.Tensor:
    """Load sample ``i`` at native video resolution for pose detection.

    Returns (T, C, H, W) float32 in [0, 1] RGB at the original frame size
    (no 224x224 resize).  Falls back to the dataset's own loader if the video
    cannot be opened.
    """
    import cv2
    import numpy as np

    sample = dataset_raw.samples[i]
    video_path = sample["video_path"]
    start_frame = sample["start_frame"]
    end_frame = sample["end_frame"]

    duration = end_frame - start_frame
    if duration <= 0:
        return None

    indices = np.linspace(start_frame, end_frame - 1, T).astype(int)

    cap = dataset_raw._get_cap(video_path)
    if cap is None:
        return None

    frames: list[torch.Tensor] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if ok and bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(
                torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            )
        else:
            return None

    return torch.stack(frames)


def _pick_primary_pose_index(pose_landmarks_list) -> int:
    """Choose the main player when MediaPipe returns multiple poses."""
    best_i = 0
    best_score = -1.0
    for i, pose in enumerate(pose_landmarks_list):
        vis_sum = 0.0
        xs: list[float] = []
        ys: list[float] = []
        for lm in pose:
            vis_sum += float(lm.visibility)
            if lm.visibility > 0.3:
                xs.append(float(lm.x))
                ys.append(float(lm.y))
        if len(xs) >= 4:
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            score = vis_sum + area
        else:
            score = vis_sum
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


def _poses_from_frames_tensor(
    frames: torch.Tensor,
    pose_estimator,
    *,
    pick_primary: bool = False,
) -> torch.Tensor:
    """Per-frame MediaPipe on ``(T, C, H, W)`` RGB [0,1] -> ``(T, 33, 3)``."""
    import mediapipe as mp

    T = int(frames.shape[0])
    out = torch.zeros((T, 33, 3), dtype=torch.float32)
    frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()
    frames_np = (frames_np * 255.0).clip(0, 255).astype(np.uint8)

    for t in range(T):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frames_np[t])
        result = pose_estimator.detector.detect(mp_image)
        if not result.pose_landmarks:
            continue
        idx = (
            _pick_primary_pose_index(result.pose_landmarks)
            if pick_primary and len(result.pose_landmarks) > 1
            else 0
        )
        person = result.pose_landmarks[idx]
        for j, lm in enumerate(person):
            out[t, j, 0] = float(lm.x)
            out[t, j, 1] = float(lm.y)
            out[t, j, 2] = float(lm.z)
    return out


def _pose_row_for_index(
    dataset_raw,
    pose_estimator,
    i: int,
    T: int,
    *,
    native_res: bool = False,
    pick_primary: bool = False,
) -> torch.Tensor:
    """Decode sample ``i`` and return pose tensor (T, 33, 3) float32.

    When *native_res* is True, frames are loaded at the original video
    resolution instead of the dataset's default 224x224 — dramatically
    improving MediaPipe detection on wide-angle / distant-player footage.
    """
    if native_res:
        frames = _load_native_res_frames(dataset_raw, i, T)
        if frames is None:
            return torch.zeros((T, 33, 3), dtype=torch.float32)
    else:
        frames, _ = dataset_raw[i]
    if pick_primary:
        row = _poses_from_frames_tensor(frames, pose_estimator, pick_primary=True)
    else:
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
    return row


def media_pipe_fill_pose_cache(
    dataset_raw,
    pose_estimator,
    *,
    native_res: bool = False,
    pick_primary: bool = False,
) -> torch.Tensor:
    """
    Fill a single float32 tensor of shape (len(dataset_raw), T, 33, 3) in index order.

    ``dataset_raw`` must match ``FineBadmintonDataset`` layout (``sequence_length``, ``__getitem__``).
    """
    n = len(dataset_raw)
    T = int(dataset_raw.sequence_length)
    out = torch.empty((n, T, 33, 3), dtype=torch.float32)
    for i in tqdm_pose_cache_build(n):
        out[i].copy_(
            _pose_row_for_index(
                dataset_raw,
                pose_estimator,
                i,
                T,
                native_res=native_res,
                pick_primary=pick_primary,
            )
        )
    return out.contiguous()


def list_file_fingerprint(list_file: str) -> Tuple[float, int]:
    """Mtime + size for detecting annotation changes across resume."""
    st = os.stat(list_file)
    return (float(st.st_mtime), int(st.st_size))


def _atomic_torch_save(obj: Dict[str, Any], path: str) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def media_pipe_fill_pose_cache_resumable(
    dataset_raw,
    pose_estimator,
    *,
    checkpoint_path: str,
    checkpoint_every: int,
    list_file: str,
    sequence_length: int,
    frame_interval: int,
    force: bool = False,
    native_res: bool = False,
    pick_primary: bool = False,
    start_index: Optional[int] = None,
) -> torch.Tensor:
    """
    Same output shape as ``media_pipe_fill_pose_cache``, with periodic checkpoints so a
    crash can be resumed by re-running the same command.

    Checkpoint file stores ``pose_cache``, ``next_index``, and fingerprint fields so a
    changed annotation list is not silently continued.
    """
    n = len(dataset_raw)
    T = int(dataset_raw.sequence_length)
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1")

    fp = list_file_fingerprint(list_file)
    ckpt_path = os.path.abspath(checkpoint_path)

    if force and os.path.isfile(ckpt_path):
        os.remove(ckpt_path)

    start = 0
    out: Optional[torch.Tensor] = None
    had_checkpoint = False

    if os.path.isfile(ckpt_path):
        had_checkpoint = True
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(ck, dict) or "pose_cache" not in ck:
            raise ValueError(f"Invalid checkpoint (missing pose_cache): {ckpt_path}")
        if ck.get("n") != n or int(ck.get("T", -1)) != T:
            raise ValueError(
                f"Checkpoint n/T ({ck.get('n')}, {ck.get('T')}) != dataset ({n}, {T}). "
                "Use --force to discard checkpoint."
            )
        if int(ck.get("sequence_length", -1)) != sequence_length or int(
            ck.get("frame_interval", -2)
        ) != frame_interval:
            raise ValueError(
                "Checkpoint sequence_length / frame_interval mismatch. Use --force to discard."
            )
        if bool(ck.get("native_res", False)) != bool(native_res) or bool(
            ck.get("pick_primary", False)
        ) != bool(pick_primary):
            raise ValueError(
                "Checkpoint native_res / pick_primary mismatch. Use --force to discard."
            )
        old_fp = ck.get("list_fingerprint")
        if old_fp is not None and tuple(old_fp) != fp:
            raise ValueError(
                "Annotation list changed since checkpoint (mtime/size). Use --force to discard."
            )
        out = ck["pose_cache"].to(torch.float32).contiguous()
        if tuple(out.shape) != (n, T, 33, 3):
            raise ValueError(
                f"Checkpoint pose_cache shape {tuple(out.shape)} != expected ({n}, {T}, 33, 3)"
            )
        start = int(ck.get("next_index", 0))
        if start < 0 or start > n:
            raise ValueError(f"Invalid next_index {start} in checkpoint")

    if out is None:
        out = torch.zeros((n, T, 33, 3), dtype=torch.float32)

    if start_index is not None:
        si = max(0, min(int(start_index), n))
        if si != start:
            print(f"--start-index: overriding resume start {start} -> {si}")
        start = si

    if start >= n:
        print(f"Checkpoint already complete (next_index={start}); nothing to fill.")
        return out

    if had_checkpoint or start > 0:
        print(f"Resuming pose cache from sample {start}/{n} (checkpoint: {ckpt_path})")

    for i in tqdm_pose_cache_build(n, start=start):
        out[i].copy_(
            _pose_row_for_index(
                dataset_raw,
                pose_estimator,
                i,
                T,
                native_res=native_res,
                pick_primary=pick_primary,
            )
        )
        done = i + 1
        if done % checkpoint_every == 0 or done == n:
            _atomic_torch_save(
                {
                    "pose_cache": out,
                    "next_index": done,
                    "n": n,
                    "T": T,
                    "sequence_length": sequence_length,
                    "frame_interval": frame_interval,
                    "list_fingerprint": list(fp),
                    "native_res": bool(native_res),
                    "pick_primary": bool(pick_primary),
                },
                ckpt_path,
            )

    return out.contiguous()
