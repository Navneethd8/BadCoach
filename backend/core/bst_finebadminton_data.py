"""
BST-style tensors for FineBadminton-20K stroke-type baseline.

Upstream BST uses MMPose COCO-17 + TrackNet shuttle + court homography (see
https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer ). Here we use
MediaPipe dual poses mapped to COCO-17, zeros for shuttle trajectory, and
image-normalized foot positions instead of court homography — documented so
numbers are comparable to other IsoCourt trainers on the same labels/split.

Normalization follows ``normalize_joints`` / ``normalize_shuttlecock`` /
``normalize_position`` logic from BST ``prepare_train_on_*.py`` (bbox-relative
joints; shuttle [0,1] in frame; position as feet midpoint — court projection
when homography is unavailable uses frame-normalized coordinates).
"""
from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch

PoseStyle = Literal["J_only", "JnB_bone"]

# BlazePose 33 indices -> COCO-17 (same topology as BST shuttleset_dataset).
_MP_TO_COCO17: Tuple[int, ...] = (
    0,
    2,
    5,
    7,
    8,
    11,
    12,
    13,
    14,
    15,
    16,
    23,
    24,
    25,
    26,
    27,
    28,
)


def get_bone_pairs_coco() -> List[Tuple[int, int]]:
    return [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]


def create_bones(joints: np.ndarray, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    """joints: (t, m, J, 2) -> bones same layout as BST shuttleset_dataset."""
    bones = []
    for start, end in pairs:
        sj = joints[:, :, start, :]
        ej = joints[:, :, end, :]
        bone = np.where((sj != 0.0) & (ej != 0.0), ej - sj, 0.0)
        bones.append(bone)
    return np.stack(bones, axis=-2)


def normalize_joints_bst(
    arr: np.ndarray,
    bbox: np.ndarray,
    *,
    center_align: bool = True,
) -> np.ndarray:
    """Match BST ``normalize_joints`` for arr (m, J, 2), bbox (m, 4) xyxy pixels."""
    dist = np.linalg.norm(bbox[:, 2:] - bbox[:, :2], axis=-1, keepdims=True)
    dist = np.where(dist > 1e-6, dist, 1.0)

    arr_x = arr[:, :, 0]
    arr_y = arr[:, :, 1]
    x_norm = np.where(arr_x != 0.0, (arr_x - bbox[:, None, 0]) / dist, 0.0)
    y_norm = np.where(arr_y != 0.0, (arr_y - bbox[:, None, 1]) / dist, 0.0)

    if center_align:
        center = (bbox[:, :2] + bbox[:, 2:]) / 2
        c_norm = (center - bbox[:, :2]) / dist
        x_norm = x_norm - c_norm[:, None, 0]
        y_norm = y_norm - c_norm[:, None, 1]

    return np.stack((x_norm, y_norm), axis=-1).astype(np.float32)


def bbox_from_joints_xy(joints_mj2: np.ndarray) -> np.ndarray:
    """joints (m, J, 2) in pixels; zeros ignored."""
    m = joints_mj2.shape[0]
    out = np.zeros((m, 4), dtype=np.float32)
    for i in range(m):
        j = joints_mj2[i]
        mask = (j[:, 0] != 0.0) | (j[:, 1] != 0.0)
        if not np.any(mask):
            out[i] = [0.0, 0.0, 1.0, 1.0]
            continue
        pts = j[mask]
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        out[i] = [x1, y1, x2, y2]
    return out


def mediapipe_to_coco17_xy(
    landmarks_33,
    width: float,
    height: float,
    vis_thresh: float = 0.3,
) -> np.ndarray:
    """One person's landmarks -> (17, 2) pixel xy; missing -> 0."""
    out = np.zeros((17, 2), dtype=np.float32)
    for ci, mi in enumerate(_MP_TO_COCO17):
        lm = landmarks_33[mi]
        if lm.visibility < vis_thresh:
            continue
        out[ci, 0] = lm.x * width
        out[ci, 1] = lm.y * height
    return out


def sort_two_poses_left_to_right(coco_a: np.ndarray, coco_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Order players by mean x of hips (COCO indices 11,12)."""
    def hip_mx(c: np.ndarray) -> float:
        hips = c[11:13]
        m = (hips[:, 0] != 0) | (hips[:, 1] != 0)
        if not np.any(m):
            return float(c[:, 0].mean()) if np.any(c) else 0.0
        return float(hips[m][:, 0].mean())

    ha, hb = hip_mx(coco_a), hip_mx(coco_b)
    if ha <= hb:
        return coco_a, coco_b
    return coco_b, coco_a


def image_normalized_feet_pos(coco17: np.ndarray, width: float, height: float) -> np.ndarray:
    """Feet midpoint (x,y) in [0,1] x [0,1] — BST-style substitute without homography."""
    ankles = coco17[15:17]
    m = (ankles[:, 0] != 0) | (ankles[:, 1] != 0)
    if not np.any(m):
        return np.array([0.5, 0.5], dtype=np.float32)
    ft = ankles[m].mean(axis=0)
    return np.array([ft[0] / max(width, 1.0), ft[1] / max(height, 1.0)], dtype=np.float32)


def stack_pose_style(
    joints_tmjp2: np.ndarray,
    pose_style: PoseStyle,
) -> np.ndarray:
    """joints (t, m, J, 2) -> (t, m, J', 2) with bones if needed."""
    pairs = get_bone_pairs_coco()
    if pose_style == "J_only":
        return joints_tmjp2.astype(np.float32)
    bones = create_bones(joints_tmjp2, pairs)
    return np.concatenate((joints_tmjp2, bones), axis=-2).astype(np.float32)


def frames_to_bst_arrays(
    frames_tc_hw: torch.Tensor,
    pose_estimator,
    *,
    pose_style: PoseStyle = "JnB_bone",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Decode ``frames_tc_hw`` (T,3,H,W) [0,1] RGB with dual MediaPipe poses.

    Returns:
        human_pose: (T, 2, J, 2)  J=17 or 17+num_bones
        pos: (T, 2, 2)  image-normalized feet proxy
        shuttle: (T, 2)  zeros (no TrackNet on this baseline)
        video_len: int  effective length (= T for dense sampling)
    """
    import mediapipe as mp

    t, _, h, w = frames_tc_hw.shape
    frames_np = frames_tc_hw.permute(0, 2, 3, 1).cpu().numpy()
    frames_np = (frames_np * 255).astype(np.uint8)

    joints_pix = np.zeros((t, 2, 17, 2), dtype=np.float32)
    pos_img = np.zeros((t, 2, 2), dtype=np.float32)

    for i in range(t):
        rgb = frames_np[i]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = pose_estimator.detector.detect(mp_image)

        if not result.pose_landmarks:
            continue

        poses_mp = list(result.pose_landmarks)[:2]
        cocos = []
        for pl in poses_mp:
            cocos.append(mediapipe_to_coco17_xy(pl, float(w), float(h)))

        if len(cocos) == 0:
            continue
        if len(cocos) == 1:
            a, b = cocos[0], np.zeros((17, 2), dtype=np.float32)
        else:
            a, b = sort_two_poses_left_to_right(cocos[0], cocos[1])

        joints_pix[i, 0] = a
        joints_pix[i, 1] = b

        for m in range(2):
            pos_img[i, m] = image_normalized_feet_pos(joints_pix[i, m], float(w), float(h))

        bbox = bbox_from_joints_xy(joints_pix[i])
        joints_pix[i] = normalize_joints_bst(joints_pix[i], bbox, center_align=True)

    shuttle = np.zeros((t, 2), dtype=np.float32)
    human_pose = stack_pose_style(joints_pix, pose_style)
    return human_pose, pos_img, shuttle, t
