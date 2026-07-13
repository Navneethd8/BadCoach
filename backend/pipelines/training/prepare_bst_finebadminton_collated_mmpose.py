#!/usr/bin/env python3
"""
Build BST-collated .npy tensors for FineBadminton-20K using **MMPose** instead
of MediaPipe — matching the BST paper's pose estimation pipeline.

Uses ``MMPoseInferencer('human')`` (RTMPose top-down with built-in person
detector) at native video resolution, producing dramatically cleaner COCO-17
keypoints than MediaPipe Lite at 224x224.

**Speed strategy**: samples are grouped by source video so each video is opened
once, unique frame indices are deduped, and MMPose processes the video in a
single streaming pass. This cuts runtime from ~20+ hours to ~1-2 hours on a T4.

Output layout is identical to ``prepare_bst_finebadminton_collated.py``::

    {pose_style}.npy, pos.npy, shuttle.npy, videos_len.npy, labels.npy

Example:

  python backend/pipelines/training/prepare_bst_finebadminton_collated_mmpose.py \\
    --data-root backend/data \\
    --list-file backend/data/transformed_combined_rounds_output_en_evals_translated.json \\
    --output-dir backend/data/bst_finebadminton_collated_mmpose \\
    --sequence-length 16
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

from core.bst_finebadminton_data import (
    PoseStyle,
    bbox_from_joints_xy,
    create_bones,
    get_bone_pairs_coco,
    normalize_joints_bst,
    sort_two_poses_left_to_right,
    image_normalized_feet_pos,
    stack_pose_style,
)
from core.dataset import FineBadmintonDataset
from core.split import video_level_split


def _hip_mx(c: np.ndarray) -> float:
    hips = c[11:13]
    m = (hips[:, 0] != 0) | (hips[:, 1] != 0)
    return float(hips[m][:, 0].mean()) if np.any(m) else float(c[:, 0].mean())


def _parse_preds(preds, w: float, h: float):
    """Parse MMPose predictions for one frame -> (2,17,2) norm joints, (2,2) pos or None."""
    if len(preds) < 1:
        return None, None

    persons = []
    for person in preds:
        kp = np.array(person["keypoints"], dtype=np.float32)
        if kp.shape[0] < 17:
            continue
        bb = np.array(person["bbox"][0], dtype=np.float32) if "bbox" in person else None
        persons.append((kp[:17], bb))

    if len(persons) == 0:
        return None, None

    if len(persons) == 1:
        a_kp, a_bb = persons[0]
        b_kp = np.zeros((17, 2), dtype=np.float32)
        b_bb = None
    else:
        a_kp, a_bb = persons[0]
        b_kp, b_bb = persons[1]
        a_kp, b_kp = sort_two_poses_left_to_right(a_kp, b_kp)
        if a_bb is not None and b_bb is not None:
            if _hip_mx(a_kp) > _hip_mx(b_kp):
                a_bb, b_bb = b_bb, a_bb

    kps = np.stack([a_kp, b_kp])

    pos = np.zeros((2, 2), dtype=np.float32)
    for m_i in range(2):
        pos[m_i] = image_normalized_feet_pos(kps[m_i], w, h)

    if a_bb is not None or b_bb is not None:
        bbox = np.zeros((2, 4), dtype=np.float32)
        bbox[0] = a_bb if a_bb is not None else bbox_from_joints_xy(kps[0:1])[0]
        bbox[1] = b_bb if b_bb is not None else bbox_from_joints_xy(kps[1:2])[0]
    else:
        bbox = bbox_from_joints_xy(kps)

    kps = normalize_joints_bst(kps, bbox, center_align=True)
    return kps, pos


def _process_video(
    video_path: str,
    needed_frames: set,
    inferencer,
):
    """Read only the needed frames via cv2 seek and run MMPose on each.

    Returns dict mapping frame_index -> (joints (2,17,2), pos (2,2)) or None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    sorted_indices = sorted(needed_frames)
    frame_poses = {}
    vid_name = os.path.basename(video_path)

    for fidx in tqdm(sorted_indices, desc=f"  {vid_name}", leave=False, unit="f"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            frame_poses[fidx] = (None, None)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = next(inferencer(rgb, show=False))
        preds = result.get("predictions", [[]])[0]
        kps, pos = _parse_preds(preds, float(w), float(h))
        frame_poses[fidx] = (kps, pos)

    cap.release()
    return frame_poses


def _collect_split(
    dataset: FineBadmintonDataset,
    indices: list,
    inferencer,
    pose_style: PoseStyle,
    seq_len: int,
) -> dict:
    # --- Pass 1: group samples by video, collect needed frame indices --------
    video_to_clips = defaultdict(list)
    for i in indices:
        sample = dataset.samples[i]
        vp = sample["video_path"]
        start, end = sample["start_frame"], sample["end_frame"]
        if end - start <= 0:
            frame_indices = []
        else:
            frame_indices = np.linspace(start, end - 1, seq_len).astype(int).tolist()
        label = dataset._map_labels(sample)["stroke_type"]
        video_to_clips[vp].append((i, frame_indices, label))

    # --- Pass 2: for each video, run MMPose once on the whole file -----------
    video_order = sorted(video_to_clips.keys())
    results_by_idx = {}

    total_unique = 0
    for vp in video_order:
        s = set()
        for _, fi, _ in video_to_clips[vp]:
            s.update(fi)
        total_unique += len(s)
    print(f"Total unique frames to process: {total_unique} across {len(video_order)} videos")

    frames_done = 0
    for vp in tqdm(video_order, desc="MMPose videos"):
        clips = video_to_clips[vp]
        needed = set()
        for _, frame_indices, _ in clips:
            needed.update(frame_indices)
        if not needed:
            for orig_i, frame_indices, label in clips:
                results_by_idx[orig_i] = (None, label)
            continue

        frame_poses = _process_video(vp, needed, inferencer)

        for orig_i, frame_indices, label in clips:
            T = len(frame_indices)
            joints_clip = np.zeros((T, 2, 17, 2), dtype=np.float32)
            pos_clip = np.zeros((T, 2, 2), dtype=np.float32)
            for t, fidx in enumerate(frame_indices):
                entry = frame_poses.get(fidx)
                if entry is not None:
                    kps, pos = entry
                    if kps is not None:
                        joints_clip[t] = kps
                        pos_clip[t] = pos
            human_pose = stack_pose_style(joints_clip, pose_style)
            shuttle = np.zeros((T, 2), dtype=np.float32)
            results_by_idx[orig_i] = ((human_pose, pos_clip, shuttle, T), label)

    # --- Pass 3: assemble in original index order ----------------------------
    poses, poss, shuttles, lens, labels = [], [], [], [], []
    j_count = 17 + (len(get_bone_pairs_coco()) if pose_style == "JnB_bone" else 0)
    for i in indices:
        entry = results_by_idx[i]
        tup, label = entry
        if tup is None:
            poses.append(np.zeros((seq_len, 2, j_count, 2), dtype=np.float32))
            poss.append(np.zeros((seq_len, 2, 2), dtype=np.float32))
            shuttles.append(np.zeros((seq_len, 2), dtype=np.float32))
            lens.append(0)
        else:
            hp, po, sh, vl = tup
            poses.append(hp)
            poss.append(po)
            shuttles.append(sh)
            lens.append(vl)
        labels.append(int(label))

    return {
        "human_pose": np.stack(poses, axis=0).astype(np.float32),
        "pos": np.stack(poss, axis=0).astype(np.float32),
        "shuttle": np.stack(shuttles, axis=0).astype(np.float32),
        "videos_len": np.array(lens, dtype=np.int64),
        "labels": np.array(labels, dtype=np.int64),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare BST collated npy for FineBadminton-20K (MMPose)."
    )
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--list-file", required=True)
    ap.add_argument(
        "--output-dir", required=True,
        help="e.g. backend/data/bst_finebadminton_collated_mmpose",
    )
    ap.add_argument("--sequence-length", type=int, default=16)
    ap.add_argument("--frame-interval", type=int, default=2)
    ap.add_argument(
        "--pose-style", choices=["J_only", "JnB_bone"], default="JnB_bone",
        help="Paper uses J+B (JnB_bone); J_only is lighter.",
    )
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument(
        "--split-ratio",
        type=float,
        default=None,
        help="Optional override: train fraction on non-test videos (default: 70/10/20 policy).",
    )
    ap.add_argument(
        "--pose2d", default="human",
        help="MMPoseInferencer model alias (default: 'human' = RTMPose).",
    )
    args = ap.parse_args()

    from mmpose.apis import MMPoseInferencer

    pose_style: PoseStyle = args.pose_style

    dataset = FineBadmintonDataset(
        args.data_root,
        args.list_file,
        sequence_length=args.sequence_length,
        frame_interval=args.frame_interval,
    )
    n = len(dataset)
    if n == 0:
        raise SystemExit("No samples loaded — check data-root and list-file.")

    split_kw = {"seed": args.split_seed}
    if args.split_ratio is not None:
        split_kw["ratio"] = args.split_ratio
    train_idx, val_idx, test_idx = video_level_split(dataset.samples, **split_kw)
    print(f"Split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test (seed={args.split_seed})")

    print(f"Initializing MMPoseInferencer('{args.pose2d}')...")
    inferencer = MMPoseInferencer(args.pose2d)

    out_root = os.path.abspath(args.output_dir)
    os.makedirs(out_root, exist_ok=True)

    for name, idx_list in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        sub = os.path.join(out_root, name)
        os.makedirs(sub, exist_ok=True)
        bundle = _collect_split(
            dataset, idx_list, inferencer, pose_style, args.sequence_length
        )
        np.save(os.path.join(sub, f"{pose_style}.npy"), bundle["human_pose"])
        np.save(os.path.join(sub, "pos.npy"), bundle["pos"])
        np.save(os.path.join(sub, "shuttle.npy"), bundle["shuttle"])
        np.save(os.path.join(sub, "videos_len.npy"), bundle["videos_len"])
        np.save(os.path.join(sub, "labels.npy"), bundle["labels"])
        print(f"Wrote {name}: {bundle['human_pose'].shape[0]} samples -> {sub}")

    meta_path = os.path.join(out_root, "meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"pose_estimator=mmpose ({args.pose2d})\n")
        f.write(f"resolution=native\n")
        f.write(f"list_file={os.path.abspath(args.list_file)}\n")
        f.write(f"sequence_length={args.sequence_length}\n")
        f.write(f"pose_style={pose_style}\n")
        f.write(f"split_seed={args.split_seed} train_ratio={args.split_ratio}\n")
        f.write(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}\n")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
