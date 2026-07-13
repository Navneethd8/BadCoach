#!/usr/bin/env python3
"""
Build BST-collated .npy tensors for FineBadminton-20K (stroke_type baseline).

Produces the same layout as BST ``Dataset_npy_collated`` (train/val subdirs):

  {pose_style}.npy, pos.npy, shuttle.npy, videos_len.npy, labels.npy

Run from repo root after data prep (``prepare_finebadminton_20k.py``). See
``core/bst_finebadminton_data.py`` for differences vs full ShuttleSet+MMPose+TrackNet.

Example:

  python backend/pipelines/training/prepare_bst_finebadminton_collated.py \\
    --data-root backend/data \\
    --list-file backend/data/transformed_combined_rounds_output_en_evals_translated.json \\
    --output-dir backend/data/bst_finebadminton_collated \\
    --sequence-length 30
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

from core.bst_finebadminton_data import PoseStyle, frames_to_bst_arrays
from core.dataset import FineBadmintonDataset
from core.pose_utils import PoseEstimator
from core.split import SPLIT_TRAIN_RATIO, video_level_split


def _collect_split(
    dataset: FineBadmintonDataset,
    indices: list,
    pose_estimator: PoseEstimator,
    pose_style: PoseStyle,
) -> dict:
    poses = []
    poss = []
    shuttles = []
    lens = []
    labels = []

    for i in tqdm(indices, desc="BST features"):
        frames, tensor_labels = dataset[i]
        human_pose, pos, shuttle, video_len = frames_to_bst_arrays(
            frames, pose_estimator, pose_style=pose_style
        )
        poses.append(human_pose)
        poss.append(pos)
        shuttles.append(shuttle)
        lens.append(video_len)
        labels.append(int(tensor_labels["stroke_type"].item()))

    return {
        "human_pose": np.stack(poses, axis=0).astype(np.float32),
        "pos": np.stack(poss, axis=0).astype(np.float32),
        "shuttle": np.stack(shuttles, axis=0).astype(np.float32),
        "videos_len": np.array(lens, dtype=np.int64),
        "labels": np.array(labels, dtype=np.int64),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare BST collated npy for FineBadminton-20K.")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--list-file", required=True)
    ap.add_argument("--output-dir", required=True, help="e.g. backend/data/bst_finebadminton_collated")
    ap.add_argument("--sequence-length", type=int, default=30)
    ap.add_argument("--frame-interval", type=int, default=2)
    ap.add_argument(
        "--pose-style",
        choices=["J_only", "JnB_bone"],
        default="JnB_bone",
        help="Paper uses J+B (JnB_bone); J_only is lighter.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="MediaPipe .task path (default: lite under backend/models).",
    )
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--split-ratio", type=float, default=SPLIT_TRAIN_RATIO)
    args = ap.parse_args()

    pose_style: PoseStyle = args.pose_style
    model_path = args.model
    if model_path is None:
        model_path = os.path.join(_backend_root, "models", "pose_landmarker_lite.task")

    dataset = FineBadmintonDataset(
        args.data_root,
        args.list_file,
        sequence_length=args.sequence_length,
        frame_interval=args.frame_interval,
    )
    n = len(dataset)
    if n == 0:
        raise SystemExit("No samples loaded — check data-root and list-file.")

    all_idx = list(range(n))
    train_idx, val_idx, test_idx = video_level_split(dataset.samples, seed=args.split_seed, ratio=args.split_ratio)

    pose_estimator = PoseEstimator(model_path=model_path, num_poses=2)

    out_root = os.path.abspath(args.output_dir)
    os.makedirs(out_root, exist_ok=True)

    for name, idx_list in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        sub = os.path.join(out_root, name)
        os.makedirs(sub, exist_ok=True)
        bundle = _collect_split(dataset, idx_list, pose_estimator, pose_style)
        np.save(os.path.join(sub, f"{pose_style}.npy"), bundle["human_pose"])
        np.save(os.path.join(sub, "pos.npy"), bundle["pos"])
        np.save(os.path.join(sub, "shuttle.npy"), bundle["shuttle"])
        np.save(os.path.join(sub, "videos_len.npy"), bundle["videos_len"])
        np.save(os.path.join(sub, "labels.npy"), bundle["labels"])
        print(f"Wrote {name}: {bundle['human_pose'].shape[0]} samples -> {sub}")

    meta_path = os.path.join(out_root, "meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"list_file={os.path.abspath(args.list_file)}\n")
        f.write(f"sequence_length={args.sequence_length}\n")
        f.write(f"pose_style={pose_style}\n")
        f.write(f"split_seed={args.split_seed} train_ratio={args.split_ratio}\n")
        f.write(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}\n")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
