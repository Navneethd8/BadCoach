"""
Build the full (N, T, 33, 3) MediaPipe pose cache for a FineBadminton list — no training.

Use after ``prepare_finebadminton_20k.py`` (Hugging Face) or any layout compatible with
``FineBadmintonDataset``. Output is ``{"pose_cache": Tensor}`` compatible with
``load_pose_cache_bundle`` / all IsoCourt trainers.

Example (from repo root):

  python backend/pipelines/training/build_full_pose_cache.py \\
    --data-root backend/data \\
    --list-file backend/data/transformed_combined_rounds_output_en_evals_translated.json

Colab (save to Drive so a disconnect does not lose work):

  python backend/pipelines/training/build_full_pose_cache.py \\
    --data-root backend/data \\
    --list-file backend/data/transformed_combined_rounds_output_en_evals_translated.json \\
    --output /content/drive/MyDrive/IsoCourt/checkpoints/pose_cache_mediapipe.pt
"""
from __future__ import annotations

import argparse
import os
import sys

_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

import torch

from core.dataset import FineBadmintonDataset
from core.pose_cache_build import (
    default_pose_cache_path,
    load_pose_cache_bundle,
    media_pipe_fill_pose_cache,
)
from core.pose_utils import PoseEstimator
from core.seed_utils import set_seed


def main() -> None:
    default_out = default_pose_cache_path(_backend_root)
    p = argparse.ArgumentParser(description="Build full MediaPipe pose cache (no ML training).")
    p.add_argument("--data-root", required=True, help="Dataset root (videos under data/ or FineBadminton-20K/videos/).")
    p.add_argument("--list-file", required=True, help="Merged annotations JSON (e.g. transformed_combined_...json).")
    p.add_argument(
        "--output",
        default=None,
        help=f"Output .pt path. Default: {default_out}",
    )
    p.add_argument("--sequence-length", type=int, default=16)
    p.add_argument("--frame-interval", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if an existing cache matches dataset length.",
    )
    args = p.parse_args()

    data_root = os.path.abspath(args.data_root)
    list_file = os.path.abspath(args.list_file)
    out_path = os.path.abspath(args.output) if args.output else default_out

    if not os.path.isfile(list_file):
        raise SystemExit(f"Missing list file: {list_file}")

    dataset = FineBadmintonDataset(
        data_root,
        list_file,
        transform=None,
        sequence_length=args.sequence_length,
        frame_interval=args.frame_interval,
    )
    n = len(dataset)
    if n == 0:
        raise SystemExit("Dataset is empty — check data_root and list_file.")

    if not args.force and os.path.isfile(out_path):
        loaded = load_pose_cache_bundle(out_path)
        if loaded is not None:
            pc = loaded["pose_cache"]
            if pc.shape[0] == n:
                print(f"Skipping: existing cache matches N={n}: {out_path}")
                print("Pass --force to rebuild.")
                return
            print(
                f"Existing cache has {pc.shape[0]} rows but dataset has {n}; rebuilding."
            )

    set_seed(args.seed)
    print(f"Building pose cache for N={n}, T={args.sequence_length} -> {out_path}")
    estimator = PoseEstimator()
    pose_cache = media_pipe_fill_pose_cache(dataset, estimator)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({"pose_cache": pose_cache}, out_path)
    print(f"Saved {tuple(pose_cache.shape)} to {out_path}")


if __name__ == "__main__":
    main()
