"""
Build the full (N, T, 33, 3) MediaPipe pose cache for a FineBadminton list — no training.

Example (from repo root):

  python backend/pipelines/training/build_full_pose_cache.py \\
    --data-root backend/data \\
    --list-file backend/data/transformed_combined_rounds_output_en_evals_translated.json \\
    --output backend/models/pose_cache_span_linspace.pt \\
    --sampling span_linspace \\
    --checkpoint-every 500
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
    list_file_fingerprint,
    load_pose_cache_bundle,
    media_pipe_fill_pose_cache,
    media_pipe_fill_pose_cache_resumable,
)
from core.pose_utils import PoseEstimator
from core.seed_utils import set_seed


def main() -> None:
    default_out = default_pose_cache_path(_backend_root)
    p = argparse.ArgumentParser(description="Build full MediaPipe pose cache (no ML training).")
    p.add_argument("--data-root", required=True)
    p.add_argument("--list-file", required=True)
    p.add_argument("--output", default=None, help=f"Default: {default_out}")
    p.add_argument("--sequence-length", type=int, default=16)
    p.add_argument("--frame-interval", type=int, default=2)
    p.add_argument(
        "--sampling",
        choices=["span_linspace", "hit_centered"],
        default="span_linspace",
        help="Frame indices for pose extraction (must match training --sampling).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default=None, help="Optional .task pose landmarker model path.")
    p.add_argument("--native-res", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--checkpoint-path", default=None)
    p.add_argument("--keep-checkpoint", action="store_true")
    p.add_argument("--start-index", type=int, default=None)
    args = p.parse_args()

    data_root = os.path.abspath(args.data_root)
    list_file = os.path.abspath(args.list_file)
    out_path = os.path.abspath(args.output) if args.output else default_out
    ckpt_path = (
        os.path.abspath(args.checkpoint_path)
        if args.checkpoint_path
        else f"{out_path}.inprogress"
    )

    if not os.path.isfile(list_file):
        raise SystemExit(f"Missing list file: {list_file}")

    dataset = FineBadmintonDataset(
        data_root,
        list_file,
        transform=None,
        sequence_length=args.sequence_length,
        frame_interval=args.frame_interval,
        sampling_mode=args.sampling,
    )
    n = len(dataset)
    if n == 0:
        raise SystemExit("Dataset is empty — check data_root and list_file.")

    if args.force:
        if os.path.isfile(out_path):
            print(f"--force: removing existing output {out_path}")
            os.remove(out_path)
        if os.path.isfile(ckpt_path):
            print(f"--force: removing checkpoint {ckpt_path}")
            os.remove(ckpt_path)

    if not args.force and os.path.isfile(out_path):
        loaded = load_pose_cache_bundle(out_path)
        if loaded is not None:
            pc = loaded["pose_cache"]
            cached_sampling = loaded.get("sampling_mode", "span_linspace")
            if pc.shape[0] == n and cached_sampling == args.sampling:
                print(f"Skipping: existing cache matches N={n}, sampling={args.sampling}: {out_path}")
                return
            print(
                f"Existing cache N={pc.shape[0]} sampling={cached_sampling!r}; "
                f"need N={n} sampling={args.sampling!r}; will rebuild."
            )

    set_seed(args.seed)
    print(f"Building pose cache for N={n}, T={args.sequence_length}, sampling={args.sampling}")
    print(f"Output -> {out_path}")
    estimator = PoseEstimator(model_path=args.model)

    if args.checkpoint_every <= 0:
        pose_cache = media_pipe_fill_pose_cache(dataset, estimator, native_res=args.native_res)
    else:
        print(f"Checkpoints every {args.checkpoint_every} -> {ckpt_path}")
        pose_cache = media_pipe_fill_pose_cache_resumable(
            dataset,
            estimator,
            checkpoint_path=ckpt_path,
            checkpoint_every=args.checkpoint_every,
            list_file=list_file,
            sequence_length=args.sequence_length,
            frame_interval=args.frame_interval,
            force=False,
            native_res=args.native_res,
            start_index=args.start_index,
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save({"pose_cache": pose_cache, "sampling_mode": args.sampling}, out_path)
    print(f"Saved {tuple(pose_cache.shape)} to {out_path}")

    if args.checkpoint_every > 0 and os.path.isfile(ckpt_path) and not args.keep_checkpoint:
        os.remove(ckpt_path)
        print(f"Removed checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
