"""
Build the full (N, T, 33, 3) MediaPipe pose cache for a FineBadminton list — no training.

Use after ``prepare_finebadminton_20k.py`` (Hugging Face) or any layout compatible with
``FineBadmintonDataset``. Output is ``{"pose_cache": Tensor}`` compatible with
``load_pose_cache_bundle`` / all IsoCourt trainers.

Progress is written to a **checkpoint** file (default: ``<output>.inprogress``) every
``--checkpoint-every`` samples so you can **resume** after a crash by re-running the same
command (no extra flags). Use ``--force`` to discard checkpoint + existing output and
rebuild from scratch.

Example (from repo root):

  python backend/pipelines/training/build_full_pose_cache.py \\
    --data-root backend/data \\
    --list-file backend/data/transformed_combined_rounds_output_en_evals_translated.json

Remote / long runs (point ``--output`` at durable storage so checkpoints survive restarts):

  python backend/pipelines/training/build_full_pose_cache.py \\
    --data-root backend/data \\
    --list-file backend/data/transformed_combined_rounds_output_en_evals_translated.json \\
    --output /path/to/durable/checkpoints/pose_cache_mediapipe.pt

Heavy landmarker + native resolution (better on wide-angle footage; slower):

  python backend/pipelines/training/build_full_pose_cache.py \\
    --data-root backend/data \\
    --list-file backend/data/transformed_combined_rounds_output_en_evals_translated.json \\
    --output backend/models/pose_cache_mediapipe.pt \\
    --model backend/models/pose_landmarker_heavy.task \\
    --native-res \\
    --checkpoint-every 500 \\
    --force
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
        "--model",
        default=None,
        help="Path to a .task pose landmarker model (default: models/pose_landmarker_lite.task). "
             "Use models/pose_landmarker_heavy.task for better detection on wide-angle footage.",
    )
    p.add_argument(
        "--native-res",
        action="store_true",
        help="Run MediaPipe on native video resolution instead of 224x224. "
             "Greatly improves detection on distant/small players.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Delete checkpoint and existing output (if present), then rebuild from scratch.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        metavar="N",
        help="Save resume checkpoint every N samples (0 = no checkpoints, single shot; not resumable).",
    )
    p.add_argument(
        "--checkpoint-path",
        default=None,
        help="Resume checkpoint path. Default: <output>.inprogress",
    )
    p.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="Keep the .inprogress file after a successful build (default: delete it).",
    )
    p.add_argument(
        "--start-index",
        type=int,
        default=None,
        metavar="I",
        help="Override resume start: fill samples from index I onward (uses existing checkpoint tensor). "
        "Useful to re-process a suffix after a bad segment without --force.",
    )
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
            if pc.shape[0] == n:
                print(f"Skipping: existing cache matches N={n}: {out_path}")
                print("Pass --force to rebuild.")
                return
            print(
                f"Existing output has {pc.shape[0]} rows but dataset has {n}; will rebuild."
            )

    # Finalize if a previous run wrote checkpoint to completion but final copy failed
    if not args.force and os.path.isfile(ckpt_path):
        ck_try = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ck_try, dict) and int(ck_try.get("next_index", 0)) >= n:
            pc = ck_try["pose_cache"]
            if tuple(pc.shape) == (n, int(dataset.sequence_length), 33, 3):
                lf = ck_try.get("list_fingerprint")
                if lf is not None and list(lf) != list(list_file_fingerprint(list_file)):
                    print(
                        "Checkpoint is complete but annotation fingerprint changed; "
                        "not auto-finalizing. Use --force to rebuild."
                    )
                else:
                    print(f"Found completed checkpoint; writing final file -> {out_path}")
                    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                    torch.save({"pose_cache": pc}, out_path)
                    if not args.keep_checkpoint:
                        os.remove(ckpt_path)
                    print(f"Saved {tuple(pc.shape)} to {out_path}")
                    return

    set_seed(args.seed)
    print(f"Building pose cache for N={n}, T={args.sequence_length} -> {out_path}")
    if args.native_res:
        print("Native resolution mode: pose detection at original video size")
    if args.checkpoint_every > 0:
        print(f"Checkpoints every {args.checkpoint_every} samples -> {ckpt_path}")
    estimator = PoseEstimator(model_path=args.model)

    if args.checkpoint_every <= 0:
        pose_cache = media_pipe_fill_pose_cache(dataset, estimator, native_res=args.native_res)
    else:
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
    torch.save({"pose_cache": pose_cache}, out_path)
    print(f"Saved {tuple(pose_cache.shape)} to {out_path}")

    if args.checkpoint_every > 0 and os.path.isfile(ckpt_path) and not args.keep_checkpoint:
        os.remove(ckpt_path)
        print(f"Removed checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
