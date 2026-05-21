#!/usr/bin/env python3
"""
Build a MediaPipe pose cache for **upstream ST-TR** (``train_gcn_st_tr.py``).

Unlike BST collate (COCO-17 + bones + separate .npy), this writes the standard
``(N, T, 33, 3)`` BlazePose tensor expected by ``IsoCourtOfficialSTTR``, with
ST-TR-friendly extraction:

- **Native video resolution** (not 224×224) for better joints on wide court footage
- **Dual-pose** MediaPipe with **primary-player** selection (largest / most visible)
- Resumable checkpoints (re-run the same command after a crash)

Output default: ``backend/models/pose_cache_st_tr_collated.pt``

Example (repo root, EC2 tmux)::

  ./scripts/ec2/run_train_tmux.sh st_tr_prep \\
    --data-root backend/data \\
    --list-file backend/data/transformed_combined_rounds_output_en_evals_translated.json \\
    --sequence-length 16

Then train::

  ./scripts/ec2/run_train_tmux.sh gcn_st_tr \\
    --pose-cache backend/models/pose_cache_st_tr_collated.pt \\
    --epochs 60 --batch-size 4 --lr 1e-4 --stroke-only-epochs 12
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
    default_st_tr_pose_cache_path,
    list_file_fingerprint,
    load_pose_cache_bundle,
    media_pipe_fill_pose_cache,
    media_pipe_fill_pose_cache_resumable,
)
from core.pose_utils import PoseEstimator
from core.seed_utils import set_seed


def main() -> None:
    default_out = default_st_tr_pose_cache_path(_backend_root)
    p = argparse.ArgumentParser(
        description="Collate MediaPipe pose cache for official ST-TR training."
    )
    p.add_argument("--data-root", required=True)
    p.add_argument("--list-file", required=True)
    p.add_argument("--output", default=None, help=f"Default: {default_out}")
    p.add_argument("--sequence-length", type=int, default=16)
    p.add_argument("--frame-interval", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--model",
        default=None,
        help="MediaPipe .task landmarker (default: models/pose_landmarker_heavy.task).",
    )
    p.add_argument(
        "--lite-model",
        action="store_true",
        help="Use pose_landmarker_lite.task instead of heavy (faster, weaker joints).",
    )
    p.add_argument(
        "--no-native-res",
        action="store_true",
        help="Use 224×224 dataset frames instead of native video resolution.",
    )
    p.add_argument(
        "--num-poses",
        type=int,
        default=2,
        help="MediaPipe max poses per frame (default 2 for primary-player pick).",
    )
    p.add_argument(
        "--no-pick-primary",
        action="store_true",
        help="Always use the first detected pose (legacy behaviour).",
    )
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
    native_res = not args.no_native_res
    pick_primary = not args.no_pick_primary and args.num_poses > 1

    if args.model is None:
        name = (
            "pose_landmarker_lite.task"
            if args.lite_model
            else "pose_landmarker_heavy.task"
        )
        args.model = os.path.join(_backend_root, "models", name)

    if not os.path.isfile(list_file):
        raise SystemExit(f"Missing list file: {list_file}")
    if not os.path.isfile(args.model):
        raise SystemExit(f"Missing MediaPipe model: {args.model}")

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
        for path in (out_path, ckpt_path):
            if os.path.isfile(path):
                print(f"--force: removing {path}")
                os.remove(path)

    if not args.force and os.path.isfile(out_path):
        loaded = load_pose_cache_bundle(out_path)
        if loaded is not None:
            pc = loaded["pose_cache"]
            if pc.shape[0] == n:
                print(f"Skipping: existing cache matches N={n}: {out_path}")
                print("Pass --force to rebuild.")
                return
            print(f"Existing cache rows {pc.shape[0]} != dataset {n}; rebuilding.")

    if not args.force and os.path.isfile(ckpt_path):
        ck_try = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ck_try, dict) and int(ck_try.get("next_index", 0)) >= n:
            pc = ck_try["pose_cache"]
            if tuple(pc.shape) == (n, int(dataset.sequence_length), 33, 3):
                lf = ck_try.get("list_fingerprint")
                if lf is not None and list(lf) != list(list_file_fingerprint(list_file)):
                    print(
                        "Checkpoint complete but list file changed; "
                        "use --force to rebuild."
                    )
                else:
                    print(f"Finalizing completed checkpoint -> {out_path}")
                    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                    torch.save(
                        {
                            "pose_cache": pc,
                            "collate": "st_tr_mediapipe",
                            "native_res": native_res,
                            "pick_primary": pick_primary,
                            "num_poses": args.num_poses,
                        },
                        out_path,
                    )
                    if not args.keep_checkpoint:
                        os.remove(ckpt_path)
                    print(f"Saved {tuple(pc.shape)} to {out_path}")
                    return

    set_seed(args.seed)
    print(
        f"ST-TR collate | N={n} T={args.sequence_length} | native_res={native_res} | "
        f"num_poses={args.num_poses} pick_primary={pick_primary} | model={args.model}"
    )
    print(f"Output -> {out_path}")
    if args.checkpoint_every > 0:
        print(f"Checkpoint every {args.checkpoint_every} -> {ckpt_path}")

    estimator = PoseEstimator(model_path=args.model, num_poses=max(1, args.num_poses))

    if args.checkpoint_every <= 0:
        pose_cache = media_pipe_fill_pose_cache(
            dataset,
            estimator,
            native_res=native_res,
            pick_primary=pick_primary,
        )
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
            native_res=native_res,
            pick_primary=pick_primary,
            start_index=args.start_index,
        )

    task_classes = {k: len(v) for k, v in dataset.classes.items()}
    task_classes["quality"] = 7
    if "stroke_subtype" in task_classes:
        del task_classes["stroke_subtype"]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(
        {
            "pose_cache": pose_cache,
            "task_classes": task_classes,
            "collate": "st_tr_mediapipe",
            "native_res": native_res,
            "pick_primary": pick_primary,
            "num_poses": args.num_poses,
            "mediapipe_model": os.path.basename(args.model),
        },
        out_path,
    )
    print(f"Saved {tuple(pose_cache.shape)} to {out_path}")

    if args.checkpoint_every > 0 and os.path.isfile(ckpt_path) and not args.keep_checkpoint:
        os.remove(ckpt_path)
        print(f"Removed checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
