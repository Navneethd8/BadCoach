#!/usr/bin/env python3
"""
Build placeholder shuttle trajectory cache aligned with pose cache indices.

Shape: ``(N, T, 2)`` normalized (x, y) in [0, 1] per frame.
v1 writes zeros — replace with TrackNet inference when available.

Run from repo root after pose cache exists:

  python backend/scripts/build_shuttle_cache.py \\
    --pose-cache backend/models/pose_cache_mediapipe.pt \\
    --output backend/models/shuttle_cache.pt
"""
from __future__ import annotations

import argparse
import os
import sys

_backend = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _backend not in sys.path:
    sys.path.insert(0, _backend)

import torch

from core.dataset import FineBadmintonDataset
from core.pose_cache_build import load_pose_cache_bundle
from core.shuttle_cache import save_shuttle_cache


def main() -> None:
    p = argparse.ArgumentParser(description="Build shuttle cache (placeholder zeros v1).")
    p.add_argument("--data-root", default=os.path.join(_backend, "data"))
    p.add_argument(
        "--list-file",
        default=os.path.join(
            _backend,
            "data",
            "transformed_combined_rounds_output_en_evals_translated.json",
        ),
    )
    p.add_argument("--pose-cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sampling", choices=["span_linspace", "hit_centered"], default="span_linspace")
    args = p.parse_args()

    bundle = load_pose_cache_bundle(args.pose_cache)
    if bundle is None:
        raise SystemExit(f"Missing pose cache: {args.pose_cache}")
    pose_cache = bundle["pose_cache"]
    n, t = pose_cache.shape[0], pose_cache.shape[1]

    ds = FineBadmintonDataset(
        args.data_root,
        args.list_file,
        sampling_mode=args.sampling,
    )
    if len(ds) != n:
        raise SystemExit(
            f"Dataset length {len(ds)} != pose cache {n}. Rebuild pose cache with same sampling."
        )

    shuttle_cache = torch.zeros(n, t, 2, dtype=torch.float32)
    save_shuttle_cache(args.output, shuttle_cache)
    print(
        "Placeholder shuttle cache (zeros). Plug TrackNet into this script for real trajectories."
    )


if __name__ == "__main__":
    main()
