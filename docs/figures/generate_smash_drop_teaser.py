#!/usr/bin/env python3
"""
Smash vs Drop teaser: 2×16 frames with aligned MediaPipe overlay.

See ``teaser_pose_utils.py`` for pose selection / drawing (cache + live verify, crop).

Usage (from repo root):
  PYTHONPATH=backend:docs/figures python docs/figures/generate_smash_drop_teaser.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from matplotlib.gridspec import GridSpec

FIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIG_DIR.parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
for p in (str(BACKEND_ROOT), str(FIG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.dataset import FineBadmintonDataset  # noqa: E402
from core.pose_cache_build import load_pose_cache_bundle  # noqa: E402
from teaser_pose_utils import (  # noqa: E402
    contact_timestep,
    create_teaser_pose_estimator,
    pick_best_panel,
    render_panel,
)

DEFAULT_JSON = BACKEND_ROOT / "data" / "transformed_combined_rounds_output_en_evals_translated.json"
DEFAULT_DATA_ROOT = BACKEND_ROOT / "data" / "FineBadminton-20K" / "videos"
DEFAULT_CACHE = BACKEND_ROOT / "models" / "pose_cache_mediapipe.pt"
DEFAULT_POSE_MODEL = BACKEND_ROOT / "models" / "pose_landmarker_lite.task"
DEFAULT_OUT = FIG_DIR / "smash_drop_teaser.png"

NUM_FRAMES = 16
SMASH_COLOR = "#D9480F"
DROP_COLOR = "#1864AB"
CONTACT_GOLD = "#FAB005"
CELL_SIZE = 168


def _pair_score(
    smash: Tuple[Dict[str, Any], str, int],
    drop: Tuple[Dict[str, Any], str, int],
) -> Tuple[int, int, int]:
    _, smash_sub, smash_span = smash
    _, drop_sub, drop_span = drop
    overhead_smash = smash_sub in {
        "Jump_Smash", "Full_Smash", "Slice_Smash", "Common_Smash", "Stick_Smash",
    }
    overhead_drop = drop_sub in {
        "Slice_Drop", "Reverse_Slice_Drop", "Stop_Drop", "Blocked_Drop",
    }
    return (
        int(overhead_smash and overhead_drop),
        -max(smash_span, drop_span),
        -abs(smash_span - drop_span),
    )


def _pick_pair(ds: FineBadmintonDataset) -> Tuple[int, int]:
    by_video: Dict[str, Dict[str, List[Tuple[Dict[str, Any], str, int]]]] = {}
    for i, sample in enumerate(ds.samples):
        labels = ds._map_labels(sample)
        stroke = ds.classes["stroke_type"][labels["stroke_type"]]
        if stroke not in ("Smash", "Drop"):
            continue
        if ds.classes["technique"][labels["technique"]] != "Forehand":
            continue
        subtype = ds.classes["stroke_subtype"][labels["stroke_subtype"]]
        if subtype == "None":
            continue
        span = int(sample["end_frame"]) - int(sample["start_frame"])
        if span < 8 or span > 48:
            continue
        vid = os.path.basename(sample["video_path"])
        by_video.setdefault(vid, {"Smash": [], "Drop": []})[stroke].append((sample, subtype, span, i))

    ranked: List[Tuple[Tuple[int, int, int], int, int]] = []
    for groups in by_video.values():
        if not groups["Smash"] or not groups["Drop"]:
            continue
        for smash in groups["Smash"]:
            for drop in groups["Drop"]:
                ranked.append((_pair_score(smash, drop), smash[3], drop[3]))

    if not ranked:
        raise RuntimeError("No suitable Smash/Drop pair found.")
    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1], ranked[0][2]


def _load_clip_frames(
    ds: FineBadmintonDataset,
    pose_cache: torch.Tensor,
    estimator,
    sample_index: int,
    cell_size: int,
) -> Tuple[List[Any], int]:
    contact_col = contact_timestep(ds.samples[sample_index])
    frames = []
    for t in range(NUM_FRAMES):
        rgb = render_panel(
            ds, pose_cache, estimator, sample_index, t,
            display_size=cell_size, crop=True,
        )
        frames.append(rgb)
    return frames, contact_col


def _draw_grid_row(fig, gs, row, frames, row_label, accent, contact_col):
    ax_label = fig.add_subplot(gs[row, 0])
    ax_label.axis("off")
    ax_label.text(0.5, 0.5, row_label, ha="center", va="center", fontsize=11,
                  fontweight="bold", color=accent, rotation=90)
    for col in range(NUM_FRAMES):
        ax = fig.add_subplot(gs[row, col + 1])
        ax.imshow(frames[col])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(CONTACT_GOLD if col == contact_col else "#DEE2E6")
            spine.set_linewidth(2.8 if col == contact_col else 0.8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pose-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    ds = FineBadmintonDataset(str(args.data_root), str(args.json), transform=None)
    pose_cache = load_pose_cache_bundle(str(args.pose_cache))["pose_cache"]
    estimator = create_teaser_pose_estimator(str(args.pose_model))

    smash_i, drop_i = _pick_pair(ds)
    # Prefer clips with verified foreground pose
    smash_i, _ = pick_best_panel(ds, pose_cache, estimator, "Smash")
    drop_i, _ = pick_best_panel(ds, pose_cache, estimator, "Drop")

    smash_frames, smash_contact = _load_clip_frames(ds, pose_cache, estimator, smash_i, CELL_SIZE)
    drop_frames, drop_contact = _load_clip_frames(ds, pose_cache, estimator, drop_i, CELL_SIZE)

    fig_w = 0.55 + NUM_FRAMES * (CELL_SIZE / 100.0)
    fig = plt.figure(figsize=(fig_w, 2.05), facecolor="white")
    gs = GridSpec(2, NUM_FRAMES + 1, figure=fig, width_ratios=[0.28] + [1.0] * NUM_FRAMES,
                  hspace=0.0, wspace=0.03, left=0.04, right=0.995, top=0.98, bottom=0.02)
    _draw_grid_row(fig, gs, 0, smash_frames, "Smash", SMASH_COLOR, smash_contact)
    _draw_grid_row(fig, gs, 1, drop_frames, "Drop", DROP_COLOR, drop_contact)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print(f"Smash idx={smash_i} Drop idx={drop_i}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
