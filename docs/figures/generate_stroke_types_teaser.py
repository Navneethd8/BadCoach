#!/usr/bin/env python3
"""
8-panel stroke-type teaser (4×2): fixed clips + MediaPipe on native video frames.

Uses curated (sample_index, timestep) picks. MediaPipe runs on the raw broadcast
frame (upscaled; near-court crop when helpful) — not the 224×224 training resize.
Falls back to cache only if native detection fails on a panel.

Usage (from repo root):
  PYTHONPATH=backend:docs/figures python docs/figures/generate_stroke_types_teaser.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
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
    create_teaser_pose_estimator,
    pick_all_stroke_panels,
    render_curated_panel,
)

DEFAULT_JSON = BACKEND_ROOT / "data" / "transformed_combined_rounds_output_en_evals_translated.json"
DEFAULT_DATA_ROOT = BACKEND_ROOT / "data" / "FineBadminton-20K" / "videos"
DEFAULT_POSE_MODEL = BACKEND_ROOT / "models" / "pose_landmarker_lite.task"
DEFAULT_CACHE = BACKEND_ROOT / "models" / "pose_cache_mediapipe.pt"
DEFAULT_OUT = FIG_DIR / "stroke_types_teaser.png"

STROKE_ORDER = (
    "Serve", "Clear", "Smash", "Drop", "Drive", "Net_Shot", "Lob", "Defensive_Shot",
)

# Clips chosen so live MediaPipe on native frames finds the near-court striker.
CURATED_STROKE_PANELS: Dict[str, Tuple[int, int]] = {
    "Serve": (816, 0),
    "Clear": (985, 0),
    "Smash": (778, 0),
    "Drop": (1362, 0),
    "Drive": (1442, 0),
    "Net_Shot": (1118, 0),
    "Lob": (586, 0),
    "Defensive_Shot": (1110, 0),
}

PANEL_Y_START: Dict[str, float] = {}

POSE_TUNING: Dict[str, Mapping[str, float]] = {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pose-model", type=Path, default=DEFAULT_POSE_MODEL)
    parser.add_argument("--pose-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--panel-size", type=int, default=384)
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan full dataset for best native pose per stroke (slow; default uses curated picks).",
    )
    args = parser.parse_args()

    ds = FineBadmintonDataset(str(args.data_root), str(args.json), transform=None)
    pose_cache = load_pose_cache_bundle(str(args.pose_cache))["pose_cache"]
    estimator = create_teaser_pose_estimator(str(args.pose_model))

    if args.scan:
        picks = pick_all_stroke_panels(ds, estimator, STROKE_ORDER)
    else:
        picks = dict(CURATED_STROKE_PANELS)

    for stroke in STROKE_ORDER:
        i, t = picks[stroke]
        s = ds.samples[i]
        print(f"{stroke:16} idx={i} t={t} vid={os.path.basename(s['video_path'])}")

    ncols, nrows = 4, 2
    fig = plt.figure(
        figsize=(0.35 + ncols * (args.panel_size / 100.0), 0.15 + nrows * (args.panel_size / 100.0) * 1.08),
        facecolor="white",
    )
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.08, wspace=0.04, left=0.02, right=0.98, top=0.98, bottom=0.06)
    for slot, stroke in enumerate(STROKE_ORDER):
        row, col = divmod(slot, ncols)
        i, t = picks[stroke]
        tune: Optional[Mapping[str, float]] = POSE_TUNING.get(stroke)
        y_start = PANEL_Y_START.get(stroke, 0.38)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(
            render_curated_panel(
                ds, estimator, i, t,
                display_size=args.panel_size,
                y_start=y_start,
                pose_tune=tune,
                pose_cache=pose_cache,
            )
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#DEE2E6")
        ax.set_xlabel(stroke.replace("_", " "), fontsize=11, fontweight="bold", color="#343A40", labelpad=4)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
