#!/usr/bin/env python3
"""Sample or scan training JPEGs for near-black / unreadable frames (e.g. after h264 glitches)."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np


def gray_stats(bgr: np.ndarray) -> tuple[float, float]:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(g.mean()), float(g.std())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "image_dir",
        type=Path,
        nargs="?",
        default=Path("backend/data/FineBadminton-20K/dataset/image"),
        help="Folder with {stem}_{frame}.jpg (default: FineBadminton training cache).",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="If >0, only check N random files (faster). Default 0 = full scan.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    d = args.image_dir.resolve()
    if not d.is_dir():
        print(f"Not a directory: {d}", file=sys.stderr)
        raise SystemExit(1)

    paths = sorted(p for p in d.glob("*.jpg") if p.is_file())
    n = len(paths)
    if n == 0:
        print(f"No .jpg in {d}")
        return
    if args.sample > 0 and args.sample < n:
        rng = random.Random(args.seed)
        paths = rng.sample(paths, args.sample)

    read_fail = 0
    # Near-solid black: very low mean, nearly zero variance
    blackish = 0
    very_dark = 0
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            read_fail += 1
            if read_fail <= 5:
                print(f"read_fail: {p}", file=sys.stderr)
            continue
        m, s = gray_stats(bgr)
        if m < 1.5 and s < 0.5:
            blackish += 1
        if m < 3.0:
            very_dark += 1

    tag = f"sample={len(paths)}" if args.sample else f"all {n}"
    print(f"Scanned {tag} under {d}")
    print(f"  imread fail:        {read_fail}")
    print(f"  near-black m<1.5,s<0.5: {blackish}")
    print(f"  very_dark m<3 (gray):   {very_dark}")


if __name__ == "__main__":
    main()
