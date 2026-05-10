#!/usr/bin/env python3
"""
Print dataset / train-val split / DataLoader batch counts for FineBadminton training.

Uses the same definitions as train_full.py (FineBadmintonDataset + video_level_split).
Run after prepare_finebadminton_20k.py, or pass any merged annotations JSON.

Example:

  python3 backend/pipelines/training/print_finebadminton_dataloader_stats.py \\
    --list-file backend/data/transformed_combined_rounds_output_en_evals_translated.json \\
    --data-root backend/data
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

from core.dataset import FineBadmintonDataset  # noqa: E402
from core.split import video_level_split  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        default=os.path.join(backend_root, "data"),
        help="Dataset root (videos/ or FineBadminton-20K/videos/).",
    )
    p.add_argument(
        "--list-file",
        default=os.path.join(
            backend_root, "data", "transformed_combined_rounds_output_en_evals_translated.json"
        ),
        help="Merged annotations JSON (list of clips with hitting[]).",
    )
    p.add_argument(
        "--batch-sizes",
        default="8,4,16",
        help="Comma-separated batch sizes to report.",
    )
    args = p.parse_args()

    ds = FineBadmintonDataset(args.data_root, args.list_file, transform=None)
    n = len(ds)
    if n == 0:
        raise SystemExit(f"Dataset empty: data_root={args.data_root!r} list_file={args.list_file!r}")

    train_idx, val_idx = video_level_split(ds.samples)
    n_train, n_val = len(train_idx), len(val_idx)

    print(f"annotations list_file: {args.list_file}")
    print(f"dataset samples (strokes with start/end): {n}")
    print(f"train samples: {n_train}  val samples: {n_val}")

    bss = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    for bs in bss:
        tb = (n_train + bs - 1) // bs
        vb = (n_val + bs - 1) // bs
        print(f"batch_size={bs}: train_batches/epoch={tb}  val_batches/epoch={vb}")

    # Match train_full: WeightedRandomSampler num_samples == len(train)
    st_labels = [ds._map_labels(s)["stroke_type"] for s in ds.samples]
    train_st = torch.tensor([st_labels[i] for i in train_idx], dtype=torch.long)
    counts = torch.bincount(train_st)
    w = 1.0 / (counts.float() + 1e-6)
    sample_w = w[train_st]
    sampler = WeightedRandomSampler(weights=sample_w, num_samples=len(train_st), replacement=True)
    for bs in bss:
        loader = DataLoader(Subset(ds, train_idx), batch_size=bs, sampler=sampler, num_workers=0)
        print(f"len(train DataLoader) with WeightedRandomSampler, batch_size={bs}: {len(loader)}")


if __name__ == "__main__":
    main()
