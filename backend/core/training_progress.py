"""Shared tqdm defaults for training scripts.

The default tqdm left column (e.g. ``83%|…``) is *batch progress through the
DataLoader*, not model accuracy. We use explicit ``…/N batches`` and
``…/N samples`` wording so it is not confused with metric percentages.
"""

from __future__ import annotations

from typing import Any, Iterable

from tqdm import tqdm

# CNN-LSTM / STAEformer default; TimeSformer matches for comparable runs.
DEFAULT_TRAIN_BATCH_SIZE = 4

_TRAIN_BAR = (
    "{desc} |{bar}| {n_fmt}/{total_fmt} batches "
    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)
_POSE_BAR = (
    "{desc} |{bar}| {n_fmt}/{total_fmt} samples "
    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)


def tqdm_train_batches(loader: Iterable[Any], epoch_one_based: int, total_epochs: int, **kwargs: Any) -> tqdm:
    """Progress for one training epoch (counts minibatches)."""
    opts = {
        "desc": f"Train epoch {epoch_one_based}/{total_epochs}",
        "bar_format": _TRAIN_BAR,
        "unit": "batch",
    }
    opts.update(kwargs)
    return tqdm(loader, **opts)


def tqdm_pose_cache_build(length: int, **kwargs: Any) -> tqdm:
    """Iterate ``range(length)`` while building pose cache over the full annotated dataset."""
    opts = {
        "desc": "Pose cache (full dataset)",
        "bar_format": _POSE_BAR,
        "unit": "sample",
    }
    opts.update(kwargs)
    return tqdm(range(length), **opts)
