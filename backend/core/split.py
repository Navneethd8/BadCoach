import os
from pathlib import Path

import torch
from typing import Any, List, Tuple, Dict, Optional, Set

SPLIT_SEED = 42

# Train/val split on videos *after* benchmark test holdout (50 of 70 videos).
SPLIT_REMAINDER_TRAIN_RATIO = 0.8
SPLIT_REMAINDER_VAL_RATIO = 0.2

# Fallback when benchmark list is unavailable: random 70/20/10 by video count.
SPLIT_TRAIN_RATIO = 0.7
SPLIT_VAL_RATIO = 0.2
SPLIT_TEST_RATIO = 0.1

# Back-compat alias: train fraction for scripts that still pass ``ratio=`` to split helpers.
SPLIT_RATIO = SPLIT_REMAINDER_TRAIN_RATIO

_DEFAULT_BENCHMARK_LIST = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "FineBadminton-20K"
    / "benchmark_source_videos.txt"
)


def load_benchmark_test_videos(list_path: Optional[os.PathLike[str]] = None) -> Set[str]:
    """
    Video basenames used to build FineBadmintonBenchmark (20 of 70 in FineBadminton-20K).
    Returns empty set if the list file is missing.
    """
    path = Path(list_path) if list_path is not None else _DEFAULT_BENCHMARK_LIST
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _split_counts(n: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    """Return (n_train, n_val, n_test) partition sizes that sum to *n*."""
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0

    n_train = max(1, int(train_ratio * n))
    n_val = max(1, int(val_ratio * n))
    n_test = n - n_train - n_val

    if n >= 3 and n_test <= 0:
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1
        n_test = n - n_train - n_val

    while n_test <= 0 and n_train > 1:
        n_train -= 1
        n_test = n - n_train - n_val

    return n_train, n_val, n_test


def _indices_for_videos(
    video_to_indices: Dict[str, List[int]], video_names: List[str]
) -> List[int]:
    out: List[int] = []
    for v in video_names:
        out.extend(video_to_indices[v])
    return out


def video_level_split(
    samples: List[Dict],
    seed: int = SPLIT_SEED,
    train_ratio: float = SPLIT_REMAINDER_TRAIN_RATIO,
    val_ratio: float = SPLIT_REMAINDER_VAL_RATIO,
    *,
    ratio: float | None = None,
    use_benchmark_test: bool = True,
    benchmark_list_path: Optional[os.PathLike[str]] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split dataset indices by video to prevent data leakage.
    Same seed + same samples = same split every time, across all scripts.

    Default (FineBadminton-20K):
      - **Test:** fixed holdout of 20 ``benchmark_source_videos.txt`` matches
      - **Train / val:** 80/20 on the remaining 50 videos

    Fallback (missing benchmark list or no overlap): random 70/20/10 by video count.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    if ratio is not None:
        train_ratio = ratio

    video_to_indices: Dict[str, List[int]] = {}
    for i, sample in enumerate(samples):
        v_name = os.path.basename(sample["video_path"])
        video_to_indices.setdefault(v_name, []).append(i)

    unique_videos = sorted(video_to_indices.keys())

    if len(unique_videos) == 1:
        only = unique_videos[0]
        indices = sorted(video_to_indices[only])
        n = len(indices)
        if n <= 1:
            train_indices = list(indices)
            val_indices = list(indices)
            test_indices = list(indices)
        else:
            n_train, n_val, n_test = _split_counts(n, train_ratio, val_ratio)
            train_indices = indices[:n_train]
            val_indices = indices[n_train : n_train + n_val]
            test_indices = indices[n_train + n_val :]
        print(
            f"Split: 1 video, clip-level "
            f"{train_ratio:.0%}/{val_ratio:.0%} "
            f"({len(train_indices)} train / {len(val_indices)} val / {len(test_indices)} test samples)"
        )
        return train_indices, val_indices, test_indices

    benchmark_names = load_benchmark_test_videos(benchmark_list_path) if use_benchmark_test else set()
    test_vids = sorted(v for v in unique_videos if v in benchmark_names)
    remain_videos = sorted(v for v in unique_videos if v not in benchmark_names)

    if test_vids and remain_videos:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(remain_videos), generator=g).tolist()
        n_train = max(1, int(train_ratio * len(remain_videos)))
        if len(remain_videos) >= 2:
            n_train = min(n_train, len(remain_videos) - 1)
        n_val = len(remain_videos) - n_train
        train_vids = [remain_videos[i] for i in perm[:n_train]]
        val_vids = [remain_videos[i] for i in perm[n_train:]]

        train_indices = _indices_for_videos(video_to_indices, train_vids)
        val_indices = _indices_for_videos(video_to_indices, val_vids)
        test_indices = _indices_for_videos(video_to_indices, test_vids)

        print(
            f"Split (benchmark test holdout): {len(train_vids)} train vids ({len(train_indices)} samples) / "
            f"{len(val_vids)} val vids ({len(val_indices)} samples) / "
            f"{len(test_vids)} test vids ({len(test_indices)} samples)"
        )
        return train_indices, val_indices, test_indices

    if use_benchmark_test and benchmark_names and not test_vids:
        print(
            "Warning: benchmark test list found but no matching videos in samples; "
            "falling back to random 70/20/10 split.",
            flush=True,
        )

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(unique_videos), generator=g).tolist()
    n_train, n_val, n_test = _split_counts(len(unique_videos), SPLIT_TRAIN_RATIO, SPLIT_VAL_RATIO)
    train_vids = [unique_videos[i] for i in perm[:n_train]]
    val_vids = [unique_videos[i] for i in perm[n_train : n_train + n_val]]
    test_vids = [unique_videos[i] for i in perm[n_train + n_val :]]

    train_indices = _indices_for_videos(video_to_indices, train_vids)
    val_indices = _indices_for_videos(video_to_indices, val_vids)
    test_indices = _indices_for_videos(video_to_indices, test_vids)

    print(
        f"Split (random fallback): {len(train_vids)} train vids ({len(train_indices)} samples) / "
        f"{len(val_vids)} val vids ({len(val_indices)} samples) / "
        f"{len(test_vids)} test vids ({len(test_indices)} samples)"
    )
    return train_indices, val_indices, test_indices


def vlm_jsonl_video_level_split(
    rows: List[Dict[str, Any]],
    *,
    image_key: str = "image",
    seed: int = SPLIT_SEED,
    train_ratio: float = SPLIT_REMAINDER_TRAIN_RATIO,
    val_ratio: float = SPLIT_REMAINDER_VAL_RATIO,
    ratio: float | None = None,
    use_benchmark_test: bool = False,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Same train/val/test-by-video logic as ``video_level_split``, but groups JSONL rows by
    rally/video id parsed from the image filename (FineBadminton:
    ``image/0011_001_16363.jpg`` → video key ``0011_001``).

    Benchmark holdout is off by default here because VLM JSONL rally ids do not match
    full-match ``benchmark_source_videos.txt`` names.
    """
    if ratio is not None:
        train_ratio = ratio

    video_to_indices: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        rel = row[image_key]
        if isinstance(rel, list):
            rel = rel[0] if rel else ""
        stem = Path(rel).stem
        if "_" in stem:
            v_key = stem.rsplit("_", 1)[0]
        else:
            v_key = stem
        video_to_indices.setdefault(v_key, []).append(i)

    unique_videos = sorted(video_to_indices.keys())
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(unique_videos), generator=g).tolist()

    n_train, n_val, n_test = _split_counts(len(unique_videos), SPLIT_TRAIN_RATIO, SPLIT_VAL_RATIO)
    train_vids = [unique_videos[j] for j in perm[:n_train]]
    val_vids = [unique_videos[j] for j in perm[n_train : n_train + n_val]]
    test_vids = [unique_videos[j] for j in perm[n_train + n_val :]]

    train_indices = _indices_for_videos(video_to_indices, train_vids)
    val_indices = _indices_for_videos(video_to_indices, val_vids)
    test_indices = _indices_for_videos(video_to_indices, test_vids)

    print(
        f"VLM split: {len(train_vids)} train vids ({len(train_indices)} samples) / "
        f"{len(val_vids)} val vids ({len(val_indices)} samples) / "
        f"{len(test_vids)} test vids ({len(test_indices)} samples)"
    )

    return train_indices, val_indices, test_indices
