import os
from pathlib import Path

import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

SPLIT_SEED = 42

# FineBadminton-20K target: 70% train / 10% val / 20% test (by video count).
# With benchmark_source_videos.txt: test = fixed 20 benchmark matches (~20/70);
# val = 10% of all videos; train = remaining non-benchmark videos.
SPLIT_TRAIN_RATIO = 0.70
SPLIT_VAL_RATIO = 0.10
SPLIT_TEST_RATIO = 0.20

# Back-compat alias for scripts that pass ``ratio=`` (train fraction on non-test videos).
SPLIT_RATIO = SPLIT_TRAIN_RATIO

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


def _indices_for_videos(
    video_to_indices: Dict[str, List[int]], video_names: List[str]
) -> List[int]:
    out: List[int] = []
    for v in video_names:
        out.extend(video_to_indices[v])
    return out


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


def _split_remainder_by_val_count(
    remain_videos: List[str],
    *,
    seed: int,
    n_val: int,
) -> Tuple[List[str], List[str]]:
    """Hold out ``n_val`` shuffled non-test videos; the rest train."""
    n_val = max(1, min(len(remain_videos) - 1, n_val)) if len(remain_videos) >= 2 else 0
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(remain_videos), generator=g).tolist()
    if n_val <= 0:
        return [remain_videos[i] for i in perm], []
    val_vids = [remain_videos[i] for i in perm[:n_val]]
    train_vids = [remain_videos[i] for i in perm[n_val:]]
    return train_vids, val_vids


def _split_remainder_by_train_ratio(
    remain_videos: List[str],
    *,
    seed: int,
    train_ratio: float,
) -> Tuple[List[str], List[str]]:
    """Legacy: ``train_ratio`` of non-test videos train; remainder val."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(remain_videos), generator=g).tolist()
    n_train = max(1, int(train_ratio * len(remain_videos)))
    if len(remain_videos) >= 2:
        n_train = min(n_train, len(remain_videos) - 1)
    train_vids = [remain_videos[i] for i in perm[:n_train]]
    val_vids = [remain_videos[i] for i in perm[n_train:]]
    return train_vids, val_vids


def split_video_groups(
    video_to_indices: Dict[str, List[int]],
    *,
    seed: int = SPLIT_SEED,
    train_ratio: float = SPLIT_TRAIN_RATIO,
    val_ratio: float = SPLIT_VAL_RATIO,
    remainder_train_ratio: float | None = None,
    use_benchmark_test: bool = True,
    benchmark_list_path: Optional[os.PathLike[str]] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Core video-level split used by native trainers, BST collate, and VLM JSONL.

    Default (FineBadminton-20K + ``benchmark_source_videos.txt``):
      - **Test:** fixed benchmark source matches (~20/70)
      - **Val:** ``round(val_ratio * n_videos)`` from non-benchmark pool (10% → 7/70)
      - **Train:** remaining non-benchmark videos (~43/70)

    Fallback (missing benchmark list or no overlap): random 70/10/20 on all videos.

    Pass ``remainder_train_ratio`` (or ``video_level_split(ratio=…)``) to override with a
    train-fraction split on non-test videos instead of the fixed 10% val target.
    """
    unique_videos = sorted(video_to_indices.keys())
    n_total = len(unique_videos)

    if n_total == 1:
        only = unique_videos[0]
        indices = sorted(video_to_indices[only])
        n = len(indices)
        if n <= 1:
            return list(indices), list(indices), []
        n_train, n_val, n_test = _split_counts(n, train_ratio, val_ratio)
        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val :]
        return train_indices, val_indices, test_indices

    benchmark_names = load_benchmark_test_videos(benchmark_list_path) if use_benchmark_test else set()
    test_vids = sorted(v for v in unique_videos if v in benchmark_names)
    remain_videos = sorted(v for v in unique_videos if v not in benchmark_names)

    if test_vids and remain_videos:
        if remainder_train_ratio is not None:
            train_vids, val_vids = _split_remainder_by_train_ratio(
                remain_videos, seed=seed, train_ratio=remainder_train_ratio
            )
        else:
            n_val = max(1, round(val_ratio * n_total)) if n_total >= 3 else 1
            train_vids, val_vids = _split_remainder_by_val_count(
                remain_videos, seed=seed, n_val=n_val
            )
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
            "falling back to random 70/10/20 split.",
            flush=True,
        )

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    n_train, n_val, n_test = _split_counts(n_total, train_ratio, val_ratio)
    train_vids = [unique_videos[i] for i in perm[:n_train]]
    val_vids = [unique_videos[i] for i in perm[n_train : n_train + n_val]]
    test_vids = [unique_videos[i] for i in perm[n_train + n_val :]]

    train_indices = _indices_for_videos(video_to_indices, train_vids)
    val_indices = _indices_for_videos(video_to_indices, val_vids)
    test_indices = _indices_for_videos(video_to_indices, test_vids)

    print(
        f"Split (random 70/10/20 fallback): {len(train_vids)} train vids ({len(train_indices)} samples) / "
        f"{len(val_vids)} val vids ({len(val_indices)} samples) / "
        f"{len(test_vids)} test vids ({len(test_indices)} samples)"
    )
    return train_indices, val_indices, test_indices


def video_level_split(
    samples: List[Dict],
    seed: int = SPLIT_SEED,
    train_ratio: float = SPLIT_TRAIN_RATIO,
    val_ratio: float = SPLIT_VAL_RATIO,
    *,
    ratio: float | None = None,
    use_benchmark_test: bool = True,
    benchmark_list_path: Optional[os.PathLike[str]] = None,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split dataset indices by video to prevent data leakage.
    Same seed + same samples = same split every time, across all scripts.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    remainder_train_ratio = ratio

    video_to_indices: Dict[str, List[int]] = {}
    for i, sample in enumerate(samples):
        v_name = os.path.basename(sample["video_path"])
        video_to_indices.setdefault(v_name, []).append(i)

    return split_video_groups(
        video_to_indices,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        remainder_train_ratio=remainder_train_ratio,
        use_benchmark_test=use_benchmark_test,
        benchmark_list_path=benchmark_list_path,
    )


def _match_video_basename_from_row(row: Dict[str, Any], image_key: str = "image") -> str:
    """Map a VLM JSONL row to a full-match video basename (e.g. ``0001.mp4``)."""
    if row.get("video_stem"):
        stem = str(row["video_stem"])
        return stem if stem.endswith(".mp4") else f"{stem}.mp4"

    rel = row.get(image_key)
    if isinstance(rel, list):
        rel = rel[0] if rel else ""
    if not rel and row.get("images"):
        imgs = row["images"]
        rel = imgs[0] if isinstance(imgs, list) and imgs else imgs
    if not rel and row.get("image"):
        rel = row["image"]
    if not rel:
        raise KeyError("JSONL row needs video_stem, image, or images to infer match video")

    stem = Path(str(rel)).stem
    match = stem.split("_", 1)[0]
    return match if match.endswith(".mp4") else f"{match}.mp4"


def vlm_jsonl_video_level_split(
    rows: List[Dict[str, Any]],
    *,
    image_key: str = "image",
    seed: int = SPLIT_SEED,
    train_ratio: float = SPLIT_TRAIN_RATIO,
    val_ratio: float = SPLIT_VAL_RATIO,
    ratio: float | None = None,
    use_benchmark_test: bool = True,
    benchmark_list_path: Optional[os.PathLike[str]] = None,
    video_basename_fn: Callable[[Dict[str, Any]], str] | None = None,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Same policy as ``video_level_split``, grouping JSONL rows by full-match video.

    Rows from ``build_finebadminton_jsonl.py`` include ``video_stem`` (e.g. ``0001`` from
    ``0001.mp4``). Legacy contact JSONL falls back to the match prefix in the image filename.
    """
    basename_fn = video_basename_fn or (lambda row: _match_video_basename_from_row(row, image_key))

    video_to_indices: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        v_name = basename_fn(row)
        video_to_indices.setdefault(v_name, []).append(i)

    train_indices, val_indices, test_indices = split_video_groups(
        video_to_indices,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        remainder_train_ratio=ratio,
        use_benchmark_test=use_benchmark_test,
        benchmark_list_path=benchmark_list_path,
    )
    if test_indices:
        print(
            f"VLM split: {len(train_indices)} train rows / "
            f"{len(val_indices)} val rows / {len(test_indices)} test rows"
        )
    else:
        print(f"VLM split: {len(train_indices)} train rows / {len(val_indices)} val rows")
    return train_indices, val_indices, test_indices
