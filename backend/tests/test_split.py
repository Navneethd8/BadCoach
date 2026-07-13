"""Tests for unified FineBadminton video-level split."""

from __future__ import annotations

from pathlib import Path
import tempfile

from core.split import (
    SPLIT_RATIO,
    SPLIT_TEST_RATIO,
    SPLIT_TRAIN_RATIO,
    SPLIT_VAL_RATIO,
    split_video_groups,
    video_level_split,
    vlm_jsonl_video_level_split,
)


def _video_ids_for_indices(video_to_indices, indices):
    out = set()
    for idx in indices:
        for v, ids in video_to_indices.items():
            if idx in ids:
                out.add(v)
    return out


def test_single_split_policy_constants():
    assert SPLIT_TRAIN_RATIO == 0.70
    assert SPLIT_VAL_RATIO == 0.10
    assert SPLIT_TEST_RATIO == 0.20
    assert SPLIT_RATIO == SPLIT_TRAIN_RATIO


def test_benchmark_holdout_puts_benchmark_videos_in_test():
    benchmark = {"0001.mp4", "0002.mp4"}
    with tempfile.TemporaryDirectory() as tmp:
        bench_file = Path(tmp) / "benchmark.txt"
        bench_file.write_text("0001.mp4\n0002.mp4\n", encoding="utf-8")

        video_to_indices = {
            "0001.mp4": [0, 1],
            "0002.mp4": [2, 3],
            "0003.mp4": [4, 5],
            "0004.mp4": [6, 7],
            "0005.mp4": [8, 9],
        }
        train, val, test = split_video_groups(
            video_to_indices,
            seed=42,
            benchmark_list_path=bench_file,
        )

        assert _video_ids_for_indices(video_to_indices, test) == benchmark
        assert _video_ids_for_indices(video_to_indices, train) & benchmark == set()
        assert _video_ids_for_indices(video_to_indices, val) & benchmark == set()
        assert _video_ids_for_indices(video_to_indices, train) | _video_ids_for_indices(
            video_to_indices, val
        ) == {"0003.mp4", "0004.mp4", "0005.mp4"}
        assert len(_video_ids_for_indices(video_to_indices, val)) == 1


def test_finebadminton_70_10_20_video_counts():
    """70 videos, 20 benchmark → ~43 train / 7 val / 20 test."""
    benchmark = {f"{i:04d}.mp4" for i in range(1, 21)}
    with tempfile.TemporaryDirectory() as tmp:
        bench_file = Path(tmp) / "benchmark.txt"
        bench_file.write_text("\n".join(sorted(benchmark)) + "\n", encoding="utf-8")

        video_to_indices = {f"{i:04d}.mp4": [i] for i in range(1, 71)}
        train, val, test = split_video_groups(
            video_to_indices,
            seed=42,
            benchmark_list_path=bench_file,
        )
        assert len(_video_ids_for_indices(video_to_indices, test)) == 20
        assert len(_video_ids_for_indices(video_to_indices, val)) == 7
        assert len(_video_ids_for_indices(video_to_indices, train)) == 43


def test_video_level_split_is_deterministic():
    samples = [{"video_path": f"/data/{v}"} for v in ("0001.mp4", "0003.mp4", "0004.mp4") for _ in range(2)]
    a = video_level_split(samples, seed=42, use_benchmark_test=False)
    b = video_level_split(samples, seed=42, use_benchmark_test=False)
    assert a == b


def test_vlm_jsonl_uses_video_stem_for_benchmark_test():
    rows = [
        {"video_stem": "0001", "images": ["img/a.jpg"]},
        {"video_stem": "0003", "images": ["img/b.jpg"]},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        bench_file = Path(tmp) / "benchmark.txt"
        bench_file.write_text("0001.mp4\n", encoding="utf-8")
        train, val, test = vlm_jsonl_video_level_split(
            rows,
            seed=42,
            benchmark_list_path=bench_file,
        )
        assert test == [0]
        assert 1 in train or 1 in val
