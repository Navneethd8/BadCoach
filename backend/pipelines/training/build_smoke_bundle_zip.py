"""
Build a small on-disk bundle (and .zip) for local smoke tests: subset annotations,
copied MP4s, and MediaPipe pose cache aligned to dataset indices — same contract as
``train_full.py`` / ``default_pose_cache_path`` so you can reuse ``models/pose_cache_mediapipe.pt``
with other trainers as long as they use the same ``annotations.json`` and the same
``sequence_length`` / ``frame_interval`` (defaults: 16, 2).

Example (from repo root, with full data under backend/data):

  python backend/pipelines/training/build_smoke_bundle_zip.py \\
    --data-root backend/data \\
    --list-file backend/data/transformed_combined_rounds_output_en_evals_translated.json \\
    --max-samples 48 \\
    --output backend/pipelines/training/colab/zips/isocourt_smoke_cnn_lstm_48clips
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Tuple

_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

import torch

from core.dataset import FineBadmintonDataset
from core.pose_cache_build import media_pipe_fill_pose_cache
from core.pose_utils import PoseEstimator
from core.seed_utils import set_seed


def _resolve_source_video_path(data_root: str, video_filename: str) -> str:
    """Same search order as ``FineBadmintonDataset._resolve_video_path``."""
    candidates = [
        os.path.join(data_root, video_filename),
        os.path.join(data_root, "FineBadminton-20K", "videos", video_filename),
        os.path.join(data_root, "videos", video_filename),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def _subset_annotation_list(
    data: List[Dict[str, Any]], max_samples: int
) -> Tuple[List[Dict[str, Any]], int]:
    """Mirror ``FineBadmintonDataset._load_annotations`` ordering; cap clip count."""
    out_videos: List[Dict[str, Any]] = []
    count = 0
    for video_item in data:
        if "hitting" not in video_item:
            continue
        selected_hits: List[Dict[str, Any]] = []
        for hit in video_item["hitting"]:
            if "start_frame" not in hit or "end_frame" not in hit:
                continue
            selected_hits.append(hit)
            count += 1
            if count >= max_samples:
                break
        if selected_hits:
            new_item = dict(video_item)
            new_item["hitting"] = selected_hits
            out_videos.append(new_item)
        if count >= max_samples:
            break
    return out_videos, count


def _unique_video_basenames(subset: List[Dict[str, Any]]) -> List[str]:
    seen = []
    for v in subset:
        fn = v.get("video")
        if not fn or fn in seen:
            continue
        seen.append(fn)
    return seen


def _write_readme(
    path: str,
    *,
    num_samples: int,
    sequence_length: int,
    frame_interval: int,
    pose_rel: str,
) -> None:
    text = f"""# IsoCourt smoke bundle (CNN+LSTM–compatible)

This archive was built for **option-2** style checks: model init, dataloader, pose cache load,
and a tiny train loop without paying for cloud GPU first.

## Layout

- `annotations.json` — subset of the source list; **dataset index i** matches **pose_cache[i]**.
- `videos/` — MP4s referenced by `annotations.json` (basename only in JSON).
- `{pose_rel}` — `torch.save({{"pose_cache": Tensor}})` with shape `(N, T, 33, 3)`, `N={num_samples}`, `T={sequence_length}`.

## Reuse with other trainers (VideoMAE, TimeSformer, ST-TR, …)

Use **the same** three settings together:

1. `data_root` = directory that contains this `annotations.json` and `videos/` (usually the extracted folder).
2. `list_file` = path to **this** `annotations.json` (not the full-dataset JSON).
3. `pose_cache_path` = path to **this** `{pose_rel}`.

Also keep **`sequence_length={sequence_length}`** and **`frame_interval={frame_interval}`** in the dataset the same as when this cache was built; otherwise row alignment breaks.

## CNN+LSTM quick run (after extract)

From `backend/` (adjust paths to your extract location):

```bash
python pipelines/training/train_full.py \\
  --data-root /path/to/extracted \\
  --list-file /path/to/extracted/annotations.json \\
  --pose-cache-path /path/to/extracted/{pose_rel} \\
  --epochs 1 --batch-size 2
```

If you omit `--pose-cache-path`, training looks for the default global cache under `models/`; for smoke work, always pass the bundle cache explicitly.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def build_bundle(
    *,
    data_root: str,
    list_file: str,
    max_samples: int,
    output_base: str,
    sequence_length: int,
    frame_interval: int,
    seed: int,
    build_pose: bool,
) -> str:
    data_root = os.path.abspath(data_root)
    list_file = os.path.abspath(list_file)
    output_base = os.path.abspath(output_base)
    output_zip = output_base + ".zip"

    with open(list_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise SystemExit("Expected top-level JSON array of video objects (same as FineBadmintonDataset).")

    subset, n_clips = _subset_annotation_list(raw, max_samples)
    if n_clips == 0:
        raise SystemExit("No clips in subset — check list_file / max_samples.")

    staging_parent = tempfile.mkdtemp(prefix="isocourt_smoke_")
    staging = os.path.join(staging_parent, "smoke_bundle")
    os.makedirs(os.path.join(staging, "videos"), exist_ok=True)
    os.makedirs(os.path.join(staging, "models"), exist_ok=True)

    ann_path = os.path.join(staging, "annotations.json")
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(subset, f)

    missing = []
    for base in _unique_video_basenames(subset):
        src = _resolve_source_video_path(data_root, base)
        if not os.path.isfile(src):
            missing.append((base, src))
            continue
        dst = os.path.join(staging, "videos", os.path.basename(base))
        shutil.copy2(src, dst)

    if missing:
        shutil.rmtree(staging_parent, ignore_errors=True)
        msg = "Missing video files:\n" + "\n".join(f"  {b} -> {p}" for b, p in missing[:20])
        if len(missing) > 20:
            msg += f"\n  ... and {len(missing) - 20} more"
        raise SystemExit(msg)

    pose_name = "pose_cache_mediapipe.pt"
    pose_path = os.path.join(staging, "models", pose_name)

    ds = FineBadmintonDataset(
        staging,
        ann_path,
        transform=None,
        sequence_length=sequence_length,
        frame_interval=frame_interval,
    )
    if len(ds) != n_clips:
        shutil.rmtree(staging_parent, ignore_errors=True)
        raise SystemExit(f"Internal error: dataset len {len(ds)} != expected clips {n_clips}")

    if build_pose:
        set_seed(seed)
        estimator = PoseEstimator()
        pose_cache = media_pipe_fill_pose_cache(ds, estimator)
        torch.save({"pose_cache": pose_cache}, pose_path)
    else:
        t = sequence_length
        z = torch.zeros((len(ds), t, 33, 3), dtype=torch.float32)
        torch.save({"pose_cache": z}, pose_path)

    meta = {
        "num_samples": len(ds),
        "sequence_length": sequence_length,
        "frame_interval": frame_interval,
        "source_list_file": list_file,
        "source_data_root": data_root,
        "pose_cache_file": f"models/{pose_name}",
        "build_pose": build_pose,
    }
    with open(os.path.join(staging, "bundle_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    _write_readme(
        os.path.join(staging, "README_SMOKE.md"),
        num_samples=len(ds),
        sequence_length=sequence_length,
        frame_interval=frame_interval,
        pose_rel=f"models/{pose_name}",
    )

    if os.path.isfile(output_zip):
        os.remove(output_zip)
    archive_path = shutil.make_archive(output_base, "zip", root_dir=staging_parent, base_dir="smoke_bundle")
    shutil.rmtree(staging_parent, ignore_errors=True)

    print(f"Wrote {archive_path} ({len(ds)} clips, pose={'MediaPipe' if build_pose else 'zeros'}).")
    return archive_path


def main() -> None:
    p = argparse.ArgumentParser(description="Zip subset + pose cache for CNN+LSTM / cross-model smoke tests.")
    p.add_argument("--data-root", required=True, help="FineBadminton data root (same as training).")
    p.add_argument("--list-file", required=True, help="Full annotations JSON.")
    p.add_argument("--max-samples", type=int, default=48, help="Max number of stroke clips (dataset rows).")
    p.add_argument(
        "--output",
        default=os.path.join(_backend_root, "pipelines", "training", "colab", "zips", "isocourt_smoke_cnn_lstm"),
        help="Output path **without** .zip extension (shutil.make_archive adds it).",
    )
    p.add_argument("--sequence-length", type=int, default=16)
    p.add_argument("--frame-interval", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skip-pose",
        action="store_true",
        help="Write zero pose cache (fast); use for layout-only checks without MediaPipe.",
    )
    args = p.parse_args()

    build_bundle(
        data_root=args.data_root,
        list_file=args.list_file,
        max_samples=args.max_samples,
        output_base=args.output,
        sequence_length=args.sequence_length,
        frame_interval=args.frame_interval,
        seed=args.seed,
        build_pose=not args.skip_pose,
    )


if __name__ == "__main__":
    main()
