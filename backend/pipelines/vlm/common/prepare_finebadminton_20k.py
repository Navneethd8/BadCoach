#!/usr/bin/env python3
"""
Download Moujuruo/Finebadminton-20K from Hugging Face Hub, flatten annotations to the
same JSON shape as transformed_combined_rounds_output_en_evals_translated.json, and
optionally extract contact-frame JPEGs for VLM / IsoCourt training.

Dataset: https://huggingface.co/datasets/Moujuruo/Finebadminton-20K

Typical usage (from repo):

  pip install huggingface_hub opencv-python-headless
  python backend/pipelines/vlm/common/prepare_finebadminton_20k.py
  python backend/pipelines/vlm/common/build_finebadminton_jsonl.py

This writes merged labels to backend/data/transformed_combined_rounds_output_en_evals_translated.json
by default and stores the raw Hub snapshot under backend/data/FineBadminton-20K/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HF_DATASET_ID = "Moujuruo/Finebadminton-20K"
ANNOT_GLOB = "finebadminton-20K/*_updated.json"


def _backend_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def flatten_hit(hit: dict) -> dict:
    """Map Finebadminton-20K nested hit dict to FineBadmintonDataset / JSONL flat format."""
    if "hit_type" in hit and "Foundational Actions Level" not in hit:
        return hit

    fal = hit.get("Foundational Actions Level") or {}
    tsl = hit.get("Tactical Semantics Level") or {}
    evl = hit.get("Decision Evaluation Level") or {}

    def _list(x) -> list:
        if x is None:
            return []
        if isinstance(x, list):
            return x
        return [x]

    start_frame = int(hit["start_frame"])
    end_frame = int(hit["end_frame"])
    hit_frame = start_frame

    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "player": hit.get("player") or "",
        "hit_type": fal.get("hit type") or hit.get("hit_type") or "Other",
        "subtype": _list(fal.get("subtype")),
        "quality": evl.get("quality", 1),
        "comment": evl.get("comment") or "",
        "hit_frame": hit_frame,
        "get_point": hit.get("get_point") or [],
        "ball_area": fal.get("ball area") or "Unknown",
        "hitter": fal.get("hitter") or "",
        "player_actions": _list(tsl.get("player actions")),
        "shot_characteristics": _list(tsl.get("shot characteristics")),
        "strategies": hit.get("strategies") or [],
    }


def flatten_clip(clip: dict) -> dict:
    out = {k: v for k, v in clip.items() if k != "hitting"}
    raw_hits = clip.get("hitting") or []
    out["hitting"] = [flatten_hit(h) for h in raw_hits if isinstance(h, dict)]
    return out


def merge_annotations(snapshot_dir: Path) -> list[dict]:
    paths = sorted(snapshot_dir.glob(ANNOT_GLOB))
    if not paths:
        raise FileNotFoundError(
            f"No annotation files matching {ANNOT_GLOB!r} under {snapshot_dir}"
        )
    merged: list[dict] = []
    for p in paths:
        with p.open(encoding="utf-8") as f:
            clips = json.load(f)
        if not isinstance(clips, list):
            raise ValueError(f"Expected list in {p}, got {type(clips)}")
        merged.extend(flatten_clip(c) for c in clips if isinstance(c, dict))
    return merged


def extract_contact_frames(
    merged: list[dict],
    snapshot_dir: Path,
    image_dir: Path,
    *,
    jpeg_quality: int = 92,
) -> tuple[int, int]:
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "Frame extraction requires opencv-python or opencv-python-headless."
        ) from e

    video_dir = snapshot_dir / "videos"
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Missing videos folder: {video_dir}")

    image_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    open_captures: dict[str, cv2.VideoCapture] = {}

    def get_cap(name: str):
        if name not in open_captures:
            path = video_dir / name
            if not path.is_file():
                return None
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                cap.release()
                return None
            open_captures[name] = cap
        return open_captures[name]

    try:
        for clip in merged:
            video_name = clip.get("video") or ""
            if not video_name:
                continue
            stem = Path(video_name).stem
            cap = get_cap(video_name)
            if cap is None:
                skipped += len(clip.get("hitting") or [])
                continue
            for hit in clip.get("hitting") or []:
                hf = hit.get("hit_frame")
                if hf is None:
                    skipped += 1
                    continue
                idx = int(hf)
                out_path = image_dir / f"{stem}_{idx}.jpg"
                if out_path.is_file():
                    written += 1
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    skipped += 1
                    continue
                if not cv2.imwrite(
                    str(out_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                ):
                    skipped += 1
                    continue
                written += 1
    finally:
        for cap in open_captures.values():
            cap.release()

    return written, skipped


def main() -> None:
    backend = _backend_root()
    default_root = backend / "data" / "FineBadminton-20K"
    default_out_json = backend / "data" / "transformed_combined_rounds_output_en_evals_translated.json"
    default_image_dir = default_root / "dataset" / "image"

    ap = argparse.ArgumentParser(description="Prepare Finebadminton-20K for IsoCourt.")
    ap.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help=f"Hub snapshot directory (default: {default_root}).",
    )
    ap.add_argument(
        "--repo-id",
        type=str,
        default=HF_DATASET_ID,
        help="Hugging Face dataset repo id.",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Only merge/extract; expect --local-dir to already contain videos/ and finebadminton-20K/.",
    )
    ap.add_argument(
        "--output-json",
        type=Path,
        default=default_out_json,
        help="Merged flat annotations (FineBadmintonDataset-compatible list).",
    )
    ap.add_argument(
        "--extract-frames",
        action="store_true",
        help=f"Write contact frames to --image-dir (default {default_image_dir}).",
    )
    ap.add_argument(
        "--image-dir",
        type=Path,
        default=default_image_dir,
        help="Output directory for {{video_stem}}_{{hit_frame}}.jpg",
    )
    ap.add_argument("--jpeg-quality", type=int, default=92)
    args = ap.parse_args()

    local_dir = (args.local_dir or default_root).resolve()

    if not args.skip_download:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise SystemExit("Install huggingface_hub: pip install huggingface_hub") from e
        print(f"Downloading {args.repo_id} -> {local_dir} ...")
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
        )
    elif not local_dir.is_dir():
        raise SystemExit(f"--skip-download set but {local_dir} is missing.")

    print(f"Merging annotations under {local_dir} ...")
    merged = merge_annotations(local_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)
    print(f"Wrote {len(merged)} clips -> {args.output_json}")

    if args.extract_frames:
        w, s = extract_contact_frames(
            merged,
            local_dir,
            args.image_dir.resolve(),
            jpeg_quality=args.jpeg_quality,
        )
        print(f"Extracted frames: {w} written, {s} skipped (missing video or read failure).")


if __name__ == "__main__":
    main()
