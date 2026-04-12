#!/usr/bin/env python3
"""
Download Moujuruo/Finebadminton-20K from Hugging Face Hub, flatten annotations to the
same JSON shape as transformed_combined_rounds_output_en_evals_translated.json, and
optionally extract JPEGs for VLM / IsoCourt training:
  ``--extract-frames`` = one JPEG per stroke at ``hit_frame``;
  ``--extract-all-frames`` = every decoded frame for each video (very large).

Dataset: https://huggingface.co/datasets/Moujuruo/Finebadminton-20K

Typical usage (from repo):

  pip install huggingface_hub opencv-python-headless
  python backend/pipelines/vlm/common/prepare_finebadminton_20k.py
  python backend/pipelines/vlm/common/prepare_finebadminton_20k.py --extract-frames
  python backend/pipelines/vlm/common/build_finebadminton_jsonl.py

If the download dies with OSError errno 6 (Device not configured) on an external
volume (e.g. /Volumes/...), the disk may have slept or glitched under parallel
writes. Retry with ``--max-workers 1`` or download to an internal path with
``--local-dir`` then move the folder.

This writes merged labels to backend/data/transformed_combined_rounds_output_en_evals_translated.json
by default and stores the raw Hub snapshot under backend/data/FineBadminton-20K/.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


HF_DATASET_ID = "Moujuruo/Finebadminton-20K"
ANNOT_GLOB = "finebadminton-20K/*_updated.json"
# Hub ships 0001_updated.json … only match those (avoids AppleDouble, cache junk, etc.).
ANNOT_NAME_RE = re.compile(r"^[0-9]{4}_updated\.json$")


def _backend_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def flatten_hit(hit: dict) -> dict | None:
    """Map Finebadminton-20K nested hit dict to FineBadmintonDataset / JSONL flat format."""
    if "hit_type" in hit and "Foundational Actions Level" not in hit:
        if hit.get("start_frame") is None or hit.get("end_frame") is None:
            return None
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

    if hit.get("start_frame") is None or hit.get("end_frame") is None:
        return None
    try:
        start_frame = int(hit["start_frame"])
        end_frame = int(hit["end_frame"])
    except (TypeError, ValueError):
        return None
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


def flatten_clip(clip: dict) -> tuple[dict, int]:
    """Returns (clip_with_flat_hitting, num_skipped_bad_hits)."""
    out = {k: v for k, v in clip.items() if k != "hitting"}
    raw_hits = clip.get("hitting") or []
    skipped = 0
    flat: list[dict] = []
    for h in raw_hits:
        if not isinstance(h, dict):
            skipped += 1
            continue
        fh = flatten_hit(h)
        if fh is None:
            skipped += 1
            continue
        flat.append(fh)
    out["hitting"] = flat
    return out, skipped


def _load_annotation_json(path: Path):
    """Load JSON array; tolerate UTF-8 BOM and legacy single-byte encodings (e.g. ° as 0xB0)."""
    raw = path.read_bytes()
    last_err: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "gb18030", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            return json.loads(text)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path} (encoding {enc!r}): {e}") from e
    raise RuntimeError(
        f"Could not decode {path} as UTF-8/UTF-8-SIG: {last_err}"
    ) from last_err


def merge_annotations(snapshot_dir: Path) -> list[dict]:
    paths = sorted(
        p
        for p in snapshot_dir.glob(ANNOT_GLOB)
        if p.is_file()
        and not p.name.startswith("._")
        and ".cache" not in p.parts
        and ANNOT_NAME_RE.match(p.name)
    )
    if not paths:
        raise FileNotFoundError(
            f"No annotation files matching {ANNOT_GLOB!r} (Hub names like 0001_updated.json) "
            f"under {snapshot_dir}"
        )
    merged: list[dict] = []
    skipped_hits = 0
    for p in paths:
        clips = _load_annotation_json(p)
        if not isinstance(clips, list):
            raise ValueError(f"Expected list in {p}, got {type(clips)}")
        for c in clips:
            if not isinstance(c, dict):
                continue
            fc, nskip = flatten_clip(c)
            skipped_hits += nskip
            merged.append(fc)
    if skipped_hits:
        print(
            f"Warning: skipped {skipped_hits} hitting entries missing start_frame/end_frame or malformed.",
            file=sys.stderr,
        )
    return merged


def extract_training_frames(
    merged: list[dict],
    snapshot_dir: Path,
    image_dir: Path,
    *,
    sequence_length: int = 16,
    jpeg_quality: int = 92,
) -> tuple[int, int]:
    """Extract exactly the frames that FineBadmintonDataset samples during training.

    For each hit, np.linspace(start_frame, end_frame-1, sequence_length) gives
    the frame indices. Only these are decoded and written as JPEGs, keeping disk
    usage manageable (~326 K frames for 20K hits with T=16).
    """
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "Frame extraction requires opencv-python or opencv-python-headless."
        ) from e
    import numpy as np

    video_dir = snapshot_dir / "videos"
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Missing videos folder: {video_dir}")
    image_dir.mkdir(parents=True, exist_ok=True)

    needed: dict[str, set[int]] = {}
    for clip in merged:
        video_name = clip.get("video") or ""
        if not video_name:
            continue
        stem = Path(video_name).stem
        for hit in clip.get("hitting") or []:
            sf = hit.get("start_frame")
            ef = hit.get("end_frame")
            if sf is None or ef is None:
                continue
            sf, ef = int(sf), int(ef)
            if ef - sf <= 0:
                continue
            indices = np.linspace(sf, ef - 1, sequence_length).astype(int)
            entry = needed.setdefault(video_name, set())
            for idx in indices:
                entry.add(int(idx))

    total_needed = sum(len(v) for v in needed.values())
    print(
        f"Need {total_needed:,} unique frame-JPEGs across {len(needed)} videos -> {image_dir}",
        flush=True,
    )

    written = 0
    skipped = 0
    new_files = 0

    for video_name, frame_set in sorted(needed.items()):
        stem = Path(video_name).stem
        cap = cv2.VideoCapture(str(video_dir / video_name))
        if not cap.isOpened():
            skipped += len(frame_set)
            continue
        for idx in sorted(frame_set):
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
                str(out_path), frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            ):
                skipped += 1
                continue
            written += 1
            new_files += 1
            if new_files % 1000 == 0:
                print(
                    f"  … {written:,} on disk ({new_files:,} new), {skipped} skips",
                    flush=True,
                )
        cap.release()

    print(f"Training frames: {written:,} written, {skipped} skipped.", flush=True)
    return written, skipped


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
    total_hits = sum(len(c.get("hitting") or []) for c in merged)
    print(
        f"Extracting up to {total_hits} JPEGs -> {image_dir} "
        f"(seek+decode per frame; external disks can be slow).",
        flush=True,
    )
    written = 0
    skipped = 0
    new_files = 0
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
                new_files += 1
                if new_files % 250 == 0:
                    print(
                        f"  … {written} JPEGs on disk ({new_files} new this run), "
                        f"{skipped} skipped so far",
                        flush=True,
                    )
    finally:
        for cap in open_captures.values():
            cap.release()

    return written, skipped


def extract_all_video_frames(
    merged: list[dict],
    snapshot_dir: Path,
    image_dir: Path,
    *,
    jpeg_quality: int = 92,
) -> tuple[int, int]:
    """
    Decode every frame of each unique video referenced in ``merged`` and write
    ``{video_stem}_{frame_index}.jpg`` under ``image_dir`` (same naming as training).

    Heavy: full 20K can be millions of files. Prefer SSD and ``--max-workers`` tuning
    at the Hub download step; extraction is sequential per video.
    """
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
    unique_videos = sorted({(c.get("video") or "") for c in merged if c.get("video")})
    print(
        f"Extracting ALL frames for {len(unique_videos)} videos -> {image_dir} …",
        flush=True,
    )
    written = 0
    skipped = 0
    for video_name in unique_videos:
        stem = Path(video_name).stem
        cap_path = video_dir / video_name
        cap = cv2.VideoCapture(str(cap_path))
        if not cap.isOpened():
            skipped += 1
            continue
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for idx in range(max(n, 0)):
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
            if written % 2000 == 0:
                print(f"  … {written} JPEGs written, {skipped} skips", flush=True)
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
        "--reuse-merged-json",
        action="store_true",
        help="Skip merge and skip rewriting --output-json; load it instead (faster re-runs for --extract-frames).",
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
        help=f"Write contact (hit_frame) JPEGs to --image-dir (default {default_image_dir}).",
    )
    ap.add_argument(
        "--extract-training-frames",
        action="store_true",
        help="Extract only the frames that FineBadmintonDataset samples during training "
             "(np.linspace across each clip, ~326K for T=16). Best balance of coverage and size.",
    )
    ap.add_argument(
        "--extract-all-frames",
        action="store_true",
        help="Write every decoded frame for each video in merged JSON (large output).",
    )
    ap.add_argument(
        "--image-dir",
        type=Path,
        default=default_image_dir,
        help="Output directory for {{video_stem}}_{{hit_frame}}.jpg",
    )
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument(
        "--max-workers",
        type=int,
        default=4,
        metavar="N",
        help="Parallel Hub downloads (default: 4; Hub default is 8). Use 1 on external/USB drives that error with errno 6.",
    )
    args = ap.parse_args()

    local_dir = (args.local_dir or default_root).resolve()

    if not args.skip_download:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise SystemExit("Install huggingface_hub: pip install huggingface_hub") from e
        print(f"Downloading {args.repo_id} -> {local_dir} ...")
        print(f"(parallel workers: {args.max_workers}; use e.g. 8 on a fast local SSD)")
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            max_workers=max(1, args.max_workers),
        )
    elif not local_dir.is_dir():
        raise SystemExit(f"--skip-download set but {local_dir} is missing.")

    if args.reuse_merged_json:
        if not args.output_json.is_file():
            raise SystemExit(
                f"--reuse-merged-json requires existing file: {args.output_json}"
            )
        print(f"Loading merged clips from {args.output_json} (--reuse-merged-json) …", flush=True)
        with args.output_json.open(encoding="utf-8") as f:
            merged = json.load(f)
        if not isinstance(merged, list):
            raise SystemExit(f"Expected JSON array in {args.output_json}, got {type(merged)}")
        print(f"Loaded {len(merged)} clips.", flush=True)
    else:
        print(f"Merging annotations under {local_dir} …", flush=True)
        merged = merge_annotations(local_dir)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Writing merged JSON ({len(merged)} clips) -> {args.output_json} … "
            "can take several minutes for a large file.",
            flush=True,
        )
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False)
        print(f"Wrote {len(merged)} clips -> {args.output_json}", flush=True)

    extract_flags = sum([args.extract_frames, args.extract_training_frames, args.extract_all_frames])
    if extract_flags > 1:
        raise SystemExit("Use only one of --extract-frames / --extract-training-frames / --extract-all-frames.")

    if args.extract_frames:
        print("Starting contact-frame extraction …", flush=True)
        w, s = extract_contact_frames(
            merged, local_dir, args.image_dir.resolve(),
            jpeg_quality=args.jpeg_quality,
        )
        print(f"Extracted contact frames: {w} written, {s} skipped.")

    if args.extract_training_frames:
        print("Starting training-frame extraction (T=16 linspace per hit) …", flush=True)
        w, s = extract_training_frames(
            merged, local_dir, args.image_dir.resolve(),
            sequence_length=16, jpeg_quality=args.jpeg_quality,
        )
        print(f"Extracted training frames: {w} written, {s} skipped.")

    if args.extract_all_frames:
        print("Starting full-video frame extraction (very large) …", flush=True)
        w, s = extract_all_video_frames(
            merged, local_dir, args.image_dir.resolve(),
            jpeg_quality=args.jpeg_quality,
        )
        print(f"Extracted all frames: {w} JPEGs on disk, {s} skips.")


if __name__ == "__main__":
    main()
