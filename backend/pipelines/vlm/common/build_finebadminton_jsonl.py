#!/usr/bin/env python3
"""
Build VLM JSONL from FineBadminton labels + dataset/image/*.jpg.

Annotations default to the merged file produced from Hugging Face
Moujuruo/Finebadminton-20K (run common/prepare_finebadminton_20k.py first).

Targets match FineBadmintonDataset / non-VLM multitask heads: same normalization
(raw hit_type → stroke_type, first subtype / action / characteristic / strategy,
ball_area → position, quality → ordinal band). See core/dataset.py _map_labels.

Default labels: backend/data/transformed_combined_rounds_output_en_evals_translated.json
Default image/labels root: backend/data/FineBadminton-20K/dataset
Output: .../dataset/finebadminton_vlm_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _backend_root() -> Path:
    # .../backend/pipelines/vlm/common/this_file.py → backend
    return Path(__file__).resolve().parent.parent.parent.parent


_br = _backend_root()
if str(_br) not in sys.path:
    sys.path.insert(0, str(_br))
from core.dataset import FineBadmintonDataset  # noqa: E402


def _stem(video: str) -> str:
    return Path(video).stem


def _coerce_quality(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _hit_to_label_sample(hit: dict) -> dict:
    """Same fields FineBadmintonDataset._map_labels expects (raw annotation strings)."""
    return {
        "hit_type": hit.get("hit_type", "Other"),
        "subtype": hit.get("subtype") or [],
        "player_actions": hit.get("player_actions") or [],
        "shot_characteristics": hit.get("shot_characteristics") or [],
        "ball_area": hit.get("ball_area", "Unknown"),
        "strategies": hit.get("strategies") or [],
        "quality": _coerce_quality(hit.get("quality", 1)),
    }


def multitask_vlm_instruction(ds: FineBadmintonDataset) -> str:
    """
    Prompt that asks for IsoCourt's discrete labels (aligned with CNN / VideoMAE / Timesformer heads).
    Vocabularies come from ``ds.classes`` (same source as non-VLM training).
    """

    def line(name: str, task: str) -> str:
        opts = ", ".join(ds.classes[task])
        return f"{name}: one of [{opts}]"

    blocks = [
        "You are the IsoCourt badminton analyst. The image is a single frame at shuttle contact.",
        "Reply in one line using exactly these fields in this order. Use each label exactly as written (including underscores).",
        line("Stroke", "stroke_type"),
        line("Subtype", "stroke_subtype"),
        line("Technique", "technique"),
        line("Placement", "placement"),
        line("Position", "position"),
        line("Intent", "intent"),
        line("Quality", "quality"),
        "After those fields you may add: Player: <name>; Hitter side: top or bottom; Comment: <short note> — all optional.",
    ]
    return " ".join(blocks)


def _format_multitask_response(ds: FineBadmintonDataset, hit: dict) -> str:
    """Ground-truth line: canonical multitask labels + optional raw identity/comment."""
    sample = _hit_to_label_sample(hit)
    idx = ds._map_labels(sample)
    parts: list[str] = [
        f"Stroke: {ds.classes['stroke_type'][idx['stroke_type']]}",
        f"Subtype: {ds.classes['stroke_subtype'][idx['stroke_subtype']]}",
        f"Technique: {ds.classes['technique'][idx['technique']]}",
        f"Placement: {ds.classes['placement'][idx['placement']]}",
        f"Position: {ds.classes['position'][idx['position']]}",
        f"Intent: {ds.classes['intent'][idx['intent']]}",
        f"Quality: {ds.classes['quality'][idx['quality']]}",
    ]
    player = (hit.get("player") or "").strip()
    hitter = (hit.get("hitter") or "").strip()
    if player:
        parts.append(f"Player: {player}")
    if hitter:
        parts.append(f"Hitter side: {hitter}")
    cmt = (hit.get("comment") or "").strip()
    if cmt:
        parts.append(f"Comment: {cmt}")
    return " ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="FineBadminton combined rounds JSON (English).",
    )
    p.add_argument(
        "--dataset_dir",
        type=Path,
        default=None,
        help="Folder containing image/ and where output JSONL is written.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .jsonl path (default: dataset_dir/finebadminton_vlm_train.jsonl).",
    )
    p.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Custom instruction. Default: auto prompt aligned with FineBadmintonDataset / non-VLM heads.",
    )
    p.add_argument(
        "--skip_missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rows when image file is missing.",
    )
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    backend = here.parent.parent.parent
    default_dataset = backend / "data" / "FineBadminton-20K" / "dataset"
    dataset_dir = (args.dataset_dir or default_dataset).resolve()
    default_labels = (
        backend / "data" / "transformed_combined_rounds_output_en_evals_translated.json"
    )
    labels_path = (args.labels or default_labels).resolve()
    out_path = (args.output or dataset_dir / "finebadminton_vlm_train.jsonl").resolve()
    image_root = dataset_dir / "image"

    if not labels_path.is_file():
        raise SystemExit(f"Labels not found: {labels_path}")
    if not image_root.is_dir():
        raise SystemExit(f"Image folder not found: {image_root}")

    bogus_list = str(dataset_dir / ".__vlm_jsonl_no_list__.json")
    label_ds = FineBadmintonDataset(str(dataset_dir), bogus_list, transform=None)
    instruction = (
        args.instruction if args.instruction is not None else multitask_vlm_instruction(label_ds)
    )

    with labels_path.open(encoding="utf-8") as f:
        rounds = json.load(f)

    written = 0
    skipped_no_frame = 0
    skipped_missing_file = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for clip in rounds:
            video = clip.get("video") or ""
            stem = _stem(video)
            for hit in clip.get("hitting") or []:
                hf = hit.get("hit_frame")
                if hf is None:
                    skipped_no_frame += 1
                    continue
                rel = f"image/{stem}_{int(hf)}.jpg"
                img_path = dataset_dir / rel
                if not img_path.is_file():
                    if args.skip_missing:
                        skipped_missing_file += 1
                        continue
                    raise FileNotFoundError(f"Missing image: {img_path}")
                row = {
                    "image": rel,
                    "instruction": instruction,
                    "response": _format_multitask_response(label_ds, hit),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} lines -> {out_path}")
    print(f"Skipped (no hit_frame): {skipped_no_frame}")
    print(f"Skipped (missing image): {skipped_missing_file}")


if __name__ == "__main__":
    main()
