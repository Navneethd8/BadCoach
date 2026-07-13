#!/usr/bin/env python3
"""
Qualitative val predictions for paper figures: contact-window strips + GT/pred labels.

``--faithful`` (evaluation): 224×224 span_linspace training frames only — same pixels
the vision encoder sees; no skeleton overlay, no crop, no live MediaPipe.

Usage (repo root):
  ./scripts/cluster/run_qualitative_render_only.sh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

FIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIG_DIR.parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
for p in (str(BACKEND_ROOT), str(FIG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.dataset import FineBadmintonDataset  # noqa: E402
from core.jvc_no_xattn import build_jvc_no_xattn, load_jvc_no_xattn_partial  # noqa: E402
from core.jvc import build_jvc, load_jvc_partial  # noqa: E402
from core.jvc_no_xattn import default_jvc_no_xattn_pose_cache_path  # noqa: E402
from core.pose_cache_build import load_pose_cache_bundle  # noqa: E402
from core.split import video_level_split  # noqa: E402
from teaser_pose_utils import (  # noqa: E402
    MEDIAPIPE_BONE_PAIRS,
    contact_timestep,
    create_teaser_pose_estimator,
    crop_frame_and_pose,
    frame_index_for_timestep,
    infer_pose_on_224_frame,
    infer_striker_pose_rgb_with_fallback,
    load_native_rgb,
    score_pose_for_teaser,
)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NUM_FRAMES = 16
CONTACT_GOLD = "#FAB005"
OK_GREEN = "#2B8A3E"
ERR_RED = "#C92A2A"
DEFAULT_JSON = BACKEND_ROOT / "data" / "transformed_combined_rounds_output_en_evals_translated.json"
DEFAULT_DATA_ROOT = BACKEND_ROOT / "data"
DEFAULT_POSE_CACHE = Path(default_jvc_no_xattn_pose_cache_path(str(BACKEND_ROOT)))
DEFAULT_OUT = FIG_DIR / "qualitative_jvc_val.png"

CONFUSABLE = {
    frozenset({"Smash", "Drop"}),
    frozenset({"Clear", "Lob"}),
    frozenset({"Drive", "Net_Shot"}),
    frozenset({"Drop", "Net_Shot"}),
}

_CLIP_BAR = (
    "{desc} |{bar}| {n_fmt}/{total_fmt} clips "
    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
)

# LaTeX row-strip layout (only used when generating *_rows.tex).
DEFAULT_ROW_IMG_WIDTH_LATEX = 0.52
DEFAULT_ROW_MAX_HEIGHT_LATEX = r"0.052\textheight"
# Labeled PNG layout (drop-in image for the paper — not the .tex snippets).
DEFAULT_PNG_STRIP_SCALE = 0.36
DEFAULT_PAPER_PNG_WIDTH_PX = 1050  # ~3.5 in at 300 dpi (single-column figure)
DEFAULT_FIGURE_STRIP_SCALE = DEFAULT_PNG_STRIP_SCALE
# Vertical spacing between qualitative rows (paper layout).
DEFAULT_ROW_GAP_LATEX = "0em"
DEFAULT_ROW_GAP_IN = 0.0
DEFAULT_MODEL_BLOCK_GAP_LATEX = "0em"
DEFAULT_MODEL_TITLE_GAP_LATEX = "0em"
_PANEL_BAR = (
    "{desc} |{bar}| {n_fmt}/{total_fmt} panels "
    "[{elapsed}<{remaining}, {rate_fmt}]"
)


@dataclass
class Example:
    val_pos: int
    dataset_idx: int
    gt: str
    pred: str
    conf: float
    correct: bool


class EvalDataset(Dataset):
    def __init__(self, base: FineBadmintonDataset, pose_cache: torch.Tensor):
        self.base = base
        self.pose_cache = pose_cache

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        frames, labels = self.base[idx]
        return frames, self.pose_cache[idx], labels


def _norm(frames: torch.Tensor, device: torch.device) -> torch.Tensor:
    b, t, c, h, w = frames.shape
    x = frames.view(b * t, c, h, w).to(device)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return ((x - mean) / std).view(b, t, c, h, w)


def _task_classes(ds: FineBadmintonDataset, bundle: Optional[Dict[str, Any]]) -> Dict[str, int]:
    if bundle and bundle.get("task_classes"):
        tc = dict(bundle["task_classes"])
    else:
        tc = {k: len(v) for k, v in ds.classes.items()}
    tc["quality"] = 7
    tc.pop("stroke_subtype", None)
    return tc


def _load_checkpoint_meta(path: str) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = {k: ckpt[k] for k in ckpt if k not in ("jvc", "k_st_vit", "jvc_no_xattn", "model", "optimizer", "scheduler")}
    meta["_ckpt"] = ckpt
    if "jvc" in ckpt:
        meta["architecture"] = "jvc"
    elif "k_st_vit" in ckpt:
        meta["architecture"] = "jvc"
    elif "jvc_no_xattn" in ckpt:
        meta["architecture"] = "jvc_no_xattn"
    else:
        raise KeyError(f"Unknown checkpoint keys in {path}: {list(ckpt.keys())}")
    meta.setdefault("sampling_mode", "span_linspace")
    meta.setdefault("vision_backbone", "conv3d")
    meta.setdefault("video_backbone", "r2plus1d_18")
    meta.setdefault("four_stream", True)
    return meta


def _build_model(meta: Dict[str, Any], task_classes: Dict[str, int], device: torch.device):
    arch = meta["architecture"]
    if arch in ("jvc", "k_st_vit"):
        model = build_jvc(
            task_classes,
            vision_backbone=meta.get("vision_backbone", "conv3d"),
            video_backbone=meta.get("video_backbone", "r2plus1d_18"),
            embed_dim=int(meta.get("embed_dim", 128)),
            st_depth=int(meta.get("st_depth", 4)),
            num_cross_layers=int(meta.get("num_cross_layers", 2)),
            four_stream=bool(meta.get("four_stream", True)),
            use_shuttle=bool(meta.get("use_shuttle", False)),
        )
        load_jvc_partial(model, meta["_ckpt"], device=device)
    else:
        model = build_jvc_no_xattn(
            task_classes,
            embed_dim=int(meta.get("embed_dim", 64)),
            num_heads=int(meta.get("skel_num_heads", 16)),
            four_stream=bool(meta.get("four_stream", True)),
            video_backbone=meta.get("video_backbone", "r2plus1d_18"),
        )
        load_jvc_no_xattn_partial(model, meta["_ckpt"], device=device)
    return model.to(device).eval()


@torch.no_grad()
def _predict_all(
    model,
    loader: DataLoader,
    device: torch.device,
    stroke_names: Sequence[str],
    *,
    total_clips: int,
    show_progress: bool = True,
) -> List[Tuple[int, str, float]]:
    out: List[Tuple[int, str, float]] = []
    correct = 0
    pbar = tqdm(
        total=total_clips,
        desc="Val inference",
        unit="clip",
        bar_format=_CLIP_BAR,
        disable=not show_progress,
    )
    for frames, pose, labels in loader:
        logits = model(_norm(frames, device), pose.to(device))["stroke_type"]
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        gt = labels["stroke_type"].to(device)
        bs = gt.size(0)
        for i in range(bs):
            out.append(
                (
                    int(gt[i].item()),
                    stroke_names[int(pred[i].item())],
                    float(conf[i].item()),
                )
            )
        correct += int((pred == gt).sum().item())
        seen = len(out)
        pbar.update(bs)
        pbar.set_postfix(acc=f"{100.0 * correct / max(seen, 1):.1f}%", refresh=False)
    pbar.close()
    return out


def _label_name(ds: FineBadmintonDataset, dataset_idx: int, task: str = "stroke_type") -> str:
    labels = ds._map_labels(ds.samples[dataset_idx])
    return ds.classes[task][labels[task]]


def _bbox_on_court(bbox: Tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = bbox
    cy = (y0 + y1) * 0.5
    h = y1 - y0
    return cy >= 0.42 and h >= 0.10 and h <= 0.75


def _score_clip_visual(
    ds: FineBadmintonDataset,
    dataset_idx: int,
    pose_estimator,
) -> float:
    """Rank clips for figure aesthetics: pose coverage on court + striker size."""
    sample = ds.samples[dataset_idx]
    hits = 0
    score_sum = 0.0
    for t in range(NUM_FRAMES):
        rgb = load_native_rgb(sample, frame_index_for_timestep(sample, t))
        if rgb is None:
            continue
        pose = infer_striker_pose_rgb_with_fallback(rgb, pose_estimator)
        if pose is None:
            continue
        bbox = _pose_bbox_relaxed(pose, min_joints=8)
        if bbox is None or not _bbox_on_court(bbox):
            continue
        teaser = score_pose_for_teaser(pose)
        if teaser is None:
            continue
        hits += 1
        score_sum += float(teaser)
    if hits < 8:
        return 0.0
    coverage = hits / float(NUM_FRAMES)
    return (score_sum / hits) * coverage


def _pick_examples(
    ds: FineBadmintonDataset,
    val_idx: List[int],
    preds: List[Tuple[int, str, float]],
    stroke_names: Sequence[str],
    num_correct: int,
    num_errors: int,
    *,
    pose_estimator=None,
    visual_shortlist: int = 40,
) -> List[Example]:
    rows: List[Example] = []
    for pos, (gt_idx, pred_name, conf) in enumerate(preds):
        gt_name = stroke_names[gt_idx]
        rows.append(
            Example(
                val_pos=pos,
                dataset_idx=int(val_idx[pos]),
                gt=gt_name,
                pred=pred_name,
                conf=conf,
                correct=gt_name == pred_name,
            )
        )

    chosen: List[Example] = []
    used = set()

    def _best_candidate(cands: List[Example]) -> Optional[Example]:
        if not cands:
            return None
        pool = sorted(cands, key=lambda r: r.conf, reverse=True)[:visual_shortlist]
        if pose_estimator is None:
            return pool[0]
        print(f"  visual-rank {len(pool)} candidates...", flush=True)
        return max(
            pool,
            key=lambda r: (_score_clip_visual(ds, r.dataset_idx, pose_estimator), r.conf),
        )

    for stroke in stroke_names:
        if stroke == "Other":
            continue
        cands = [r for r in rows if r.correct and r.gt == stroke]
        pick = _best_candidate(cands)
        if pick is None:
            continue
        if pick.dataset_idx in used:
            continue
        chosen.append(pick)
        used.add(pick.dataset_idx)
        if len([x for x in chosen if x.correct]) >= num_correct:
            break

    def error_rank(r: Example) -> Tuple[int, float]:
        pair = frozenset({r.gt, r.pred})
        confusable = 1 if pair in CONFUSABLE else 0
        return (confusable, r.conf)

    errors = [r for r in rows if not r.correct]
    errors.sort(key=error_rank, reverse=True)
    seen_pairs = set()
    for r in errors:
        if r.dataset_idx in used:
            continue
        pair = (r.gt, r.pred)
        if pair in seen_pairs:
            continue
        pool = [x for x in errors if (x.gt, x.pred) == pair and x.dataset_idx not in used]
        pool = sorted(pool, key=lambda x: x.conf, reverse=True)[:visual_shortlist]
        if not pool:
            continue
        if pose_estimator is not None:
            print(f"  visual-rank error {pair[0]}→{pair[1]} ({len(pool)} candidates)...", flush=True)
            pick = max(
                pool,
                key=lambda x: (_score_clip_visual(ds, x.dataset_idx, pose_estimator), x.conf),
            )
        else:
            pick = pool[0]
        chosen.append(pick)
        used.add(pick.dataset_idx)
        seen_pairs.add(pair)
        if len([x for x in chosen if not x.correct]) >= num_errors:
            break

    correct = [x for x in chosen if x.correct]
    wrong = [x for x in chosen if not x.correct]
    return correct + wrong


def _parse_render_panels(spec: str) -> List[Example]:
    """Parse render-only panels: idx:GT:Pred:conf,idx:GT:Pred:conf,..."""
    examples: List[Example] = []
    for i, part in enumerate(spec.split(",")):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        if len(fields) != 4:
            raise ValueError(f"Bad panel spec {part!r}; want idx:GT:Pred:conf")
        idx, gt, pred, conf = int(fields[0]), fields[1], fields[2], float(fields[3])
        examples.append(
            Example(val_pos=i, dataset_idx=idx, gt=gt, pred=pred, conf=conf, correct=(gt == pred))
        )
    return examples


def _examples_for_indices(
    val_idx: List[int],
    preds: List[Tuple[int, str, float]],
    stroke_names: Sequence[str],
    indices: Sequence[int],
) -> List[Example]:
    lookup: Dict[int, Example] = {}
    for pos, (gt_idx, pred_name, conf) in enumerate(preds):
        gt_name = stroke_names[gt_idx]
        didx = int(val_idx[pos])
        lookup[didx] = Example(
            val_pos=pos,
            dataset_idx=didx,
            gt=gt_name,
            pred=pred_name,
            conf=conf,
            correct=gt_name == pred_name,
        )
    out: List[Example] = []
    for i, didx in enumerate(indices):
        if didx not in lookup:
            raise KeyError(f"dataset_idx {didx} not in val split")
        ex = lookup[didx]
        out.append(
            Example(
                val_pos=i,
                dataset_idx=ex.dataset_idx,
                gt=ex.gt,
                pred=ex.pred,
                conf=ex.conf,
                correct=ex.correct,
            )
        )
    return out


def _contact_timesteps(contact: int, display_frames: int, clip_frames: int = NUM_FRAMES) -> List[int]:
    """Uniform clip indices centered on contact (gold border frame)."""
    display_frames = max(3, min(display_frames, clip_frames))
    half = display_frames // 2
    start = max(0, contact - half)
    end = min(clip_frames, start + display_frames)
    start = max(0, end - display_frames)
    return list(range(start, end))


def _panel_timesteps(
    contact: int,
    contact_frames: int,
    frame_stride: int = 1,
    clip_frames: int = NUM_FRAMES,
) -> Tuple[List[int], int]:
    """Row panel indices; ``frame_stride=4`` on a 16-frame clip shows timesteps 0, 4, 8, 12."""
    full = _contact_timesteps(contact, contact_frames, clip_frames)
    if frame_stride <= 1:
        contact_col = full.index(contact) if contact in full else len(full) // 2
        return full, contact_col
    subs = full[::frame_stride]
    if contact in subs:
        contact_col = subs.index(contact)
    else:
        contact_col = int(np.argmin([abs(t - contact) for t in subs]))
    return subs, contact_col


def _row_label_stacked(gt: str, pred: str, conf: float, *, correct: bool) -> Tuple[str, List[str]]:
    tag = "Correct" if correct else "Failure"
    gt_s = gt.replace("_", " ")
    pred_s = pred.replace("_", " ")
    return tag, [f"GT: {gt_s}", f"Pred: {pred_s}", f"Confidence: {conf:.2f}"]


def _latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in text.replace("_", " "):
        out.append(repl.get(ch, ch))
    return "".join(out)


def _draw_panel_row(
    fig: plt.Figure,
    gs: GridSpec,
    row: int,
    col_offset: int,
    frames: Sequence[np.ndarray],
    contact_col: int,
) -> None:
    for col, frame in enumerate(frames):
        ax = fig.add_subplot(gs[row, col + col_offset])
        ax.imshow(frame, aspect="equal", interpolation="bilinear")
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(CONTACT_GOLD if col == contact_col else "#DEE2E6")
            spine.set_linewidth(2.6 if col == contact_col else 0.8)


def _save_row_strip(
    frames: Sequence[np.ndarray],
    contact_col: int,
    out_path: Path,
    *,
    cell_size: int,
    dpi: int,
) -> None:
    ncols = len(frames)
    fig_w, fig_h, width_ratios, margins = _figure_inches(
        1, ncols, cell_size, dpi, with_labels=False
    )
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = GridSpec(
        1,
        ncols,
        figure=fig,
        width_ratios=width_ratios,
        wspace=0.02,
        left=margins["left"] / fig_w,
        right=1.0 - margins["right"] / fig_w,
        top=1.0 - margins["top"] / fig_h,
        bottom=margins["bottom"] / fig_h,
    )
    _draw_panel_row(fig, gs, 0, 0, frames, contact_col)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def _trim_vertical_whitespace(arr: np.ndarray, threshold: int = 252) -> np.ndarray:
    """Drop matplotlib padding above/below a row strip."""
    mask = np.any(arr[..., :3] < threshold, axis=-1)
    rows = np.where(np.any(mask, axis=1))[0]
    if len(rows) == 0:
        return arr
    return arr[rows.min() : rows.max() + 1]


def _write_rows_latex(
    tex_path: Path,
    stem: str,
    rows: Sequence[Dict[str, Any]],
    *,
    figures_dir: str = "figures",
    strip_scale: float = DEFAULT_FIGURE_STRIP_SCALE,
    img_w: float = DEFAULT_ROW_IMG_WIDTH_LATEX,
    row_max_height: Optional[str] = DEFAULT_ROW_MAX_HEIGHT_LATEX,
) -> None:
    """Write per-row LaTeX blocks (labels + \\includegraphics for each row strip)."""
    label_w = 0.20
    if row_max_height:
        includegraphics = (
            f"\\includegraphics[width=\\linewidth,height={row_max_height},keepaspectratio]"
        )
    else:
        includegraphics = "\\includegraphics[width=\\linewidth]"
    parts: List[str] = [
        f"% Auto-generated row strips for {stem}. Requires xcolor in preamble.\n",
        "\\begingroup\\setlength{\\parskip}{0pt}\\lineskip=0pt\\lineskiplimit=0pt\n",
    ]
    for i, row in enumerate(rows):
        tag, lines = _row_label_stacked(
            row["gt"], row["pred"], float(row["conf"]), correct=bool(row["correct"])
        )
        color = "qualOk" if row["correct"] else "qualErr"
        stacked = "\\\\\n".join(
            f"{{\\footnotesize {_latex_escape(line)}}}" for line in lines
        )
        parts.append(
            f"\\noindent\\begin{{minipage}}[c]{{{label_w:.2f}\\textwidth}}\n"
            f"{{\\bfseries\\color{{{color}}}{_latex_escape(tag)}}}\\\\[0.35em]\n"
            f"{stacked}\n"
            f"\\end{{minipage}}\\hfill\n"
            f"\\begin{{minipage}}[c]{{{img_w:.2f}\\textwidth}}\n"
            f"{includegraphics}"
            f"{{{figures_dir}/{stem}_row{i}.pdf}}\n"
            f"\\end{{minipage}}\n"
        )
    parts.append("\\endgroup\n")
    tex_path.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote LaTeX rows {tex_path}", flush=True)


def _export_labeled_png(
    stem: str,
    rows: Sequence[Dict[str, Any]],
    fig_dir: Path,
    *,
    dpi: int = 200,
    out_path: Optional[Path] = None,
    strip_scale: float = DEFAULT_PNG_STRIP_SCALE,
    paper_width_px: int = DEFAULT_PAPER_PNG_WIDTH_PX,
) -> None:
    """Labeled figure: Correct/Failure + stacked GT/Pred/Confidence beside row strips."""
    from PIL import Image, ImageDraw, ImageFont

    OK_GREEN = (43, 138, 62)
    ERR_RED = (201, 42, 42)

    strips: List[Image.Image] = []
    for ri in range(len(rows)):
        arr = _trim_vertical_whitespace(np.array(Image.open(fig_dir / f"{stem}_row{ri}.png")))
        img = Image.fromarray(arr)
        if strip_scale != 1.0:
            new_w = max(1, int(round(img.width * strip_scale)))
            new_h = max(1, int(round(img.height * strip_scale)))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        strips.append(img)

    strip_w = max(s.width for s in strips)
    strip_h_total = sum(s.height for s in strips)
    label_w_px = int(round(0.72 * dpi))
    gap_px = int(round(0.04 * dpi))
    margin_l = int(round(0.06 * dpi))
    margin_r = int(round(0.06 * dpi))
    margin_t = int(round(0.04 * dpi))
    margin_b = int(round(0.05 * dpi))

    canvas_w = margin_l + label_w_px + gap_px + strip_w + margin_r
    canvas_h = margin_t + strip_h_total + margin_b
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font_bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 16)
        font_reg = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font_bold = ImageFont.load_default()
        font_reg = font_bold

    y = margin_t
    for ri, row in enumerate(rows):
        tag, lines = _row_label_stacked(
            row["gt"], row["pred"], float(row["conf"]), correct=bool(row["correct"])
        )
        color = OK_GREEN if row["correct"] else ERR_RED
        x_label = margin_l
        draw.text((x_label, y + 2), tag, fill=color, font=font_bold)
        line_y = y + 24
        for line in lines:
            draw.text((x_label, line_y), line, fill=color, font=font_reg)
            line_y += 18

        strip = strips[ri]
        x_strip = margin_l + label_w_px + gap_px
        canvas.paste(strip, (x_strip, y))
        y += strip.height

    save_dpi = 300
    if paper_width_px and canvas.width > 0 and canvas.width != paper_width_px:
        new_h = max(1, int(round(canvas.height * paper_width_px / canvas.width)))
        canvas = canvas.resize((paper_width_px, new_h), Image.Resampling.LANCZOS)

    out = out_path if out_path is not None else fig_dir / f"{stem}_labeled.png"
    canvas.save(out, dpi=(save_dpi, save_dpi))
    print(f"Wrote labeled figure {out}", flush=True)
    labeled_alias = fig_dir / f"{stem}_labeled.png"
    if out.resolve() != labeled_alias.resolve():
        import shutil
        shutil.copy2(out, labeled_alias)
        print(f"Wrote labeled figure {labeled_alias}", flush=True)


def compose_qualitative_latex(
    out_path: Path,
    blocks: Sequence[Tuple[str, str]],
    *,
    figures_dir: str = "figures",
) -> None:
    """Combine per-model row snippets into one figure* for the paper."""
    lines = [
        "% Auto-generated qualitative figure. Requires: graphicx, xcolor\n"
        "\\providecolor{qualOk}{RGB}{43,138,62}\n"
        "\\providecolor{qualErr}{RGB}{201,42,42}\n"
        "\\begin{figure*}[t]\n"
        "  \\centering\n",
    ]
    for i, (stem, title) in enumerate(blocks):
        if i and DEFAULT_MODEL_BLOCK_GAP_LATEX not in ("0em", "0pt", "0"):
            lines.append(f"  \\vspace{{{DEFAULT_MODEL_BLOCK_GAP_LATEX}}}\n\n")
        lines.append(f"  {{\\normalsize\\textbf{{{_latex_escape(title)}}}}}\\par\n")
        if DEFAULT_MODEL_TITLE_GAP_LATEX not in ("0em", "0pt", "0"):
            lines.append(f"  \\vspace{{{DEFAULT_MODEL_TITLE_GAP_LATEX}}}\n")
        lines.append(f"  \\input{{{figures_dir}/{stem}_rows.tex}}\n\n")
    lines.extend(
        [
            "  \\caption{%\n"
            "    Qualitative stroke-type predictions on validation clips.\n"
            "    Each row shows every fourth frame of the 16-frame \\texttt{span\\_linspace} input\n"
            "    (224$\\times$224 RGB seen by the vision encoder); gold border marks contact.\n"
            "    Top two rows: correct high-confidence predictions; bottom two rows: representative\n"
            "    confusions (distinct error pairs).\n"
            "  }\n"
            "  \\label{fig:qualitative_predictions}\n"
            "\\end{figure*}\n",
        ]
    )
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote LaTeX figure {out_path}", flush=True)


def _figure_inches(
    nrows: int,
    ncols: int,
    cell_size: int,
    dpi: int,
    *,
    with_labels: bool,
) -> Tuple[float, float, List[float], Dict[str, float]]:
    """Size the canvas so each frame panel is a square of ``cell_size`` px at ``dpi``."""
    cell_in = cell_size / dpi
    label_ratio = 1.55 if with_labels else 0.0
    width_ratios = ([label_ratio] + [1.0] * ncols) if with_labels else [1.0] * ncols
    grid_w_in = cell_in * (label_ratio + ncols)
    grid_h_in = cell_in * nrows
    if with_labels:
        margins = dict(left=0.30, right=0.40, top=0.10, bottom=0.14)
    else:
        margins = dict(left=0.02, right=0.02, top=0.01, bottom=0.01)
    fig_w = grid_w_in + margins["left"] + margins["right"]
    fig_h = grid_h_in + margins["top"] + margins["bottom"]
    return fig_w, fig_h, width_ratios, margins


def _draw_pose_overlay(bgr: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Bold skeleton for small qualitative crops (readable in print)."""
    h, w = bgr.shape[:2]
    scale = max(2.0, min(h, w) / 120.0)
    joint_r = max(4, int(round(5 * scale)))
    bone_w = max(3, int(round(4 * scale)))
    out = bgr.copy()
    pts: List[Any] = [None] * 33
    for j in range(33):
        x, y = float(pose[j, 0]), float(pose[j, 1])
        if x <= 1e-4 and y <= 1e-4:
            continue
        cx, cy = int(x * w), int(y * h)
        pts[j] = (cx, cy)
        cv2.circle(out, (cx, cy), joint_r + 2, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (cx, cy), joint_r, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    for a, b in MEDIAPIPE_BONE_PAIRS:
        if pts[a] is None or pts[b] is None:
            continue
        cv2.line(out, pts[a], pts[b], (0, 0, 0), bone_w + 2, lineType=cv2.LINE_AA)
        cv2.line(out, pts[a], pts[b], (0, 255, 255), bone_w, lineType=cv2.LINE_AA)
    return out


def _visible_joint_count(pose: np.ndarray) -> int:
    mask = (pose[:, 0] > 1e-4) | (pose[:, 1] > 1e-4)
    return int(mask.sum())


def _pose_bbox_relaxed(pose: np.ndarray, *, min_joints: int = 6) -> Optional[Tuple[float, float, float, float]]:
    mask = (pose[:, 0] > 1e-4) | (pose[:, 1] > 1e-4)
    if int(mask.sum()) < min_joints:
        return None
    xs, ys = pose[mask, 0], pose[mask, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _expand_bbox(
    bbox: Tuple[float, float, float, float],
    margin: float,
) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    x0 = max(0.0, x0 - margin)
    y0 = max(0.0, y0 - margin)
    x1 = min(1.0, x1 + margin)
    y1 = min(1.0, y1 + margin)
    return x0, y0, x1, y1


def _crop_with_bbox(
    rgb: np.ndarray,
    pose: np.ndarray,
    bbox: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = bbox
    if x1 - x0 < 0.08 or y1 - y0 < 0.08:
        return rgb, pose
    h, w = rgb.shape[:2]
    crop = rgb[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)].copy()
    pose_c = pose.copy()
    pose_c[:, 0] = (pose[:, 0] - x0) / max(x1 - x0, 1e-6)
    pose_c[:, 1] = (pose[:, 1] - y0) / max(y1 - y0, 1e-6)
    return crop, pose_c


def _anchor_bbox(
    poses: Sequence[np.ndarray],
    contact_col: int,
    crop_margin: float,
) -> Optional[Tuple[float, float, float, float]]:
    order = [contact_col] + sorted(
        range(len(poses)),
        key=lambda i: _visible_joint_count(poses[i]),
        reverse=True,
    )
    seen = set()
    for i in order:
        if i in seen:
            continue
        seen.add(i)
        bbox = _pose_bbox_relaxed(poses[i])
        if bbox is not None and _bbox_on_court(bbox):
            return _expand_bbox(bbox, crop_margin)
    return None


def _rgb_cache224(ds: FineBadmintonDataset, dataset_idx: int, timestep: int) -> np.ndarray:
    clip, _ = ds[dataset_idx]
    return (clip[timestep].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def _resolve_panel_rgb_pose(
    ds: FineBadmintonDataset,
    pose_cache: torch.Tensor,
    dataset_idx: int,
    timestep: int,
    *,
    frame_source: str,
    pose_estimator=None,
    faithful: bool = False,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Return display RGB + normalized pose for one timestep.

    ``faithful``: 224×224 training frames only (vision-branch pixels; no overlay).
    Default path: native broadcast frame + live MediaPipe every timestep.
    """
    cached_pose = pose_cache[dataset_idx, timestep].numpy()
    if faithful:
        return _rgb_cache224(ds, dataset_idx, timestep), cached_pose
    sample = ds.samples[dataset_idx]
    clip, _ = ds[dataset_idx]
    frame_chw = clip[timestep]

    rgb_native = load_native_rgb(sample, frame_index_for_timestep(sample, timestep))
    rgb_cache = _rgb_cache224(ds, dataset_idx, timestep)

    if pose_estimator is not None:
        if frame_source == "cache224":
            live224 = infer_pose_on_224_frame(frame_chw, pose_estimator)
            if live224 is not None:
                return rgb_cache, live224
        elif rgb_native is not None:
            live = infer_striker_pose_rgb_with_fallback(rgb_native, pose_estimator)
            if live is not None:
                return rgb_native, live
            live224 = infer_pose_on_224_frame(frame_chw, pose_estimator)
            if live224 is not None:
                return rgb_native, live224
        else:
            live224 = infer_pose_on_224_frame(frame_chw, pose_estimator)
            if live224 is not None:
                return rgb_cache, live224

    if frame_source == "cache224":
        return rgb_cache, cached_pose
    if rgb_native is not None:
        return rgb_native, cached_pose
    return None, cached_pose


def _render_panel_from_rgb_pose(
    rgb: Optional[np.ndarray],
    pose: np.ndarray,
    cell_size: int,
    *,
    crop_pose: bool,
    crop_margin: float,
    fixed_bbox: Optional[Tuple[float, float, float, float]] = None,
    pose_overlay: bool = True,
) -> np.ndarray:
    if rgb is None:
        return np.zeros((cell_size, cell_size, 3), dtype=np.uint8)

    if crop_pose:
        if fixed_bbox is not None:
            rgb, pose = _crop_with_bbox(rgb, pose, fixed_bbox)
        elif _visible_joint_count(pose) >= 6:
            rgb, pose = crop_frame_and_pose(rgb, pose, margin=crop_margin)

    if pose_overlay:
        bgr = _draw_pose_overlay(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), pose)
        rgb_out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb_out = rgb
    return cv2.resize(rgb_out, (cell_size, cell_size), interpolation=cv2.INTER_CUBIC)


def _render_row(
    ds: FineBadmintonDataset,
    pose_cache: torch.Tensor,
    dataset_idx: int,
    cell_size: int,
    timesteps: Sequence[int],
    *,
    crop_pose: bool,
    crop_margin: float,
    frame_source: str,
    pose_estimator=None,
    pose_overlay: bool = True,
    faithful: bool = False,
) -> Tuple[List[np.ndarray], int]:
    sample = ds.samples[dataset_idx]
    contact = contact_timestep(sample)
    contact_col = timesteps.index(contact) if contact in timesteps else len(timesteps) // 2

    resolved: List[Tuple[Optional[np.ndarray], np.ndarray]] = [
        _resolve_panel_rgb_pose(
            ds,
            pose_cache,
            dataset_idx,
            t,
            frame_source=frame_source,
            pose_estimator=pose_estimator,
            faithful=faithful,
        )
        for t in timesteps
    ]
    poses = [p for _, p in resolved]
    fixed_bbox = _anchor_bbox(poses, contact_col, crop_margin) if crop_pose else None

    frames = [
        _render_panel_from_rgb_pose(
            rgb,
            pose,
            cell_size,
            crop_pose=crop_pose,
            crop_margin=crop_margin,
            fixed_bbox=fixed_bbox,
            pose_overlay=pose_overlay,
        )
        for rgb, pose in resolved
    ]
    return frames, contact_col


def _draw_figure(
    examples: Sequence[Example],
    ds: FineBadmintonDataset,
    pose_cache: torch.Tensor,
    model_label: str,
    out_path: Path,
    dpi: int,
    cell_size: int,
    contact_frames: int,
    *,
    frame_stride: int = 1,
    crop_pose: bool,
    crop_margin: float,
    frame_source: str,
    pose_estimator=None,
    show_progress: bool = True,
    with_labels: bool = False,
    strip_scale: float = DEFAULT_FIGURE_STRIP_SCALE,
    pose_overlay: bool = True,
    faithful: bool = False,
) -> None:
    nrows = len(examples)
    ex0_contact = contact_timestep(ds.samples[examples[0].dataset_idx])
    ncols = len(_panel_timesteps(ex0_contact, contact_frames, frame_stride)[0])

    meta_rows: List[Dict[str, Any]] = []
    row_frames: List[Tuple[List[np.ndarray], int]] = []

    for row, ex in enumerate(
        tqdm(
            examples,
            desc="Rendering panels",
            unit="panel",
            bar_format=_PANEL_BAR,
            disable=not show_progress,
        )
    ):
        contact = contact_timestep(ds.samples[ex.dataset_idx])
        row_steps, contact_col = _panel_timesteps(contact, contact_frames, frame_stride)
        frames, _ = _render_row(
            ds,
            pose_cache,
            ex.dataset_idx,
            cell_size,
            row_steps,
            crop_pose=crop_pose,
            crop_margin=crop_margin,
            frame_source=frame_source,
            pose_estimator=pose_estimator,
            pose_overlay=pose_overlay,
            faithful=faithful,
        )
        row_frames.append((frames, contact_col))
        meta_rows.append(
            {
                "row": row,
                "dataset_idx": ex.dataset_idx,
                "gt": ex.gt,
                "pred": ex.pred,
                "conf": ex.conf,
                "correct": ex.correct,
                "contact_col": contact_col,
                "timesteps": row_steps,
                "stroke_subtype": _label_name(ds, ex.dataset_idx, "stroke_subtype"),
                "technique": _label_name(ds, ex.dataset_idx, "technique"),
            }
        )

    stem = out_path.stem
    for i, (frames, contact_col) in enumerate(row_frames):
        row_png = out_path.with_name(f"{stem}_row{i}.png")
        _save_row_strip(frames, contact_col, row_png, cell_size=cell_size, dpi=dpi)

    label_col = 1 if with_labels else 0
    fig_w, fig_h, width_ratios, margins = _figure_inches(
        nrows, ncols, cell_size, dpi, with_labels=with_labels
    )
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = GridSpec(
        nrows,
        ncols + label_col,
        figure=fig,
        width_ratios=width_ratios,
        hspace=0.06 if with_labels else 0.02,
        wspace=0.10 if with_labels else 0.02,
        left=margins["left"] / fig_w,
        right=1.0 - margins["right"] / fig_w,
        top=1.0 - margins["top"] / fig_h,
        bottom=margins["bottom"] / fig_h,
    )

    for row, (frames, contact_col) in enumerate(row_frames):
        if with_labels:
            ax_label = fig.add_subplot(gs[row, 0])
            ax_label.axis("off")
            ex = examples[row]
            color = OK_GREEN if ex.correct else ERR_RED
            tag, lines = _row_label_stacked(ex.gt, ex.pred, ex.conf, correct=ex.correct)
            ax_label.text(0.0, 0.82, tag, ha="left", va="center", fontsize=10, fontweight="bold", color=color)
            ytxt = 0.58
            for line in lines:
                ax_label.text(0.0, ytxt, line, ha="left", va="center", fontsize=8.5, color=color)
                ytxt -= 0.22
        _draw_panel_row(fig, gs, row, label_col, frames, contact_col)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path = out_path.with_name(f"{stem}_grid.png")
    fig.savefig(grid_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Wrote grid (no labels) {grid_path}", flush=True)

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "model_label": model_label,
                "render_mode": "faithful" if faithful else "illustrative",
                "frame_source": "cache224" if faithful else frame_source,
                "pose_overlay": pose_overlay,
                "clip_frames": contact_frames,
                "frame_stride": frame_stride,
                "display_frames": ncols,
                "cell_size": cell_size,
                "rows": meta_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote metadata {meta_path}", flush=True)
    _write_rows_latex(
        out_path.with_name(f"{stem}_rows.tex"), stem, meta_rows, strip_scale=strip_scale
    )
    _export_labeled_png(
        stem, meta_rows, out_path.parent, dpi=dpi, out_path=out_path, strip_scale=strip_scale
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pose-cache", type=Path, default=DEFAULT_POSE_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=BACKEND_ROOT / "models" / "badminton_model_k_st_vit.pth")
    parser.add_argument("--model-label", default="JVC")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-correct", type=int, default=2, help="Correct examples to show (default: 2)")
    parser.add_argument("--num-errors", type=int, default=2, help="Error examples to show (default: 2)")
    parser.add_argument(
        "--contact-frames",
        type=int,
        default=16,
        help="Frames shown per row (default: 16, matching training clip length)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Show every Nth frame from the clip window (default: 1; use 4 for 4 panels from 16)",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.18,
        help="Padding around pose bbox when cropping (default: 0.18)",
    )
    parser.add_argument("--no-crop", action="store_true", help="Disable pose-centered crop")
    parser.add_argument(
        "--frame-source",
        choices=("native", "cache224"),
        default="native",
        help="native = broadcast MP4 (default); cache224 = 224×224 training frames",
    )
    parser.add_argument(
        "--no-live-pose",
        action="store_true",
        help="Skip per-frame MediaPipe (sparse cache only; not recommended for figures)",
    )
    parser.add_argument("--pose-model", type=Path, default=BACKEND_ROOT / "models" / "pose_landmarker_lite.task")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--cell-size", type=int, default=140)
    parser.add_argument(
        "--figure-scale",
        type=float,
        default=DEFAULT_FIGURE_STRIP_SCALE,
        help="Labeled PNG frame scale (default: 0.36)",
    )
    parser.add_argument(
        "--paper-png-width",
        type=int,
        default=DEFAULT_PAPER_PNG_WIDTH_PX,
        help="Resize labeled PNG to this pixel width at 300 dpi (~3.5in; 0 to disable)",
    )
    parser.add_argument(
        "--row-img-width",
        type=float,
        default=DEFAULT_ROW_IMG_WIDTH_LATEX,
        help="LaTeX minipage width for row strips (default: 0.52)",
    )
    parser.add_argument(
        "--row-max-height",
        default=DEFAULT_ROW_MAX_HEIGHT_LATEX,
        help="LaTeX max height per row strip, e.g. 0.052\\textheight (empty to disable)",
    )
    parser.add_argument(
        "--render-panels",
        default=None,
        help="Skip inference; render only. Format: idx:GT:Pred:conf,... (from prior log)",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument(
        "--with-labels",
        action="store_true",
        help="Burn GT/pred text and title into the figure (default: image-only for LaTeX)",
    )
    parser.add_argument(
        "--pick-visual",
        action="store_true",
        help="Among high-conf val candidates, pick clips with best pose coverage on court",
    )
    parser.add_argument(
        "--curated-idx",
        default=None,
        help="Fixed dataset indices (comma-separated). Still runs inference for this model's preds.",
    )
    parser.add_argument(
        "--no-pose-overlay",
        action="store_true",
        help="RGB only (vision-branch pixels); no skeleton drawn on frames",
    )
    parser.add_argument(
        "--model-view",
        action="store_true",
        help="224×224 training frames + no crop (still illustrative if live pose on)",
    )
    parser.add_argument(
        "--faithful",
        action="store_true",
        help="Evaluation mode: 224×224 training frames, no crop, no live pose, no overlay",
    )
    parser.add_argument(
        "--visual-shortlist",
        type=int,
        default=40,
        help="Top-N by confidence to visual-rank when --pick-visual (default: 40)",
    )
    parser.add_argument(
        "--compose-latex",
        type=Path,
        default=None,
        help="Write combined figure* .tex from --latex-block entries (no render)",
    )
    parser.add_argument(
        "--latex-block",
        action="append",
        default=[],
        help="stem:Model title for --compose-latex (repeatable)",
    )
    parser.add_argument(
        "--reexport-layout",
        action="store_true",
        help="Regenerate *_rows.tex and labeled PNG from existing row strips + .meta.json (no render)",
    )
    args = parser.parse_args()
    if args.compose_latex:
        if not args.latex_block:
            raise SystemExit("--compose-latex requires at least one --latex-block stem:title")
        blocks = []
        for item in args.latex_block:
            stem, title = item.split(":", 1)
            blocks.append((stem.strip(), title.strip()))
        compose_qualitative_latex(args.compose_latex, blocks)
        return

    if args.reexport_layout:
        stem = args.out.stem
        meta_path = args.out.with_name(f"{stem}.meta.json")
        if not meta_path.is_file():
            raise SystemExit(f"Missing metadata: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows = meta["rows"]
        fig_dir = args.out.parent
        row_max_height = args.row_max_height or None
        _write_rows_latex(
            fig_dir / f"{stem}_rows.tex",
            stem,
            rows,
            strip_scale=args.figure_scale,
            img_w=args.row_img_width,
            row_max_height=row_max_height,
        )
        _export_labeled_png(
            stem,
            rows,
            fig_dir,
            dpi=args.dpi,
            out_path=args.out,
            strip_scale=args.figure_scale,
            paper_width_px=args.paper_png_width,
        )
        return

    if args.faithful:
        args.frame_source = "cache224"
        args.no_crop = True
        args.no_live_pose = True
        args.no_pose_overlay = True
    elif args.model_view:
        args.frame_source = "cache224"
        args.no_crop = True

    show_progress = not args.no_progress
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "(all visible)")
        print(f"GPU 0: {gpu_name}  CUDA_VISIBLE_DEVICES={cvd}", flush=True)

    ds = FineBadmintonDataset(str(args.data_root), str(args.json), transform=None, sampling_mode="span_linspace")
    pose_path = str(args.pose_cache)
    bundle = load_pose_cache_bundle(pose_path)
    if bundle is None:
        raise SystemExit(f"Missing pose cache: {pose_path}")
    pose_cache = bundle["pose_cache"]
    stroke_names = ds.classes["stroke_type"]

    pose_estimator = None
    if args.faithful:
        print("Faithful render: 224×224 training frames, no pose overlay", flush=True)
    elif not args.no_live_pose:
        try:
            pose_estimator = create_teaser_pose_estimator(str(args.pose_model))
            print("Live MediaPipe enabled (per-frame pose on native frames)", flush=True)
        except OSError as exc:
            print(
                f"WARNING: MediaPipe unavailable ({exc}); pose will be sparse (cache only). "
                "On cluster: ./scripts/cluster/setup_mediapipe_gl.sh && "
                "source scripts/cluster/mediapipe_gl.env",
                flush=True,
            )

    if args.render_panels:
        examples = _parse_render_panels(args.render_panels)
        print(f"Render-only: {len(examples)} panels (skipping inference)", flush=True)
    else:
        task_classes = _task_classes(ds, bundle)
        _, val_idx, _test_idx = video_level_split(ds.samples)
        val_ds = EvalDataset(Subset(ds, val_idx), pose_cache)
        loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        meta = _load_checkpoint_meta(str(args.checkpoint))
        model = _build_model(meta, task_classes, device)
        print(f"Scanning val split ({len(val_idx)} clips)...", flush=True)
        raw_preds = _predict_all(
            model,
            loader,
            device,
            stroke_names,
            total_clips=len(val_idx),
            show_progress=show_progress,
        )
        if args.curated_idx:
            indices = [int(x.strip()) for x in args.curated_idx.split(",") if x.strip()]
            examples = _examples_for_indices(val_idx, raw_preds, stroke_names, indices)
            print(f"Using curated indices: {indices}", flush=True)
        else:
            pick_estimator = pose_estimator if args.pick_visual else None
            if args.pick_visual and pick_estimator is None:
                print("WARNING: --pick-visual needs live MediaPipe; falling back to confidence pick", flush=True)
            examples = _pick_examples(
                ds,
                val_idx,
                raw_preds,
                stroke_names,
                args.num_correct,
                args.num_errors,
                pose_estimator=pick_estimator,
                visual_shortlist=args.visual_shortlist,
            )

    print(
        f"Selected {len(examples)} panels ({sum(e.correct for e in examples)} correct, "
        f"{sum(not e.correct for e in examples)} errors)",
        flush=True,
    )
    for ex in examples:
        print(f"  idx={ex.dataset_idx} GT={ex.gt} Pred={ex.pred} conf={ex.conf:.2f}", flush=True)

    _draw_figure(
        examples,
        ds,
        pose_cache,
        args.model_label,
        args.out,
        args.dpi,
        args.cell_size,
        args.contact_frames,
        frame_stride=args.frame_stride,
        crop_pose=not args.no_crop,
        crop_margin=args.crop_margin,
        frame_source=args.frame_source,
        pose_estimator=pose_estimator,
        show_progress=show_progress,
        with_labels=args.with_labels,
        pose_overlay=not args.no_pose_overlay,
        faithful=args.faithful,
        strip_scale=args.figure_scale,
    )
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
