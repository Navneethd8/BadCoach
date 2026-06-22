"""Eval metrics for VLM SFT: token accuracy and FineBadminton stroke-type accuracy."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_STROKE_ANCHOR = re.compile(r"Stroke:\s*", re.IGNORECASE)
_FORMAT_PLACEHOLDER = re.compile(r"^<[^>]+>$")


def parse_stroke_type(text: str) -> str | None:
    """
    Main stroke label from FineBadminton-style captions, e.g.
    ``Stroke: push shot Subtype: flat lift`` → ``push shot``.
    Stops before ``Subtype:``, ``Player:``, ``Hitter side:``, or newline.

    Uses the **last** ``Stroke:`` match so format hints in the prompt
    (``Stroke: <hit type>``) are not mistaken for the model answer.
    """
    if not text:
        return None
    matches = list(_STROKE_ANCHOR.finditer(text))
    if not matches:
        return None
    rest = text[matches[-1].end() :]
    for stop in (" Subtype:", " Player:", " Hitter side:", "\n", "\r"):
        if stop in rest:
            rest = rest.split(stop, 1)[0]
    s = rest.strip()
    if not s or _FORMAT_PLACEHOLDER.match(s):
        return None
    return s


def infer_stroke_label_from_text(text: str) -> str | None:
    """Fallback when the model skips the ``Stroke:`` prefix."""
    if not text:
        return None
    lowered = text.lower()
    # Prefer longer multi-word hit types first (e.g. "push shot" before "shot").
    from core.label_maps import HIT_TYPE_TO_STROKE_TYPE, STROKE_TYPE_CLASSES

    candidates: list[tuple[int, str]] = []
    for raw in HIT_TYPE_TO_STROKE_TYPE:
        if raw in lowered:
            candidates.append((len(raw), raw))
    for cls in STROKE_TYPE_CLASSES:
        for form in (cls.lower(), cls.lower().replace("_", " ")):
            if form in lowered:
                candidates.append((len(form), form))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def extract_stroke_label(text: str) -> str | None:
    """Parse ``Stroke: ...`` or fall back to keyword search in free text."""
    raw = parse_stroke_type(text)
    if raw is not None:
        return raw
    return infer_stroke_label_from_text(text)


def _normalize_stroke(s: str) -> str:
    return " ".join(s.lower().split())


def build_sft_eval_compute_metrics(tokenizer: PreTrainedTokenizerBase):
    import numpy as np

    def compute_metrics(eval_pred):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = np.asarray(logits)
        labels = np.asarray(labels)
        if logits.ndim == 3:
            preds = np.argmax(logits, axis=-1)
        else:
            preds = logits

        mask = labels != -100
        pad = tokenizer.pad_token_id
        if pad is not None:
            mask &= labels != pad
        denom = float(mask.sum())
        if denom <= 0:
            return {"eval_accuracy": 0.0, "eval_stroke_accuracy": 0.0}

        acc = float(((preds == labels) & mask).sum() / denom)

        n_stroke = 0
        n_stroke_correct = 0
        for i in range(labels.shape[0]):
            row_mask = mask[i]
            if not row_mask.any():
                continue
            t_ids = labels[i][row_mask].tolist()
            p_ids = preds[i][row_mask].tolist()
            ref = tokenizer.decode(t_ids, skip_special_tokens=True)
            hyp = tokenizer.decode(p_ids, skip_special_tokens=True)
            gt = parse_stroke_type(ref)
            if gt is None:
                continue
            n_stroke += 1
            pr = parse_stroke_type(hyp)
            if pr is not None and _normalize_stroke(gt) == _normalize_stroke(pr):
                n_stroke_correct += 1

        stroke_acc = float(n_stroke_correct / n_stroke) if n_stroke else 0.0
        return {"eval_accuracy": acc, "eval_stroke_accuracy": stroke_acc}

    return compute_metrics
