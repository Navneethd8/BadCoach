"""Eval metrics for VLM SFT: token accuracy and FineBadminton stroke-type accuracy."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_STROKE_ANCHOR = re.compile(r"Stroke:\s*", re.IGNORECASE)


def parse_stroke_type(text: str) -> str | None:
    """
    Main stroke label from FineBadminton-style captions, e.g.
    ``Stroke: push shot Subtype: flat lift`` → ``push shot``.
    Stops before ``Subtype:``, ``Player:``, ``Hitter side:``, or newline.
    """
    if not text:
        return None
    m = _STROKE_ANCHOR.search(text)
    if not m:
        return None
    rest = text[m.end() :]
    for stop in (" Subtype:", " Player:", " Hitter side:", "\n", "\r"):
        if stop in rest:
            rest = rest.split(stop, 1)[0]
    s = rest.strip()
    return s if s else None


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
