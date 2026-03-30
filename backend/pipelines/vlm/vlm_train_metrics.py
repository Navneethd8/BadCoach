"""Token-level eval accuracy for SFT (use with ``eval_loss`` from Trainer)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def build_sft_eval_compute_metrics(tokenizer: PreTrainedTokenizerBase):
    import numpy as np

    def compute_metrics(eval_pred):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
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
            return {"eval_accuracy": 0.0}
        acc = float(((preds == labels) & mask).sum() / denom)
        return {"eval_accuracy": acc}

    return compute_metrics
