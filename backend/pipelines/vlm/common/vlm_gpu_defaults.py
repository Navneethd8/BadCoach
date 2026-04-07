"""CUDA capability–aware defaults (e.g. A100 / Ampere sm_80+ prefers bf16)."""

from __future__ import annotations


def default_mixed_precision_flags() -> tuple[bool, bool]:
    """
    Return (bf16, fp16) for Hugging Face ``TrainingArguments``-style kwargs.

    Ampere and newer (compute capability major >= 8, e.g. A100, A10, L4, RTX 30xx+):
    prefer bf16. Older GPUs (e.g. T4 = 7.5): fp16.
    """
    import torch

    if not torch.cuda.is_available():
        return False, True
    major, _minor = torch.cuda.get_device_capability(0)
    if major >= 8:
        return True, False
    return False, True


def resolve_train_amp(
    bf16: bool | None,
    fp16: bool | None,
) -> tuple[bool, bool]:
    """
    Merge CLI overrides with auto defaults.

    ``None`` means "not specified". Explicit ``bf16=True`` always wins;
    ``bf16=False`` and ``fp16=False`` together disables both (fp32).
    """
    auto_bf16, auto_fp16 = default_mixed_precision_flags()
    if bf16 is True:
        return True, False
    if bf16 is False and fp16 is False:
        return False, False
    if bf16 is False:
        return False, True
    if fp16 is True:
        return False, True
    if fp16 is False:
        return False, False
    return auto_bf16, auto_fp16
