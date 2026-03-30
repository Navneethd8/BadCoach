"""Align Qwen3-VL / Unsloth tokenizer vision preprocessor with training (max_pixels, etc.)."""

from __future__ import annotations

from typing import Any


def _collect_image_processors(tokenizer: Any) -> list[Any]:
    seen: set[int] = set()
    out: list[Any] = []

    def add(obj: Any) -> None:
        if obj is None:
            return
        i = id(obj)
        if i in seen:
            return
        seen.add(i)
        out.append(obj)

    add(getattr(tokenizer, "image_processor", None))
    proc = getattr(tokenizer, "processor", None)
    if proc is not None:
        add(getattr(proc, "image_processor", None))
    return out


def apply_vision_processor_limits(
    tokenizer: Any,
    *,
    max_pixels: int | None,
    min_pixels: int | None = None,
) -> None:
    """
    Set ``max_pixels`` / ``min_pixels`` on Hugging Face image processors attached to the
    tokenizer so preprocessing matches PIL resize and avoids implicit defaults (e.g. 512).

    No-op if no processor exposes these attributes.
    """
    if max_pixels is None and min_pixels is None:
        return
    for ip in _collect_image_processors(tokenizer):
        if max_pixels is not None and hasattr(ip, "max_pixels"):
            ip.max_pixels = max_pixels
        if min_pixels is not None and hasattr(ip, "min_pixels"):
            ip.min_pixels = min_pixels
