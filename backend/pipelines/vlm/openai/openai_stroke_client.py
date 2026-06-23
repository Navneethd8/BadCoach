"""Chat Completions client for 16-frame stroke eval."""

from __future__ import annotations

import base64
import io
import time
from typing import Any

from PIL import Image

from openai_vlm_config import (
    DEFAULT_IMAGE_DETAIL,
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_REQUEST_TIMEOUT,
)


def _is_unsupported_token_limit_param(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) != 400:
        return False
    msg = str(exc).lower()
    return "max_tokens" in msg or "max_completion_tokens" in msg


def _is_unsupported_reasoning_effort(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) != 400:
        return False
    return "reasoning_effort" in str(exc).lower()


def _model_uses_reasoning(model: str) -> bool:
    m = (model or "").lower()
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4")


def _chat_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    timeout: float,
    reasoning_effort: str | None,
) -> Any:
    """Newer models (e.g. gpt-5.x) use max_completion_tokens; older use max_tokens."""
    last_err: Exception | None = None
    for token_kw in (
        {"max_completion_tokens": max_tokens},
        {"max_tokens": max_tokens},
    ):
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "timeout": timeout,
            **token_kw,
        }
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            if _is_unsupported_token_limit_param(e):
                last_err = e
                continue
            if reasoning_effort is not None and _is_unsupported_reasoning_effort(e):
                kwargs.pop("reasoning_effort", None)
                return client.chat.completions.create(**kwargs)
            raise
    assert last_err is not None
    raise last_err


def _message_text_or_empty(resp: Any) -> str:
    choice = resp.choices[0]
    msg = choice.message
    text = (getattr(msg, "content", None) or "").strip()
    if text:
        return text
    refusal = getattr(msg, "refusal", None)
    if refusal:
        return str(refusal).strip()
    return ""


def _completion_budgets(max_tokens: int, *, cap: int = 4096) -> list[int]:
    budgets: list[int] = []
    t = max(64, int(max_tokens))
    while True:
        if t not in budgets:
            budgets.append(t)
        if t >= cap:
            break
        t = min(t * 2, cap)
    return budgets


def _reasoning_efforts(model: str, preferred: str | None) -> list[str | None]:
    if not _model_uses_reasoning(model):
        return [None]
    ordered: list[str | None] = []
    for effort in (preferred, "none", "minimal", "low", None):
        if effort not in ordered:
            ordered.append(effort)
    return ordered


def _image_to_data_url(img: Image.Image, *, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=92)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def build_stroke_messages(
    instruction: str,
    images: list[Image.Image],
    *,
    image_detail: str = DEFAULT_IMAGE_DETAIL,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
    for img in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_to_data_url(img),
                    "detail": image_detail,
                },
            }
        )
    return [{"role": "user", "content": content}]


def generate_stroke_caption(
    client: Any,
    instruction: str,
    images: list[Image.Image],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    image_detail: str = DEFAULT_IMAGE_DETAIL,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    max_retries: int = 5,
) -> str:
    messages = build_stroke_messages(instruction, images, image_detail=image_detail)
    delay = 2.0
    last_empty: dict[str, Any] | None = None

    for effort in _reasoning_efforts(model, reasoning_effort):
        for budget in _completion_budgets(max_tokens):
            for attempt in range(max_retries):
                try:
                    resp = _chat_completion(
                        client,
                        model=model,
                        messages=messages,
                        max_tokens=budget,
                        timeout=timeout,
                        reasoning_effort=effort,
                    )
                    text = _message_text_or_empty(resp)
                    if text:
                        return text
                    choice = resp.choices[0]
                    usage = getattr(resp, "usage", None)
                    last_empty = {
                        "finish_reason": getattr(choice, "finish_reason", None),
                        "completion_tokens": getattr(usage, "completion_tokens", None),
                        "reasoning_effort": effort,
                        "max_tokens": budget,
                    }
                    break
                except Exception as e:
                    status = getattr(e, "status_code", None)
                    if status in (429, 503) and attempt + 1 < max_retries:
                        time.sleep(delay)
                        delay = min(delay * 2, 60.0)
                        continue
                    raise

    detail = last_empty or {}
    raise RuntimeError(
        "OpenAI returned empty message content after retries "
        f"(last attempt: {detail}). "
        "Try --reasoning_effort none and --max_completion_tokens 4096, or OPENAI_VLM_MODEL=gpt-4o."
    )
