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
    effort_attempts: list[str | None] = [None]
    if reasoning_effort and _model_uses_reasoning(model):
        effort_attempts = [reasoning_effort, "low", "none", None]

    last_err: Exception | None = None
    for effort in effort_attempts:
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
            if effort is not None:
                kwargs["reasoning_effort"] = effort
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as e:
                if _is_unsupported_token_limit_param(e):
                    last_err = e
                    continue
                if effort is not None and _is_unsupported_reasoning_effort(e):
                    last_err = e
                    break
                raise
    assert last_err is not None
    raise last_err


def _extract_message_text(resp: Any) -> str:
    choice = resp.choices[0]
    msg = choice.message
    text = (getattr(msg, "content", None) or "").strip()
    if text:
        return text
    refusal = getattr(msg, "refusal", None)
    if refusal:
        return str(refusal).strip()
    finish = getattr(choice, "finish_reason", None)
    usage = getattr(resp, "usage", None)
    completion = getattr(usage, "completion_tokens", None) if usage else None
    raise RuntimeError(
        "OpenAI returned empty message content "
        f"(finish_reason={finish!r}, completion_tokens={completion!r}). "
        "For gpt-5.x, use reasoning_effort=minimal and a larger max_completion_tokens."
    )


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
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = _chat_completion(
                client,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                timeout=timeout,
                reasoning_effort=reasoning_effort,
            )
            return _extract_message_text(resp)
        except Exception as e:
            last_err = e
            status = getattr(e, "status_code", None)
            if status in (429, 503) and attempt + 1 < max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            raise
    raise RuntimeError(f"OpenAI request failed after retries: {last_err}")
