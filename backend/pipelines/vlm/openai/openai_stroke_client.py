"""Chat Completions client for 16-frame stroke eval."""

from __future__ import annotations

import base64
import io
import time
from typing import Any

from PIL import Image

from openai_vlm_config import DEFAULT_IMAGE_DETAIL, DEFAULT_MODEL, DEFAULT_REQUEST_TIMEOUT


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
    max_tokens: int = 256,
    image_detail: str = DEFAULT_IMAGE_DETAIL,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
    max_retries: int = 5,
) -> str:
    messages = build_stroke_messages(instruction, images, image_detail=image_detail)
    delay = 2.0
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            status = getattr(e, "status_code", None)
            if status in (429, 503) and attempt + 1 < max_retries:
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            raise
    raise RuntimeError(f"OpenAI request failed after retries: {last_err}")
