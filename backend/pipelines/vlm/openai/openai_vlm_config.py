"""OpenAI GPT multimodal defaults for IsoCourt VLM eval."""

import os

DEFAULT_MODEL = os.environ.get("OPENAI_VLM_MODEL", "gpt-5.5")
DEFAULT_IMAGE_DETAIL = "low"
DEFAULT_REQUEST_TIMEOUT = 120.0
