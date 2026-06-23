"""OpenAI GPT multimodal defaults for IsoCourt VLM eval."""

import os

DEFAULT_MODEL = os.environ.get("OPENAI_VLM_MODEL", "gpt-5.5")
DEFAULT_IMAGE_DETAIL = "low"
DEFAULT_REQUEST_TIMEOUT = 120.0
# gpt-5.x may spend the whole budget on hidden reasoning before visible text.
DEFAULT_MAX_COMPLETION_TOKENS = 2048
DEFAULT_REASONING_EFFORT = os.environ.get("OPENAI_VLM_REASONING_EFFORT", "none")
