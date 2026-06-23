"""OpenAI GPT multimodal defaults for IsoCourt VLM eval."""

import os

DEFAULT_MODEL = os.environ.get("OPENAI_VLM_MODEL", "gpt-5.5")
DEFAULT_IMAGE_DETAIL = "low"
DEFAULT_REQUEST_TIMEOUT = 120.0
# Reasoning models (gpt-5.x) can burn the whole budget on hidden reasoning tokens.
DEFAULT_MAX_COMPLETION_TOKENS = 512
DEFAULT_REASONING_EFFORT = os.environ.get("OPENAI_VLM_REASONING_EFFORT", "minimal")
