"""16-frame FineBadminton clip protocol for VLM train/eval (shared with OpenAI)."""

from __future__ import annotations

from core.label_maps import STROKE_TYPE_CLASSES
from core.training_standards import SEQUENCE_LENGTH

FRAME_SIZE = 224

_STROKE_CLASS_LIST = ", ".join(STROKE_TYPE_CLASSES)

# Legacy IsoCourt contact-frame / open-caption style (JSONL SFT, not stroke benchmark).
DEFAULT_16FRAME_INSTRUCTION = (
    "You are a badminton analyst. You are shown 16 frames uniformly sampled across "
    "a hitting motion (not a single contact frame). Using the images and any pose "
    "data provided, describe the stroke type, technique, player (name or top/bottom), "
    "court area, quality out of 5, and tactical intent. Answer in concise English."
)

# Research benchmark: 9-class stroke_type only (matches native Conv3D / JVC metric).
DEFAULT_16FRAME_STROKE_CLASSIFY_INSTRUCTION = (
    "You are a badminton stroke classifier. You are shown 16 frames uniformly sampled "
    "across a hitting motion (not a single contact frame), plus optional pose landmarks. "
    f"Classify the hitter's stroke into exactly ONE of these types: {_STROKE_CLASS_LIST}. "
    "Use only the clip and pose evidence. Do not invent player names or extra labels."
)

DEFAULT_POSE_PREAMBLE = (
    "The [Pose sequence] below lists MediaPipe BlazePose landmarks (normalized x, y, z) "
    "for each frame t=0..15. Use them together with the images."
)

DEFAULT_RESPONSE_FORMAT_HINT = (
    "Format your answer starting with: Stroke: <hit type> then Subtype:, Player:, "
    "Hitter side:, Ball area:, Quality:, etc."
)

STROKE_CLASSIFY_FORMAT_HINT = (
    "Reply with exactly one line and nothing else:\n"
    f"Stroke: <class>\n"
    f"where <class> is exactly one of: {_STROKE_CLASS_LIST}."
)

DEFAULT_CONTACT_INSTRUCTION = (
    "You are a badminton analyst. This frame is captured at shuttle contact. "
    "Describe the stroke type, technique, which player (name or top/bottom), "
    "court area, quality score out of 5, and any tactical intent. Answer in concise English."
)


def build_user_instruction(
    task_instruction: str,
    *,
    pose_text: str | None = None,
    include_format_hint: bool = True,
    format_hint: str | None = None,
) -> str:
    """Compose the user text block (pose + task + optional format hint)."""
    parts: list[str] = []
    if pose_text and pose_text.strip():
        parts.append(pose_text.strip())
        parts.append(DEFAULT_POSE_PREAMBLE)
    parts.append(task_instruction.strip())
    if include_format_hint:
        parts.append(format_hint or DEFAULT_RESPONSE_FORMAT_HINT)
    return "\n\n".join(parts)


def build_stroke_classify_instruction(
    *,
    pose_text: str | None = None,
    include_format_hint: bool = True,
) -> str:
    """User prompt for zero-shot / API stroke_type benchmark (9 classes only)."""
    return build_user_instruction(
        DEFAULT_16FRAME_STROKE_CLASSIFY_INSTRUCTION,
        pose_text=pose_text,
        include_format_hint=include_format_hint,
        format_hint=STROKE_CLASSIFY_FORMAT_HINT,
    )
