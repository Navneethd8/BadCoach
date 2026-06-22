"""16-frame FineBadminton clip protocol for VLM train/eval (shared with OpenAI)."""

from __future__ import annotations

from core.training_standards import SEQUENCE_LENGTH

FRAME_SIZE = 224

DEFAULT_16FRAME_INSTRUCTION = (
    "You are a badminton analyst. You are shown 16 frames uniformly sampled across "
    "a hitting motion (not a single contact frame). Using the images and any pose "
    "data provided, describe the stroke type, technique, player (name or top/bottom), "
    "court area, quality out of 5, and tactical intent. Answer in concise English."
)

DEFAULT_POSE_PREAMBLE = (
    "The [Pose sequence] below lists MediaPipe BlazePose landmarks (normalized x, y, z) "
    "for each frame t=0..15. Use them together with the images."
)

DEFAULT_RESPONSE_FORMAT_HINT = (
    "Format your answer starting with: Stroke: <hit type> then Subtype:, Player:, "
    "Hitter side:, Ball area:, Quality:, etc."
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
) -> str:
    """Compose the user text block (pose + task + optional format hint)."""
    parts: list[str] = []
    if pose_text and pose_text.strip():
        parts.append(pose_text.strip())
        parts.append(DEFAULT_POSE_PREAMBLE)
    parts.append(task_instruction.strip())
    if include_format_hint:
        parts.append(DEFAULT_RESPONSE_FORMAT_HINT)
    return "\n\n".join(parts)
