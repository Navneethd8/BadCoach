"""Shared FineBadminton label mappings (stroke trainers + VLM eval)."""

from __future__ import annotations

STROKE_TYPE_CLASSES: tuple[str, ...] = (
    "Serve",
    "Clear",
    "Smash",
    "Drop",
    "Drive",
    "Net_Shot",
    "Lob",
    "Defensive_Shot",
    "Other",
)

HIT_TYPE_TO_STROKE_TYPE: dict[str, str] = {
    "serve": "Serve",
    "clear": "Clear",
    "smash": "Smash",
    "kill": "Smash",
    "net kill": "Smash",
    "drop": "Drop",
    "drop shot": "Drop",
    "drive": "Drive",
    "net shot": "Net_Shot",
    "cross-court net shot": "Net_Shot",
    "lob": "Lob",
    "push shot": "Lob",
    "net lift": "Lob",
    "block": "Defensive_Shot",
    "defensive shot": "Defensive_Shot",
}


def map_hit_type_to_stroke_class(hit_type: str) -> str:
    """Map raw annotation ``hit_type`` to 9-class ``stroke_type`` name."""
    return HIT_TYPE_TO_STROKE_TYPE.get((hit_type or "").lower(), "Other")


def stroke_class_to_index(name: str) -> int:
    normalized = (name or "").strip()
    for i, cls in enumerate(STROKE_TYPE_CLASSES):
        if cls.lower() == normalized.lower():
            return i
    return STROKE_TYPE_CLASSES.index("Other")


def map_hit_type_to_stroke_index(hit_type: str) -> int:
    return stroke_class_to_index(map_hit_type_to_stroke_class(hit_type))
