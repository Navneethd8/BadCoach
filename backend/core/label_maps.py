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
    """Map raw annotation ``hit_type``, free text, or 9-class name to ``stroke_type``."""
    raw = (hit_type or "").strip()
    if not raw:
        return "Other"
    mapped = HIT_TYPE_TO_STROKE_TYPE.get(raw.lower())
    if mapped is not None:
        return mapped
    lowered = raw.lower()
    compact = lowered.replace("_", " ")
    for cls in STROKE_TYPE_CLASSES:
        if cls.lower() == lowered or cls.lower().replace("_", " ") == compact:
            return cls
    # e.g. "Backhand Clear" / "Forehand Smash" -> Clear / Smash
    best: str | None = None
    best_len = 0
    for cls in STROKE_TYPE_CLASSES:
        for form in (cls.lower(), cls.lower().replace("_", " ")):
            if len(form) < 3:
                continue
            if form in lowered and len(form) > best_len:
                best = cls
                best_len = len(form)
    return best if best is not None else "Other"


def stroke_class_to_index(name: str) -> int:
    normalized = (name or "").strip()
    for i, cls in enumerate(STROKE_TYPE_CLASSES):
        if cls.lower() == normalized.lower():
            return i
    return STROKE_TYPE_CLASSES.index("Other")


def map_hit_type_to_stroke_index(hit_type: str) -> int:
    return stroke_class_to_index(map_hit_type_to_stroke_class(hit_type))
