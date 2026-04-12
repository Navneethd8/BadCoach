"""
Stroke model registry (training catalog) + inference selection (prod switch).

Training writes ``model_registry.json`` under ``backend/models/`` using the v2 layout:
``models.architectures.<category>`` with a single ``primary`` and a list of
``registrations`` (experiments). Non-experiment saves overwrite ``primary``;
experiment saves append without touching ``primary``.

Inference uses *only* ``primary`` for the active category (see ``resolve_inference_model_path``).
Which category is active is chosen via CLI / ``inference_selection.json`` / env — not by mutating
the training registry during normal training.
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, List, MutableMapping, Optional, Tuple

from core.finebadminton_dataset_spec import merge_registry_dataset

SCHEMA_VERSION = 2
INFERENCE_SELECTION_FILENAME = "inference_selection.json"
ENV_INFERENCE_CATEGORY = "ISOCOURT_INFERENCE_CATEGORY"

# Keys under models.architectures.* (CLI + training must use these).
ARCHITECTURE_CATEGORIES: Tuple[str, ...] = (
    "cnn_lstm",
    "conv3d_pose",
    "staeformer",
    "timesformer",
    "videomae_pose",
    "videomae_timesformer",
    "vit_gcn",
)

# Optional grouping for ``list`` output (research families).
CATEGORY_GROUPS: Dict[str, str] = {
    "cnn_lstm": "cnn",
    "conv3d_pose": "video_cnn",
    "staeformer": "graph",
    "vit_gcn": "graph",
    "timesformer": "video_transformer",
    "videomae_pose": "video_transformer",
    "videomae_timesformer": "hybrid",
}

SCRIPT_TO_CATEGORY: Dict[str, str] = {
    "train_full.py": "cnn_lstm",
    "train_conv3d.py": "conv3d_pose",
    "train_staeformer.py": "staeformer",
    "train_timesformer.py": "timesformer",
    "train_videomae.py": "videomae_pose",
    "train_videomae_timesformer.py": "videomae_timesformer",
    "train_vit_gcn.py": "vit_gcn",
}

# When nothing is configured, prefer first hit (API-safe: skip staeformer primary).
DEFAULT_CATEGORY_FALLBACK_ORDER: Tuple[str, ...] = (
    "videomae_timesformer",
    "timesformer",
    "cnn_lstm",
    "conv3d_pose",
    "videomae_pose",
    "vit_gcn",
    "staeformer",
)


def _empty_slot() -> Dict[str, Any]:
    return {"primary": None, "registrations": []}


def default_registry_v2() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "models": {"architectures": {k: _empty_slot() for k in ARCHITECTURE_CATEGORIES}},
    }


def _is_v2_layout(data: Dict[str, Any]) -> bool:
    if data.get("schema_version") != SCHEMA_VERSION:
        return False
    m = data.get("models")
    if not isinstance(m, dict):
        return False
    arch = m.get("architectures")
    return isinstance(arch, dict)


def infer_category_from_meta(filename: str, meta: Dict[str, Any]) -> str:
    arch = (meta.get("architecture") or "").strip().lower().replace("-", "_")
    if arch in ARCHITECTURE_CATEGORIES:
        return arch
    script = meta.get("script") or ""
    if script in SCRIPT_TO_CATEGORY:
        return SCRIPT_TO_CATEGORY[script]
    fn = filename.lower()
    if "videomae" in fn and "timesformer" in fn:
        return "videomae_timesformer"
    if "videomae" in fn:
        return "videomae_pose"
    if "conv3d" in fn:
        return "conv3d_pose"
    if "timesformer" in fn:
        return "timesformer"
    if "vit_gcn" in fn or "vitgcn" in fn:
        return "vit_gcn"
    if "staeformer" in fn:
        return "staeformer"
    return "cnn_lstm"


def migrate_v1_to_v2(v1: Dict[str, Any]) -> Dict[str, Any]:
    out = default_registry_v2()
    arch = out["models"]["architectures"]
    flat: Dict[str, Any] = dict(v1.get("models") or {})
    active = v1.get("active_model")

    for fname, meta in flat.items():
        if not isinstance(meta, dict):
            meta = {}
        cat = infer_category_from_meta(fname, meta)
        entry = merge_registry_dataset({**deepcopy(meta), "file": fname, "experiment": True})
        if fname == active:
            entry["experiment"] = False
            arch[cat]["primary"] = entry
        else:
            arch[cat]["registrations"].append(entry)

    # v1 migration: lone experiment per category becomes primary so CLI category switches work.
    for _cat, slot in arch.items():
        if slot.get("primary") is not None:
            continue
        regs = list(slot.get("registrations") or [])
        if len(regs) == 1:
            slot["primary"] = regs.pop()
            if isinstance(slot["primary"], dict):
                slot["primary"]["experiment"] = False
            slot["registrations"] = regs

    return out


def ensure_all_entries_have_dataset(reg: Dict[str, Any]) -> Dict[str, Any]:
    """Attach default FineBadminton-20K ``dataset`` metadata to every checkpoint entry."""
    reg = normalize_registry(reg) if not _is_v2_layout(reg) else reg
    if not _is_v2_layout(reg):
        return reg
    arch = reg["models"]["architectures"]
    for _cat, slot in arch.items():
        if not isinstance(slot, dict):
            continue
        p = slot.get("primary")
        if isinstance(p, dict) and p.get("file"):
            slot["primary"] = merge_registry_dataset(p)
        regs = slot.get("registrations") or []
        for i, r in enumerate(list(regs)):
            if isinstance(r, dict) and r.get("file"):
                regs[i] = merge_registry_dataset(r)
        slot["registrations"] = regs
    return reg


def normalize_registry(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a v2 registry dict (mutates nothing if already v2 — caller may copy)."""
    if not data:
        return default_registry_v2()
    if _is_v2_layout(data):
        merged = default_registry_v2()
        src_arch = data["models"]["architectures"]
        for k in ARCHITECTURE_CATEGORIES:
            slot = src_arch.get(k)
            if isinstance(slot, dict):
                merged["models"]["architectures"][k] = {
                    "primary": slot.get("primary"),
                    "registrations": list(slot.get("registrations") or []),
                }
        merged["schema_version"] = SCHEMA_VERSION
        return ensure_all_entries_have_dataset(merged)
    return migrate_v1_to_v2(data)


def load_registry_file(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return default_registry_v2()
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return default_registry_v2()
    return normalize_registry(raw)


def save_registry_file(path: str, registry: Dict[str, Any]) -> None:
    reg = ensure_all_entries_have_dataset(normalize_registry(registry))
    reg["schema_version"] = SCHEMA_VERSION
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)


def _iter_indexed_entries(
    registry: Dict[str, Any],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Yield (category, role, entry) for primary and registrations."""
    reg = normalize_registry(registry)
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    arch = reg["models"]["architectures"]
    for cat, slot in arch.items():
        if not isinstance(slot, dict):
            continue
        p = slot.get("primary")
        if isinstance(p, dict) and p.get("file"):
            out.append((cat, "primary", p))
        for i, r in enumerate(slot.get("registrations") or []):
            if isinstance(r, dict) and r.get("file"):
                out.append((cat, f"experiment[{i}]", r))
    return out


def flatten_models_by_file(registry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """basename -> merged metadata (for model_loader)."""
    by_file: Dict[str, Dict[str, Any]] = {}
    for cat, _role, entry in _iter_indexed_entries(registry):
        fname = entry["file"]
        base = os.path.basename(fname)
        meta = {k: v for k, v in entry.items() if k != "file"}
        meta.setdefault("architecture", cat if cat in ARCHITECTURE_CATEGORIES else infer_category_from_meta(base, meta))
        by_file[base] = meta
    return by_file


def registry_meta_for_checkpoint(registry: Dict[str, Any], model_path: str) -> Dict[str, Any]:
    name = os.path.basename(model_path)
    flat = flatten_models_by_file(registry)
    return merge_registry_dataset(dict(flat.get(name, {})))


def inference_selection_path(models_dir: str) -> str:
    return os.path.join(models_dir, INFERENCE_SELECTION_FILENAME)


def load_inference_selection(models_dir: str) -> Dict[str, Any]:
    p = inference_selection_path(models_dir)
    if not os.path.isfile(p):
        return {}
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, dict) else {}


def save_inference_selection(models_dir: str, category: str) -> None:
    if category not in ARCHITECTURE_CATEGORIES:
        raise ValueError(f"Unknown category {category!r}. Expected one of: {ARCHITECTURE_CATEGORIES}")
    p = inference_selection_path(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"category": category}, f, indent=2)


def resolve_inference_category(models_dir: str, registry: Dict[str, Any]) -> Optional[str]:
    env = (os.environ.get(ENV_INFERENCE_CATEGORY) or "").strip()
    if env:
        if env not in ARCHITECTURE_CATEGORIES:
            raise ValueError(
                f"{ENV_INFERENCE_CATEGORY}={env!r} is not a valid category. "
                f"Choose one of: {', '.join(ARCHITECTURE_CATEGORIES)}"
            )
        return env
    sel = load_inference_selection(models_dir)
    cat = (sel.get("category") or "").strip()
    if cat:
        if cat not in ARCHITECTURE_CATEGORIES:
            raise ValueError(
                f"inference_selection.json category {cat!r} invalid. "
                f"Expected one of: {', '.join(ARCHITECTURE_CATEGORIES)}"
            )
        return cat
    # Legacy: v1 active_model basename -> infer category
    reg = normalize_registry(registry)
    legacy_active = registry.get("active_model") if isinstance(registry, dict) else None
    if legacy_active and isinstance(legacy_active, str):
        flat_old = registry.get("models") if isinstance(registry.get("models"), dict) else {}
        meta = flat_old.get(legacy_active) if isinstance(flat_old, dict) else {}
        if isinstance(meta, dict):
            return infer_category_from_meta(legacy_active, meta)
    reg_n = normalize_registry(registry)
    arch = reg_n["models"]["architectures"]
    for c in DEFAULT_CATEGORY_FALLBACK_ORDER:
        slot = arch.get(c) or {}
        prim = slot.get("primary")
        if isinstance(prim, dict) and prim.get("file"):
            return c
    for c in ARCHITECTURE_CATEGORIES:
        slot = arch.get(c) or {}
        prim = slot.get("primary")
        if isinstance(prim, dict) and prim.get("file"):
            return c
    return None


def primary_entry_for_category(registry: Dict[str, Any], category: str) -> Optional[Dict[str, Any]]:
    reg = normalize_registry(registry)
    if category not in ARCHITECTURE_CATEGORIES:
        return None
    slot = reg["models"]["architectures"].get(category) or {}
    p = slot.get("primary")
    return p if isinstance(p, dict) else None


def resolve_primary_checkpoint_path(models_dir: str, registry: Dict[str, Any], category: str) -> Optional[str]:
    p = primary_entry_for_category(registry, category)
    if not p or not p.get("file"):
        return None
    rel = p["file"]
    cand = rel if os.path.isabs(rel) else os.path.join(models_dir, rel)
    return cand if os.path.isfile(cand) else None


def resolve_inference_model_path(models_dir: str, registry: Dict[str, Any]) -> Optional[str]:
    """
    Path to the checkpoint used for API inference: always ``primary`` for the resolved category.
    """
    reg = normalize_registry(registry)
    cat = resolve_inference_category(models_dir, registry)
    if not cat:
        return None
    path = resolve_primary_checkpoint_path(models_dir, reg, cat)
    return path


def register_training_checkpoint(
    models_dir: str,
    *,
    category: str,
    file_basename: str,
    meta: Dict[str, Any],
    experiment: bool,
) -> None:
    """
    Update ``model_registry.json`` after a training save.

    If ``experiment`` is False, sets ``architectures[category].primary`` (overwrites).
    If True, appends to ``registrations`` (never replaces primary).
    """
    if category not in ARCHITECTURE_CATEGORIES:
        raise ValueError(f"Invalid category {category!r}")
    reg_path = os.path.join(models_dir, "model_registry.json")
    if os.path.isfile(reg_path):
        with open(reg_path, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = {}
    reg = normalize_registry(raw if isinstance(raw, dict) else {})
    arch = reg["models"]["architectures"]
    slot = arch[category]
    entry = merge_registry_dataset({**deepcopy(meta), "file": file_basename})
    entry["experiment"] = bool(experiment)
    if experiment:
        # Dedupe by file basename
        regs: List[Dict[str, Any]] = list(slot.get("registrations") or [])
        regs = [r for r in regs if isinstance(r, dict) and os.path.basename(r.get("file", "")) != file_basename]
        regs.append(entry)
        slot["registrations"] = regs
    else:
        entry["experiment"] = False
        slot["primary"] = entry
    save_registry_file(reg_path, reg)


def promote_experiment_to_primary(models_dir: str, category: str, index: int) -> None:
    """Move ``registrations[index]`` to ``primary``; old primary becomes an experiment registration."""
    if category not in ARCHITECTURE_CATEGORIES:
        raise ValueError(f"Invalid category {category!r}")
    reg_path = os.path.join(models_dir, "model_registry.json")
    reg = load_registry_file(reg_path)
    slot = reg["models"]["architectures"][category]
    regs: List[Dict[str, Any]] = list(slot.get("registrations") or [])
    if index < 0 or index >= len(regs):
        raise IndexError(f"No experiment at index {index} for category {category!r}")
    new_primary = deepcopy(regs.pop(index))
    new_primary["experiment"] = False
    old_p = slot.get("primary")
    if isinstance(old_p, dict) and old_p.get("file"):
        old_copy = deepcopy(old_p)
        old_copy["experiment"] = True
        regs.insert(0, old_copy)
    slot["primary"] = new_primary
    slot["registrations"] = regs
    save_registry_file(reg_path, reg)


def promote_experiment_by_filename(models_dir: str, category: str, filename: str) -> None:
    reg = load_registry_file(os.path.join(models_dir, "model_registry.json"))
    slot = reg["models"]["architectures"][category]
    regs = list(slot.get("registrations") or [])
    base = os.path.basename(filename)
    for i, r in enumerate(regs):
        if isinstance(r, dict) and os.path.basename(r.get("file", "")) == base:
            promote_experiment_to_primary(models_dir, category, i)
            return
    raise ValueError(f"No experiment registration with file {filename!r} under {category!r}")
