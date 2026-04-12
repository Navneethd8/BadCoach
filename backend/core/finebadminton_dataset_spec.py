"""
Canonical FineBadminton-20K layout for training metadata and API inference.

``prepare_finebadminton_20k.py`` writes merged labels under ``backend/data/`` and
expects videos under ``backend/data/FineBadminton-20K/videos/``. JPEGs for training
live under ``backend/data/FineBadminton-20K/dataset/image/`` (contact frames by default;
``--extract-all-frames`` fills every frame index per video).
"""
from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

# Registry / docs
DATASET_ID_FINEBADMINTON_20K = "finebadminton_20k"

# Frame policies (stored on registry ``dataset.frames``)
FRAMES_UNIFORM_16_IN_HIT_SPAN = "uniform_16_in_hit_span"
FRAMES_JPEG_ALL_VIDEO = "jpeg_all_video_frames"

DEFAULT_REGISTRY_DATASET: Dict[str, Any] = {
    "id": DATASET_ID_FINEBADMINTON_20K,
    "source": "Moujuruo/Finebadminton-20K",
    "list_file": os.path.join("data", "transformed_combined_rounds_output_en_evals_translated.json"),
    "data_root_relative": os.path.join("data", "FineBadminton-20K", "videos"),
    "frames": FRAMES_UNIFORM_16_IN_HIT_SPAN,
}


def backend_root_from_models_dir(models_dir: str) -> str:
    """``.../backend/models`` -> ``.../backend``."""
    return os.path.normpath(os.path.join(os.path.abspath(models_dir), ".."))


def merge_registry_dataset(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``meta`` with a filled ``dataset`` block (defaults + user overrides)."""
    out = dict(meta)
    base = deepcopy(DEFAULT_REGISTRY_DATASET)
    user = out.get("dataset")
    if isinstance(user, dict):
        base.update(user)
    out["dataset"] = base
    return out


def resolve_dataset_paths(backend_root: str, dataset: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Resolve ``data_root`` and ``list_file`` to absolute paths.

    Returns:
        (dataset_id, data_root_abs, list_file_abs)
    """
    ds_id = str(dataset.get("id") or DATASET_ID_FINEBADMINTON_20K)
    lf = dataset.get("list_file") or DEFAULT_REGISTRY_DATASET["list_file"]
    dr = dataset.get("data_root_relative") or DEFAULT_REGISTRY_DATASET["data_root_relative"]
    if not os.path.isabs(lf):
        lf = os.path.normpath(os.path.join(backend_root, lf))
    if not os.path.isabs(dr):
        dr = os.path.normpath(os.path.join(backend_root, dr))
    return ds_id, dr, lf


def resolve_inference_dataset_layout(
    models_dir: str,
    registry_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Paths and flags for API startup (FineBadminton-20K by default).

    ``registry_meta`` should be the flattened primary metadata for the loaded
    checkpoint (from ``registry_meta_for_checkpoint``).
    """
    backend_root = backend_root_from_models_dir(models_dir)
    merged = merge_registry_dataset(dict(registry_meta or {}))
    ds = merged["dataset"]
    ds_id, data_root, list_file = resolve_dataset_paths(backend_root, ds)
    layout = {
        "dataset_id": ds_id,
        "data_root": data_root,
        "list_file": list_file,
        "frames": ds.get("frames", FRAMES_UNIFORM_16_IN_HIT_SPAN),
        "source": ds.get("source", DEFAULT_REGISTRY_DATASET["source"]),
    }
    # Optional prod overrides (paths absolute or relative to cwd)
    ev_root = os.environ.get("ISOCOURT_INFERENCE_DATA_ROOT", "").strip()
    if ev_root:
        layout["data_root"] = os.path.normpath(os.path.expanduser(ev_root))
    ev_list = os.environ.get("ISOCOURT_INFERENCE_LIST_FILE", "").strip()
    if ev_list:
        layout["list_file"] = os.path.normpath(os.path.expanduser(ev_list))
    ev_id = os.environ.get("ISOCOURT_INFERENCE_DATASET_ID", "").strip()
    if ev_id:
        layout["dataset_id"] = ev_id
    return layout
