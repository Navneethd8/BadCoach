"""
Load the stroke model from disk using checkpoint metadata + model_registry.json.

Checkpoints from training may be a raw state_dict (CNN-LSTM) or a dict envelope with
``model``, ``architecture``, ``task_classes``, and constructor hints. Registry entries
can override or supply ``architecture`` / ``inference`` when older checkpoints omit fields.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from core.model import CNN_LSTM_Model

ARCH_CNN_LSTM = "cnn_lstm"
ARCH_TIMESFORMER = "timesformer"
ARCH_VIDEOMAE_POSE = "videomae_pose"
ARCH_VIDEOMAE_TIMESFORMER = "videomae_timesformer"
ARCH_STAE = "staeformer"

_SCRIPT_TO_ARCH = {
    "train_full.py": ARCH_CNN_LSTM,
    "train_timesformer.py": ARCH_TIMESFORMER,
    "train_videomae.py": ARCH_VIDEOMAE_POSE,
    "train_videomae_timesformer.py": ARCH_VIDEOMAE_TIMESFORMER,
    "train_staeformer.py": ARCH_STAE,
}


def split_checkpoint(raw: Any) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        meta = {k: v for k, v in raw.items() if k != "model"}
        return meta, raw["model"]
    if isinstance(raw, dict):
        return {}, raw
    raise TypeError(f"Unexpected checkpoint type: {type(raw)}")


def _merge_inference(registry_meta: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(registry_meta.get("inference") or {})
    for k in ("architecture", "hf_model_id", "embed_dim", "depth", "num_heads"):
        if k in registry_meta and k not in out:
            out[k] = registry_meta[k]
    return out


def resolve_architecture(
    ckpt_meta: Dict[str, Any],
    registry_meta: Dict[str, Any],
    filename: str,
) -> str:
    for src in (ckpt_meta, _merge_inference(registry_meta), registry_meta):
        a = src.get("architecture")
        if a:
            return str(a).lower().replace("-", "_")
    script = registry_meta.get("script")
    if script in _SCRIPT_TO_ARCH:
        return _SCRIPT_TO_ARCH[script]
    fn = filename.lower()
    if "videomae" in fn and "timesformer" in fn:
        return ARCH_VIDEOMAE_TIMESFORMER
    if "videomae" in fn:
        return ARCH_VIDEOMAE_POSE
    if "timesformer" in fn:
        return ARCH_TIMESFORMER
    if "staeformer" in fn and "timesformer" not in fn:
        return ARCH_STAE
    return ARCH_CNN_LSTM


def build_model(
    arch: str,
    task_classes: Dict[str, int],
    ckpt_meta: Dict[str, Any],
    registry_meta: Dict[str, Any],
) -> nn.Module:
    inf = _merge_inference(registry_meta)

    def _i(key: str, default: Any = None) -> Any:
        if key in ckpt_meta:
            return ckpt_meta[key]
        if key in inf:
            return inf[key]
        return default

    if arch == ARCH_STAE:
        raise RuntimeError(
            "STAEformer checkpoints are not supported by the /analyze API "
            "(they need per-frame CNN features). Switch active_model to a CNN-LSTM, "
            "TimeSformer, or VideoMAE checkpoint."
        )

    if arch == ARCH_CNN_LSTM:
        hidden = int(_i("hidden_size", registry_meta.get("hidden_size", 128)))
        use_pose = bool(_i("use_pose", False))
        return CNN_LSTM_Model(task_classes=task_classes, hidden_size=hidden, pretrained=False, use_pose=use_pose)

    if arch == ARCH_TIMESFORMER:
        from core.timesformer import TimeSformerPoseModel

        return TimeSformerPoseModel(
            task_classes=task_classes,
            img_size=224,
            patch_size=16,
            num_frames=int(_i("num_frames", 16)),
            embed_dim=int(_i("embed_dim", 128)),
            num_heads=int(_i("num_heads", 4)),
            depth=int(_i("depth", 4)),
            backbone=str(_i("backbone", "vit")),
            vit_model_name=str(_i("vit_model_name", "vit_small_patch16_224")),
            vit_unfreeze_last_n=int(_i("vit_unfreeze_last_n", 0)),
        )

    if arch == ARCH_VIDEOMAE_POSE:
        from core.videomae_pose import VideoMAEPoseModel

        return VideoMAEPoseModel(
            task_classes=task_classes,
            hf_model_id=str(_i("hf_model_id", "MCG-NJU/videomae-base")),
            num_frames=int(_i("num_frames", 16)),
            freeze_backbone=bool(_i("freeze_videomae", _i("freeze_backbone", True))),
            unfreeze_last_n=int(_i("videomae_unfreeze_last_n", _i("unfreeze_last_n", 0))),
        )

    if arch == ARCH_VIDEOMAE_TIMESFORMER:
        from core.videomae_timesformer import VideoMAETimeSformerPoseModel

        return VideoMAETimeSformerPoseModel(
            task_classes=task_classes,
            hf_model_id=str(_i("hf_model_id", "MCG-NJU/videomae-base")),
            num_frames=int(_i("num_frames", 16)),
            embed_dim=int(_i("embed_dim", 128)),
            num_heads=int(_i("num_heads", 4)),
            depth=int(_i("depth", 4)),
            freeze_videomae=bool(_i("freeze_videomae", True)),
            videomae_unfreeze_last_n=int(_i("videomae_unfreeze_last_n", 0)),
        )

    raise ValueError(f"Unknown architecture {arch!r}")


def load_stroke_model(
    model_path: str,
    task_classes: Dict[str, int],
    registry: Dict[str, Any],
    device: str,
) -> Tuple[nn.Module, str]:
    raw = torch.load(model_path, map_location=device, weights_only=False)
    ckpt_meta, state_dict = split_checkpoint(raw)
    name = os.path.basename(model_path)
    active = registry.get("active_model")
    reg_models = registry.get("models") or {}
    registry_meta = reg_models.get(name, {}) if name in reg_models else {}

    arch = resolve_architecture(ckpt_meta, registry_meta, name)

    if ckpt_meta.get("task_classes"):
        tc = ckpt_meta["task_classes"]
        if isinstance(tc, dict):
            task_classes = {k: int(v) for k, v in tc.items()}

    model = build_model(arch, task_classes, ckpt_meta, registry_meta)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING: Missing keys (partial load / random init for those layers): {len(missing)} keys")
    if unexpected:
        print(f"WARNING: Unexpected keys ignored: {len(unexpected)} keys")

    return model, arch


def load_registry(models_dir: str) -> Dict[str, Any]:
    path = os.path.join(models_dir, "model_registry.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_model_path(models_dir: str, registry: Dict[str, Any]) -> str | None:
    """Prefer active_model; else highest-accuracy non-STAEformer file (dev / multi-weight trees)."""
    active = registry.get("active_model")
    if active:
        p = os.path.join(models_dir, active)
        if os.path.isfile(p):
            return p

    best_name, best_acc = None, -1.0
    for name, meta in (registry.get("models") or {}).items():
        nl = name.lower()
        if "staeformer" in nl and "timesformer" not in nl:
            continue
        path = os.path.join(models_dir, name)
        if not os.path.isfile(path):
            continue
        acc = float(meta.get("accuracy", 0.0))
        if acc > best_acc:
            best_acc = acc
            best_name = name
    if best_name:
        return os.path.join(models_dir, best_name)

    fallback = os.path.join(models_dir, "badminton_model.pth")
    return fallback if os.path.isfile(fallback) else None
