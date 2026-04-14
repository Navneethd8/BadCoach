#!/usr/bin/env python3
"""
Quick local smoke test: verify every training pipeline can import, load the
dataset, build the model, and survive one forward+backward pass.

Run from repo root:
    python backend/pipelines/training/colab/smoke_test.py

Requires the merged JSON + extracted images on disk (same as training).
Does NOT run a full epoch — typically finishes in 1–3 minutes.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import torch

DATA_ROOT = os.path.join(_BACKEND, "data")
LIST_FILE = os.path.join(_BACKEND, "data", "transformed_combined_rounds_output_en_evals_translated.json")
POSE_CACHE = os.path.join(_BACKEND, "models", "pose_cache_mediapipe.pt")

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

results: list[tuple[str, str, float]] = []


def _smoke(name: str, fn):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        fn()
        dt = time.time() - t0
        results.append((name, "PASS", dt))
        print(f"  -> PASS ({dt:.1f}s)")
    except Exception:
        dt = time.time() - t0
        results.append((name, "FAIL", dt))
        traceback.print_exc()
        print(f"  -> FAIL ({dt:.1f}s)")


def _test_cnn_lstm():
    from core.dataset import FineBadmintonDataset
    from core.model import CNN_LSTM_Model

    ds = FineBadmintonDataset(DATA_ROOT, LIST_FILE, transform=None)
    assert len(ds) > 0, f"Dataset empty (LIST_FILE={LIST_FILE})"
    task_classes = {k: len(v) for k, v in ds.classes.items()}
    task_classes["quality"] = 7
    del task_classes["stroke_subtype"]

    model = CNN_LSTM_Model(task_classes=task_classes, hidden_size=128, use_pose=False).to(DEVICE)
    frames, labels = ds[0]
    if isinstance(frames, list):
        frames = torch.stack(frames)
    x = frames.unsqueeze(0).to(DEVICE)
    out = model(x)
    loss = sum(o.sum() for o in out.values())
    loss.backward()
    print(f"  dataset={len(ds)} samples, forward OK, heads={list(out.keys())}")


def _test_conv3d():
    from core.dataset import FineBadmintonDataset
    from core.conv3d_pose import Conv3DPoseMultitaskModel

    ds = FineBadmintonDataset(DATA_ROOT, LIST_FILE, transform=None)
    task_classes = {k: len(v) for k, v in ds.classes.items()}
    task_classes["quality"] = 7
    if "stroke_subtype" in task_classes:
        del task_classes["stroke_subtype"]

    model = Conv3DPoseMultitaskModel(
        task_classes=task_classes, video_backbone="r2plus1d_18",
        spatial_size=112, pretrained=False, use_pose=False,
    ).to(DEVICE)
    B, T = 1, 16
    x = torch.randn(B, T, 3, 112, 112, device=DEVICE)
    out = model(x)
    loss = sum(o.sum() for o in out.values())
    loss.backward()
    print(f"  forward OK, heads={list(out.keys())}")


def _test_staeformer():
    from core.staeformer import STAEformerModel

    task_classes = {"stroke_type": 9, "technique": 3, "placement": 10, "position": 10, "intent": 10, "quality": 7}
    model = STAEformerModel(task_classes=task_classes, use_cnn=False).to(DEVICE)
    B, T = 1, 16
    pose = torch.randn(B, T, 33, 3, device=DEVICE)
    out = model(pose)
    loss = sum(o.sum() for o in out.values())
    loss.backward()
    print(f"  forward OK (pose-only), heads={list(out.keys())}")


def _test_timesformer():
    from core.timesformer import TimeSformerPoseModel

    task_classes = {"stroke_type": 9, "technique": 3, "placement": 10, "position": 10, "intent": 10, "quality": 7}
    model = TimeSformerPoseModel(
        task_classes=task_classes, embed_dim=64, depth=2, num_heads=4,
        num_frames=16, backbone="scratch", use_pose=False,
    ).to(DEVICE)
    B, T = 1, 16
    x = torch.randn(B, T, 3, 224, 224, device=DEVICE)
    out = model(x)
    loss = sum(o.sum() for o in out.values())
    loss.backward()
    print(f"  forward OK, heads={list(out.keys())}")


def _test_videomae():
    from core.videomae_pose import VideoMAEPoseModel

    task_classes = {"stroke_type": 9, "technique": 3, "placement": 10, "position": 10, "intent": 10, "quality": 7}
    model = VideoMAEPoseModel(
        task_classes=task_classes, hf_model_id="MCG-NJU/videomae-base",
        freeze_backbone=True, use_pose=False,
    ).to(DEVICE)
    B, T = 1, 16
    x = torch.randn(B, T, 3, 224, 224, device=DEVICE)
    out = model(x)
    loss = sum(o.sum() for o in out.values())
    loss.backward()
    print(f"  forward OK, heads={list(out.keys())}")


def _test_videomae_timesformer():
    from core.videomae_timesformer import VideoMAETimeSformerPoseModel

    task_classes = {"stroke_type": 9, "technique": 3, "placement": 10, "position": 10, "intent": 10, "quality": 7}
    model = VideoMAETimeSformerPoseModel(
        task_classes=task_classes, hf_model_id="MCG-NJU/videomae-base",
        freeze_videomae=True, embed_dim=64, depth=2, num_heads=4, use_pose=False,
    ).to(DEVICE)
    B, T = 1, 16
    x = torch.randn(B, T, 3, 224, 224, device=DEVICE)
    out = model(x)
    loss = sum(o.sum() for o in out.values())
    loss.backward()
    print(f"  forward OK, heads={list(out.keys())}")


def _test_vit_gcn():
    from core.vit_gcn import ViTGCNMultitaskModel

    task_classes = {"stroke_type": 9, "technique": 3, "placement": 10, "position": 10, "intent": 10, "quality": 7}
    model = ViTGCNMultitaskModel(
        task_classes=task_classes, embed_dim=64, gcn_layers=2,
        vit_model_name="vit_tiny_patch16_224", use_pose=False,
    ).to(DEVICE)
    B, T = 1, 16
    x = torch.randn(B, T, 3, 224, 224, device=DEVICE)
    out = model(x)
    loss = sum(o.sum() for o in out.values())
    loss.backward()
    print(f"  forward OK, heads={list(out.keys())}")


def _test_st_tr():
    from core.st_tr import STTRModel

    task_classes = {"stroke_type": 9, "technique": 3, "placement": 10, "position": 10, "intent": 10, "quality": 7}
    model = STTRModel(
        task_classes=task_classes, embed_dim=64, num_heads=4, num_layers=2, fusion="concat",
    ).to(DEVICE)
    B, T = 1, 16
    pose = torch.randn(B, T, 33, 3, device=DEVICE)
    out = model(pose)
    loss = sum(o.sum() for o in out.values())
    loss.backward()
    print(f"  forward OK, heads={list(out.keys())}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Data root: {DATA_ROOT}")
    print(f"List file: {LIST_FILE}")
    assert os.path.isfile(LIST_FILE), f"Missing {LIST_FILE}"

    _smoke("cnn_lstm      (ResNet50+LSTM)", _test_cnn_lstm)
    _smoke("conv3d_pose   (R(2+1)D)", _test_conv3d)
    _smoke("staeformer    (pose graph)", _test_staeformer)
    _smoke("timesformer   (div ST)", _test_timesformer)
    _smoke("videomae_pose (VideoMAE)", _test_videomae)
    _smoke("videomae_tf   (VideoMAE+ST)", _test_videomae_timesformer)
    _smoke("vit_gcn       (ViT+GCN)", _test_vit_gcn)
    _smoke("st_tr         (ST-TR)", _test_st_tr)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, status, dt in results:
        mark = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{mark}] {name:30s} ({dt:.1f}s)")
    n_fail = sum(1 for _, s, _ in results if s != "PASS")
    print()
    if n_fail:
        print(f"  {n_fail} FAILED — fix before packaging for Colab.")
        sys.exit(1)
    else:
        print("  All models OK. Safe to package.")
        sys.exit(0)


if __name__ == "__main__":
    main()
