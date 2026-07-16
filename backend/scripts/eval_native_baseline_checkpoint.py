#!/usr/bin/env python3
"""
Evaluate native IsoCourt stroke models on the video-level **test** split.

Supports: CNN-LSTM (train_full), Conv3D+pose, TimeSformer, JVC.

  python backend/scripts/eval_native_baseline_checkpoint.py \\
    --model conv3d \\
    --checkpoint backend/models/badminton_model_conv3d_pose_20260714T030509Z.pth \\
    --split test --per-class
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple

_backend = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _backend not in sys.path:
    sys.path.insert(0, _backend)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from core.conv3d_pose import Conv3DPoseMultitaskModel
from core.dataset import FineBadmintonDataset
from core.jvc import build_jvc, load_jvc_partial
from core.model import CNN_LSTM_Model
from core.pose_cache_build import default_pose_cache_path, load_pose_cache_bundle
from core.shuttle_cache import load_shuttle_cache_bundle
from core.split import video_level_split
from core.timesformer import TimeSformerPoseModel

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
_RESNET_FEAT_DIM = 2048


def _imagenet_norm_video(frames: torch.Tensor, device: torch.device) -> torch.Tensor:
    B, T, C, H, W = frames.shape
    x = frames.view(B * T, C, H, W).to(device)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.view(B, T, C, H, W)


class _FramePoseDataset(Dataset):
    """Frames + (T, 33, 3) pose cache for Conv3D / TimeSformer."""

    def __init__(self, base, pose_cache):
        self.base = base
        self.pose_cache = pose_cache

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        frames, labels = self.base[idx]
        pose = self.pose_cache[idx]
        if isinstance(pose, torch.Tensor):
            pose = pose.clone()
        return frames, pose, labels


class _CnnLstmDataset(Dataset):
    """Frames + flattened pose (T, 99) for CNN-LSTM."""

    def __init__(self, base, pose_cache):
        self.base = base
        self.pose_cache = pose_cache

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        frames, labels = self.base[idx]
        pose = self.pose_cache[idx].clone().view(self.pose_cache[idx].shape[0], -1)
        return frames, pose, labels


class _JvcEvalDataset(Dataset):
    def __init__(self, base, pose_cache, shuttle_cache=None):
        self.base = base
        self.pose_cache = pose_cache
        self.shuttle_cache = shuttle_cache

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        frames, labels = self.base[idx]
        pose = self.pose_cache[idx]
        if isinstance(pose, torch.Tensor):
            pose = pose.clone()
        if self.shuttle_cache is not None:
            return frames, pose, self.shuttle_cache[idx], labels
        return frames, pose, labels


def _task_classes_from_dataset(ds: FineBadmintonDataset) -> Dict[str, int]:
    task_classes = {k: len(v) for k, v in ds.classes.items()}
    task_classes["quality"] = 7
    if "stroke_subtype" in task_classes:
        del task_classes["stroke_subtype"]
    return task_classes


def _infer_cnn_lstm_hparams(state: Dict[str, torch.Tensor]) -> Tuple[int, bool]:
    w_ih = state.get("lstm.weight_ih_l0")
    w_hh = state.get("lstm.weight_hh_l0")
    if w_ih is None or w_hh is None:
        return 256, True
    hidden_size = int(w_hh.shape[1])
    use_pose = int(w_ih.shape[1]) > _RESNET_FEAT_DIM
    return hidden_size, use_pose


def _load_cnn_lstm(
    ckpt,
    task_classes: Dict[str, int],
    device: torch.device,
    hidden_size: int | None,
    use_pose: bool | None,
    pretrained: bool,
) -> CNN_LSTM_Model:
    state = ckpt
    if isinstance(ckpt, dict):
        if "cnn_lstm" in ckpt:
            state = ckpt["cnn_lstm"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
    if not isinstance(state, dict):
        raise SystemExit("CNN-LSTM checkpoint must be a state_dict or dict with model weights.")

    inferred_hidden, inferred_pose = _infer_cnn_lstm_hparams(state)
    hidden_size = hidden_size if hidden_size is not None else inferred_hidden
    use_pose = use_pose if use_pose is not None else inferred_pose

    model = CNN_LSTM_Model(
        task_classes=task_classes,
        hidden_size=hidden_size,
        pretrained=pretrained,
        use_pose=use_pose,
    ).to(device)
    model.load_state_dict(state, strict=True)
    return model


def _load_conv3d(ckpt: dict, device: torch.device) -> Conv3DPoseMultitaskModel:
    if "model" not in ckpt:
        raise SystemExit("Conv3D checkpoint missing 'model' state dict.")
    task_classes = ckpt.get("task_classes")
    if not task_classes:
        raise SystemExit("Conv3D checkpoint missing task_classes.")
    model = Conv3DPoseMultitaskModel(
        task_classes=task_classes,
        num_frames=int(ckpt.get("num_frames", 16)),
        video_backbone=str(ckpt.get("video_backbone", "r2plus1d_18")),
        spatial_size=int(ckpt.get("spatial_size", 224)),
        pretrained=bool(ckpt.get("pretrained", True)),
        freeze_backbone=bool(ckpt.get("freeze_3d", True)),
        unfreeze_layer4=bool(ckpt.get("unfreeze_layer4", True)),
        use_pose=bool(ckpt.get("use_pose", True)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    return model


def _load_timesformer(ckpt: dict, device: torch.device) -> TimeSformerPoseModel:
    if "model" not in ckpt:
        raise SystemExit("TimeSformer checkpoint missing 'model' state dict.")
    task_classes = ckpt.get("task_classes")
    if not task_classes:
        raise SystemExit("TimeSformer checkpoint missing task_classes.")
    model = TimeSformerPoseModel(
        task_classes=task_classes,
        num_frames=int(ckpt.get("num_frames", 16)),
        embed_dim=int(ckpt.get("embed_dim", 128)),
        num_heads=int(ckpt.get("num_heads", 4)),
        depth=int(ckpt.get("depth", 4)),
        backbone=str(ckpt.get("backbone", "scratch")),
        vit_model_name=str(ckpt.get("vit_model_name", "vit_small_patch16_224")),
        vit_unfreeze_last_n=int(ckpt.get("vit_unfreeze_last_n", 0)),
        use_pose=bool(ckpt.get("use_pose", True)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    return model


def _load_jvc(
    ckpt: dict,
    task_classes: Dict[str, int],
    device: torch.device,
    window_size: int,
) -> Tuple[nn.Module, bool]:
    use_shuttle = bool(ckpt.get("use_shuttle", False))
    model = build_jvc(
        task_classes,
        window_size=window_size,
        embed_dim=int(ckpt.get("embed_dim", 128)),
        skel_embed_dim=int(ckpt.get("skel_embed_dim", 64)),
        skel_num_heads=int(ckpt.get("skel_num_heads", 16)),
        num_heads=int(ckpt.get("num_heads", 4)),
        st_depth=int(ckpt.get("st_depth", 4)),
        num_cross_layers=int(ckpt.get("num_cross_layers", 2)),
        vision_backbone=str(ckpt.get("vision_backbone", "conv3d")),
        video_backbone=str(ckpt.get("video_backbone", "r2plus1d_18")),
        spatial_size=int(ckpt.get("spatial_size", 224)),
        vit_model_name=str(ckpt.get("vit_model_name", "vit_small_patch16_224")),
        vit_unfreeze_last_n=int(ckpt.get("vit_unfreeze_last_n", 4)),
        use_shuttle=use_shuttle,
    ).to(device)
    load_jvc_partial(model, ckpt, device=device)
    return model, use_shuttle


@torch.no_grad()
def _evaluate_stroke(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
    use_pose: bool,
    use_shuttle: bool,
    tta_flip: bool,
) -> Tuple[float, Dict[int, Tuple[int, int]]]:
    model.eval()
    correct = 0
    total = 0
    per_class: Dict[int, Tuple[int, int]] = {}

    for batch in tqdm(loader, desc=f"Eval {model_name}", unit="batch", file=sys.stdout):
        if model_name == "cnn_lstm":
            frames, poses, labels = batch
            frames = frames.to(device)
            poses = poses.to(device) if use_pose else None
            logits = model(frames, poses)["stroke_type"]
        elif model_name == "jvc":
            if use_shuttle:
                frames, poses, shuttle, labels = batch
                shuttle = shuttle.to(device)
            else:
                frames, poses, labels = batch
                shuttle = None
            frames = _imagenet_norm_video(frames.to(device), device)
            poses = poses.to(device)
            logits = model(frames, poses, shuttle=shuttle)["stroke_type"]
            if tta_flip:
                frames_f = frames.flip(-1)
                logits_f = model(frames_f, poses, shuttle=shuttle)["stroke_type"]
                logits = 0.5 * (logits + logits_f)
        else:
            frames, poses, labels = batch
            frames = _imagenet_norm_video(frames.to(device), device)
            poses = poses.to(device)
            logits = model(frames, poses if use_pose else None)["stroke_type"]
            if tta_flip:
                frames_f = frames.flip(-1)
                logits_f = model(frames_f, poses if use_pose else None)["stroke_type"]
                logits = 0.5 * (logits + logits_f)

        pred = logits.argmax(dim=1)
        y = labels["stroke_type"].to(device)
        correct += int((pred == y).sum().item())
        total += y.size(0)
        for p, t in zip(pred.cpu().tolist(), y.cpu().tolist()):
            c, n = per_class.get(t, (0, 0))
            per_class[t] = (c + int(p == t), n + 1)

    return 100.0 * correct / max(total, 1), per_class


def main() -> None:
    p = argparse.ArgumentParser(description="Eval native stroke models on train/val/test split.")
    p.add_argument(
        "--model",
        required=True,
        choices=["cnn_lstm", "conv3d", "conv3d_pose", "timesformer", "jvc"],
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--data-root", default=None)
    p.add_argument("--list-file", default=None)
    p.add_argument("--pose-cache", default=None)
    p.add_argument("--shuttle-cache", default=None)
    p.add_argument(
        "--sampling",
        choices=["span_linspace", "hit_centered"],
        default=None,
        help="Frame sampling (default: checkpoint sampling_mode, else span_linspace).",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--hidden-size", type=int, default=None, help="CNN-LSTM only (auto if omitted).")
    p.add_argument("--no-pose", action="store_true", help="CNN-LSTM only; default auto from checkpoint.")
    p.add_argument("--tta-flip", action="store_true", help="Horizontal flip TTA on RGB (video models).")
    p.add_argument("--per-class", action="store_true")
    args = p.parse_args()

    model_name = args.model
    if model_name == "conv3d_pose":
        model_name = "conv3d"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = args.data_root or os.path.join(_backend, "data")
    list_file = args.list_file or os.path.join(
        _backend, "data", "transformed_combined_rounds_output_en_evals_translated.json"
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sampling = args.sampling
    if sampling is None:
        if isinstance(ckpt, dict) and ckpt.get("sampling_mode"):
            sampling = ckpt["sampling_mode"]
        else:
            sampling = "span_linspace"

    ds = FineBadmintonDataset(data_root, list_file, sampling_mode=sampling)
    frame_src = (
        f"JPEG ({ds.image_dir})"
        if ds.image_dir
        else "MP4 decode (slow — ensure backend/data/FineBadminton-20K/dataset/image exists)"
    )
    print(f"Frame source: {frame_src}", flush=True)
    task_classes = _task_classes_from_dataset(ds)

    pose_path = args.pose_cache or default_pose_cache_path(_backend)
    bundle = load_pose_cache_bundle(pose_path)
    if bundle is None:
        raise SystemExit(f"Missing pose cache: {pose_path}")
    pose_cache = bundle["pose_cache"]
    if pose_cache.shape[0] != len(ds):
        raise SystemExit(
            f"Pose cache rows ({pose_cache.shape[0]}) != dataset ({len(ds)}). "
            f"Rebuild cache for {list_file}"
        )

    shuttle_cache = None
    use_shuttle = False
    if args.shuttle_cache and os.path.isfile(args.shuttle_cache):
        sb = load_shuttle_cache_bundle(args.shuttle_cache)
        if sb is not None:
            shuttle_cache = sb["shuttle_cache"]
            use_shuttle = True
    elif isinstance(ckpt, dict) and ckpt.get("use_shuttle"):
        raise SystemExit("Checkpoint expects shuttle features; pass --shuttle-cache PATH.")

    train_idx, val_idx, test_idx = video_level_split(ds.samples)
    split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
    indices = split_map[args.split]

    if model_name == "cnn_lstm":
        eval_ds = _CnnLstmDataset(Subset(ds, indices), pose_cache)
        use_pose_arg = False if args.no_pose else None
        model = _load_cnn_lstm(
            ckpt,
            task_classes,
            device,
            args.hidden_size,
            use_pose_arg,
            pretrained=True,
        )
        use_pose = model.use_pose
        use_shuttle = False
    elif model_name == "jvc":
        eval_ds = _JvcEvalDataset(Subset(ds, indices), pose_cache, shuttle_cache)
        model, use_shuttle = _load_jvc(ckpt, task_classes, device, int(ds.sequence_length))
        use_pose = True
    elif model_name == "conv3d":
        eval_ds = _FramePoseDataset(Subset(ds, indices), pose_cache)
        model = _load_conv3d(ckpt, device)
        use_pose = model.use_pose
        use_shuttle = False
    elif model_name == "timesformer":
        eval_ds = _FramePoseDataset(Subset(ds, indices), pose_cache)
        model = _load_timesformer(ckpt, device)
        use_pose = model.use_pose
        use_shuttle = False
    else:
        raise SystemExit(f"Unsupported model {args.model}")

    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    print(
        f"Loaded {args.model} on {device}; evaluating {len(eval_ds)} {args.split} clips "
        f"(batch_size={args.batch_size}, ~{len(loader)} batches)...",
        flush=True,
    )

    acc, per_class = _evaluate_stroke(
        model,
        loader,
        device,
        model_name,
        use_pose,
        use_shuttle,
        args.tta_flip,
    )

    print(
        f"{args.model} {args.split}: {acc:.2f}%  "
        f"(n={len(eval_ds)}, T={ds.sequence_length}, sampling={sampling})"
    )
    if isinstance(ckpt, dict) and ckpt.get("best_acc") is not None:
        print(f"  checkpoint best_val_acc={float(ckpt['best_acc']):.2f}% epoch={ckpt.get('epoch')}")
    if args.tta_flip and model_name != "cnn_lstm":
        print("  (with horizontal-flip TTA on RGB)")
    if args.per_class:
        for cls_idx in sorted(per_class):
            c, n = per_class[cls_idx]
            print(f"  class {cls_idx}: {100.0 * c / max(n, 1):.1f}% ({c}/{n})")


if __name__ == "__main__":
    main()
