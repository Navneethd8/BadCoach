#!/usr/bin/env python3
"""
Evaluate stroke_type accuracy for K-STViT / Conv3D checkpoints on the val split.

Optional horizontal-flip TTA on RGB (pose unchanged).

  python backend/scripts/eval_stroke_checkpoint.py \\
    --checkpoint backend/models/badminton_model_k_st_vit.pth \\
    --tta-flip
"""
from __future__ import annotations

import argparse
import os
import sys

_backend = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _backend not in sys.path:
    sys.path.insert(0, _backend)

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from core.dataset import FineBadmintonDataset
from core.k_st_vit import build_k_st_vit, load_k_st_vit_partial
from core.pose_cache_build import default_pose_cache_path, load_pose_cache_bundle
from core.shuttle_cache import load_shuttle_cache_bundle
from core.split import video_level_split

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _norm(frames: torch.Tensor, device: torch.device) -> torch.Tensor:
    B, T, C, H, W = frames.shape
    x = frames.view(B * T, C, H, W).to(device)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.view(B, T, C, H, W)


class EvalDataset(Dataset):
    def __init__(self, base, pose_cache, shuttle_cache=None):
        self.base = base
        self.pose_cache = pose_cache
        self.shuttle_cache = shuttle_cache

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        frames, labels = self.base[idx]
        pose = self.pose_cache[idx]
        if self.shuttle_cache is not None:
            return frames, pose, self.shuttle_cache[idx], labels
        return frames, pose, labels


@torch.no_grad()
def evaluate(model, loader, device, tta_flip: bool, use_shuttle: bool) -> tuple[float, dict]:
    model.eval()
    correct = 0
    total = 0
    per_class = {}
    for batch in loader:
        if use_shuttle:
            frames, pose, shuttle, labels = batch
            shuttle = shuttle.to(device)
        else:
            frames, pose, labels = batch
            shuttle = None
        frames = _norm(frames, device)
        pose = pose.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}

        logits = model(frames, pose, shuttle=shuttle)["stroke_type"]
        if tta_flip:
            frames_f = frames.flip(-1)
            logits_f = model(frames_f, pose, shuttle=shuttle)["stroke_type"]
            logits = 0.5 * (logits + logits_f)

        pred = logits.argmax(dim=1)
        y = labels["stroke_type"]
        correct += (pred == y).sum().item()
        total += y.size(0)
        for p, t in zip(pred.cpu().tolist(), y.cpu().tolist()):
            per_class.setdefault(t, [0, 0])
            per_class[t][1] += 1
            if p == t:
                per_class[t][0] += 1

    acc = 100.0 * correct / max(total, 1)
    return acc, per_class


def main() -> None:
    p = argparse.ArgumentParser(description="Eval stroke_type on val split.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--sampling", choices=["span_linspace", "hit_centered"], default="hit_centered")
    p.add_argument("--vision-backbone", choices=["conv3d", "vit"], default="conv3d")
    p.add_argument("--shuttle-cache", default=None)
    p.add_argument("--tta-flip", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = os.path.join(_backend, "data")
    list_file = os.path.join(
        _backend, "data", "transformed_combined_rounds_output_en_evals_translated.json"
    )
    ds = FineBadmintonDataset(data_root, list_file, sampling_mode=args.sampling)
    pose_path = default_pose_cache_path(_backend)
    bundle = load_pose_cache_bundle(pose_path)
    if bundle is None:
        raise SystemExit(f"Missing pose cache: {pose_path}")
    pose_cache = bundle["pose_cache"]
    task_classes = bundle.get("task_classes")
    if task_classes is None:
        task_classes = {k: len(v) for k, v in ds.classes.items()}
        task_classes["quality"] = 7
        if "stroke_subtype" in task_classes:
            del task_classes["stroke_subtype"]

    shuttle_cache = None
    use_shuttle = False
    if args.shuttle_cache and os.path.isfile(args.shuttle_cache):
        sb = load_shuttle_cache_bundle(args.shuttle_cache)
        if sb is not None:
            shuttle_cache = sb["shuttle_cache"]
            use_shuttle = True

    _, val_idx = video_level_split(ds.samples)
    val_ds = EvalDataset(Subset(ds, val_idx), pose_cache, shuttle_cache)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_k_st_vit(
        task_classes,
        vision_backbone=args.vision_backbone,
        use_shuttle=use_shuttle,
    ).to(device)
    load_k_st_vit_partial(model, ckpt, device=device)

    acc, per_class = evaluate(model, loader, device, args.tta_flip, use_shuttle)
    print(f"Val stroke_type: {acc:.2f}%  (T={ds.sequence_length}, sampling={args.sampling})")
    if args.tta_flip:
        print("  (with horizontal-flip TTA on RGB)")
    for cls_idx in sorted(per_class):
        c, n = per_class[cls_idx]
        print(f"  class {cls_idx}: {100.0 * c / max(n, 1):.1f}% ({c}/{n})")


if __name__ == "__main__":
    main()
