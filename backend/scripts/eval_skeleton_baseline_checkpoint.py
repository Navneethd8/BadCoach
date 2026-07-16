#!/usr/bin/env python3
"""
Evaluate BST / TemPose_V / ST-GCN checkpoints on collated FineBadminton .npy splits.

These external skeleton baselines are not wired into ``api/model_loader.py``; use this
script for val/test accuracy (and optional per-class breakdown) after training.

Examples (from repo root):

  python backend/scripts/eval_skeleton_baseline_checkpoint.py \\
    --model bst \\
    --checkpoint backend/models/badminton_model_bst_baseline.pth \\
    --collated-root backend/data/bst_finebadminton_collated_mmpose_16 \\
    --split test

  python backend/scripts/eval_skeleton_baseline_checkpoint.py \\
    --model tempose --checkpoint backend/models/badminton_model_tempose_baseline.pth \\
    --collated-root backend/data/bst_finebadminton_collated_mmpose_16

  python backend/scripts/eval_skeleton_baseline_checkpoint.py \\
    --model stgcn --checkpoint backend/models/badminton_model_stgcn_baseline.pth \\
    --collated-root backend/data/bst_finebadminton_collated_mmpose_16 --per-class
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple

_backend = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _backend not in sys.path:
    sys.path.insert(0, _backend)

BST_ROOT = os.path.join(
    _backend,
    "third_party",
    "BST-Badminton-Stroke-type-Transformer",
    "stroke_classification",
)


def _ensure_bst_root() -> None:
    if not os.path.isdir(BST_ROOT):
        raise SystemExit(
            f"Missing BST checkout at {BST_ROOT}. Clone:\n"
            "  git clone --depth 1 https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer.git "
            f"{os.path.dirname(BST_ROOT)}/BST-Badminton-Stroke-type-Transformer"
        )
    if BST_ROOT not in sys.path:
        sys.path.insert(0, BST_ROOT)


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from core.bst_finebadminton_data import get_bone_pairs_coco
from core.bst_finebadminton_loader import FineBadmintonBSTCollatedDataset
from core.stgcn_compat import ST_GCN_18


class _STGCNCollatedDataset(Dataset):
    def __init__(self, root_dir: str, split: str, pose_style: str):
        branch = os.path.join(root_dir, split)
        raw = np.load(os.path.join(branch, f"{pose_style}.npy"), mmap_mode="r")
        self.human_pose = raw
        self.labels = np.load(os.path.join(branch, "labels.npy"), mmap_mode="r")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int):
        hp = np.asarray(self.human_pose[i], dtype=np.float32)
        hp = hp[:, :, :17, :]
        x = torch.from_numpy(hp).permute(3, 0, 2, 1).contiguous()
        y = int(self.labels[i])
        return x, torch.tensor(y, dtype=torch.long)


def _infer_in_dim(pose_style: str, in_channels: int = 2) -> int:
    n_joints = 17
    n_bones = len(get_bone_pairs_coco())
    if pose_style == "J_only":
        extra = 0
    elif pose_style == "JnB_bone":
        extra = 1
    else:
        raise ValueError(pose_style)
    return (n_joints + n_bones * extra) * in_channels


def _build_bst(model_name: str, in_dim: int, seq_len: int, n_classes: int):
    from model.bst import BST, BST_AP, BST_CG, BST_CG_AP

    kw = dict(
        in_dim=in_dim,
        seq_len=seq_len,
        n_class=n_classes,
        d_model=100,
        d_head=128,
        n_head=6,
        depth_tem=2,
        depth_inter=1,
        drop_p=0.3,
        mlp_d_scale=4,
        tcn_kernel_size=5,
    )
    name = model_name.upper().replace("-", "_")
    if name == "BST":
        return BST(**kw)
    if name == "BST_CG":
        return BST_CG(**kw)
    if name == "BST_AP":
        return BST_AP(**kw)
    if name == "BST_CG_AP":
        return BST_CG_AP(**kw)
    raise ValueError(f"Unknown BST model_name {model_name}")


def _resolve_hparams(
    ckpt: dict,
    model: str,
    collated_root: str | None,
    pose_style: str | None,
    sequence_length: int | None,
    num_classes: int | None,
    model_name: str | None,
    dropout: float | None,
) -> dict:
    saved = ckpt.get("args") or {}
    out = {
        "collated_root": collated_root or saved.get("collated_root"),
        "pose_style": pose_style or saved.get("pose_style", "JnB_bone"),
        "sequence_length": sequence_length or saved.get("sequence_length", 16),
        "num_classes": num_classes or saved.get("num_classes", 9),
        "model_name": model_name or saved.get("model_name", "BST_CG_AP"),
        "dropout": dropout if dropout is not None else saved.get("dropout", 0.5),
    }
    if not out["collated_root"]:
        raise SystemExit("Pass --collated-root (not found in checkpoint args).")
    return out


def _load_model(model: str, ckpt: dict, hparams: dict, device: torch.device) -> nn.Module:
    key = model
    if key not in ckpt:
        raise SystemExit(f"Checkpoint missing state dict key '{key}'. Keys: {list(ckpt)}")

    if model == "bst":
        _ensure_bst_root()
        in_dim = ckpt.get("in_dim") or _infer_in_dim(hparams["pose_style"])
        net = _build_bst(
            hparams["model_name"],
            in_dim,
            hparams["sequence_length"],
            hparams["num_classes"],
        )
    elif model == "tempose":
        _ensure_bst_root()
        from model.tempose import TemPose_V

        in_dim = ckpt.get("in_dim") or _infer_in_dim(hparams["pose_style"])
        net = TemPose_V(
            in_dim=in_dim,
            seq_len=hparams["sequence_length"],
            n_class=hparams["num_classes"],
            n_people=2,
            d_model=100,
            d_head=128,
            n_head=6,
            depth_tem=2,
            depth_inter=2,
            drop_p=0.3,
            mlp_d_scale=4,
        )
    elif model == "stgcn":
        net = ST_GCN_18(
            in_channels=2,
            num_class=hparams["num_classes"],
            graph_cfg={"layout": "coco", "strategy": "spatial"},
            edge_importance_weighting=True,
            data_bn=True,
            tem_kernel_size=9,
            dropout=hparams["dropout"],
        )
    else:
        raise SystemExit(f"Unknown --model {model} (use bst, tempose, or stgcn)")

    net.load_state_dict(ckpt[key], strict=True)
    return net.to(device)


@torch.no_grad()
def _evaluate_bst(model, loader, device) -> Tuple[float, Dict[int, Tuple[int, int]]]:
    model.eval()
    correct = 0
    total = 0
    per_class: Dict[int, Tuple[int, int]] = {}
    for hp, pos, shuttle, vlen, y in loader:
        hp = hp.to(device)
        pos = pos.to(device)
        shuttle = shuttle.to(device)
        vlen = vlen.to(device)
        y = y.to(device)
        pred = model(hp, shuttle, pos, vlen).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += y.size(0)
        for p, t in zip(pred.cpu().tolist(), y.cpu().tolist()):
            c, n = per_class.get(t, (0, 0))
            per_class[t] = (c + int(p == t), n + 1)
    return 100.0 * correct / max(total, 1), per_class


@torch.no_grad()
def _evaluate_tempose(model, loader, device) -> Tuple[float, Dict[int, Tuple[int, int]]]:
    model.eval()
    correct = 0
    total = 0
    per_class: Dict[int, Tuple[int, int]] = {}
    for hp, _pos, _shuttle, vlen, y in loader:
        hp = hp.to(device)
        vlen = vlen.to(device)
        y = y.to(device)
        pred = model(hp, vlen).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += y.size(0)
        for p, t in zip(pred.cpu().tolist(), y.cpu().tolist()):
            c, n = per_class.get(t, (0, 0))
            per_class[t] = (c + int(p == t), n + 1)
    return 100.0 * correct / max(total, 1), per_class


@torch.no_grad()
def _evaluate_stgcn(model, loader, device) -> Tuple[float, Dict[int, Tuple[int, int]]]:
    model.eval()
    correct = 0
    total = 0
    per_class: Dict[int, Tuple[int, int]] = {}
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += y.size(0)
        for p, t in zip(pred.cpu().tolist(), y.cpu().tolist()):
            c, n = per_class.get(t, (0, 0))
            per_class[t] = (c + int(p == t), n + 1)
    return 100.0 * correct / max(total, 1), per_class


def main() -> None:
    p = argparse.ArgumentParser(description="Eval BST / TemPose / ST-GCN on collated npy.")
    p.add_argument("--model", required=True, choices=["bst", "tempose", "stgcn"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--collated-root", default=None)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--pose-style", choices=["J_only", "JnB_bone"], default=None)
    p.add_argument("--sequence-length", type=int, default=None)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--model-name", default=None, help="BST variant (BST_CG_AP, etc.)")
    p.add_argument("--dropout", type=float, default=None, help="ST-GCN dropout")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--per-class", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    hparams = _resolve_hparams(
        ckpt,
        args.model,
        args.collated_root,
        args.pose_style,
        args.sequence_length,
        args.num_classes,
        args.model_name,
        args.dropout,
    )
    model = _load_model(args.model, ckpt, hparams, device)

    if args.model in ("bst", "tempose"):
        ds = FineBadmintonBSTCollatedDataset(
            hparams["collated_root"], args.split, hparams["pose_style"]
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        if args.model == "bst":
            acc, per_class = _evaluate_bst(model, loader, device)
        else:
            acc, per_class = _evaluate_tempose(model, loader, device)
    else:
        ds = _STGCNCollatedDataset(
            hparams["collated_root"], args.split, hparams["pose_style"]
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        acc, per_class = _evaluate_stgcn(model, loader, device)

    ckpt_acc = ckpt.get("best_acc")
    ckpt_epoch = ckpt.get("epoch")
    print(
        f"{args.model} {args.split}: {acc:.2f}%  "
        f"(n={len(ds)}, collated={hparams['collated_root']}, pose={hparams['pose_style']}, "
        f"T={hparams['sequence_length']})"
    )
    if ckpt_acc is not None:
        print(f"  checkpoint best_val_acc={float(ckpt_acc) * 100:.2f}% epoch={ckpt_epoch}")
    if args.per_class:
        for cls_idx in sorted(per_class):
            c, n = per_class[cls_idx]
            print(f"  class {cls_idx}: {100.0 * c / max(n, 1):.1f}% ({c}/{n})")


if __name__ == "__main__":
    main()
