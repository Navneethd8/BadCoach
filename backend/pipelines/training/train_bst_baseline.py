"""
Train BST variants on FineBadminton-20K collated tensors from
``prepare_bst_finebadminton_collated.py`` — skeleton baseline comparable to other IsoCourt models.

Uses upstream BST under ``backend/third_party/BST-Badminton-Stroke-type-Transformer``
(Jing-Yuan Chang, MIT). Stroke-type classification only (9 classes).

Example:

  python backend/pipelines/training/train_bst_baseline.py \\
    --collated-root backend/data/bst_finebadminton_collated \\
    --sequence-length 30 \\
    --pose-style JnB_bone \\
    --model-name BST_CG_AP \\
    --epochs 80 \\
    --batch-size 64
"""
from __future__ import annotations

import argparse
import os
import sys

_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

BST_ROOT = os.path.join(_backend_root, "third_party", "BST-Badminton-Stroke-type-Transformer", "stroke_classification")
if not os.path.isdir(BST_ROOT):
    raise RuntimeError(
        f"Missing BST checkout at {BST_ROOT}. Clone:\n"
        "  git clone --depth 1 https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer.git "
        f"{os.path.dirname(BST_ROOT)}/BST-Badminton-Stroke-type-Transformer"
    )
if BST_ROOT not in sys.path:
    sys.path.insert(0, BST_ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.bst_finebadminton_data import get_bone_pairs_coco
from core.bst_finebadminton_loader import FineBadmintonBSTCollatedDataset
from core.seed_utils import set_seed

from model.bst import BST, BST_AP, BST_CG, BST_CG_AP


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


def _build_net(model_name: str, in_dim: int, seq_len: int, n_classes: int):
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
    raise ValueError(f"Unknown model_name {model_name}")


def train_one_epoch(model, loader, opt, device, criterion):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for hp, pos, shuttle, vlen, y in loader:
        hp = hp.to(device)
        pos = pos.to(device)
        shuttle = shuttle.to(device)
        vlen = vlen.to(device)
        y = y.to(device)

        opt.zero_grad()
        logits = model(hp, shuttle, pos, vlen)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()

        total_loss += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        total_correct += int((pred == y).sum().item())
        total += y.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for hp, pos, shuttle, vlen, y in loader:
        hp = hp.to(device)
        pos = pos.to(device)
        shuttle = shuttle.to(device)
        vlen = vlen.to(device)
        y = y.to(device)
        logits = model(hp, shuttle, pos, vlen)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        total_correct += int((pred == y).sum().item())
        total += y.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Train BST baseline on FineBadminton BST collated npy.")
    p.add_argument("--collated-root", required=True, help="Directory with train/ and val/ npy.")
    p.add_argument("--sequence-length", type=int, default=30)
    p.add_argument("--pose-style", choices=["J_only", "JnB_bone"], default="JnB_bone")
    p.add_argument("--model-name", default="BST_CG_AP", help="BST | BST_CG | BST_AP | BST_CG_AP")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--save-path",
        default=None,
        help="Checkpoint .pth (default: backend/models/badminton_model_bst_baseline.pth)",
    )
    p.add_argument("--num-classes", type=int, default=9)
    args = p.parse_args()

    set_seed(args.seed)

    save_path = args.save_path or os.path.join(_backend_root, "models", "badminton_model_bst_baseline.pth")

    train_ds = FineBadmintonBSTCollatedDataset(args.collated_root, "train", args.pose_style)
    val_ds = FineBadmintonBSTCollatedDataset(args.collated_root, "val", args.pose_style)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    in_dim = _infer_in_dim(args.pose_style)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_net(args.model_name, in_dim, args.sequence_length, args.num_classes).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    print(f"device={device} model={args.model_name} in_dim={in_dim} train={len(train_ds)} val={len(val_ds)}")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device, criterion)
        va_loss, va_acc = evaluate(model, val_loader, device, criterion)
        print(f"epoch {epoch:03d}  train loss={tr_loss:.4f} acc={tr_acc:.4f}  val loss={va_loss:.4f} acc={va_acc:.4f}")
        if va_acc >= best_acc:
            best_acc = va_acc
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": va_acc,
                    "args": vars(args),
                    "in_dim": in_dim,
                },
                save_path,
            )
            print(f"  saved {save_path} (best val acc={best_acc:.4f})")

    print(f"Done. Best val acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
