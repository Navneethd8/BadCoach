"""
Train TemPose_V on FineBadminton-20K collated tensors — skeleton-only baseline.

Uses TemPose_V (Ibh et al., 2023) from the BST third-party repo.  The "V"
variant takes only skeleton joint+bone features and video length (no shuttle
trajectory, no court position), making it a fair architectural comparison.

Example:

  python backend/pipelines/training/train_tempose_baseline.py \\
    --collated-root backend/data/bst_finebadminton_collated \\
    --sequence-length 16 \\
    --pose-style JnB_bone \\
    --epochs 60 \\
    --batch-size 64
"""
from __future__ import annotations

import argparse
import datetime
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

import mlflow

from core.bst_finebadminton_data import get_bone_pairs_coco
from core.bst_finebadminton_loader import FineBadmintonBSTCollatedDataset
from core.model_registry import (
    default_registry_checkpoint_path,
    register_training_checkpoint,
    resolve_training_save_path,
)
from core.seed_utils import set_seed

from model.tempose import TemPose_V

REGISTRY_CATEGORY = "tempose"


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


def train_one_epoch(model, loader, opt, device, criterion):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for hp, _pos, _shuttle, vlen, y in loader:
        hp, vlen, y = hp.to(device), vlen.to(device), y.to(device)
        opt.zero_grad()
        logits = model(hp, vlen)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * y.size(0)
        total_correct += int((logits.argmax(1) == y).sum().item())
        total += y.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for hp, _pos, _shuttle, vlen, y in loader:
        hp, vlen, y = hp.to(device), vlen.to(device), y.to(device)
        logits = model(hp, vlen)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * y.size(0)
        total_correct += int((logits.argmax(1) == y).sum().item())
        total += y.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Train TemPose_V baseline on FineBadminton BST collated npy.")
    p.add_argument("--collated-root", required=True, help="Directory with train/ and val/ npy.")
    p.add_argument("--sequence-length", type=int, default=16)
    p.add_argument("--pose-style", choices=["J_only", "JnB_bone"], default="JnB_bone")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-path", default=None,
                   help=f"Checkpoint .pth (default: backend/models/badminton_model_{REGISTRY_CATEGORY}.pth)")
    p.add_argument("--num-classes", type=int, default=9)
    p.add_argument(
        "--registry-experiment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"Save to a timestamped .pth and append to registry registrations (default). "
        f"Use --no-registry-experiment to overwrite {REGISTRY_CATEGORY} primary instead.",
    )
    args = p.parse_args()

    set_seed(args.seed)

    save_path = resolve_training_save_path(
        args.save_path or default_registry_checkpoint_path(_backend_root, REGISTRY_CATEGORY),
        args.registry_experiment,
    )
    models_dir = os.path.dirname(save_path) or os.path.join(_backend_root, "models")

    in_dim = _infer_in_dim(args.pose_style)

    train_ds = FineBadmintonBSTCollatedDataset(args.collated_root, "train", args.pose_style)
    val_ds = FineBadmintonBSTCollatedDataset(args.collated_root, "val", args.pose_style)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TemPose_V(
        in_dim=in_dim,
        seq_len=args.sequence_length,
        n_class=args.num_classes,
        n_people=2,
        d_model=100,
        d_head=128,
        n_head=6,
        depth_tem=2,
        depth_inter=2,
        drop_p=0.3,
        mlp_d_scale=4,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device={device} model=TemPose_V in_dim={in_dim} params={n_params:,} "
          f"train={len(train_ds)} val={len(val_ds)}")

    _coll = os.path.normpath(os.path.abspath(args.collated_root))
    _coll_tag = "mmpose" if "mmpose" in _coll.lower() else "other"

    mlflow.set_experiment("IsoCourt_Training_TemPose_Baseline")
    with mlflow.start_run():
        mlflow.set_tag("collated_data", _coll_tag)
        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "pose_style": args.pose_style,
            "sequence_length": args.sequence_length,
            "num_classes": args.num_classes,
            "in_dim": in_dim,
            "n_params": n_params,
            "collated_root": _coll,
            "script": "train_tempose_baseline.py",
        })

        best_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device, criterion)
            va_loss, va_acc = evaluate(model, val_loader, device, criterion)

            mlflow.log_metrics({
                "train_loss": tr_loss,
                "train_acc": tr_acc * 100,
                "val_loss": va_loss,
                "val_acc": va_acc * 100,
            }, step=epoch)

            print(
                f"epoch {epoch:03d}  train loss={tr_loss:.4f} acc={tr_acc:.4f}  "
                f"val loss={va_loss:.4f} acc={va_acc:.4f}"
            )

            if va_acc >= best_acc:
                best_acc = va_acc
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                torch.save({
                    REGISTRY_CATEGORY: model.state_dict(),
                    "best_acc": va_acc,
                    "epoch": epoch,
                    "args": vars(args),
                    "in_dim": in_dim,
                }, save_path)
                print(f"  saved {save_path} (best val acc={best_acc:.4f})")

                register_training_checkpoint(
                    models_dir,
                    category=REGISTRY_CATEGORY,
                    file_basename=os.path.basename(save_path),
                    meta={
                        "accuracy": round(best_acc * 100, 2),
                        "epoch": epoch,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "script": "train_tempose_baseline.py",
                        "architecture": REGISTRY_CATEGORY,
                        "inference": {
                            "variant": "TemPose_V",
                            "pose_style": args.pose_style,
                            "sequence_length": args.sequence_length,
                            "in_dim": in_dim,
                            "num_classes": args.num_classes,
                        },
                    },
                    experiment=args.registry_experiment,
                )

        print(f"Done. Best val acc={best_acc:.4f} ({best_acc * 100:.2f}%)")


if __name__ == "__main__":
    main()
