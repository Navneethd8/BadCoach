"""
Train ST-GCN on FineBadminton-20K collated tensors — skeleton-only baseline.

Uses ST-GCN (Yan et al., 2018) from the BST third-party repo. Takes only
skeleton joint coordinates (no shuttle, no court position), making it a fair
architectural comparison against video-based IsoCourt models.

Example:

  python backend/pipelines/training/train_stgcn_baseline.py \\
    --collated-root backend/data/bst_finebadminton_collated \\
    --sequence-length 16 \\
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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from core.model_registry import (
    default_registry_checkpoint_path,
    register_training_checkpoint,
    resolve_training_save_path,
)
from core.seed_utils import set_seed
from core.stgcn_compat import ST_GCN_18
from core.training_standards import (
    mlflow_log_metrics,
    mlflow_log_params,
    mlflow_log_tag,
    mlflow_training_context,
)

REGISTRY_CATEGORY = "stgcn"


class _STGCNCollatedDataset(Dataset):
    """Load collated .npy and return joint-only tensor in ST-GCN layout.

    ST-GCN forward expects ``(N, C, T, V, M)`` with C=2 (x,y), V=17 joints,
    M=2 players.  The collated file ``{pose_style}.npy`` has shape
    ``(N, T, M=2, J, C=2)`` where J >= 17 (17 joints + optional bones).
    We slice to the first 17 joints regardless of pose_style.
    """

    def __init__(self, root_dir: str, split: str, pose_style: str):
        branch = os.path.join(root_dir, split)
        raw = np.load(os.path.join(branch, f"{pose_style}.npy"), mmap_mode="r")
        self.human_pose = raw
        self.labels = np.load(os.path.join(branch, "labels.npy"), mmap_mode="r")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int):
        hp = np.asarray(self.human_pose[i], dtype=np.float32)  # (T, M, J, C)
        hp = hp[:, :, :17, :]  # joints only, drop bones
        # -> (C, T, V, M) for ST-GCN
        x = torch.from_numpy(hp).permute(3, 0, 2, 1).contiguous()
        y = int(self.labels[i])
        return x, torch.tensor(y, dtype=torch.long)


def train_one_epoch(model, loader, opt, device, criterion):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
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
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * y.size(0)
        total_correct += int((logits.argmax(1) == y).sum().item())
        total += y.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Train ST-GCN baseline on FineBadminton BST collated npy.")
    p.add_argument("--collated-root", required=True, help="Directory with train/ and val/ npy.")
    p.add_argument("--sequence-length", type=int, default=16)
    p.add_argument("--pose-style", choices=["J_only", "JnB_bone"], default="JnB_bone",
                   help="Which collated file to load (joints-only slice is used regardless).")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.5)
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

    train_ds = _STGCNCollatedDataset(args.collated_root, "train", args.pose_style)
    val_ds = _STGCNCollatedDataset(args.collated_root, "val", args.pose_style)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ST_GCN_18(
        in_channels=2,
        num_class=args.num_classes,
        graph_cfg={"layout": "coco", "strategy": "spatial"},
        edge_importance_weighting=True,
        data_bn=True,
        tem_kernel_size=9,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device={device} model=ST_GCN_18 params={n_params:,} train={len(train_ds)} val={len(val_ds)}")

    _coll = os.path.normpath(os.path.abspath(args.collated_root))
    _coll_tag = "mmpose" if "mmpose" in _coll.lower() else "other"

    with mlflow_training_context("IsoCourt_Training_STGCN_Baseline", _backend_root):
        mlflow_log_tag("collated_data", _coll_tag)
        mlflow_log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "seed": args.seed,
            "pose_style": args.pose_style,
            "sequence_length": args.sequence_length,
            "num_classes": args.num_classes,
            "n_params": n_params,
            "collated_root": _coll,
            "script": "train_stgcn_baseline.py",
        })

        best_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device, criterion)
            va_loss, va_acc = evaluate(model, val_loader, device, criterion)

            mlflow_log_metrics({
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
                        "script": "train_stgcn_baseline.py",
                        "architecture": REGISTRY_CATEGORY,
                        "inference": {
                            "in_channels": 2,
                            "num_joints": 17,
                            "graph_layout": "coco",
                            "sequence_length": args.sequence_length,
                            "num_classes": args.num_classes,
                        },
                    },
                    experiment=args.registry_experiment,
                )

        print(f"Done. Best val acc={best_acc:.4f} ({best_acc * 100:.2f}%)")


if __name__ == "__main__":
    main()
