"""
ST-TR (upstream skeleton trunk) + fine-tuned timm ViT late fusion on FineBadminton clips.

Loads RGB frames from the dataset and MediaPipe pose from ``pose_cache_mediapipe.pt``.
Optionally warm-starts the skeleton trunk from ``badminton_model_st_tr.pth``.

Example (EC2):

  python backend/pipelines/training/train_st_tr_vit.py \\
    --batch-size 4 --stream both \\
    --resume-skeleton backend/models/badminton_model_st_tr.pth \\
    --stroke-only-epochs 8 --stroke-loss-weight 5 --aux-loss-weight 0.25 \\
    --scheduler-t0 30 --vit-unfreeze-last-n 2 --registry-experiment
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision.transforms import v2

from core.dataset import FineBadmintonDataset
from core.model_registry import make_experiment_checkpoint_path, register_training_checkpoint
from core.pose_cache_build import (
    default_pose_cache_path,
    load_pose_cache_bundle,
    media_pipe_fill_pose_cache,
)
from core.pose_utils import PoseEstimator
from core.seed_utils import set_seed
from core.split import video_level_split
from core.st_tr_vit_fusion import (
    build_st_tr_vit_fusion,
    default_st_tr_vit_checkpoint_path,
    load_st_tr_skeleton_branch,
)
from core.training_progress import tqdm_train_batches

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _imagenet_norm_video(frames: torch.Tensor, device: torch.device) -> torch.Tensor:
    B, T, C, H, W = frames.shape
    x = frames.view(B * T, C, H, W)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.view(B, T, C, H, W)


def _build_train_transform(aug_strength: str):
    if aug_strength == "strong":
        return v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                v2.RandomGrayscale(p=0.1),
                v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),
            ]
        )
    if aug_strength == "medium":
        return v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.92, 1.08)),
                v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.06),
                v2.RandomGrayscale(p=0.06),
                v2.RandomErasing(p=0.15, scale=(0.02, 0.1)),
            ]
        )
    if aug_strength == "mild":
        return v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)),
                v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.04),
                v2.RandomGrayscale(p=0.03),
                v2.RandomErasing(p=0.08, scale=(0.02, 0.08)),
            ]
        )
    raise ValueError(f"aug_strength must be strong|medium|mild, got {aug_strength!r}")


class FramePoseDataset(Dataset):
    def __init__(self, frame_dataset, pose_cache):
        self.frame_dataset = frame_dataset
        self.pose_cache = pose_cache

    def __len__(self):
        return len(self.frame_dataset)

    def __getitem__(self, idx):
        frames, labels = self.frame_dataset[idx]
        pose = self.pose_cache[idx].clone()
        return frames, pose, labels


def _build_pose_cache(dataset, list_file, cache_path, seed=42):
    n_expected = len(dataset)
    out = load_pose_cache_bundle(cache_path)
    if out is not None:
        pose_cache = out["pose_cache"]
        if pose_cache.shape[0] == n_expected:
            return pose_cache, out.get("task_classes")
        print(
            f"Pose cache length ({pose_cache.shape[0]}) != dataset ({n_expected}); rebuilding."
        )

    set_seed(seed)
    pose_estimator = PoseEstimator()
    dataset_raw = FineBadmintonDataset(
        dataset.data_root,
        list_file,
        transform=None,
        sequence_length=dataset.sequence_length,
        frame_interval=dataset.frame_interval,
    )
    pose_cache = media_pipe_fill_pose_cache(dataset_raw, pose_estimator)

    task_classes = {k: len(v) for k, v in dataset.classes.items()}
    task_classes["quality"] = 7
    if "stroke_subtype" in task_classes:
        del task_classes["stroke_subtype"]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"pose_cache": pose_cache, "task_classes": task_classes}, cache_path)
    print(f"Saved pose cache to {cache_path}")
    return pose_cache, task_classes


def _task_loss_weights(stroke_loss_weight, aux_loss_weight):
    return {
        "stroke_type": stroke_loss_weight,
        "position": aux_loss_weight,
        "technique": aux_loss_weight,
        "placement": aux_loss_weight,
        "intent": aux_loss_weight,
        "quality": aux_loss_weight,
    }


def _multitask_batch_loss(
    logits_dict,
    labels,
    *,
    loss_weights,
    criterion_st,
    criterion_default,
    stroke_only=False,
):
    batch_loss = torch.tensor(0.0, device=next(iter(logits_dict.values())).device)
    for task, logits in logits_dict.items():
        if stroke_only and task != "stroke_type":
            continue
        crit = criterion_st if task == "stroke_type" else criterion_default
        w = loss_weights.get(task, 1.0)
        if w <= 0:
            continue
        batch_loss = batch_loss + w * crit(logits, labels[task])
    return batch_loss


def train_st_tr_vit(
    data_root,
    list_file,
    epochs=60,
    batch_size=4,
    lr=1e-4,
    device="cpu",
    save_path=None,
    pose_cache_path=None,
    resume_checkpoint=None,
    resume_skeleton=None,
    start_epoch=0,
    seed=42,
    stream="both",
    registry_experiment=False,
    balanced_sampler=False,
    stroke_loss_weight=5.0,
    aux_loss_weight=0.25,
    stroke_only_epochs=8,
    scheduler_t0=30,
    scheduler_t_mult=2,
    scheduler_eta_min=1e-5,
    label_smoothing=0.1,
    drop_connect=False,
    accumulation_steps=4,
    vit_model_name="vit_small_patch16_224",
    vit_embed_dim=128,
    vit_unfreeze_last_n=2,
    vit_pretrained=True,
    skel_lr_mult=0.25,
    vit_lr_mult=1.0,
    fusion_lr_mult=5.0,
    weight_decay=1e-2,
    aug_strength="medium",
    freeze_skeleton_epochs=0,
):
    set_seed(seed)

    backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if save_path is None:
        save_path = default_st_tr_vit_checkpoint_path(backend_root)
    if pose_cache_path is None:
        pose_cache_path = default_pose_cache_path(backend_root)
    if registry_experiment:
        save_path = make_experiment_checkpoint_path(save_path)

    loss_weights = _task_loss_weights(stroke_loss_weight, aux_loss_weight)
    mlflow.set_experiment("IsoCourt_Training_ST_TR_ViT")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "seed": seed,
                "stream": stream,
                "script": "train_st_tr_vit.py",
                "balanced_sampler": balanced_sampler,
                "stroke_loss_weight": stroke_loss_weight,
                "aux_loss_weight": aux_loss_weight,
                "stroke_only_epochs": stroke_only_epochs,
                "scheduler_t0": scheduler_t0,
                "vit_model_name": vit_model_name,
                "vit_unfreeze_last_n": vit_unfreeze_last_n,
                "skel_lr_mult": skel_lr_mult,
                "vit_lr_mult": vit_lr_mult,
                "fusion_lr_mult": fusion_lr_mult,
                "aug_strength": aug_strength,
                "freeze_skeleton_epochs": freeze_skeleton_epochs,
            }
        )

        train_transform = _build_train_transform(aug_strength)
        dataset = FineBadmintonDataset(data_root, list_file, transform=train_transform)
        val_dataset = FineBadmintonDataset(data_root, list_file, transform=None)

        pose_cache, task_classes = _build_pose_cache(
            dataset, list_file, pose_cache_path, seed=seed
        )
        if task_classes is None:
            task_classes = {k: len(v) for k, v in dataset.classes.items()}
            task_classes["quality"] = 7
            if "stroke_subtype" in task_classes:
                del task_classes["stroke_subtype"]

        wrapper_train = FramePoseDataset(dataset, pose_cache)
        wrapper_val = FramePoseDataset(val_dataset, pose_cache)

        train_indices, val_indices = video_level_split(dataset.samples)
        train_subset = Subset(wrapper_train, train_indices)
        val_subset = Subset(wrapper_val, val_indices)

        loader_kw = dict(batch_size=batch_size, num_workers=0, pin_memory=(device == "cuda"))
        if balanced_sampler:
            st_labels = [
                dataset._map_labels(dataset.samples[i])["stroke_type"] for i in train_indices
            ]
            train_st_labels = torch.tensor(st_labels)
            class_counts = torch.bincount(train_st_labels)
            class_weights = 1.0 / (class_counts.float() + 1e-6)
            sample_weights = class_weights[train_st_labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_loader = DataLoader(train_subset, sampler=sampler, **loader_kw)
        else:
            gen = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                train_subset, shuffle=True, generator=gen, **loader_kw
            )
        val_loader = DataLoader(val_subset, shuffle=False, **loader_kw)

        T = int(dataset.sequence_length)
        model = build_st_tr_vit_fusion(
            task_classes,
            window_size=T,
            stream=stream,
            dropout=0.1,
            model_kwargs={"drop_connect": drop_connect},
            vit_model_name=vit_model_name,
            vit_embed_dim=vit_embed_dim,
            vit_unfreeze_last_n=vit_unfreeze_last_n,
            vit_pretrained=vit_pretrained,
        ).to(device)

        best_acc = 0.0
        if resume_skeleton and os.path.exists(resume_skeleton):
            load_st_tr_skeleton_branch(model, resume_skeleton, device=device)
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
            if "st_tr_vit" in ckpt:
                model.load_state_dict(ckpt["st_tr_vit"], strict=False)
                best_acc = float(ckpt.get("best_acc", 0.0))
                print(
                    f"Loaded ST-TR+ViT from {resume_checkpoint} "
                    f"(best val stroke {best_acc:.1f}%)"
                )

        vit_params, skel_params, fusion_params = [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("vit_encoder."):
                vit_params.append(p)
            elif name.startswith("st_tr."):
                skel_params.append(p)
            else:
                fusion_params.append(p)

        optimizer = optim.AdamW(
            [
                {"params": vit_params, "lr": lr * vit_lr_mult, "weight_decay": weight_decay},
                {"params": skel_params, "lr": lr * skel_lr_mult, "weight_decay": weight_decay},
                {
                    "params": fusion_params,
                    "lr": lr * fusion_lr_mult,
                    "weight_decay": weight_decay,
                },
            ]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_t0,
            T_mult=scheduler_t_mult,
            eta_min=scheduler_eta_min,
        )

        weights_st = torch.tensor(
            [1.0, 1.5, 1.3, 2.0, 1.5, 1.5, 1.5, 2.0, 5.0],
            dtype=torch.float32,
            device=device,
        )
        criterion_st = nn.CrossEntropyLoss(weight=weights_st, label_smoothing=label_smoothing)
        criterion_default = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"\nST-TR+ViT | T={T} stream={stream} | trainable params: {trainable:,} | "
            f"vit={vit_model_name} unfreeze_last={vit_unfreeze_last_n}"
        )
        print(
            f"LR base={lr} | skel×{skel_lr_mult} vit×{vit_lr_mult} fusion×{fusion_lr_mult} | "
            f"stroke/aux={stroke_loss_weight}/{aux_loss_weight} | "
            f"stroke_only_epochs={stroke_only_epochs} | aug={aug_strength}"
        )

        for epoch in range(start_epoch, epochs):
            stroke_only = epoch < stroke_only_epochs
            freeze_skel = epoch < freeze_skeleton_epochs

            if freeze_skel:
                for p in model.st_tr.parameters():
                    p.requires_grad = False
            else:
                for p in model.st_tr.parameters():
                    p.requires_grad = True
                for p in model.vit_encoder.vit.parameters():
                    n = vit_unfreeze_last_n
                    p.requires_grad = False
                if vit_unfreeze_last_n > 0:
                    for blk in model.vit_encoder.vit.blocks[-vit_unfreeze_last_n:]:
                        for p in blk.parameters():
                            p.requires_grad = True

            model.train()
            running_loss = 0.0
            train_correct = {k: 0 for k in task_classes}
            train_total = 0
            optimizer.zero_grad(set_to_none=True)

            pbar = tqdm_train_batches(train_loader, epoch + 1, epochs)
            for batch_idx, (frames, poses, labels) in enumerate(pbar):
                frames = _imagenet_norm_video(frames.to(device), device)
                poses = poses.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}

                logits_dict = model(frames, poses)
                batch_loss = _multitask_batch_loss(
                    logits_dict,
                    labels,
                    loss_weights=loss_weights,
                    criterion_st=criterion_st,
                    criterion_default=criterion_default,
                    stroke_only=stroke_only,
                )
                for task, logits in logits_dict.items():
                    _, pred = logits.max(1)
                    train_correct[task] += (pred == labels[task]).sum().item()
                    if task == "stroke_type":
                        train_total += labels[task].size(0)

                (batch_loss / accumulation_steps).backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                running_loss += batch_loss.item()
                pbar.set_postfix(loss=running_loss / (batch_idx + 1))

            if (batch_idx + 1) % accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss = running_loss / len(train_loader)
            train_acc = 100.0 * train_correct["stroke_type"] / max(train_total, 1)
            scheduler.step(epoch)

            model.eval()
            val_correct = {k: 0 for k in task_classes}
            val_total = 0
            val_running_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for frames, poses, labels in val_loader:
                    frames = _imagenet_norm_video(frames.to(device), device)
                    poses = poses.to(device)
                    labels = {k: v.to(device) for k, v in labels.items()}
                    logits_dict = model(frames, poses)
                    val_running_loss += _multitask_batch_loss(
                        logits_dict,
                        labels,
                        loss_weights=loss_weights,
                        criterion_st=criterion_st,
                        criterion_default=criterion_default,
                        stroke_only=False,
                    ).item()
                    val_batches += 1
                    val_total += poses.size(0)
                    for task, logits in logits_dict.items():
                        _, pred = logits.max(1)
                        val_correct[task] += (pred == labels[task]).sum().item()

            val_acc = 100.0 * val_correct["stroke_type"] / val_total
            val_pos = 100.0 * val_correct["position"] / val_total
            val_loss = val_running_loss / max(val_batches, 1)
            mlflow.log_metrics(
                {
                    "train_loss": epoch_loss,
                    "train_type_acc": train_acc,
                    "val_loss": val_loss,
                    "val_type_acc": val_acc,
                    "val_pos_acc": val_pos,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "stroke_only_phase": float(stroke_only),
                },
                step=epoch,
            )
            phase = " [stroke-only]" if stroke_only else ""
            print(
                f"Epoch {epoch+1:3d}{phase} | Loss: {epoch_loss:.4f} | ValLoss: {val_loss:.4f} | "
                f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | "
                f"Val Pos: {val_pos:.1f}% | LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(
                    {
                        "st_tr_vit": model.state_dict(),
                        "task_classes": task_classes,
                        "stream": stream,
                        "vit_model_name": vit_model_name,
                        "vit_embed_dim": vit_embed_dim,
                        "best_acc": best_acc,
                        "epoch": epoch + 1,
                    },
                    save_path,
                )
                print(f"  -> Saved best ({best_acc:.1f}%)")
                register_training_checkpoint(
                    os.path.dirname(save_path),
                    category="st_tr_vit",
                    file_basename=os.path.basename(save_path),
                    meta={
                        "accuracy": round(best_acc, 2),
                        "epoch": epoch + 1,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "script": "train_st_tr_vit.py",
                        "architecture": "st_tr_vit",
                        "stream": stream,
                        "vit_model_name": vit_model_name,
                    },
                    experiment=registry_experiment,
                )

        print(f"\nTraining finished! Best stroke_type accuracy: {best_acc:.1f}%")


def main() -> None:
    p = argparse.ArgumentParser(description="Train ST-TR + ViT late fusion.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--stream", choices=["spatial", "temporal", "both"], default="both")
    p.add_argument("--balanced-sampler", action="store_true")
    p.add_argument("--stroke-loss-weight", type=float, default=5.0)
    p.add_argument("--aux-loss-weight", type=float, default=0.25)
    p.add_argument("--stroke-only-epochs", type=int, default=8)
    p.add_argument("--scheduler-t0", type=int, default=30)
    p.add_argument("--scheduler-t-mult", type=int, default=2)
    p.add_argument("--scheduler-eta-min", type=float, default=1e-5)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--drop-connect", action="store_true")
    p.add_argument("--accum-steps", type=int, default=4)
    p.add_argument("--vit-model", default="vit_small_patch16_224")
    p.add_argument("--vit-embed-dim", type=int, default=128)
    p.add_argument("--vit-unfreeze-last-n", type=int, default=2)
    p.add_argument("--no-vit-pretrained", action="store_true")
    p.add_argument("--skel-lr-mult", type=float, default=0.25)
    p.add_argument("--vit-lr-mult", type=float, default=1.0)
    p.add_argument("--fusion-lr-mult", type=float, default=5.0)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--aug", choices=["strong", "medium", "mild"], default="medium")
    p.add_argument(
        "--resume-skeleton",
        type=str,
        default=None,
        help="Pose-only ST-TR .pth (key st_tr) to warm-start skeleton trunk.",
    )
    p.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Full ST-TR+ViT .pth (key st_tr_vit).",
    )
    p.add_argument("--start-epoch", type=int, default=0)
    p.add_argument("--registry-experiment", action="store_true")
    args = p.parse_args()

    backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_root = os.path.join(backend_root, "data")
    list_file = os.path.join(
        backend_root, "data", "transformed_combined_rounds_output_en_evals_translated.json"
    )
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")
    train_st_tr_vit(
        data_root=data_root,
        list_file=list_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        stream=args.stream,
        registry_experiment=args.registry_experiment,
        balanced_sampler=args.balanced_sampler,
        stroke_loss_weight=args.stroke_loss_weight,
        aux_loss_weight=args.aux_loss_weight,
        stroke_only_epochs=args.stroke_only_epochs,
        scheduler_t0=args.scheduler_t0,
        scheduler_t_mult=args.scheduler_t_mult,
        scheduler_eta_min=args.scheduler_eta_min,
        label_smoothing=args.label_smoothing,
        drop_connect=args.drop_connect,
        accumulation_steps=args.accum_steps,
        vit_model_name=args.vit_model,
        vit_embed_dim=args.vit_embed_dim,
        vit_unfreeze_last_n=args.vit_unfreeze_last_n,
        vit_pretrained=not args.no_vit_pretrained,
        skel_lr_mult=args.skel_lr_mult,
        vit_lr_mult=args.vit_lr_mult,
        fusion_lr_mult=args.fusion_lr_mult,
        weight_decay=args.weight_decay,
        aug_strength=args.aug,
        resume_skeleton=args.resume_skeleton,
        resume_checkpoint=args.resume_checkpoint,
        start_epoch=args.start_epoch,
    )


if __name__ == "__main__":
    main()
