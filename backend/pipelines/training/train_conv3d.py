"""
R(2+1)D / R3D / MC3 (torchvision) + MediaPipe late fusion — same training contract as ``train_videomae.py``:

``video_level_split``, weighted sampler, augmentations, multitask losses, MLflow, registry category
``conv3d_pose``, checkpoint ``badminton_model_conv3d_pose.pth`` by default.

**Interpretability:** explicit RGB 3D trunk vs skeleton MLP; see ``core/conv3d_pose.Conv3DPoseMultitaskModel.grad_cam_target_module``.
"""
import os
import sys
import datetime
import importlib.util

_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import mlflow

from core.conv3d_pose import Conv3DPoseMultitaskModel, backbone_parameter_groups
from core.dataset import FineBadmintonDataset
from core.seed_utils import set_seed
from core.split import video_level_split
from core.training_progress import DEFAULT_TRAIN_BATCH_SIZE, tqdm_train_batches
from core.model_registry import register_training_checkpoint

_tf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_timesformer.py")
_spec = importlib.util.spec_from_file_location("train_timesformer", _tf_path)
_tf = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_tf)

_multitask_loss = _tf._multitask_loss
_build_train_transform = _tf._build_train_transform
FramePoseDataset = _tf.FramePoseDataset
_build_pose_cache = _tf._build_pose_cache
_imagenet_norm_video = _tf._imagenet_norm_video


def train_conv3d(
    data_root,
    list_file,
    epochs=60,
    batch_size=DEFAULT_TRAIN_BATCH_SIZE,
    lr=1e-4,
    device="cpu",
    save_path=None,
    pose_cache_path=None,
    resume_checkpoint=None,
    start_epoch=0,
    seed=42,
    max_train_batches=None,
    video_backbone="r2plus1d_18",
    spatial_size=112,
    pretrained=True,
    freeze_3d=True,
    unfreeze_layer4=True,
    lr_mult=5.0,
    backbone_lr_mult=0.1,
    weight_decay=1e-2,
    label_smoothing=0.1,
    scheduler_t0=10,
    scheduler_t_mult=2,
    accumulation_steps=4,
    stroke_loss_weight=2.0,
    aug_strength="strong",
    registry_experiment=False,
    use_pose=True,
):
    set_seed(seed)

    _dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(_dir))
    if save_path is None:
        save_path = os.path.join(backend_root, "models", "badminton_model_conv3d_pose.pth")
    if pose_cache_path is None:
        pose_cache_path = os.path.join(backend_root, "models", "pose_cache_staeformer.pt")

    mlflow.set_experiment("IsoCourt_Training_Conv3D_Pose")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "seed": seed,
                "video_backbone": video_backbone,
                "spatial_size": spatial_size,
                "pretrained": pretrained,
                "freeze_3d": freeze_3d,
                "unfreeze_layer4": unfreeze_layer4,
                "lr_mult": lr_mult,
                "backbone_lr_mult": backbone_lr_mult,
                "weight_decay": weight_decay,
                "label_smoothing": label_smoothing,
                "scheduler_t0": scheduler_t0,
                "scheduler_t_mult": scheduler_t_mult,
                "accumulation_steps": accumulation_steps,
                "stroke_loss_weight": stroke_loss_weight,
                "aug_strength": aug_strength,
                "use_pose": use_pose,
                "script": "train_conv3d.py",
            }
        )

        train_transform = _build_train_transform(aug_strength)
        print("Loading dataset...")
        dataset = FineBadmintonDataset(data_root, list_file, transform=train_transform)

        pose_cache, task_classes = _build_pose_cache(
            dataset, list_file, pose_cache_path, seed=seed, use_pose=use_pose
        )
        if task_classes is None:
            task_classes = {k: len(v) for k, v in dataset.classes.items()}
            task_classes["quality"] = 7
            if "stroke_subtype" in task_classes:
                del task_classes["stroke_subtype"]

        wrapper = FramePoseDataset(dataset, pose_cache)
        st_labels = []
        for sample in dataset.samples:
            labels = dataset._map_labels(sample)
            st_labels.append(labels["stroke_type"])
        train_indices, val_indices = video_level_split(dataset.samples)
        train_subset = Subset(wrapper, train_indices)
        val_subset = Subset(wrapper, val_indices)

        train_st_labels = torch.tensor([st_labels[i] for i in train_indices])
        class_counts = torch.bincount(train_st_labels)
        class_weights = 1.0 / (class_counts.float() + 1e-6)
        sample_weights = class_weights[train_st_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=(device == "cuda"),
            generator=train_generator,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device == "cuda"),
        )

        model = Conv3DPoseMultitaskModel(
            task_classes=task_classes,
            num_frames=dataset.sequence_length,
            video_backbone=video_backbone,
            spatial_size=spatial_size,
            pretrained=pretrained,
            freeze_backbone=freeze_3d,
            unfreeze_layer4=unfreeze_layer4,
            use_pose=use_pose,
        ).to(device)

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=False)
                print("Loaded Conv3D+pose from checkpoint")

        trainable = [p for p in model.parameters() if p.requires_grad]
        print(
            f"Conv3D {video_backbone} | T={dataset.sequence_length} | spatial_in->{spatial_size} | "
            f"trainable params: {sum(p.numel() for p in trainable):,}"
        )

        bb_params, other_params = backbone_parameter_groups(model)
        if bb_params:
            optimizer = optim.AdamW(
                [
                    {"params": bb_params, "lr": lr * backbone_lr_mult, "weight_decay": weight_decay},
                    {"params": other_params, "lr": lr * lr_mult, "weight_decay": weight_decay},
                ]
            )
            print(
                f"Optimizer: 3D backbone lr={lr * backbone_lr_mult:.6f} | "
                f"fusion/heads lr={lr * lr_mult:.6f}"
            )
        else:
            optimizer = optim.AdamW(trainable, lr=lr * lr_mult, weight_decay=weight_decay)
            print(f"Optimizer: single group lr={lr * lr_mult:.6f}")
        _clip_params = trainable
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=scheduler_t0, T_mult=scheduler_t_mult
        )

        weights_st = torch.tensor(
            [1.0, 1.5, 1.3, 2.0, 1.5, 1.5, 1.5, 2.0, 5.0], dtype=torch.float32, device=device
        )
        criterion_st = nn.CrossEntropyLoss(weight=weights_st, label_smoothing=label_smoothing)
        criterion_default = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        loss_weights = {
            "stroke_type": stroke_loss_weight,
            "position": 1.0,
            "technique": 0.5,
            "placement": 0.5,
            "intent": 0.5,
            "quality": 0.5,
        }
        best_acc = 0.0

        print(f"\nStarting Conv3D + pose training (use_pose={use_pose})...")
        print(
            f"lr_mult={lr_mult} | backbone_lr_mult={backbone_lr_mult} | aug={aug_strength} | "
            f"stroke_loss_weight={stroke_loss_weight} | "
            f"cosine T_0={scheduler_t0} T_mult={scheduler_t_mult} | "
            f"accumulation_steps={accumulation_steps}"
        )

        for epoch in range(start_epoch, epochs):
            model.train()
            running_loss = 0.0
            train_correct = {k: 0 for k in task_classes}
            train_total = 0
            optimizer.zero_grad()

            pbar = tqdm_train_batches(train_loader, epoch + 1, epochs)
            for batch_idx, (frames, poses, labels) in enumerate(pbar):
                frames = _imagenet_norm_video(frames.to(device), device)
                poses = poses.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}

                logits_dict = model(frames, poses if use_pose else None)
                batch_loss = _multitask_loss(
                    logits_dict, labels, device, criterion_st, criterion_default, loss_weights
                )
                for task, logits in logits_dict.items():
                    _, pred = logits.max(1)
                    train_correct[task] += (pred == labels[task]).sum().item()
                    if task == "stroke_type":
                        train_total += labels[task].size(0)

                (batch_loss / accumulation_steps).backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(_clip_params, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                running_loss += batch_loss.item()
                pbar.set_postfix(loss=running_loss / (batch_idx + 1))

                if max_train_batches is not None and (batch_idx + 1) >= max_train_batches:
                    break

            if (batch_idx + 1) % accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(_clip_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            n_tb = batch_idx + 1
            epoch_loss = running_loss / n_tb
            train_acc = 100.0 * train_correct["stroke_type"] / train_total
            train_pos = 100.0 * train_correct["position"] / train_total
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
                    logits_dict = model(frames, poses if use_pose else None)
                    vloss = _multitask_loss(
                        logits_dict, labels, device, criterion_st, criterion_default, loss_weights
                    )
                    val_running_loss += vloss.item()
                    val_batches += 1
                    val_total += poses.size(0)
                    for task, logits in logits_dict.items():
                        _, pred = logits.max(1)
                        val_correct[task] += (pred == labels[task]).sum().item()

            val_acc = 100.0 * val_correct["stroke_type"] / val_total
            val_pos = 100.0 * val_correct["position"] / val_total
            val_loss_epoch = val_running_loss / max(val_batches, 1)

            lrs = [float(g["lr"]) for g in optimizer.param_groups]
            metrics = {
                "train_loss": float(epoch_loss),
                "val_loss": float(val_loss_epoch),
                "train_type_acc": float(train_acc),
                "train_pos_acc": float(train_pos),
                "val_type_acc": float(val_acc),
                "val_pos_acc": float(val_pos),
                "learning_rate": lrs[0],
            }
            if len(lrs) > 1:
                metrics["lr_3d"] = lrs[0]
                metrics["lr_other"] = lrs[1]
            mlflow.log_metrics(metrics, step=epoch)

            lr_str = ", ".join(f"{x:.6f}" for x in lrs)
            print(
                f"Epoch {epoch+1:3d} | train_loss: {epoch_loss:.4f} | val_loss: {val_loss_epoch:.4f} | "
                f"Train Type Acc: {train_acc:.1f}% | Train Pos Acc: {train_pos:.1f}% | "
                f"Val Type Acc: {val_acc:.1f}% | Val Pos Acc: {val_pos:.1f}% | "
                f"LR: [{lr_str}]"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "task_classes": task_classes,
                        "architecture": "conv3d_pose",
                        "num_frames": model.num_frames,
                        "video_backbone": video_backbone,
                        "spatial_size": spatial_size,
                        "pretrained": pretrained,
                        "freeze_3d": freeze_3d,
                        "unfreeze_layer4": unfreeze_layer4,
                        "use_pose": use_pose,
                    },
                    save_path,
                )
                print(f"  -> Saved best ({best_acc:.1f}%)")
                name = os.path.basename(save_path)
                register_training_checkpoint(
                    os.path.dirname(save_path),
                    category="conv3d_pose",
                    file_basename=name,
                    meta={
                        "accuracy": round(best_acc, 2),
                        "epoch": epoch + 1,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "script": "train_conv3d.py",
                        "architecture": "conv3d_pose",
                        "inference": {
                            "num_frames": model.num_frames,
                            "video_backbone": video_backbone,
                            "spatial_size": spatial_size,
                            "pretrained": pretrained,
                            "freeze_3d": freeze_3d,
                            "unfreeze_layer4": unfreeze_layer4,
                            "use_pose": use_pose,
                        },
                    },
                    experiment=registry_experiment,
                )

            if (epoch + 1) % 10 == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "task_classes": task_classes,
                        "architecture": "conv3d_pose",
                        "num_frames": model.num_frames,
                        "video_backbone": video_backbone,
                        "spatial_size": spatial_size,
                        "pretrained": pretrained,
                        "freeze_3d": freeze_3d,
                        "unfreeze_layer4": unfreeze_layer4,
                        "use_pose": use_pose,
                    },
                    f"{save_path}_epoch_{epoch+1}.pth",
                )

        print(f"\nTraining finished! Best stroke_type accuracy: {best_acc:.1f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="R(2+1)D/R3D/MC3 + MediaPipe (registry: conv3d_pose, badminton_model_conv3d_pose.pth)"
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pose-cache", type=str, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument(
        "--video-backbone",
        type=str,
        default="r2plus1d_18",
        choices=("r2plus1d_18", "r3d_18", "mc3_18"),
        help="torchvision.models.video Kinetics400-pretrained trunk (default: R(2+1)D-18).",
    )
    parser.add_argument("--spatial-size", type=int, default=112, help="Square spatial size fed to 3D CNN")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Random-init 3D trunk (no Kinetics download; for smoke tests / small data).",
    )
    parser.add_argument(
        "--no-freeze-3d",
        action="store_true",
        help="Train the full 3D backbone (heavy; lower backbone_lr_mult recommended).",
    )
    parser.add_argument(
        "--no-unfreeze-layer4",
        action="store_true",
        help="When 3D is frozen: do not train layer4 (only fusion+heads).",
    )
    parser.add_argument("--lr-mult", type=float, default=5.0)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--scheduler-t0", type=int, default=10)
    parser.add_argument("--scheduler-t-mult", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--stroke-loss-weight", type=float, default=2.0)
    parser.add_argument("--aug", type=str, choices=("strong", "medium", "mild"), default="strong")
    parser.add_argument(
        "--registry-experiment",
        action="store_true",
        help="Append best checkpoint to registry experiments instead of overwriting conv3d_pose primary.",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="RGB 3D trunk only; skips MediaPipe cache build.",
    )
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--start-epoch", type=int, default=0)

    args = parser.parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(current_dir))
    data_root = os.path.join(backend_root, "data")
    list_file = os.path.join(
        backend_root, "data", "transformed_combined_rounds_output_en_evals_translated.json"
    )
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    train_conv3d(
        data_root=data_root,
        list_file=list_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        pose_cache_path=args.pose_cache,
        max_train_batches=args.max_train_batches,
        resume_checkpoint=args.resume_checkpoint,
        start_epoch=args.start_epoch,
        video_backbone=args.video_backbone,
        spatial_size=args.spatial_size,
        pretrained=not args.no_pretrained,
        freeze_3d=not args.no_freeze_3d,
        unfreeze_layer4=not args.no_unfreeze_layer4,
        lr_mult=args.lr_mult,
        backbone_lr_mult=args.backbone_lr_mult,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        scheduler_t0=args.scheduler_t0,
        scheduler_t_mult=args.scheduler_t_mult,
        accumulation_steps=args.accum_steps,
        stroke_loss_weight=args.stroke_loss_weight,
        aug_strength=args.aug,
        registry_experiment=args.registry_experiment,
        use_pose=not args.no_pose,
    )
