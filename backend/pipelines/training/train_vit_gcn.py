"""
timm ViT (per-frame CLS) + fixed skeleton GCN on cached MediaPipe joints.

Matches train_timesformer.py: same dataset, video_level_split, augmentations, multitask
losses (stroke_type weighted), sampler, and val stroke_type accuracy for checkpointing.
Reuses the default pose cache path so MediaPipe runs once if you already trained STAEformer/TimeSformer.

Default ViT is ``vit_tiny_patch16_224`` (faster per epoch than ``vit_small_patch16_224``). Pass
``--vit-model vit_small_patch16_224`` to align with TimeSformer’s default backbone.
"""
import os
import sys
import datetime

_backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision.transforms import v2
import mlflow
from core.dataset import FineBadmintonDataset
from core.pose_utils import PoseEstimator
from core.seed_utils import set_seed
from core.split import video_level_split
from core.vit_gcn import ViTGCNMultitaskModel
from core.training_progress import DEFAULT_TRAIN_BATCH_SIZE, tqdm_pose_cache_build, tqdm_train_batches
from core.model_registry import register_training_checkpoint

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _multitask_loss(logits_dict, labels, device, criterion_st, criterion_default, loss_weights):
    total = torch.tensor(0.0, device=device)
    for task, logits in logits_dict.items():
        crit = criterion_st if task == "stroke_type" else criterion_default
        total = total + loss_weights.get(task, 1.0) * crit(logits, labels[task])
    return total


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


def _imagenet_norm_video(frames, device):
    B, T, C, H, W = frames.shape
    x = frames.view(B * T, C, H, W)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.view(B, T, C, H, W)


class FramePoseDataset(Dataset):
    def __init__(self, frame_dataset, pose_cache):
        self.frame_dataset = frame_dataset
        self.pose_cache = pose_cache

    def __len__(self):
        return len(self.frame_dataset)

    def __getitem__(self, idx):
        frames, labels = self.frame_dataset[idx]
        pose = self.pose_cache[idx]
        if isinstance(pose, torch.Tensor):
            pose = pose.clone()
        return frames, pose, labels


def _build_pose_cache(dataset, list_file, cache_path, seed=42, use_pose=True):
    n_expected = len(dataset)
    T = dataset.sequence_length
    if not use_pose:
        print("use_pose=False: skipping MediaPipe; using zero pose tensors for the dataloader.")
        return torch.zeros(n_expected, T, 33, 3), None
    if os.path.exists(cache_path):
        print(f"Loading pose cache from {cache_path}...")
        out = torch.load(cache_path, map_location="cpu", weights_only=False)
        pose_cache = out["pose_cache"]
        if pose_cache.shape[0] == n_expected:
            return pose_cache, out.get("task_classes")
        print(
            f"Pose cache length ({pose_cache.shape[0]}) does not match dataset ({n_expected}); "
            "rebuilding."
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

    pose_list = []
    for i in tqdm_pose_cache_build(len(dataset_raw)):
        frames, _ = dataset_raw[i]
        with torch.no_grad():
            p = pose_estimator.extract_tensor_poses(frames)
        if p.dim() == 2:
            p = p.view(-1, 33, 3)
        pose_list.append(p.cpu())
    pose_cache = torch.stack(pose_list)

    task_classes = {k: len(v) for k, v in dataset.classes.items()}
    task_classes["quality"] = 7
    if "stroke_subtype" in task_classes:
        del task_classes["stroke_subtype"]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"pose_cache": pose_cache, "task_classes": task_classes}, cache_path)
    print(f"Saved pose cache to {cache_path}")
    return pose_cache, task_classes


def train_vit_gcn(
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
    embed_dim=128,
    gcn_layers=2,
    max_train_batches=None,
    vit_model_name="vit_tiny_patch16_224",
    vit_unfreeze_last_n=0,
    weight_decay=1e-2,
    label_smoothing=0.1,
    lr_mult=5.0,
    vit_lr_mult=0.25,
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
        save_path = os.path.join(backend_root, "models", "badminton_model_vit_gcn.pth")
    if pose_cache_path is None:
        pose_cache_path = os.path.join(backend_root, "models", "pose_cache_staeformer.pt")

    mlflow.set_experiment("IsoCourt_Training_ViT_GCN")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "seed": seed,
                "embed_dim": embed_dim,
                "gcn_layers": gcn_layers,
                "max_train_batches": max_train_batches,
                "vit_model_name": vit_model_name,
                "vit_unfreeze_last_n": vit_unfreeze_last_n,
                "weight_decay": weight_decay,
                "label_smoothing": label_smoothing,
                "lr_mult": lr_mult,
                "vit_lr_mult": vit_lr_mult,
                "scheduler_t0": scheduler_t0,
                "scheduler_t_mult": scheduler_t_mult,
                "accumulation_steps": accumulation_steps,
                "stroke_loss_weight": stroke_loss_weight,
                "aug_strength": aug_strength,
                "use_pose": use_pose,
                "script": "train_vit_gcn.py",
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

        model = ViTGCNMultitaskModel(
            task_classes=task_classes,
            num_frames=dataset.sequence_length,
            embed_dim=embed_dim,
            gcn_layers=gcn_layers,
            vit_model_name=vit_model_name,
            vit_unfreeze_last_n=vit_unfreeze_last_n,
            use_pose=use_pose,
        ).to(device)

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=False)
                print("Loaded ViT+GCN from checkpoint")

        trainable = [p for p in model.parameters() if p.requires_grad]
        print(
            f"ViT+GCN | trainable params: {sum(p.numel() for p in trainable):,} "
            f"(vit_unfreeze_last_n={vit_unfreeze_last_n}, gcn_layers={gcn_layers})"
        )

        vit_params = []
        other_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("vit."):
                vit_params.append(p)
            else:
                other_params.append(p)
        if vit_params:
            optimizer = optim.AdamW(
                [
                    {"params": vit_params, "lr": lr * vit_lr_mult, "weight_decay": weight_decay},
                    {"params": other_params, "lr": lr * lr_mult, "weight_decay": weight_decay},
                ]
            )
            print(
                f"Optimizer: ViT params lr={lr * vit_lr_mult:.6f} | "
                f"other params lr={lr * lr_mult:.6f}"
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

        print(f"\nStarting ViT + {'GCN + ' if use_pose else ''}multitask training (use_pose={use_pose})...")
        print(
            f"lr_mult={lr_mult} | vit_lr_mult={vit_lr_mult} | aug={aug_strength} | "
            f"stroke_loss_weight={stroke_loss_weight} | "
            f"cosine T_0={scheduler_t0} T_mult={scheduler_t_mult} | "
            f"accumulation_steps={accumulation_steps}"
        )
        if max_train_batches is not None:
            print(
                f"NOTE: max_train_batches={max_train_batches} (partial train epochs; "
                "val still uses full val set)"
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
                if poses.dim() == 3:
                    poses = poses.view(poses.size(0), frames.size(1), 33, 3)
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
                    if poses.dim() == 3:
                        poses = poses.view(poses.size(0), frames.size(1), 33, 3)
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
                metrics["lr_vit"] = lrs[0]
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
                        "architecture": "vit_gcn",
                        "num_frames": model.num_frames,
                        "embed_dim": embed_dim,
                        "gcn_layers": gcn_layers,
                        "vit_model_name": vit_model_name,
                        "vit_unfreeze_last_n": vit_unfreeze_last_n,
                        "use_pose": use_pose,
                    },
                    save_path,
                )
                print(f"  -> Saved best ({best_acc:.1f}%)")
                name = os.path.basename(save_path)
                register_training_checkpoint(
                    os.path.dirname(save_path),
                    category="vit_gcn",
                    file_basename=name,
                    meta={
                        "accuracy": round(best_acc, 2),
                        "epoch": epoch + 1,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "script": "train_vit_gcn.py",
                        "architecture": "vit_gcn",
                        "inference": {
                            "num_frames": model.num_frames,
                            "embed_dim": embed_dim,
                            "gcn_layers": gcn_layers,
                            "vit_model_name": vit_model_name,
                            "vit_unfreeze_last_n": vit_unfreeze_last_n,
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
                        "architecture": "vit_gcn",
                        "num_frames": model.num_frames,
                        "embed_dim": embed_dim,
                        "gcn_layers": gcn_layers,
                        "vit_model_name": vit_model_name,
                        "vit_unfreeze_last_n": vit_unfreeze_last_n,
                        "use_pose": use_pose,
                    },
                    f"{save_path}_epoch_{epoch+1}.pth",
                )

        print(f"\nTraining finished! Best stroke_type accuracy: {best_acc:.1f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Same data and metrics as train_timesformer.py; compares ViT+GCN fusion.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_TRAIN_BATCH_SIZE,
        help=f"Train/val minibatch size (default {DEFAULT_TRAIN_BATCH_SIZE})",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument(
        "--gcn-layers",
        type=int,
        default=2,
        help="Number of fixed-graph GCN layers (each: A @ X @ W)",
    )
    parser.add_argument(
        "--pose-cache",
        type=str,
        default=None,
        help="Path to pose .pt cache (default: models/pose_cache_staeformer.pt)",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Stop each train epoch after this many batches (smoke / faster runs; val unchanged)",
    )
    parser.add_argument(
        "--vit-model",
        type=str,
        default="vit_tiny_patch16_224",
        help=(
            "timm ViT backbone. Default vit_tiny_patch16_224 is faster for long epochs; "
            "use vit_small_patch16_224 to match TimeSformer’s default ViT."
        ),
    )
    parser.add_argument(
        "--vit-unfreeze-last-n",
        type=int,
        default=0,
        help="Unfreeze last N ViT transformer blocks (0=frozen ViT)",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--lr-mult", type=float, default=5.0)
    parser.add_argument("--vit-lr-mult", type=float, default=0.25)
    parser.add_argument("--scheduler-t0", type=int, default=10)
    parser.add_argument("--scheduler-t-mult", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--stroke-loss-weight", type=float, default=2.0)
    parser.add_argument(
        "--aug",
        type=str,
        choices=("strong", "medium", "mild"),
        default="strong",
    )
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument(
        "--registry-experiment",
        action="store_true",
        help="Append best checkpoint to registry experiments instead of overwriting vit_gcn primary.",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="ViT-only (no skeleton GCN); skips MediaPipe cache build.",
    )
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(current_dir))
    data_root = os.path.join(backend_root, "data")
    list_file = os.path.join(
        backend_root, "data", "transformed_combined_rounds_output_en_evals_translated.json"
    )
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    resume = args.resume_checkpoint
    if resume is None and args.start_epoch > 0:
        resume = os.path.join(backend_root, "models", "badminton_model_vit_gcn.pth")
        print(f"--start-epoch {args.start_epoch} without --resume-checkpoint: defaulting to {resume}")
    if args.start_epoch > 0:
        if not resume or not os.path.isfile(resume):
            print(f"ERROR: need an existing checkpoint when --start-epoch > 0; missing: {resume!r}")
            sys.exit(1)
    train_vit_gcn(
        data_root=data_root,
        list_file=list_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        resume_checkpoint=resume,
        start_epoch=args.start_epoch,
        pose_cache_path=args.pose_cache,
        embed_dim=args.embed_dim,
        gcn_layers=args.gcn_layers,
        max_train_batches=args.max_train_batches,
        vit_model_name=args.vit_model,
        vit_unfreeze_last_n=args.vit_unfreeze_last_n,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        lr_mult=args.lr_mult,
        vit_lr_mult=args.vit_lr_mult,
        scheduler_t0=args.scheduler_t0,
        scheduler_t_mult=args.scheduler_t_mult,
        accumulation_steps=args.accum_steps,
        stroke_loss_weight=args.stroke_loss_weight,
        aug_strength=args.aug,
        registry_experiment=args.registry_experiment,
        use_pose=not args.no_pose,
    )
