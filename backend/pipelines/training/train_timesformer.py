"""
TimeSformer (divided space-time) + MediaPipe pose token training.

Same dataset, video_level_split, augmentations, losses, and sampler as train_staeformer.py.
Default pose cache path matches STAEformer so MediaPipe runs once if you already cached.

Train/val indices come only from video_level_split (no random clip leakage). Pretrained ViT
weights are ImageNet — they do not encode your labels. Watch val_loss vs train_loss to spot
overfitting; val_type_acc is used for checkpointing.

Hyperparameter sweep ranges: see TIMESFORMER_HYPERPARAMS.md next to this script.
"""
import os
import sys
import json
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
from core.timesformer import TimeSformerPoseModel
from core.training_progress import DEFAULT_TRAIN_BATCH_SIZE, tqdm_pose_cache_build, tqdm_train_batches

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _multitask_loss(logits_dict, labels, device, criterion_st, criterion_default, loss_weights):
    total = torch.tensor(0.0, device=device)
    for task, logits in logits_dict.items():
        crit = criterion_st if task == "stroke_type" else criterion_default
        total = total + loss_weights.get(task, 1.0) * crit(logits, labels[task])
    return total


def _build_train_transform(aug_strength: str):
    """strong = same as STAEformer; medium/mild shrink aug for val alignment experiments."""
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
    """frames: (B, T, C, H, W) in [0,1]"""
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


def _build_pose_cache(dataset, list_file, cache_path, seed=42):
    if os.path.exists(cache_path):
        print(f"Loading pose cache from {cache_path}...")
        out = torch.load(cache_path, map_location="cpu", weights_only=False)
        return out["pose_cache"], out.get("task_classes")

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


def train_timesformer(
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
    depth=4,
    num_heads=4,
    max_train_batches=None,
    backbone="scratch",
    vit_model_name="vit_small_patch16_224",
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
):
    set_seed(seed)

    _dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(_dir))
    if save_path is None:
        save_path = os.path.join(backend_root, "models", "badminton_model_timesformer.pth")
    if pose_cache_path is None:
        # Share cache with STAEformer training when present
        pose_cache_path = os.path.join(backend_root, "models", "pose_cache_staeformer.pt")

    mlflow.set_experiment("IsoCourt_Training_TimeSformer")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "seed": seed,
                "embed_dim": embed_dim,
                "depth": depth,
                "num_heads": num_heads,
                "max_train_batches": max_train_batches,
                "backbone": backbone,
                "vit_model_name": vit_model_name if backbone == "vit" else "n/a",
                "vit_unfreeze_last_n": vit_unfreeze_last_n if backbone == "vit" else 0,
                "weight_decay": weight_decay,
                "label_smoothing": label_smoothing,
                "lr_mult": lr_mult,
                "vit_lr_mult": vit_lr_mult,
                "scheduler_t0": scheduler_t0,
                "scheduler_t_mult": scheduler_t_mult,
                "accumulation_steps": accumulation_steps,
                "stroke_loss_weight": stroke_loss_weight,
                "aug_strength": aug_strength,
                "script": "train_timesformer.py",
            }
        )

        train_transform = _build_train_transform(aug_strength)

        print("Loading dataset...")
        dataset = FineBadmintonDataset(data_root, list_file, transform=train_transform)

        pose_cache, task_classes = _build_pose_cache(dataset, list_file, pose_cache_path, seed=seed)
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

        model = TimeSformerPoseModel(
            task_classes=task_classes,
            num_frames=dataset.sequence_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            backbone=backbone,
            vit_model_name=vit_model_name,
            vit_unfreeze_last_n=vit_unfreeze_last_n,
        ).to(device)

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=False)
                print("Loaded TimeSformer from checkpoint")

        trainable = [p for p in model.parameters() if p.requires_grad]
        print(
            f"Backbone={backbone} | trainable params: {sum(p.numel() for p in trainable):,} "
            f"(vit_unfreeze_last_n={vit_unfreeze_last_n})"
        )

        vit_params = []
        other_params = []
        if backbone == "vit" and model.vit is not None:
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

        print("\nStarting TimeSformer + pose training...")
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
                labels = {k: v.to(device) for k, v in labels.items()}

                logits_dict = model(frames, poses)
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
                    logits_dict = model(frames, poses)
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
                        "architecture": "timesformer",
                        "num_frames": model.num_frames,
                        "embed_dim": embed_dim,
                        "depth": depth,
                        "num_heads": num_heads,
                        "backbone": backbone,
                        "vit_model_name": vit_model_name,
                        "vit_unfreeze_last_n": vit_unfreeze_last_n,
                    },
                    save_path,
                )
                print(f"  -> Saved best ({best_acc:.1f}%)")
                registry_path = os.path.join(os.path.dirname(save_path), "model_registry.json")
                try:
                    with open(registry_path, "r") as f:
                        registry = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    registry = {"models": {}, "active_model": None}
                name = os.path.basename(save_path)
                registry["models"][name] = {
                    "accuracy": round(best_acc, 2),
                    "epoch": epoch + 1,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "script": "train_timesformer.py",
                }
                registry["active_model"] = name
                with open(registry_path, "w") as f:
                    json.dump(registry, f, indent=2)

            if (epoch + 1) % 10 == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "task_classes": task_classes,
                        "architecture": "timesformer",
                        "num_frames": model.num_frames,
                        "embed_dim": embed_dim,
                        "depth": depth,
                        "num_heads": num_heads,
                        "backbone": backbone,
                        "vit_model_name": vit_model_name,
                        "vit_unfreeze_last_n": vit_unfreeze_last_n,
                    },
                    f"{save_path}_epoch_{epoch+1}.pth",
                )

        print(f"\nTraining finished! Best stroke_type accuracy: {best_acc:.1f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Hyperparameter sweep ranges: TIMESFORMER_HYPERPARAMS.md (this folder).",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_TRAIN_BATCH_SIZE,
        help=f"Train/val minibatch size (default {DEFAULT_TRAIN_BATCH_SIZE}, same as CNN/STAEformer)",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
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
        "--backbone",
        type=str,
        choices=("scratch", "vit"),
        default="vit",
        help="scratch=Conv patch embed; vit=timm ViT (ImageNet) per-frame tokens + projection",
    )
    parser.add_argument(
        "--vit-model",
        type=str,
        default="vit_small_patch16_224",
        help="timm ViT name when --backbone vit (e.g. vit_small_patch16_224, vit_base_patch16_224)",
    )
    parser.add_argument(
        "--vit-unfreeze-last-n",
        type=int,
        default=0,
        help="Unfreeze last N ViT transformer blocks (0=frozen ViT; try 2–4 after val plateaus)",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Lower (e.g. 0.05) can raise train acc; 0 risks overfitting on small data",
    )
    parser.add_argument(
        "--lr-mult",
        type=float,
        default=5.0,
        help="AdamW lr = --lr × this for non-ViT params (search ~3–8)",
    )
    parser.add_argument(
        "--vit-lr-mult",
        type=float,
        default=0.25,
        help="When --vit-unfreeze-last-n > 0: ViT lr = --lr × this (search ~0.1–0.5)",
    )
    parser.add_argument(
        "--scheduler-t0",
        type=int,
        default=10,
        help="CosineAnnealingWarmRestarts period in epochs (try 10–30)",
    )
    parser.add_argument(
        "--scheduler-t-mult",
        type=int,
        default=2,
        help="Restart cycle multiplier (1 = fixed-length restarts, 2 = doubles each time)",
    )
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (2–8)",
    )
    parser.add_argument(
        "--stroke-loss-weight",
        type=float,
        default=2.0,
        help="Loss multiplier for stroke_type vs other heads (try 1.5–3)",
    )
    parser.add_argument(
        "--aug",
        type=str,
        choices=("strong", "medium", "mild"),
        default="strong",
        help="Train augmentation strength (medium/mild if val lags train)",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to .pth with key 'model' (e.g. models/badminton_model_timesformer.pth). "
        "Loads weights only; use --start-epoch to continue epoch numbering toward --epochs.",
    )
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="0-based epoch index to begin the training loop (e.g. 36 after completing 36 epochs "
        "to run epochs 37..--epochs). Optimizer/scheduler state is not restored.",
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
        resume = os.path.join(backend_root, "models", "badminton_model_timesformer.pth")
        print(f"--start-epoch {args.start_epoch} without --resume-checkpoint: defaulting to {resume}")
    if args.start_epoch > 0:
        if not resume or not os.path.isfile(resume):
            print(f"ERROR: need an existing checkpoint when --start-epoch > 0; missing: {resume!r}")
            sys.exit(1)
    train_timesformer(
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
        depth=args.depth,
        num_heads=args.num_heads,
        max_train_batches=args.max_train_batches,
        backbone=args.backbone,
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
    )
