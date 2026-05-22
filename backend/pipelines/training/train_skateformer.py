"""
SkateFormer training for IsoCourt (MediaPipe skeleton, 16 frames, batch 4).

Pose-only model using the ECCV 2024 SkateFormer backbone adapted for MediaPipe
(33 joints, single player). Same data pipeline as ``train_st_tr.py``.

Reference: https://github.com/KAIST-VICLab/SkateFormer
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
import mlflow
from core.dataset import FineBadmintonDataset
from core.pose_cache_build import (
    default_pose_cache_path,
    load_pose_cache_bundle,
    media_pipe_fill_pose_cache,
)
from core.pose_utils import PoseEstimator
from core.seed_utils import set_seed
from core.split import video_level_split
from core.skateformer_model import SkateFormerMultitaskModel, default_skateformer_kwargs
from core.model_registry import register_training_checkpoint
from core.training_progress import DEFAULT_TRAIN_BATCH_SIZE, tqdm_train_batches
from core.training_standards import (
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_MULTITASK_LOSS_WEIGHTS,
    DEFAULT_SEED,
    GRAD_ACCUMULATION_STEPS,
    GRAD_CLIP_NORM,
    SEQUENCE_LENGTH,
    build_task_classes,
    common_mlflow_clip_params,
    configure_mlflow,
    default_list_file,
    default_multitask_criteria,
    load_training_dataset,
    validate_pose_cache,
)


class PoseOnlyDataset(Dataset):
    """Wraps a frame dataset and attaches cached pose (skips loading images)."""

    def __init__(self, frame_dataset, pose_cache):
        self.frame_dataset = frame_dataset
        self.pose_cache = pose_cache

    def __len__(self):
        return len(self.frame_dataset)

    def __getitem__(self, idx):
        _, labels = self.frame_dataset[idx]
        pose = self.pose_cache[idx].clone()
        return pose, labels


def _build_pose_cache(dataset, list_file, device, cache_path, seed=42):
    """Build or load pose cache (same format as other pose-fusion trainers)."""
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
        dataset.data_root, list_file, transform=None,
        sequence_length=dataset.sequence_length, frame_interval=dataset.frame_interval,
    )

    pose_cache = media_pipe_fill_pose_cache(dataset_raw, pose_estimator)

    task_classes = build_task_classes(dataset)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"pose_cache": pose_cache, "task_classes": task_classes}, cache_path)
    print(f"Saved pose cache to {cache_path}")
    return pose_cache, task_classes


def train_skateformer(
    data_root,
    list_file,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_TRAIN_BATCH_SIZE,
    lr=DEFAULT_LR,
    device="cpu",
    save_path=None,
    pose_cache_path=None,
    resume_checkpoint=None,
    start_epoch=0,
    seed=DEFAULT_SEED,
    embed_dim=64,
    num_heads=16,
    registry_experiment=False,
):
    set_seed(seed)

    _dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(_dir))
    if save_path is None:
        save_path = os.path.join(backend_root, "models", "badminton_model_skateformer.pth")
    if pose_cache_path is None:
        pose_cache_path = default_pose_cache_path(backend_root)

    skate_kw = default_skateformer_kwargs(embed_dim=embed_dim, num_heads=num_heads)
    tracking_uri = configure_mlflow(backend_root)

    mlflow.set_experiment("IsoCourt_Training_SkateFormer")
    with mlflow.start_run():
        mlflow.log_params(common_mlflow_clip_params(
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            seed=seed,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_frames=skate_kw["num_frames"],
            partition_type_1=list(skate_kw["type_1_size"]),
            partition_type_2=list(skate_kw["type_2_size"]),
            mlflow_tracking_uri=tracking_uri,
            script="train_skateformer.py",
        ))

        print("Loading dataset...")
        dataset = load_training_dataset(data_root, list_file, transform=None)

        pose_cache, task_classes = _build_pose_cache(
            dataset, list_file, device, pose_cache_path, seed=seed
        )
        validate_pose_cache(pose_cache)
        if task_classes is None:
            task_classes = build_task_classes(dataset)

        wrapper = PoseOnlyDataset(dataset, pose_cache)

        st_labels = [dataset._map_labels(s)["stroke_type"] for s in dataset.samples]
        train_indices, val_indices = video_level_split(dataset.samples)
        train_subset = Subset(wrapper, train_indices)
        val_subset = Subset(wrapper, val_indices)

        train_st_labels = torch.tensor([st_labels[i] for i in train_indices])
        class_counts = torch.bincount(train_st_labels)
        class_weights = 1.0 / (class_counts.float() + 1e-6)
        sample_weights = class_weights[train_st_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True,
        )
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, sampler=sampler,
            num_workers=0, pin_memory=(device == "cuda"),
            generator=torch.Generator().manual_seed(seed),
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=(device == "cuda"),
        )

        model = SkateFormerMultitaskModel(
            task_classes=task_classes,
            skateformer_kwargs=skate_kw,
        ).to(device)

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
            if "skateformer" in ckpt:
                model.load_state_dict(ckpt["skateformer"], strict=False)
                print("Loaded SkateFormer from checkpoint")

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"SkateFormer params: {total_params:,}")

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        criterion_st, criterion_default = default_multitask_criteria(device)
        loss_weights = dict(DEFAULT_MULTITASK_LOSS_WEIGHTS)
        accumulation_steps = GRAD_ACCUMULATION_STEPS
        best_acc = 0.0

        print(
            f"\nStarting SkateFormer training (T={skate_kw['num_frames']}, "
            f"J={skate_kw['num_points']})..."
        )
        print(f"LR: {lr} | Batch: {batch_size} | Embed: {embed_dim} | Heads: {num_heads}")

        for epoch in range(start_epoch, epochs):
            model.train()
            running_loss = 0.0
            train_correct = {k: 0 for k in task_classes}
            train_total = 0
            optimizer.zero_grad()

            pbar = tqdm_train_batches(train_loader, epoch + 1, epochs)
            for batch_idx, (poses, labels) in enumerate(pbar):
                poses = poses.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}

                logits_dict = model(poses)

                batch_loss = torch.tensor(0.0, device=device)
                for task, logits in logits_dict.items():
                    crit = criterion_st if task == "stroke_type" else criterion_default
                    batch_loss += loss_weights.get(task, 1.0) * crit(logits, labels[task])
                    _, pred = logits.max(1)
                    train_correct[task] += (pred == labels[task]).sum().item()
                    if task == "stroke_type":
                        train_total += labels[task].size(0)

                (batch_loss / accumulation_steps).backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                    optimizer.step()
                    optimizer.zero_grad()
                running_loss += batch_loss.item()
                pbar.set_postfix(loss=running_loss / (batch_idx + 1))

            if (batch_idx + 1) % accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss = running_loss / len(train_loader)
            train_acc = 100.0 * train_correct["stroke_type"] / train_total
            scheduler.step(epoch)

            model.eval()
            val_correct = {k: 0 for k in task_classes}
            val_total = 0
            with torch.no_grad():
                for poses, labels in val_loader:
                    poses = poses.to(device)
                    labels = {k: v.to(device) for k, v in labels.items()}
                    logits_dict = model(poses)
                    val_total += poses.size(0)
                    for task, logits in logits_dict.items():
                        _, pred = logits.max(1)
                        val_correct[task] += (pred == labels[task]).sum().item()

            val_acc = 100.0 * val_correct["stroke_type"] / val_total
            val_pos = 100.0 * val_correct["position"] / val_total
            mlflow.log_metrics({
                "train_loss": epoch_loss, "train_type_acc": train_acc,
                "val_type_acc": val_acc, "val_pos_acc": val_pos,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }, step=epoch)
            print(
                f"Epoch {epoch+1:3d} | Loss: {epoch_loss:.4f} | "
                f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | "
                f"Val Pos: {val_pos:.1f}% | LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    "skateformer": model.state_dict(),
                    "task_classes": task_classes,
                    "skateformer_kwargs": skate_kw,
                }, save_path)
                print(f"  -> Saved best ({best_acc:.1f}%)")
                register_training_checkpoint(
                    os.path.dirname(save_path),
                    category="skateformer",
                    file_basename=os.path.basename(save_path),
                    meta={
                        "accuracy": round(best_acc, 2),
                        "epoch": epoch + 1,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "script": "train_skateformer.py",
                        "architecture": "skateformer",
                        "embed_dim": embed_dim,
                        "num_heads": num_heads,
                    },
                    experiment=registry_experiment,
                )
            if (epoch + 1) % 10 == 0:
                torch.save({
                    "skateformer": model.state_dict(),
                    "task_classes": task_classes,
                    "skateformer_kwargs": skate_kw,
                }, f"{save_path}_epoch_{epoch+1}.pth")

        print(f"\nTraining finished! Best stroke_type accuracy: {best_acc:.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SkateFormer on FineBadminton (MediaPipe pose)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument(
        "--registry-experiment", action="store_true",
        help="Append best checkpoint to registry experiments instead of overwriting skateformer primary.",
    )
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(current_dir))
    data_root = os.path.join(backend_root, "data")
    list_file = default_list_file(backend_root)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    train_skateformer(
        data_root=data_root,
        list_file=list_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        registry_experiment=args.registry_experiment,
    )
