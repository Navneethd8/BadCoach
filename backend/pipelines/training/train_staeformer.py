"""
STAEformer training script.

Two modes controlled by --pose-only / use_cnn flag:
  use_cnn=True  (default)  – CNN backbone provides 2048-dim features as node 34.
  use_cnn=False (pose-only) – 33 pose joints only; no CNN, no frames at training time.

Same data, split, and training setup as train_full.py so results are comparable.
Pose is cached once (from unaugmented frames).
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
from torchvision import models
from torchvision.transforms import v2
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
from core.staeformer import STAEformerModel
from core.model_registry import register_training_checkpoint
from core.training_progress import tqdm_train_batches


# ImageNet normalization (same as CNN_LSTM_Model)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _cnn_sequence(cnn, frames, device):
    """Run CNN on (B, T, C, H, W) -> (B, T, 2048). Frames in [0,1]."""
    B, T, C, H, W = frames.shape
    x = frames.view(B * T, C, H, W)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    feat = cnn(x)
    return feat.view(B, T, -1)


class FramePoseDataset(Dataset):
    """Wraps a frame dataset and attaches pose from cache (same index order)."""

    def __init__(self, frame_dataset, pose_cache):
        self.frame_dataset = frame_dataset
        self.pose_cache = pose_cache  # tensor (N, 16, 33, 3) or list of tensors

    def __len__(self):
        return len(self.frame_dataset)

    def __getitem__(self, idx):
        frames, labels = self.frame_dataset[idx]
        pose = self.pose_cache[idx]
        if isinstance(pose, torch.Tensor):
            pose = pose.clone()
        return frames, pose, labels


def _build_pose_cache(dataset, list_file, device, cache_path, seed=42, use_pose=True):
    """Extract pose for every sample; save to cache. Uses dataset with transform=None."""
    n_expected = len(dataset)
    T = dataset.sequence_length
    if not use_pose:
        print("use_pose=False: skipping MediaPipe; using zero pose tensors for the dataloader.")
        return torch.zeros(n_expected, T, 33, 3), None
    out = load_pose_cache_bundle(cache_path)
    if out is not None:
        pose_cache = out["pose_cache"]
        if pose_cache.shape[0] == n_expected:
            return pose_cache, out.get("task_classes")
        print(
            f"Pose cache length ({pose_cache.shape[0]}) does not match dataset ({n_expected}); "
            "rebuilding."
        )

    set_seed(seed)
    pose_estimator = PoseEstimator()
    # Same data/split as main dataset but no transform, so pose is deterministic
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


def train_staeformer(
    data_root,
    list_file,
    epochs=60,
    batch_size=4,
    lr=1e-4,
    device="cpu",
    save_path=None,
    pose_cache_path=None,
    resume_checkpoint=None,
    start_epoch=0,
    seed=42,
    use_cnn=True,
    registry_experiment=False,
    use_pose=True,
):
    set_seed(seed)
    if not use_pose:
        use_cnn = True

    _dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(_dir))
    if not use_pose:
        suffix = "staeformer_cnn_only"
    elif use_cnn:
        suffix = "staeformer"
    else:
        suffix = "staeformer_pose_only"
    if save_path is None:
        save_path = os.path.join(backend_root, "models", f"badminton_model_{suffix}.pth")
    if pose_cache_path is None:
        pose_cache_path = default_pose_cache_path(backend_root)

    if not use_pose:
        experiment_name = "IsoCourt_Training_STAEformer_CNNOnly"
    elif use_cnn:
        experiment_name = "IsoCourt_Training_STAEformer"
    else:
        experiment_name = "IsoCourt_Training_STAEformer_PoseOnly"
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "use_cnn": use_cnn,
            "use_pose": use_pose,
            "script": "train_staeformer.py",
        })

        train_transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            v2.RandomGrayscale(p=0.1),
            v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])

        print("Loading dataset...")
        dataset = FineBadmintonDataset(data_root, list_file, transform=train_transform)

        # Build or load pose cache (aligned to dataset indices)
        pose_cache, task_classes = _build_pose_cache(
            dataset, list_file, device, pose_cache_path, seed=seed, use_pose=use_pose
        )
        if task_classes is None:
            task_classes = {k: len(v) for k, v in dataset.classes.items()}
            task_classes["quality"] = 7
            if "stroke_subtype" in task_classes:
                del task_classes["stroke_subtype"]

        wrapper = FramePoseDataset(dataset, pose_cache)

        # Video-level split (same as train_full)
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

        cnn = None
        if use_cnn:
            weights = models.ResNet50_Weights.DEFAULT
            resnet = models.resnet50(weights=weights)
            cnn = nn.Sequential(*list(resnet.children())[:-1]).to(device)
            cnn.eval()
            for name, param in cnn.named_parameters():
                param.requires_grad = "7" in name
            if resume_checkpoint and os.path.exists(resume_checkpoint):
                ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
                if "cnn" in ckpt:
                    cnn.load_state_dict(ckpt["cnn"], strict=True)
                    print("Loaded CNN from checkpoint")

        staeformer = STAEformerModel(
            task_classes=task_classes, embed_dim=128, use_cnn=use_cnn, use_pose=use_pose
        ).to(device)
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
            if "staeformer" in ckpt:
                staeformer.load_state_dict(ckpt["staeformer"], strict=False)
                print("Loaded STAEformer from checkpoint")

        param_groups = [{"params": staeformer.parameters(), "lr": lr * 5}]
        if use_cnn:
            cnn_layer4 = [p for n, p in cnn.named_parameters() if "7" in n]
            param_groups.insert(0, {"params": cnn_layer4, "lr": lr * 0.5})
        optimizer = optim.AdamW(param_groups, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        weights_st = torch.tensor([1.0, 1.5, 1.3, 2.0, 1.5, 1.5, 1.5, 2.0, 5.0], dtype=torch.float32, device=device)
        criterion_st = nn.CrossEntropyLoss(weight=weights_st, label_smoothing=0.1)
        criterion_default = nn.CrossEntropyLoss(label_smoothing=0.1)
        loss_weights = {
            "stroke_type": 2.0, "position": 1.0, "technique": 0.5,
            "placement": 0.5, "intent": 0.5, "quality": 0.5,
        }
        accumulation_steps = 4
        best_acc = 0.0

        if not use_pose:
            mode_label = "CNN-only STAEformer (no MediaPipe graph)"
        elif use_cnn:
            mode_label = "CNN + STAEformer"
        else:
            mode_label = "Pose-Only STAEformer"
        print(f"\nStarting {mode_label} training...")
        if use_cnn:
            print(f"Layer4 LR: {lr*0.5:.6f} | STAEformer LR: {lr*5:.6f}")
        else:
            print(f"STAEformer LR: {lr*5:.6f}")

        for epoch in range(start_epoch, epochs):
            if cnn is not None:
                cnn.train()
            staeformer.train()
            running_loss = 0.0
            train_correct = {k: 0 for k in task_classes}
            train_total = 0
            optimizer.zero_grad()

            pbar = tqdm_train_batches(train_loader, epoch + 1, epochs)
            for batch_idx, (frames, poses, labels) in enumerate(pbar):
                poses = poses.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}

                if use_cnn and not use_pose:
                    frames = frames.to(device)
                    cnn_seq = _cnn_sequence(cnn, frames, device)
                    logits_dict = staeformer(cnn_seq=cnn_seq)
                elif use_cnn:
                    frames = frames.to(device)
                    cnn_seq = _cnn_sequence(cnn, frames, device)
                    logits_dict = staeformer(poses, cnn_seq)
                else:
                    logits_dict = staeformer(poses)

                batch_loss = torch.tensor(0.0, device=device)
                for task, logits in logits_dict.items():
                    crit = criterion_st if task == "stroke_type" else criterion_default
                    batch_loss += loss_weights.get(task, 1.0) * crit(logits, labels[task])
                    _, pred = logits.max(1)
                    train_correct[task] += (pred == labels[task]).sum().item()
                    if task == "stroke_type":
                        train_total += labels[task].size(0)

                (batch_loss / accumulation_steps).backward()
                all_params = list(cnn.parameters()) + list(staeformer.parameters()) if cnn is not None else list(staeformer.parameters())
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                running_loss += batch_loss.item()
                pbar.set_postfix(loss=running_loss / (batch_idx + 1))

            if (batch_idx + 1) % accumulation_steps != 0:
                all_params = list(cnn.parameters()) + list(staeformer.parameters()) if cnn is not None else list(staeformer.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss = running_loss / len(train_loader)
            train_acc = 100.0 * train_correct["stroke_type"] / train_total
            scheduler.step(epoch)

            if cnn is not None:
                cnn.eval()
            staeformer.eval()
            val_correct = {k: 0 for k in task_classes}
            val_total = 0
            with torch.no_grad():
                for frames, poses, labels in val_loader:
                    poses = poses.to(device)
                    labels = {k: v.to(device) for k, v in labels.items()}
                    if use_cnn and not use_pose:
                        frames = frames.to(device)
                        cnn_seq = _cnn_sequence(cnn, frames, device)
                        logits_dict = staeformer(cnn_seq=cnn_seq)
                    elif use_cnn:
                        frames = frames.to(device)
                        cnn_seq = _cnn_sequence(cnn, frames, device)
                        logits_dict = staeformer(poses, cnn_seq)
                    else:
                        logits_dict = staeformer(poses)
                    val_total += poses.size(0)
                    for task, logits in logits_dict.items():
                        _, pred = logits.max(1)
                        val_correct[task] += (pred == labels[task]).sum().item()

            val_acc = 100.0 * val_correct["stroke_type"] / val_total
            val_pos = 100.0 * val_correct["position"] / val_total
            mlflow.log_metrics({
                "train_loss": epoch_loss,
                "train_type_acc": train_acc,
                "val_type_acc": val_acc,
                "val_pos_acc": val_pos,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }, step=epoch)
            print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss:.4f} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Val Pos: {val_pos:.1f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                ckpt_data = {
                    "staeformer": staeformer.state_dict(),
                    "task_classes": task_classes,
                    "use_cnn": use_cnn,
                    "use_pose": use_pose,
                }
                if cnn is not None:
                    ckpt_data["cnn"] = cnn.state_dict()
                torch.save(ckpt_data, save_path)
                print(f"  -> Saved best ({best_acc:.1f}%)")
                name = os.path.basename(save_path)
                register_training_checkpoint(
                    os.path.dirname(save_path),
                    category="staeformer",
                    file_basename=name,
                    meta={
                        "accuracy": round(best_acc, 2),
                        "epoch": epoch + 1,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "script": "train_staeformer.py",
                        "architecture": "staeformer",
                        "use_cnn": use_cnn,
                        "use_pose": use_pose,
                    },
                    experiment=registry_experiment,
                )
            if (epoch + 1) % 10 == 0:
                periodic_ckpt = {
                    "staeformer": staeformer.state_dict(),
                    "task_classes": task_classes,
                    "use_cnn": use_cnn,
                    "use_pose": use_pose,
                }
                if cnn is not None:
                    periodic_ckpt["cnn"] = cnn.state_dict()
                torch.save(periodic_ckpt, f"{save_path}_epoch_{epoch+1}.pth")

        print(f"\nTraining finished! Best stroke_type accuracy: {best_acc:.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-only", action="store_true", help="Train on pose only (no CNN backbone)")
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="CNN-only STAEformer (no MediaPipe joints in the graph); implies CNN path.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--registry-experiment",
        action="store_true",
        help="Append best checkpoint to registry experiments instead of overwriting staeformer primary.",
    )
    args = parser.parse_args()
    if args.pose_only and args.no_pose:
        parser.error("Cannot combine --pose-only with --no-pose")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(current_dir))
    data_root = os.path.join(backend_root, "data")
    list_file = os.path.join(backend_root, "data", "transformed_combined_rounds_output_en_evals_translated.json")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    train_staeformer(
        data_root=data_root,
        list_file=list_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        use_cnn=not args.pose_only,
        registry_experiment=args.registry_experiment,
        use_pose=not args.no_pose,
    )
