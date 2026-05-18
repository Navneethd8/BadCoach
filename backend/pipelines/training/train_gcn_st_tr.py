"""
ST-TR training (GCN path) — **same upstream** as ``train_st_tr.py``.

Uses Chiaraplizz/ST-TR ``st_gcn.net.st_gcn.Model`` via ``core.st_tr_official``.
This script differs only in default hyperparameters / checkpoint name (``gcn_st_tr``).

Reference: https://github.com/Chiaraplizz/ST-TR
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
from core.st_tr_official import build_official_st_tr
from core.model_registry import make_experiment_checkpoint_path, register_training_checkpoint
from core.training_progress import tqdm_train_batches


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
    """Build or load MediaPipe pose cache."""
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

    task_classes = {k: len(v) for k, v in dataset.classes.items()}
    task_classes["quality"] = 7
    if "stroke_subtype" in task_classes:
        del task_classes["stroke_subtype"]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"pose_cache": pose_cache, "task_classes": task_classes}, cache_path)
    print(f"Saved pose cache to {cache_path}")
    return pose_cache, task_classes


def train_gcn_st_tr(
    data_root,
    list_file,
    epochs=80,
    batch_size=8,
    lr=5e-4,
    device="cpu",
    save_path=None,
    pose_cache_path=None,
    resume_checkpoint=None,
    start_epoch=0,
    seed=42,
    stream="both",
    registry_experiment=False,
):
    set_seed(seed)

    _dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(_dir))
    if save_path is None:
        save_path = os.path.join(backend_root, "models", "badminton_model_gcn_st_tr.pth")
    if pose_cache_path is None:
        pose_cache_path = default_pose_cache_path(backend_root)
    if registry_experiment:
        save_path = make_experiment_checkpoint_path(save_path)

    mlflow.set_experiment("IsoCourt_Training_GCN_ST_TR")
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "seed": seed, "stream": stream,
            "upstream": "Chiaraplizz/ST-TR",
            "script": "train_gcn_st_tr.py",
        })

        print("Loading dataset...")
        dataset = FineBadmintonDataset(data_root, list_file, transform=None)

        pose_cache, task_classes = _build_pose_cache(
            dataset, list_file, device, pose_cache_path, seed=seed
        )
        if task_classes is None:
            task_classes = {k: len(v) for k, v in dataset.classes.items()}
            task_classes["quality"] = 7
            if "stroke_subtype" in task_classes:
                del task_classes["stroke_subtype"]

        # Only train on tasks learnable from pose: stroke_type, position, hand
        pose_tasks = {k: task_classes[k] for k in ("stroke_type", "position", "hand")
                      if k in task_classes}

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

        T = int(dataset.sequence_length)
        model = build_official_st_tr(
            pose_tasks, window_size=T, stream=stream, dropout=0.1,
        ).to(device)

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
            key = "gcn_st_tr" if "gcn_st_tr" in ckpt else "st_tr"
            if key in ckpt:
                model.load_state_dict(ckpt[key], strict=False)
                print(f"Loaded weights from checkpoint key '{key}'")

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"ST-TR (upstream) params: {total_params:,}  stream={stream}")

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        weights_st = torch.tensor(
            [1.0, 1.5, 1.3, 2.0, 1.5, 1.5, 1.5, 2.0, 5.0],
            dtype=torch.float32, device=device,
        )
        criterion_st = nn.CrossEntropyLoss(weight=weights_st, label_smoothing=0.1)
        criterion_default = nn.CrossEntropyLoss(label_smoothing=0.1)
        loss_weights = {"stroke_type": 1.0, "position": 0.2, "hand": 0.1}
        accumulation_steps = 4
        best_acc = 0.0

        print(f"\nStarting ST-TR (upstream) training | stream={stream} | T={T}")
        print(f"LR: {lr} | Batch: {batch_size}")

        for epoch in range(start_epoch, epochs):
            model.train()
            running_loss = 0.0
            train_correct = {k: 0 for k in pose_tasks}
            train_total = 0
            optimizer.zero_grad()

            pbar = tqdm_train_batches(train_loader, epoch + 1, epochs)
            for batch_idx, (poses, labels) in enumerate(pbar):
                poses = poses.to(device)
                labels = {k: v.to(device) for k, v in labels.items()
                          if k in pose_tasks}

                logits_dict = model(poses)

                batch_loss = torch.tensor(0.0, device=device)
                for task, logits in logits_dict.items():
                    crit = criterion_st if task == "stroke_type" else criterion_default
                    batch_loss += loss_weights.get(task, 0.0) * crit(logits, labels[task])
                    _, pred = logits.max(1)
                    train_correct[task] += (pred == labels[task]).sum().item()
                    if task == "stroke_type":
                        train_total += labels[task].size(0)

                (batch_loss / accumulation_steps).backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

            # Validation
            model.eval()
            val_correct = {k: 0 for k in pose_tasks}
            val_total = 0
            val_loss_sum = 0.0
            with torch.no_grad():
                for poses, labels in val_loader:
                    poses = poses.to(device)
                    labels = {k: v.to(device) for k, v in labels.items()
                              if k in pose_tasks}
                    logits_dict = model(poses)
                    val_total += poses.size(0)
                    for task, logits in logits_dict.items():
                        crit = criterion_st if task == "stroke_type" else criterion_default
                        val_loss_sum += loss_weights.get(task, 0.0) * crit(logits, labels[task]).item()
                        _, pred = logits.max(1)
                        val_correct[task] += (pred == labels[task]).sum().item()

            val_acc = 100.0 * val_correct["stroke_type"] / val_total
            val_pos = 100.0 * val_correct["position"] / val_total
            val_loss = val_loss_sum / len(val_loader)
            mlflow.log_metrics({
                "train_loss": epoch_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "val_pos_acc": val_pos,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }, step=epoch + 1)
            print(
                f"epoch {epoch+1:03d} train loss={epoch_loss:.4f} acc={train_acc:.4f} "
                f"val loss={val_loss:.4f} acc={val_acc:.4f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    "gcn_st_tr": model.state_dict(),
                    "task_classes": pose_tasks,
                    "stream": stream,
                    "upstream_st_tr": True,
                }, save_path)
                print(
                    f"  saved {save_path} (best val acc={best_acc:.4f})"
                )
                register_training_checkpoint(
                    os.path.dirname(save_path),
                    category="gcn_st_tr",
                    file_basename=os.path.basename(save_path),
                    meta={
                        "accuracy": round(best_acc, 2),
                        "epoch": epoch + 1,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "script": "train_gcn_st_tr.py",
                        "architecture": "gcn_st_tr",
                        "stream": stream,
                        "upstream_st_tr": True,
                    },
                    experiment=registry_experiment,
                )
            if (epoch + 1) % 10 == 0:
                torch.save({
                    "gcn_st_tr": model.state_dict(),
                    "task_classes": pose_tasks,
                    "stream": stream,
                    "upstream_st_tr": True,
                }, f"{save_path}_epoch_{epoch+1}.pth")

        print(f"\nTraining finished! Best stroke_type accuracy: {best_acc:.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--stream",
        choices=["spatial", "temporal", "both"],
        default="both",
        help="Upstream ST-TR stream (see Chiaraplizz/ST-TR README).",
    )
    parser.add_argument(
        "--registry-experiment", action="store_true",
        help="Append best checkpoint to registry experiments instead of overwriting primary; "
        "weights use a timestamped filename next to the default checkpoint.",
    )
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(current_dir))
    data_root = os.path.join(backend_root, "data")
    list_file = os.path.join(backend_root, "data", "transformed_combined_rounds_output_en_evals_translated.json")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    train_gcn_st_tr(
        data_root=data_root,
        list_file=list_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        stream=args.stream,
        registry_experiment=args.registry_experiment,
    )
