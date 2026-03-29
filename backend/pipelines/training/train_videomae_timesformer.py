"""
VideoMAE encoder + pose + **divided space–time** blocks (`DividedSTBlock` from `timesformer.py`).

Same **training pipeline** as `train_videomae.py` / `train_timesformer.py` (split, cache, losses, registry).

**vs** `train_videomae.py`: that script **pools** VideoMAE tokens and MLP-fuses pose. **This** script
reshapes VideoMAE tokens to **(B, T_tube, spatial_patches, D)** (e.g. 8×196 for tubelet 2, 16 frames),
aligns pose by **averaging** pairs of frames per tube, prepends a pose token per tube, then runs
the same **spatial then temporal** attention stack as TimeSformer.

Checkpoint: `badminton_model_videomae_timesformer.pth` — registry `script: train_videomae_timesformer.py`.

**Conservative retry (method 3)** — smaller ST + regularization + lower head LR + light VideoMAE unfreeze.
Main checkpoint still follows **val stroke-type accuracy** (use --checkpoint-metric val_loss to switch).

    python3 train_videomae_timesformer.py --preset conservative
"""
import os
import sys
import json
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

from core.dataset import FineBadmintonDataset
from core.seed_utils import set_seed
from core.split import video_level_split
from core.videomae_timesformer import VideoMAETimeSformerPoseModel
from core.training_progress import DEFAULT_TRAIN_BATCH_SIZE, tqdm_pose_cache_build, tqdm_train_batches

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


def train_videomae_timesformer(
    data_root,
    list_file,
    epochs=30,
    batch_size=DEFAULT_TRAIN_BATCH_SIZE,
    lr=1e-4,
    device="cpu",
    save_path=None,
    pose_cache_path=None,
    resume_checkpoint=None,
    start_epoch=0,
    seed=42,
    max_train_batches=None,
    hf_model_id="MCG-NJU/videomae-base",
    freeze_videomae=True,
    videomae_unfreeze_last_n=0,
    lr_mult=5.0,
    videomae_lr_mult=0.1,
    weight_decay=1e-2,
    label_smoothing=0.1,
    scheduler_t0=10,
    scheduler_t_mult=2,
    accumulation_steps=4,
    stroke_loss_weight=2.0,
    aug_strength="strong",
    embed_dim=128,
    depth=4,
    num_heads=4,
    checkpoint_metric="val_type_acc",
):
    set_seed(seed)

    _dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(_dir))
    if save_path is None:
        save_path = os.path.join(backend_root, "models", "badminton_model_videomae_timesformer.pth")
    if pose_cache_path is None:
        pose_cache_path = os.path.join(backend_root, "models", "pose_cache_staeformer.pt")

    mlflow.set_experiment("IsoCourt_Training_VideoMAE_TimeSformer")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "seed": seed,
                "hf_model_id": hf_model_id,
                "freeze_videomae": freeze_videomae,
                "videomae_unfreeze_last_n": videomae_unfreeze_last_n,
                "lr_mult": lr_mult,
                "videomae_lr_mult": videomae_lr_mult,
                "weight_decay": weight_decay,
                "label_smoothing": label_smoothing,
                "scheduler_t0": scheduler_t0,
                "scheduler_t_mult": scheduler_t_mult,
                "accumulation_steps": accumulation_steps,
                "stroke_loss_weight": stroke_loss_weight,
                "aug_strength": aug_strength,
                "embed_dim": embed_dim,
                "depth": depth,
                "num_heads": num_heads,
                "checkpoint_metric": checkpoint_metric,
                "script": "train_videomae_timesformer.py",
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

        model = VideoMAETimeSformerPoseModel(
            task_classes=task_classes,
            hf_model_id=hf_model_id,
            num_frames=dataset.sequence_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            freeze_videomae=freeze_videomae,
            videomae_unfreeze_last_n=videomae_unfreeze_last_n,
        ).to(device)

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=False)
                print("Loaded VideoMAE+TimeSformer from checkpoint")

        trainable = [p for p in model.parameters() if p.requires_grad]
        print(
            f"VideoMAE+ST | {hf_model_id} | trainable params: {sum(p.numel() for p in trainable):,} "
            f"(unfreeze_last_n={videomae_unfreeze_last_n}) | embed_dim={embed_dim} depth={depth}"
        )

        backbone_params = []
        other_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(p)
            else:
                other_params.append(p)

        if backbone_params:
            optimizer = optim.AdamW(
                [
                    {"params": backbone_params, "lr": lr * videomae_lr_mult, "weight_decay": weight_decay},
                    {"params": other_params, "lr": lr * lr_mult, "weight_decay": weight_decay},
                ]
            )
            print(
                f"Optimizer: VideoMAE lr={lr * videomae_lr_mult:.6f} | "
                f"ST/heads lr={lr * lr_mult:.6f}"
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
        best_val_loss = float("inf")
        peak_val_acc = 0.0
        min_val_loss_seen = float("inf")

        print("\nStarting VideoMAE + pose + divided TimeSformer stack...")
        print(
            f"checkpoint_metric={checkpoint_metric} | "
            f"lr_mult={lr_mult} | videomae_lr_mult={videomae_lr_mult} | aug={aug_strength} | "
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
                metrics["lr_videomae"] = lrs[0]
                metrics["lr_other"] = lrs[1]
            mlflow.log_metrics(metrics, step=epoch)

            peak_val_acc = max(peak_val_acc, val_acc)
            min_val_loss_seen = min(min_val_loss_seen, val_loss_epoch)

            lr_str = ", ".join(f"{x:.6f}" for x in lrs)
            print(
                f"Epoch {epoch+1:3d} | train_loss: {epoch_loss:.4f} | val_loss: {val_loss_epoch:.4f} | "
                f"Train Type Acc: {train_acc:.1f}% | Train Pos Acc: {train_pos:.1f}% | "
                f"Val Type Acc: {val_acc:.1f}% | Val Pos Acc: {val_pos:.1f}% | "
                f"LR: [{lr_str}]"
            )

            should_save = False
            if checkpoint_metric == "val_type_acc":
                if val_acc > best_acc:
                    best_acc = val_acc
                    should_save = True
            elif checkpoint_metric == "val_loss":
                if val_loss_epoch < best_val_loss:
                    best_val_loss = val_loss_epoch
                    should_save = True
            else:
                raise ValueError(f"Unknown checkpoint_metric: {checkpoint_metric}")

            if should_save:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "task_classes": task_classes,
                        "architecture": "videomae_timesformer",
                        "hf_model_id": hf_model_id,
                        "freeze_videomae": freeze_videomae,
                        "videomae_unfreeze_last_n": videomae_unfreeze_last_n,
                        "embed_dim": embed_dim,
                        "depth": depth,
                        "num_heads": num_heads,
                        "checkpoint_metric": checkpoint_metric,
                    },
                    save_path,
                )
                if checkpoint_metric == "val_type_acc":
                    print(f"  -> Saved best val_type_acc ({best_acc:.1f}%)")
                else:
                    print(
                        f"  -> Saved best val_loss ({best_val_loss:.4f}) "
                        f"(val_type_acc this epoch: {val_acc:.1f}%)"
                    )
                registry_path = os.path.join(os.path.dirname(save_path), "model_registry.json")
                try:
                    with open(registry_path, "r") as f:
                        registry = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    registry = {"models": {}, "active_model": None}
                name = os.path.basename(save_path)
                reg_entry = {
                    "accuracy": round(val_acc, 2),
                    "val_loss": round(val_loss_epoch, 4),
                    "epoch": epoch + 1,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "script": "train_videomae_timesformer.py",
                    "hf_model_id": hf_model_id,
                    "checkpoint_metric": checkpoint_metric,
                }
                if checkpoint_metric == "val_loss":
                    reg_entry["best_val_loss"] = round(best_val_loss, 4)
                else:
                    reg_entry["best_val_type_acc"] = round(best_acc, 2)
                registry["models"][name] = reg_entry
                registry["active_model"] = name
                with open(registry_path, "w") as f:
                    json.dump(registry, f, indent=2)

            if (epoch + 1) % 10 == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "task_classes": task_classes,
                        "architecture": "videomae_timesformer",
                        "hf_model_id": hf_model_id,
                        "freeze_videomae": freeze_videomae,
                        "videomae_unfreeze_last_n": videomae_unfreeze_last_n,
                        "embed_dim": embed_dim,
                        "depth": depth,
                        "num_heads": num_heads,
                        "checkpoint_metric": checkpoint_metric,
                    },
                    f"{save_path}_epoch_{epoch+1}.pth",
                )

        print(
            f"\nTraining finished! checkpoint_metric={checkpoint_metric} | "
            f"peak val_type_acc: {peak_val_acc:.1f}% | min val_loss: {min_val_loss_seen:.4f}"
        )
        if checkpoint_metric == "val_type_acc":
            print(f"  (saved checkpoint: best val_type_acc {best_acc:.1f}%)")
        else:
            print(
                f"  (saved checkpoint: best val_loss {best_val_loss:.4f} — "
                f"compare peak acc {peak_val_acc:.1f}% on a fixed val if needed)"
            )


if __name__ == "__main__":
    import argparse
    import sys

    def _argv_has_flag(argv: list[str], flag: str) -> bool:
        """True if the user passed ``flag`` or ``flag=value`` (not argparse defaults)."""
        for a in argv:
            if a == flag or a.startswith(flag + "="):
                return True
        return False

    _argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="VideoMAE + pose + divided ST. Registry: badminton_model_videomae_timesformer.pth"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pose-cache", type=str, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--hf-model-id", type=str, default="MCG-NJU/videomae-base")
    parser.add_argument("--no-freeze-videomae", action="store_true")
    parser.add_argument("--videomae-unfreeze-last-n", type=int, default=0)
    parser.add_argument("--lr-mult", type=float, default=5.0)
    parser.add_argument("--videomae-lr-mult", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--scheduler-t0", type=int, default=10)
    parser.add_argument("--scheduler-t-mult", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--stroke-loss-weight", type=float, default=2.0)
    parser.add_argument("--aug", type=str, choices=("strong", "medium", "mild"), default="strong")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument(
        "--checkpoint-metric",
        choices=("val_type_acc", "val_loss"),
        default="val_type_acc",
        help="Which metric drives the main checkpoint (default: val_type_acc).",
    )
    parser.add_argument(
        "--preset",
        choices=("none", "conservative"),
        default="none",
        help="conservative: embed 128/4/4, WD 0.02, lr_mult 4, unfreeze last 2 VideoMAE blocks at "
        "0.05× backbone LR, label_smoothing 0.05 (checkpoint metric unchanged; default val_type_acc).",
    )

    args = parser.parse_args()
    if args.preset == "conservative":
        if not _argv_has_flag(_argv, "--embed-dim"):
            args.embed_dim = 128
        if not _argv_has_flag(_argv, "--depth"):
            args.depth = 4
        if not _argv_has_flag(_argv, "--num-heads"):
            args.num_heads = 4
        if not _argv_has_flag(_argv, "--weight-decay"):
            args.weight_decay = 0.02
        if not _argv_has_flag(_argv, "--lr-mult"):
            args.lr_mult = 4.0
        if not _argv_has_flag(_argv, "--videomae-unfreeze-last-n"):
            args.videomae_unfreeze_last_n = 2
        if not _argv_has_flag(_argv, "--videomae-lr-mult"):
            args.videomae_lr_mult = 0.05
        if not _argv_has_flag(_argv, "--label-smoothing"):
            args.label_smoothing = 0.05
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(current_dir))
    data_root = os.path.join(backend_root, "data")
    list_file = os.path.join(
        backend_root, "data", "transformed_combined_rounds_output_en_evals_translated.json"
    )
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    train_videomae_timesformer(
        data_root=data_root,
        list_file=list_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        pose_cache_path=args.pose_cache,
        max_train_batches=args.max_train_batches,
        hf_model_id=args.hf_model_id,
        freeze_videomae=not args.no_freeze_videomae,
        videomae_unfreeze_last_n=args.videomae_unfreeze_last_n,
        lr_mult=args.lr_mult,
        videomae_lr_mult=args.videomae_lr_mult,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        scheduler_t0=args.scheduler_t0,
        scheduler_t_mult=args.scheduler_t_mult,
        accumulation_steps=args.accum_steps,
        stroke_loss_weight=args.stroke_loss_weight,
        aug_strength=args.aug,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        checkpoint_metric=args.checkpoint_metric,
    )
