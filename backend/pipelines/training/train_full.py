import os
import sys

# Add backend directory to sys.path so we can import core and pipelines
backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_root not in sys.path:
    sys.path.append(backend_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision.transforms import v2
import mlflow
import mlflow.pytorch
from core.dataset import FineBadmintonDataset, default_training_jpeg_dir
from core.pose_cache_build import (
    default_pose_cache_path,
    load_pose_cache_bundle,
    media_pipe_fill_pose_cache,
)
from core.training_progress import tqdm_train_batches
from core.model import CNN_LSTM_Model
from core.pose_utils import PoseEstimator
from core.seed_utils import set_seed
from core.split import video_level_split
from core.model_registry import register_training_checkpoint, resolve_training_save_path


def _resample_pose_cache_time(pose_cache: torch.Tensor, t_new: int) -> torch.Tensor:
    """(N, T_old, 33, 3) -> (N, t_new, 33, 3) via linear interpolation along time (matches video T change)."""
    n, t_old, j, c = pose_cache.shape
    if t_old == t_new:
        return pose_cache
    if t_new < 1:
        raise ValueError(f"t_new must be >= 1, got {t_new}")
    # (N, T, 99) -> (N, 99, T) for 1D interpolate on the time dim
    x = pose_cache.reshape(n, t_old, -1).permute(0, 2, 1).contiguous()
    x = torch.nn.functional.interpolate(x, size=t_new, mode="linear", align_corners=True)
    return x.permute(0, 2, 1).reshape(n, t_new, j, c).contiguous()


class FramePoseDataset(Dataset):
    """Wraps a frame dataset and attaches pre-cached pose tensors."""

    def __init__(self, frame_dataset, pose_cache):
        self.frame_dataset = frame_dataset
        self.pose_cache = pose_cache  # (N, T, 33, 3)

    def __len__(self):
        return len(self.frame_dataset)

    def __getitem__(self, idx):
        frames, labels = self.frame_dataset[idx]
        pose = self.pose_cache[idx].clone()
        # Flatten (T, 33, 3) -> (T, 99) for CNN-LSTM's pose input
        pose_flat = pose.view(pose.shape[0], -1)
        return frames, pose_flat, labels


def _build_pose_cache(dataset, list_file, cache_path, seed=42, use_pose=True):
    """Build or load cached pose tensors aligned to dataset indices."""
    n_expected = len(dataset)
    T = dataset.sequence_length
    if use_pose and n_expected == 0:
        raise ValueError(
            "Cannot build or match pose cache: dataset has 0 samples. "
            "Confirm list_file exists and data_root contains the videos (avoid README placeholders like /path/to/...)."
        )
    if not use_pose:
        print("use_pose=False: skipping MediaPipe; using zero pose tensors for the dataloader.")
        return torch.zeros(n_expected, T, 33, 3)
    out = load_pose_cache_bundle(cache_path)
    if out is not None:
        pose_cache = out["pose_cache"]
        if pose_cache.shape[0] == n_expected:
            if pose_cache.shape[1] == T:
                return pose_cache
            print(
                f"Pose cache time dim ({pose_cache.shape[1]}) != requested sequence_length ({T}); "
                "resampling in memory (linear along time; original file unchanged)."
            )
            return _resample_pose_cache_time(pose_cache, T)
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
        image_dir=dataset.image_dir,
        prefer_video=dataset.prefer_video,
    )

    pose_cache = media_pipe_fill_pose_cache(dataset_raw, pose_estimator)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"pose_cache": pose_cache}, cache_path)
    print(f"Saved pose cache to {cache_path}")
    return pose_cache

def train_full(
    data_root,
    list_file,
    epochs=50,
    batch_size=4,
    lr=1e-4,
    device="cpu",
    hidden_size=128,
    save_path=None,
    pose_cache_path=None,
    resume_checkpoint=None,
    start_epoch=0,
    seed=42,
    registry_experiment=True,
    use_pose=True,
    pretrained=True,
    training_jpeg_dir=None,
    disable_training_jpegs=False,
    num_workers=-1,
    sequence_length: int = 16,
):
    set_seed(seed)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    _dir = os.path.dirname(os.path.abspath(__file__))
    _backend_root = os.path.dirname(os.path.dirname(_dir))
    if save_path is None:
        save_path = os.path.join(_backend_root, "models", "badminton_model.pth")
    save_path = resolve_training_save_path(save_path, registry_experiment)
    if pose_cache_path is None:
        pose_cache_path = default_pose_cache_path(_backend_root)

    # Set up MLFlow tracking
    mlflow.set_experiment("IsoCourt_Training_Full")
    with mlflow.start_run():
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "hidden_size": hidden_size,
            "device": device,
            "seed": seed,
            "use_pose": use_pose,
            "pretrained_backbone": pretrained,
            "sequence_length": sequence_length,
        })

        train_transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            v2.RandomGrayscale(p=0.1),
            v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])

        print(
            f"Loading dataset from {data_root} (sequence_length={sequence_length} frames per clip)..."
        )
        if not os.path.isfile(list_file):
            raise FileNotFoundError(
                f"Annotations not found: {list_file}\n"
                "Use the real path to your merged annotations JSON (e.g. "
                "backend/data/transformed_combined_rounds_output_en_evals_translated.json), "
                "not a placeholder like /path/to/..."
            )
        dataset = FineBadmintonDataset(
            data_root,
            list_file,
            transform=train_transform,
            sequence_length=sequence_length,
            image_dir=training_jpeg_dir if training_jpeg_dir else None,
            prefer_video=disable_training_jpegs,
        )
        _use_jpeg = dataset.image_dir is not None
        if num_workers < 0:
            n_cpu = os.cpu_count() or 1
            if _use_jpeg:
                # e.g. g4dn.xlarge: 4 workers = 4 vCPUs (cap avoids huge worker pools on 32+ vCPU)
                _nw = min(4, n_cpu)
            else:
                # MP4: parallel decode (2 is a good default on 4 vCPUs: 2 workers + main).
                # Set --num-workers 0 if you hit RAM pressure (pose cache + workers).
                _nw = min(2, max(0, n_cpu - 1))
        else:
            _nw = num_workers
        mlflow.log_param("training_jpegs", _use_jpeg)
        mlflow.log_param("dataloader_num_workers", _nw)
        if _use_jpeg:
            print(f"Frame source: training JPEGs under {dataset.image_dir} (num_workers={_nw}).")
        else:
            print(
                f"Frame source: MP4 decode per sample; num_workers={_nw} "
                "(default uses multiple CPUs for loading; set --num-workers 0 to disable). "
                "For fastest I/O run prepare_finebadminton_20k.py --extract-training-frames."
            )
        if len(dataset) == 0:
            raise ValueError(
                f"Dataset is empty for list_file={list_file!r}, data_root={data_root!r}. "
                "Check JSON content and that videos exist under data_root (e.g. videos/*.mp4)."
            )

        # Build or load pose cache (aligned to dataset indices)
        pose_cache = _build_pose_cache(dataset, list_file, pose_cache_path, seed=seed, use_pose=use_pose)
        wrapper = FramePoseDataset(dataset, pose_cache)

        # --- WeightedRandomSampler for Class Balance ---
        st_labels = []
        print("Pre-calculating class weights for balanced sampling...")
        for sample in dataset.samples:
            labels = dataset._map_labels(sample)
            st_labels.append(labels['stroke_type'])

        # --- Video-Level Train/Val Split ---
        train_indices, val_indices, _test_indices = video_level_split(dataset.samples)
        
        train_subset = Subset(wrapper, train_indices)
        val_subset = Subset(wrapper, val_indices)
        
        # WeightedRandomSampler on train split only
        train_st_labels = torch.tensor([st_labels[i] for i in train_indices], dtype=torch.long)
        if train_st_labels.numel() == 0:
            raise ValueError(
                "Train split is empty after video_level_split. "
                "Need at least one training sample (check dataset size and split logic)."
            )
        class_counts = torch.bincount(train_st_labels)
        class_weights = 1. / (class_counts.float() + 1e-6)
        sample_weights = class_weights[train_st_labels]
        
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        
        train_generator = torch.Generator().manual_seed(seed)
        _dl_common = {
            "num_workers": _nw,
            "persistent_workers": _nw > 0,
            "pin_memory": device == "cuda",
        }
        if _nw > 0:
            _dl_common["prefetch_factor"] = 2
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            sampler=sampler,
            generator=train_generator,
            **_dl_common,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            **_dl_common,
        )
        
        task_classes = {k: len(v) for k, v in dataset.classes.items()}
        task_classes["quality"] = 7
        del task_classes["stroke_subtype"]
        model = CNN_LSTM_Model(
            task_classes=task_classes,
            hidden_size=hidden_size,
            pretrained=pretrained,
            use_pose=use_pose,
        ).to(device)

        if pretrained:
            # Partial freeze: only layer4 is trainable for domain adaptation (ImageNet init).
            print("Freezing CNN backbone (layer4 unfrozen for domain adaptation)...")
            for name, param in model.cnn.named_parameters():
                param.requires_grad = "7" in name  # nn.Sequential index 7 = layer4
            cnn_trainable_params = [p for n, p in model.cnn.named_parameters() if "7" in n]
            cnn_lr = lr * 0.5
        else:
            print(
                "ResNet pretrained=False: training full backbone from scratch (all CNN layers unfrozen)."
            )
            for param in model.cnn.parameters():
                param.requires_grad = True
            cnn_trainable_params = list(model.cnn.parameters())
            # Base LR on the whole CNN; frozen-early-layer policy does not apply.
            cnn_lr = lr

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"Resuming training from checkpoint: {resume_checkpoint}")
            model.load_state_dict(torch.load(resume_checkpoint, map_location=device, weights_only=True))

        optimizer = optim.AdamW([
            {'params': cnn_trainable_params, 'lr': cnn_lr},
            {'params': model.lstm.parameters(), 'lr': lr * 5},
            {'params': model.heads.parameters(), 'lr': lr * 5}
        ], weight_decay=1e-2)
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        weights_st = torch.tensor([1.0, 1.5, 1.3, 2.0, 1.5, 1.5, 1.5, 2.0, 5.0], dtype=torch.float32, device=device)
        criterion_st = nn.CrossEntropyLoss(weight=weights_st, label_smoothing=0.1)
        criterion_default = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_acc = 0.0
        accumulation_steps = 4

        print(f"\nStarting End-to-End Training ({epochs} epochs)...")
        _cnn_label = "Layer4" if pretrained else "CNN (full)"
        print(f"{_cnn_label} LR: {cnn_lr:.6f} | LSTM LR: {lr*5:.6f} | Heads LR: {lr*5:.6f}")
        
        for epoch in range(start_epoch, epochs):
            # --- Training Phase ---
            model.train()
            running_loss = 0.0
            train_correct = {k: 0 for k in task_classes.keys()}
            train_total = 0
            
            optimizer.zero_grad()
            
            pbar = tqdm_train_batches(train_loader, epoch + 1, epochs)
            for batch_idx, (frames, poses, labels) in enumerate(pbar):
                frames = frames.to(device)
                poses = poses.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}
                
                outputs = model(frames, poses=poses if use_pose else None)
                
                batch_loss = torch.tensor(0.0, device=device)
                loss_weights = {
                    "stroke_type": 2.0, "position": 1.0, "technique": 0.5,
                    "placement": 0.5, "intent": 0.5, "quality": 0.5
                }
                
                for task, logits in outputs.items():
                    crit = criterion_st if task == "stroke_type" else criterion_default
                    loss = crit(logits, labels[task])
                    batch_loss += loss * loss_weights.get(task, 1.0)
                    
                    _, predicted = torch.max(logits.data, 1)
                    train_correct[task] += (predicted == labels[task]).sum().item()
                    if task == "stroke_type":
                        train_total += labels[task].size(0)
                
                (batch_loss / accumulation_steps).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                running_loss += batch_loss.item()
                pbar.set_postfix({'loss': running_loss/(batch_idx+1)})

            # Flush leftover accumulated gradients
            if (batch_idx + 1) % accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss = running_loss / len(train_loader)
            train_acc = 100 * train_correct["stroke_type"] / train_total
            train_pos = 100 * train_correct["position"] / train_total
            scheduler.step(epoch)
            
            # --- Validation Phase ---
            model.eval()
            val_correct = {k: 0 for k in task_classes.keys()}
            val_total = 0
            
            with torch.no_grad():
                for frames, poses, labels in val_loader:
                    frames = frames.to(device)
                    poses = poses.to(device)
                    labels = {k: v.to(device) for k, v in labels.items()}
                    outputs = model(frames, poses=poses if use_pose else None)
                    
                    val_total += frames.size(0)
                    for task, logits in outputs.items():
                        _, predicted = torch.max(logits.data, 1)
                        val_correct[task] += (predicted == labels[task]).sum().item()
            
            val_acc = 100 * val_correct["stroke_type"] / val_total
            pos_acc = 100 * val_correct["position"] / val_total
            
            mlflow.log_metrics({
                "train_loss": epoch_loss,
                "train_type_acc": train_acc,
                "train_pos_acc": train_pos,
                "val_type_acc": val_acc,
                "val_pos_acc": pos_acc,
                "learning_rate": optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss:.4f} | Train Type Acc: {train_acc:.1f}% | Val Type Acc: {val_acc:.1f}% | Val Pos Acc: {pos_acc:.1f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Save Checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"  -> Saved best model (Type Acc: {best_acc:.1f}%)")
                
                import datetime

                model_name = os.path.basename(save_path)
                register_training_checkpoint(
                    os.path.dirname(save_path),
                    category="cnn_lstm",
                    file_basename=model_name,
                    meta={
                        "accuracy": round(best_acc, 2),
                        "epoch": epoch + 1,
                        "hidden_size": hidden_size,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "script": "train_full.py",
                        "architecture": "cnn_lstm",
                        "inference": {
                            "use_pose": use_pose,
                            "hidden_size": hidden_size,
                            "sequence_length": sequence_length,
                            "pretrained_backbone": pretrained,
                        },
                    },
                    experiment=registry_experiment,
                )
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), f"{save_path}_epoch_{epoch+1}.pth")

        print(f"\nTraining finished! Best stroke_type accuracy: {best_acc:.1f}%")

if __name__ == "__main__":
    import argparse

    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(current_dir))
    default_data_root = os.path.join(backend_root, "data")
    default_list_file = os.path.join(
        backend_root, "data", "transformed_combined_rounds_output_en_evals_translated.json"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default=None,
        help="Dataset root (contains videos/ or FineBadminton-20K/videos/). Default: backend/data.",
    )
    parser.add_argument(
        "--list-file",
        default=None,
        help="Annotations JSON. Default: backend/data/transformed_combined_rounds_output_en_evals_translated.json",
    )
    parser.add_argument(
        "--pose-cache-path",
        default=None,
        help="Pose cache .pt for this list. Default: models/pose_cache_mediapipe.pt",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="Training epochs (use 1 for smoke tests).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Train/val batch size.",
    )
    parser.add_argument(
        "--registry-experiment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save to a timestamped .pth and append to registry registrations (default). "
        "Use --no-registry-experiment to overwrite cnn_lstm primary instead.",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="ResNet+LSTM on RGB only (no MediaPipe stream); skips pose cache build.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Initialize ResNet50 with random weights and train the full backbone (pose vs --no-pose ablations). "
        "Default: ImageNet weights + only layer4 trainable.",
    )
    parser.add_argument(
        "--training-jpeg-dir",
        default=None,
        help="Folder with {video_stem}_{frame}.jpg from --extract-training-frames. "
        "Default: auto-detect under data_root (FineBadminton-20K/dataset/image).",
    )
    parser.add_argument(
        "--disable-training-jpegs",
        action="store_true",
        help="Always decode MP4 (ignore pre-extracted JPEG cache).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader workers. Default -1: training JPEGs -> min(4, CPU count); "
        "MP4 -> min(2, CPU-1) for parallel decode (e.g. 2 on g4dn 4vCPU). Use 0 to force single process.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=16,
        metavar="T",
        help="Frames per clip (linspace in [start,end)). Lower (e.g. 4) is faster; default 16. "
        "Pose cache is resampled in memory if a T=16 .pt is loaded.",
    )
    args = parser.parse_args()
    if args.sequence_length < 1 or args.sequence_length > 64:
        raise SystemExit("--sequence-length must be between 1 and 64")

    data_root = args.data_root or default_data_root
    list_file = args.list_file or default_list_file
    pose_cache_path = args.pose_cache_path or None

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_full(
        data_root=data_root,
        list_file=list_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        pose_cache_path=args.pose_cache_path,
        registry_experiment=args.registry_experiment,
        use_pose=not args.no_pose,
        pretrained=not args.no_pretrained,
        training_jpeg_dir=args.training_jpeg_dir,
        disable_training_jpegs=args.disable_training_jpegs,
        num_workers=args.num_workers,
        sequence_length=args.sequence_length,
        # resume_checkpoint=os.path.join(backend_root, "models", "badminton_model.pth_epoch_60.pth"),
        # start_epoch=60
    )
