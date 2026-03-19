import os
import sys

# Add backend directory to sys.path so we can import core and pipelines
backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_root not in sys.path:
    sys.path.append(backend_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2
import mlflow
import mlflow.pytorch
from tqdm import tqdm

from core.dataset import FineBadmintonDataset, get_class_weights
from core.model import CNN_LSTM_Model
from core.seed_utils import set_seed
from core.split import video_level_split

def train_full(
    data_root,
    list_file,
    epochs=50,
    batch_size=4,  # End-to-end is memory heavy
    lr=1e-4,       # Lower LR for end-to-end
    device="cpu",
    hidden_size=128,
    save_path=None,
    resume_checkpoint=None,
    start_epoch=0,
    seed=42,
):
    set_seed(seed)

    _dir = os.path.dirname(os.path.abspath(__file__))
    _backend_root = os.path.dirname(os.path.dirname(_dir))
    if save_path is None:
        save_path = os.path.join(_backend_root, "models", "badminton_model.pth")

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
        })

        train_transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            v2.RandomGrayscale(p=0.1),
            v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])

        print(f"Loading dataset from {data_root}...")
        dataset = FineBadmintonDataset(data_root, list_file, transform=train_transform)
        
        # --- WeightedRandomSampler for Class Balance ---
        st_labels = []
        print("Pre-calculating class weights for balanced sampling...")
        for sample in dataset.samples:
            labels = dataset._map_labels(sample)
            st_labels.append(labels['stroke_type'])

        # --- Video-Level Train/Val Split ---
        from torch.utils.data import Subset
        train_indices, val_indices = video_level_split(dataset.samples)
        
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        # WeightedRandomSampler on train split only
        train_st_labels = torch.tensor([st_labels[i] for i in train_indices])
        class_counts = torch.bincount(train_st_labels)
        class_weights = 1. / (class_counts.float() + 1e-6)
        sample_weights = class_weights[train_st_labels]
        
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        
        num_workers = 0
        train_generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True if device == "cuda" else False,
            generator=train_generator,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if device == "cuda" else False
        )
        
        # Step 3: Model & Loss
        task_classes = {k: len(v) for k, v in dataset.classes.items()}
        task_classes["quality"] = 7
        del task_classes["stroke_subtype"]
        model = CNN_LSTM_Model(task_classes=task_classes, hidden_size=hidden_size, pretrained=True).to(device)
        
        # Partial freeze: only layer4 is trainable for domain adaptation
        print("Freezing CNN backbone (layer4 unfrozen for domain adaptation)...")
        for name, param in model.cnn.named_parameters():
            param.requires_grad = "7" in name  # nn.Sequential index 7 = layer4
                
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"Resuming training from checkpoint: {resume_checkpoint}")
            model.load_state_dict(torch.load(resume_checkpoint, map_location=device, weights_only=True))
            
        cnn_layer4_params = [p for n, p in model.cnn.named_parameters() if "7" in n]
        optimizer = optim.AdamW([
            {'params': cnn_layer4_params, 'lr': lr * 0.5},
            {'params': model.lstm.parameters(), 'lr': lr * 5},
            {'params': model.heads.parameters(), 'lr': lr * 5}
        ], weight_decay=1e-2)
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        weights_st_list, _ = get_class_weights(data_root, list_file, task="stroke_type", cap_unseen=2.0)
        weights_st = torch.tensor(weights_st_list, dtype=torch.float32, device=device)
        criterion_st = nn.CrossEntropyLoss(weight=weights_st, label_smoothing=0.1)
        criterion_default = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_acc = 0.0
        accumulation_steps = 4

        print(f"\nStarting End-to-End Training ({epochs} epochs)...")
        print(f"Layer4 LR: {lr*0.5:.6f} | LSTM LR: {lr*5:.6f} | Heads LR: {lr*5:.6f}")
        
        for epoch in range(start_epoch, epochs):
            # --- Training Phase ---
            model.train()
            running_loss = 0.0
            train_correct = {k: 0 for k in task_classes.keys()}
            train_total = 0
            
            optimizer.zero_grad()
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, (frames, labels) in enumerate(pbar):
                frames = frames.to(device)
                labels = {k: v.to(device) for k, v in labels.items()}
                
                outputs = model(frames)
                
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
                for frames, labels in val_loader:
                    frames = frames.to(device)
                    labels = {k: v.to(device) for k, v in labels.items()}
                    outputs = model(frames)
                    
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
                
                import json, datetime
                registry_path = os.path.join(os.path.dirname(save_path), "model_registry.json")
                try:
                    with open(registry_path, "r") as f:
                        registry = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    registry = {"models": {}, "active_model": None}
                
                model_name = os.path.basename(save_path)
                registry["models"][model_name] = {
                    "accuracy": round(best_acc, 2),
                    "epoch": epoch + 1,
                    "hidden_size": hidden_size,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "script": "train_full.py"
                }
                registry["active_model"] = model_name
                with open(registry_path, "w") as f:
                    json.dump(registry, f, indent=2)
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), f"{save_path}_epoch_{epoch+1}.pth")

        print(f"\nTraining finished! Best stroke_type accuracy: {best_acc:.1f}%")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(os.path.dirname(current_dir))
    data_root = os.path.join(backend_root, "data")
    list_file = os.path.join(backend_root, "data", "transformed_combined_rounds_output_en_evals_translated.json")
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_full(
        data_root=data_root, 
        list_file=list_file, 
        epochs=60,
        device=device,
        # resume_checkpoint=os.path.join(backend_root, "models", "badminton_model.pth_epoch_60.pth"),
        # start_epoch=60
    )
