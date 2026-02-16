import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import numpy as np
import os
import sys
from tqdm import tqdm
from dataset import FineBadmintonDataset
from model import CNN_LSTM_Model

class CachedFeatureDataset(Dataset):
    def __init__(self, features, label_dict):
        self.features = features
        self.label_dict = label_dict
        self.keys = list(label_dict.keys())

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], {k: self.label_dict[k][idx] for k in self.keys}

def extract_and_train(
    data_root, 
    list_file, 
    epochs=50, 
    batch_size=8, 
    lr=0.001, 
    device="cpu",
    save_path="models/badminton_model.pth",
    cache_path="models/feature_cache_mtl.pt"
):
    # Step 1: Feature Extraction
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        # Load to CPU first for compatibility with multiprocessing
        cache = torch.load(cache_path, map_location="cpu")
        all_features = cache['features']
        all_labels = cache['labels']
        task_classes = cache['task_classes']
    else:
        print("Extracting features and multi-task labels...")
        dataset = FineBadmintonDataset(data_root=data_root, list_file=list_file)
        task_classes = {k: len(v) for k, v in dataset.classes.items()}
        task_classes["quality"] = 7
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        cnn = nn.Sequential(*modules).to(device)
        cnn.eval()
        
        all_features = []
        # Initialize label storage
        all_labels = {k: [] for k in task_classes.keys()}
        
        for i in tqdm(range(len(dataset)), desc="Extracting"):
            frames, labels = dataset[i]
            
            # Apply ImageNet normalization 
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            frames_norm = (frames - mean) / std
            
            with torch.no_grad():
                features = cnn(frames_norm.to(device)).squeeze(-1).squeeze(-1) # (16, 2048)
            
            all_features.append(features.cpu())
            for k, v in labels.items():
                all_labels[k].append(v)
        
        all_features = torch.stack(all_features)
        for k in all_labels:
            all_labels[k] = torch.stack(all_labels[k])
            
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            'features': all_features,
            'labels': all_labels,
            'task_classes': task_classes
        }, cache_path)
    
    # Step 2: MTL Training
    print(f"\nTraining Multi-Task Model ({epochs} epochs)...")
    
    # --- WeightedRandomSampler for Class Balance ---
    # We use stroke_type as the primary task for balancing
    st_labels = all_labels["stroke_type"]
    class_counts = torch.bincount(st_labels)
    # Give higher weight to minority classes
    class_weights = 1. / (class_counts.float() + 1e-6)
    sample_weights = class_weights[st_labels]
    
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Set num_workers to 0 for stability on macOS
    num_workers = 0
    dataloader = DataLoader(
        CachedFeatureDataset(all_features, all_labels), 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=num_workers
    )
    
    # Build a smaller model to prevent overfitting
    hidden_size = 128
    lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=1, batch_first=True).to(device)
    heads = nn.ModuleDict({
        task: nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5), # Increased dropout
            nn.Linear(hidden_size // 2, num_c)
        )
        for task, num_c in task_classes.items()
    }).to(device)
    
    # Class Balancing Loss
    weights_st = torch.tensor([1.0, 1.5, 1.3, 2.0, 1.5, 1.5, 1.5, 2.0, 5.0]).to(device)
    criterion_st = nn.CrossEntropyLoss(weight=weights_st, label_smoothing=0.1)
    criterion_default = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = optim.Adam(list(lstm.parameters()) + list(heads.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    best_acc = 0.0
    for epoch in range(epochs):
        lstm.train()
        heads.train()
        running_loss = 0.0
        task_correct = {task: 0 for task in task_classes.keys()}
        total = 0
        
        for features, labels in dataloader:
            features = features.to(device)
            
            # --- Feature Data Augmentation ---
            # Add subtle Gaussian noise to help generalization
            if lstm.training:
                noise = torch.randn_like(features) * 0.01 
                features = features + noise
            
            lstm_out, (h_n, c_n) = lstm(features)
            
            # Temporal Global Pooling
            avg_pool = torch.mean(lstm_out, dim=1)
            max_pool, _ = torch.max(lstm_out, dim=1)
            final_feature = torch.cat([avg_pool, max_pool], dim=1)
            
            total_loss = 0
            for task, task_labels in labels.items():
                task_labels = task_labels.to(device)
                logits = heads[task](final_feature)
                
                crit = criterion_st if task == "stroke_type" else criterion_default
                loss = crit(logits, task_labels)
                
                weight = 1.0 if task in ["stroke_type", "quality", "position"] else 0.3
                total_loss += weight * loss
                
                pred = torch.argmax(logits, dim=1)
                task_correct[task] += (pred == task_labels).sum().item()
                if task == "stroke_type":
                    total += task_labels.size(0)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * task_correct["stroke_type"] / total
        scheduler.step(epoch_loss)
        
        if (epoch + 1) % 5 == 0 or epoch_acc > best_acc:
            pos_acc = 100 * task_correct["position"] / total
            print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss:.4f} | Type Acc: {epoch_acc:.1f}% | Pos Acc: {pos_acc:.1f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            # Save as full model
            full_model = CNN_LSTM_Model(task_classes=task_classes, hidden_size=hidden_size)
            full_model.lstm.load_state_dict(lstm.state_dict())
            full_model.heads.load_state_dict(heads.state_dict())
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(full_model.state_dict(), save_path)
            print(f"  -> Saved best model (Type Acc: {best_acc:.1f}%)")

    print(f"\nTraining finished! Best stroke_type accuracy: {best_acc:.1f}%")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(current_dir, "data")
    list_file = os.path.join(current_dir, "data", "transformed_combined_rounds_output_en_evals_translated.json")
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    extract_and_train(
        data_root=data_root, 
        list_file=list_file, 
        epochs=100,
        lr=0.0005,
        device=device
    )
