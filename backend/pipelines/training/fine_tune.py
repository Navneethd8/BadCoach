import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import os
import sys
from tqdm import tqdm

from core.dataset import FineBadmintonDataset
from core.model import CNN_LSTM_Model

def fine_tune(
    data_root, 
    list_file, 
    base_model_path="models/badminton_model.pth",
    epochs=20, 
    batch_size=8, 
    lr=1e-5,      # Very low LR for fine-tuning
    device="cpu",
    save_path="models/badminton_model_tuned.pth"
):
    print("Loading Dataset for Fine-tuning...")
    # For fine-tuning, we might want less aggressive jitter since we ARE 
    # trying to learn the specific domain, but we should still keep some for robustness.
    train_transform = v2.Compose([
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    ])
    
    dataset = FineBadmintonDataset(data_root=data_root, list_file=list_file, transform=train_transform)
    task_classes = {k: len(v) for k, v in dataset.classes.items()}
    task_classes["quality"] = 7
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Step 2: Load Base Model
    model = CNN_LSTM_Model(task_classes=task_classes).to(device)
    if os.path.exists(base_model_path):
        model.load_state_dict(torch.load(base_model_path, map_location=device))
        print(f"Loaded base model from {base_model_path}")
    else:
        print("Warning: Base model not found. Training from scratch.")

    # Freeze Backbone (ResNet)
    # We only want to train the LSTM and the Task Heads
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False
    
    # Verify which layers are being trained
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable Parameters: {len(trainable_params)}")

    optimizer = optim.Adam(trainable_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\nStarting Fine-tuning ({epochs} epochs)...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (frames, labels) in enumerate(pbar):
            frames = frames.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            optimizer.zero_grad()
            outputs = model(frames)
            
            total_loss = 0
            for task, logits in outputs.items():
                total_loss += criterion(logits, labels[task])
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            pbar.set_postfix({'loss': running_loss/(batch_idx+1)})

        print(f"Epoch {epoch+1:3d} | Loss: {running_loss/len(dataloader):.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Fine-tuning complete. Saved to {save_path}")

if __name__ == "__main__":
    # Example usage for user
    # Users should create a 'local_data.json' with their court's clips
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(current_dir, "data")
    list_file = os.path.join(current_dir, "data", "local_data.json")
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    if os.path.exists(list_file):
        fine_tune(data_root=data_root, list_file=list_file, device=device)
    else:
        print(f"Error: {list_file} not found. Please create it with your labeled clips.")
