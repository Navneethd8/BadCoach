
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import FineBadmintonDataset
from model import CNN_LSTM_Model
from tqdm import tqdm

def train_model(
    data_root, 
    list_file, 
    epochs=50, 
    batch_size=8, 
    lr=0.0005, 
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_path="models/badminton_model.pth"
):
    
    # Initialize Dataset and Dataloader
    dataset = FineBadmintonDataset(data_root=data_root, list_file=list_file)
    
    if len(dataset) == 0:
        raise ValueError(f"Dataset is empty! Please check if list_file path is correct: {list_file}")
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Initialize Model
    num_actions = len(dataset.classes)
    num_qualities = 7
    
    model = CNN_LSTM_Model(num_classes=num_actions, num_quality_classes=num_qualities).to(device)
    
    # Phase 1: Freeze CNN backbone (transfer learning)
    # Only train LSTM + FC heads (~270K params instead of 23M)
    for param in model.cnn.parameters():
        param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    model.train()
    
    # Loss Functions and Optimizer (only trainable params)
    criterion_action = nn.CrossEntropyLoss()
    criterion_quality = nn.CrossEntropyLoss() 
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    print(f"Starting training on {device} ({len(dataset)} samples, {epochs} epochs)...")
    
    best_acc = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total_samples = 0
        
        for batch_idx, (frames, action_labels, quality_labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            frames = frames.to(device)
            action_labels = action_labels.to(device)
            quality_labels = quality_labels.to(device)
            
            # Forward pass
            action_preds, quality_preds = model(frames)
            
            # Compute loss
            loss_action = criterion_action(action_preds, action_labels)
            loss_quality = criterion_quality(quality_preds, quality_labels)
            
            total_loss = loss_action + 0.5 * loss_quality
            
            # Backward pass and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
            # Track accuracy
            predicted = torch.argmax(action_preds, dim=1)
            correct += (predicted == action_labels).sum().item()
            total_samples += action_labels.size(0)
            
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total_samples
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_loss)
        
        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.1f}% | LR: {current_lr:.6f}")
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (acc: {best_acc:.1f}%)")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
             torch.save(model.state_dict(), f"{save_path}_epoch_{epoch+1}.pth")

    print(f"\nTraining finished! Best accuracy: {best_acc:.1f}%")

if __name__ == "__main__":
    # Example usage (adjust paths)
    # Use absolute paths relative to this script to avoid CWD issues
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(current_dir, "data")
    list_file = os.path.join(current_dir, "data", "transformed_combined_rounds_output_en_evals_translated.json")
    
    # Check if file exists, if not try looking up one level (in case structure varies)
    if not os.path.exists(list_file):
        # Fallback for running from root
        list_file_alt = os.path.join(current_dir, "../backend/data/transformed_combined_rounds_output_en_evals_translated.json")
        if os.path.exists(list_file_alt):
            list_file = list_file_alt
            data_root = os.path.dirname(list_file)

    train_model(
        data_root=data_root, 
        list_file=list_file, 
        epochs=50
    )
