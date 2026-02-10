import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
import numpy as np
import os
import sys
from tqdm import tqdm
from dataset import FineBadmintonDataset
from model import CNN_LSTM_Model

def extract_and_train(
    data_root, 
    list_file, 
    epochs=50, 
    batch_size=8, 
    lr=0.0005, 
    device="cpu",
    save_path="models/badminton_model.pth",
    cache_path="models/feature_cache.pt"
):
    # Step 1: Extract features through frozen ResNet50 (one-time cost)
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}...")
        cache = torch.load(cache_path, map_location=device)
        all_features = cache['features']
        all_actions = cache['actions']
        all_qualities = cache['qualities']
        num_classes = cache['num_classes']
        print(f"Loaded {len(all_features)} cached feature sequences.")
    else:
        print("Extracting ResNet50 features (one-time cost)...")
        dataset = FineBadmintonDataset(data_root=data_root, list_file=list_file)
        num_classes = len(dataset.classes)
        
        # Load pretrained ResNet50 backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]
        cnn = nn.Sequential(*modules).to(device)
        cnn.eval()
        
        all_features = []
        all_actions = []
        all_qualities = []
        
        for i in tqdm(range(len(dataset)), desc="Extracting features"):
            frames, action, quality = dataset[i]
            # frames shape: (16, 3, 224, 224)
            with torch.no_grad():
                features = cnn(frames.to(device))  # (16, 2048, 1, 1)
                features = features.squeeze(-1).squeeze(-1)  # (16, 2048)
            
            all_features.append(features.cpu())
            all_actions.append(action)
            all_qualities.append(quality)
        
        all_features = torch.stack(all_features)  # (N, 16, 2048)
        all_actions = torch.stack(all_actions)     # (N,)
        all_qualities = torch.stack(all_qualities) # (N,)
        
        # Save cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            'features': all_features,
            'actions': all_actions,
            'qualities': all_qualities,
            'num_classes': num_classes
        }, cache_path)
        print(f"Cached {len(all_features)} feature sequences to {cache_path}")
    
    # Step 2: Train LSTM + FC heads on cached features (fast!)
    print(f"\nTraining LSTM + FC heads ({epochs} epochs)...")
    
    feature_dataset = TensorDataset(all_features, all_actions, all_qualities)
    dataloader = DataLoader(feature_dataset, batch_size=batch_size, shuffle=True)
    
    # Build the LSTM + FC part only
    hidden_size = 256
    num_qualities_classes = 7
    
    lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=1, batch_first=True).to(device)
    fc_action = nn.Linear(hidden_size, num_classes).to(device)
    fc_quality = nn.Linear(hidden_size, num_qualities_classes).to(device)
    
    criterion_action = nn.CrossEntropyLoss()
    criterion_quality = nn.CrossEntropyLoss()
    
    all_params = list(lstm.parameters()) + list(fc_action.parameters()) + list(fc_quality.parameters())
    optimizer = optim.Adam(all_params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    trainable = sum(p.numel() for p in all_params)
    print(f"Trainable params: {trainable:,}")
    
    best_acc = 0.0
    for epoch in range(epochs):
        lstm.train()
        fc_action.train()
        fc_quality.train()
        
        running_loss = 0.0
        correct = 0
        total_samples = 0
        
        for features, actions, qualities in dataloader:
            features = features.to(device)
            actions = actions.to(device)
            qualities = qualities.to(device)
            
            # Forward through LSTM
            lstm_out, (h_n, c_n) = lstm(features)
            final_feature = h_n[-1]  # (batch, 256)
            
            action_preds = fc_action(final_feature)
            quality_preds = fc_quality(final_feature)
            
            loss_action = criterion_action(action_preds, actions)
            loss_quality = criterion_quality(quality_preds, qualities)
            total_loss = loss_action + 0.5 * loss_quality
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            predicted = torch.argmax(action_preds, dim=1)
            correct += (predicted == actions).sum().item()
            total_samples += actions.size(0)
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total_samples
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_loss)
        
        if (epoch + 1) % 5 == 0 or epoch_acc > best_acc:
            print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.1f}% | LR: {current_lr:.6f}")
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            # Save as full CNN_LSTM_Model
            full_model = CNN_LSTM_Model(num_classes=num_classes, num_quality_classes=num_qualities_classes)
            full_model.lstm.load_state_dict(lstm.state_dict())
            full_model.fc_action.load_state_dict(fc_action.state_dict())
            full_model.fc_quality.load_state_dict(fc_quality.state_dict())
            torch.save(full_model.state_dict(), save_path)
            print(f"  -> Saved best model (acc: {best_acc:.1f}%)")
        
        if (epoch + 1) % 10 == 0:
            full_model = CNN_LSTM_Model(num_classes=num_classes, num_quality_classes=num_qualities_classes)
            full_model.lstm.load_state_dict(lstm.state_dict())
            full_model.fc_action.load_state_dict(fc_action.state_dict())
            full_model.fc_quality.load_state_dict(fc_quality.state_dict())
            torch.save(full_model.state_dict(), f"{save_path}_epoch_{epoch+1}.pth")
    
    print(f"\nTraining finished! Best accuracy: {best_acc:.1f}%")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(current_dir, "data")
    list_file = os.path.join(current_dir, "data", "transformed_combined_rounds_output_en_evals_translated.json")
    
    extract_and_train(
        data_root=data_root, 
        list_file=list_file, 
        epochs=50
    )
