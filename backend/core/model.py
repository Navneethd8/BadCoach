import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os

MODEL_VERSION = "V2.3_TEMPORAL_OPTIMIZED"

class CNN_LSTM_Model(nn.Module):
    def __init__(self, task_classes=None, hidden_size=256, num_layers=1, pretrained=True, use_pose=False):
        """
        CNN + LSTM model for Multi-Task Stroke Recognition.
        
        Args:
            task_classes (dict): Map of task name to number of classes.
            hidden_size (int): Hidden size for LSTM.
            num_layers (int): Number of LSTM layers.
            pretrained (bool): Whether to use pretrained ResNet weights.
            use_pose (bool): Whether to include MediaPipe pose vectors (99-dim).
        """
        super(CNN_LSTM_Model, self).__init__()
        self.use_pose = use_pose
        
        if task_classes is None:
            # Default fallback matching the current dataset definition
            task_classes = {
                "stroke_type": 9, "stroke_subtype": 21, "technique": 4, 
                "placement": 7, "position": 10, "intent": 10, "quality": 7
            }
        
        # 1. CNN Backbone (ResNet50)
        # Using DEFAULT (V2) for consistency with current training runs
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        modules = list(resnet.children())[:-1] 
        self.cnn = nn.Sequential(*modules)
        self.feature_dim = resnet.fc.in_features 
        
        # 2. LSTM Temporal Module
        lstm_input_size = self.feature_dim + 99 if self.use_pose else self.feature_dim
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.lstm_dropout = nn.Dropout(0.5)
        
        # 3. Multi-Task Classification Heads
        # Enhanced heads with a hidden layer. 
        # Using hidden_size * 2 because we concatenate Avg and Max pooling.
        self.heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.5), # Increased from 0.3
                nn.Linear(hidden_size, num_c)
            )
            for task, num_c in task_classes.items()
        })

    def forward(self, x, poses=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, C, H, W)
                             Assumes input range [0, 1]
            poses (torch.Tensor, optional): Input tensor of shape (batch, seq_len, 99)
        Returns:
            Dict[str, torch.Tensor]: Dictionary of logits for each task.
        """
        batch_size, seq_len, C, H, W = x.size()
        
        # 1. Normalize (ImageNet stats)
        # Reshape to (B*S, C, H, W) for efficient processing
        c_in = x.view(batch_size * seq_len, C, H, W)
        
        # Apply normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        c_in = (c_in - mean) / std
        
        # 2. CNN Feature Extraction
        features = self.cnn(c_in)
        features = features.view(batch_size, seq_len, -1) 
        
        # Add Poses if using Two-Stream
        if self.use_pose and poses is not None:
            features = torch.cat([features, poses], dim=2)
        
        # 3. Temporal Modeling
        lstm_out, (h_n, c_n) = self.lstm(features)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # 4. Temporal Global Pooling (Collapse Prevention)
        # Instead of just taking the last state, we pool across all 16 frames.
        # This captures the "peak" of the action (Max) and the "context" (Avg).
        avg_pool = torch.mean(lstm_out, dim=1) # (batch, hidden_size)
        max_pool, _ = torch.max(lstm_out, dim=1) # (batch, hidden_size)
        
        # Concatenate Avg and Max features
        final_feature = torch.cat([avg_pool, max_pool], dim=1) # (batch, hidden_size * 2)
        
        # 5. Predictions through task heads
        logits = {task: head(final_feature) for task, head in self.heads.items()}
        
        return logits

if __name__ == "__main__":
    task_classes = {
        "stroke_type": 9, "stroke_subtype": 21, "technique": 4,
        "placement": 7, "position": 10, "intent": 10, "quality": 7
    }
    model = CNN_LSTM_Model(task_classes=task_classes, use_pose=True)
    dummy_input = torch.randn(2, 16, 3, 224, 224)
    dummy_poses = torch.randn(2, 16, 99)
    outputs = model(dummy_input, poses=dummy_poses)
    for task, logits in outputs.items():
        print(f"Task '{task}' Logits Shape: {logits.shape}")
