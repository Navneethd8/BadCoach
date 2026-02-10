import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_classes=11, num_quality_classes=3, hidden_size=256, num_layers=1, pretrained=True):
        """
        CNN + LSTM model for action recognition and quality assessment.
        
        Args:
            num_classes (int): Number of action classes (default 11 for basic strokes).
            num_quality_classes (int): Number of quality levels (default 3 or regression).
            hidden_size (int): Hidden size for LSTM.
            num_layers (int): Number of LSTM layers.
            pretrained (bool): Whether to use pretrained ResNet weights.
        """
        super(CNN_LSTM_Model, self).__init__()
        
        # 1. CNN Backbone (ResNet50)
        # We remove the last fc layer to get features
        resnet = models.resnet50(pretrained=pretrained)
        modules = list(resnet.children())[:-1] # Remove fc layer
        self.cnn = nn.Sequential(*modules)
        
        # Freeze CNN weights optionally
        # for param in self.cnn.parameters():
        #     param.requires_grad = False
            
        self.feature_dim = resnet.fc.in_features # 2048 for ResNet50
        
        # 2. LSTM Temporal Module
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 3. Classification Heads
        # Action Classification Head
        self.fc_action = nn.Linear(hidden_size, num_classes)
        
        # Quality Assessment Head 
        # (Could be classification or regression, let's assume classification for now)
        self.fc_quality = nn.Linear(hidden_size, num_quality_classes) 

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, C, H, W)
        Returns:
            action_logits (torch.Tensor): (batch, num_classes)
            quality_logits (torch.Tensor): (batch, num_quality_classes)
        """
        batch_size, seq_len, C, H, W = x.size()
        
        # Flatten time dimension for CNN processing
        c_in = x.view(batch_size * seq_len, C, H, W)
        
        # Extract features with CNN
        # Output shape: (batch * seq_len, 2048, 1, 1) -> (batch * seq_len, 2048)
        features = self.cnn(c_in)
        features = features.view(batch_size, seq_len, -1) 
        
        # Pass through LSTM
        # Output shape: (batch, seq_len, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Use the hidden state of the last time step for classification
        # h_n shape: (num_layers, batch, hidden_size). We take the last layer.
        final_feature = h_n[-1] 
        
        # Predict Action
        action_logits = self.fc_action(final_feature)
        
        # Predict Quality
        quality_logits = self.fc_quality(final_feature)
        
        return action_logits, quality_logits

if __name__ == "__main__":
    # Test block
    model = CNN_LSTM_Model(num_classes=11, num_quality_classes=3)
    dummy_input = torch.randn(2, 16, 3, 224, 224) # Batch=2, Seq=16, C=3, H=224, W=224
    actions, qualities = model(dummy_input)
    print(f"Action Logits Shape: {actions.shape}") # Should be (2, 11)
    print(f"Quality Logits Shape: {qualities.shape}") # Should be (2, 3)
