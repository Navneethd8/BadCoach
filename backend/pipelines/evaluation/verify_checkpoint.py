import torch
import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.model import CNN_LSTM_Model
from core.dataset import FineBadmintonDataset

def verify():
    model_path = "models/badminton_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
        return

    print(f"Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location="cpu")
    
    # Check keys
    print(f"Keys in state_dict: {list(state_dict.keys())[:5]}... (Total: {len(state_dict)})")
    
    # Initialize model
    task_classes = {
        "stroke_type": 9, "stroke_subtype": 21, "technique": 4, 
        "placement": 7, "position": 10, "intent": 10, "quality": 7
    }
    model = CNN_LSTM_Model(task_classes=task_classes)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Check head weights for 'stroke_type'
    head_weights = state_dict['heads.stroke_type.0.weight']
    print(f"Stroke type head weights mean: {head_weights.mean().item():.4f}, std: {head_weights.std().item():.4f}")
    
    # Run dummy prediction
    dummy_input = torch.zeros((1, 16, 3, 224, 224)) # Black frames
    with torch.no_grad():
        # Step 1: Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        norm_in = (dummy_input - mean) / std
        
        # Step 2: CNN
        b, s, c, h, w = norm_in.size()
        c_in = norm_in.view(b*s, c, h, w)
        cnn_out = model.cnn(c_in).view(b, s, -1)
        print(f"CNN features mean: {cnn_out.mean().item():.4f}, std: {cnn_out.std().item():.4f}")
        
        # Manually run LSTM + Pool
        lstm_out, _ = model.lstm(cnn_out)
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        final_feature = torch.cat([avg_pool, max_pool], dim=1)
        print(f"Final feature (concat) mean: {final_feature.mean().item():.4f}, std: {final_feature.std().item():.4f}, max: {final_feature.max().item():.4f}")
        
        outputs = model(dummy_input)
        probs = torch.softmax(outputs['stroke_type'], dim=1)
        val, idx = torch.max(probs, dim=1)
        print(f"Black Frames Prediction: {idx.item()} (Conf: {val.item()*100:.1f}%)")
        print(f"Logits for stroke_type: {outputs['stroke_type'].cpu().numpy()}")
        
        # Check Biases
        st_head_bias = state_dict['heads.stroke_type.3.bias'] # Final layer bias
        print(f"Stroke type final bias mean: {st_head_bias.mean().item():.4f}, values: {st_head_bias.cpu().numpy()}")
        
    dummy_noise = torch.randn((1, 16, 3, 224, 224))
    with torch.no_grad():
        outputs = model(dummy_noise)
        probs = torch.softmax(outputs['stroke_type'], dim=1)
        val, idx = torch.max(probs, dim=1)
        print(f"Noise Frames Prediction: {idx.item()} (Conf: {val.item()*100:.1f}%)")

if __name__ == "__main__":
    verify()
