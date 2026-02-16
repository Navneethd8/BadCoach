import torch
import cv2
import numpy as np
import os
from model import CNN_LSTM_Model

def verify_video(video_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        frames.append(frame_resized)
    cap.release()
    
    if not frames:
        print("Error: No frames read.")
        return

    # Sample 16 frames uniformly
    indices = np.linspace(0, len(frames)-1, 16).astype(int)
    segment = [frames[i] for i in indices]
    
    # Preprocess
    x = torch.from_numpy(np.array(segment)).float() / 255.0
    x = x.permute(0, 3, 1, 2).unsqueeze(0).to(device)
    
    # Load Model
    task_classes = {
        "stroke_type": 9, "stroke_subtype": 21, "technique": 4, 
        "placement": 7, "position": 10, "intent": 10, "quality": 7
    }
    model = CNN_LSTM_Model(task_classes=task_classes)
    model_path = "models/badminton_model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("Error: Model not found.")
        return
        
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs['stroke_type'], dim=1)
        val, idx = torch.max(probs, dim=1)
        
        classes = ["Serve", "Clear", "Smash", "Drop", "Drive", "Net_Shot", "Lob", "Defensive_Shot", "Other"]
        print(f"Prediction: {classes[idx.item()]} ({val.item()*100:.1f}%)")
        print(f"Logits std: {torch.std(outputs['stroke_type']).item():.3f}")
        
        # Top 3
        top_v, top_i = torch.topk(probs, 3)
        top_str = ", ".join([f"{classes[top_i[0, m].item()]}: {top_v[0, m].item()*100:.1f}%" for m in range(3)])
        print(f"Top 3: {top_str}")

if __name__ == "__main__":
    v_path = "data/FineBadminton-master/benchmark/video/HITTING_PREDICTION_0010.mp4_1240_1274_111.mp4"
    if os.path.exists(v_path):
        verify_video(v_path)
    else:
        print(f"Video not found: {v_path}")
