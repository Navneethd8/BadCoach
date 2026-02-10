import os
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Optional

class FineBadmintonDataset(Dataset):
    """
    Dataset class for FineBadminton data.
    Designed to work with video files and CSV/JSON annotations.
    """
    def __init__(self, data_root: str, list_file: str, transform=None, sequence_length: int = 16, frame_interval: int = 2):
        """
        Args:
            data_root (str): Root directory containing the dataset (videos and annotations).
            list_file (str): Path to the annotation file (e.g., train_split.txt or annotations.csv).
            transform (callable, optional): Optional transform to be applied on a sample.
            sequence_length (int): Number of frames to sample for each clip.
            frame_interval (int): Interval between sampled frames.
        """
        self.data_root = data_root
        self.transform = transform
        self.sequence_length = sequence_length
        self.frame_interval = frame_interval
        
        # Placeholder: Assume list_file is a CSV with 'video_path', 'start_frame', 'end_frame', 'label', 'quality_score'
        # In a real scenario, this would parse the specific FineBadminton annotation format.
        if os.path.exists(list_file):
             self.samples = self._load_annotations(list_file)
        else:
            print(f"Warning: Annotation file {list_file} not found. Initializing empty dataset.")
            self.samples = []

        # Define class mappings (11 stroke types as per paper)
        self.classes = [
            "Serve", "Forehand_Clear", "Backhand_Clear", "Smash", "Drop", 
            "Drive", "Net_Shot", "Lob", "Defensive_Shot", "Other", "Unknown"
        ]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def _load_annotations(self, list_file: str) -> List[Dict[str, Any]]:
        import json
        with open(list_file, 'r') as f:
            data = json.load(f)
            
        samples = []
        for video_item in data:
            # video_item keys: "video", "hitting" (list of strokes), etc.
            video_filename = video_item['video'] # e.g. "0011_001.mp4"
            
            # The JSON seems to list rallies. "hitting" contains individual strokes.
            # We want to extract individual strokes as samples.
            if 'hitting' not in video_item:
                continue
                
            for hit in video_item['hitting']:
                if 'start_frame' not in hit or 'end_frame' not in hit:
                    continue
                    
                samples.append({
                    'video_path': os.path.join(self.data_root, video_filename),
                    'start_frame': hit['start_frame'],
                    'end_frame': hit['end_frame'],
                    'label': hit['hit_type'], # e.g. "serve"
                    'quality': hit.get('quality', 1) # Default to 1 (neutral/lowest) if missing
                })
        return samples
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_filename = sample['video_path'] # This holds "0011_001.mp4" based on previous logic, but we want base name
        # Actually, in _load_annotations we did os.path.join(self.data_root, video_filename). 
        # But for images, we need the base name "0011_001".
        # Let's clean this up.
        
        # We can extract base name from the stored path or just store base name.
        # simpler: define video_base
        video_base = os.path.splitext(os.path.basename(video_filename))[0]

        start_frame = sample['start_frame']
        end_frame = sample['end_frame']
        label_str = sample['label']
        if 'quality' in sample:
             try:
                 # quality is 1-7, we want 0-6
                 quality_val = int(sample['quality']) - 1 
                 # Clamp to ensure valid range just in case
                 quality_val = max(0, min(quality_val, 6))
             except (ValueError, TypeError):
                 quality_val = 0 # Default if parsing fails
        else:
             quality_val = 0 # Default if missing
        
        # Normalize label string
        label_map = {
            "serve": "Serve",
            "smash": "Smash",
            "kill": "Smash",
            "net kill": "Smash",
            "clear": "Forehand_Clear",
            "drop": "Drop",
            "drop shot": "Drop",
            "net shot": "Net_Shot",
            "cross-court net shot": "Net_Shot",
            "lob": "Lob",
            "push shot": "Lob",
            "net lift": "Lob",
            "drive": "Drive",
            "block": "Defensive_Shot",
            "defensive shot": "Defensive_Shot",
        }
        
        mapped_label = label_map.get(label_str.lower(), "Other")
        label = self.class_to_idx.get(mapped_label, self.class_to_idx["Other"])

        frames = self._load_image_frames(video_base, start_frame, end_frame)
        
        if self.transform:
            frames = self.transform(frames)

        return frames, torch.tensor(label, dtype=torch.long), torch.tensor(quality_val, dtype=torch.long)

    def _load_image_frames(self, video_base: str, start_frame: int, end_frame: int) -> torch.Tensor:
        # Construct path to the image directory
        # We assume the directory structure is data_root/image/
        
        # Resolve absolute path for data_root to avoid relative path issues
        abs_data_root = os.path.abspath(self.data_root)
        
        # Try finding the image directory
        image_dir = os.path.join(abs_data_root, "image")
        
        if not os.path.exists(image_dir):
             # check specifically for: FineBadminton-master/dataset/image
             potential_paths = [
                 os.path.join(abs_data_root, "FineBadminton-master", "dataset", "image"),
                 os.path.join(abs_data_root, "dataset", "image"), # some might have unzipped differently
                 os.path.join(abs_data_root, "../data/FineBadminton-master/dataset/image") # fallback
             ]
             for path in potential_paths:
                 if os.path.exists(path):
                     image_dir = path
                     break
        
        # Sampling strategy
        duration = end_frame - start_frame
        if duration <= 0:
             return torch.zeros((self.sequence_length, 3, 224, 224), dtype=torch.float32)
             
        indices = np.linspace(start_frame, end_frame - 1, self.sequence_length).astype(int)
        
        frames = []
        for idx in indices:
            # Construct filename: 0011_001_16362.jpg
            img_name = f"{video_base}_{idx}.jpg"
            img_path = os.path.join(image_dir, img_name)
            
            frame = cv2.imread(img_path)
            
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            else:
                # If image missing, pad with zeros
                # Only print warning for the first few to avoid spam
                if idx == indices[0]: 
                    print(f"Warning: Could not read image at {img_path}")
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        frames_np = np.array(frames)
        frames_tensor = torch.from_numpy(frames_np).float() / 255.0
        frames_tensor = frames_tensor.permute(0, 3, 1, 2) 
        
        return frames_tensor

if __name__ == "__main__":
    # Test block
    print("Dataset class defined.")
