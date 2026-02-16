import json
import os
from collections import Counter

def count_classes():
    json_path = "data/transformed_combined_rounds_output_en_evals_translated.json"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Matching dataset.py mappings
    from dataset import FineBadmintonDataset
    temp_ds = FineBadmintonDataset(data_root="data", list_file=json_path)
    
    task_counts = {task: Counter() for task in temp_ds.classes.keys()}
    
    for episode in data:
        if isinstance(episode, dict) and 'hitting' in episode:
            actions = episode['hitting']
        else:
            continue
            
        for action in actions:
            # We reuse the mapping logic by calling _map_labels on a dummy sample
            # This is easier than re-implementing it.
            # But _map_labels expects the structure from _load_annotations.
            sample = {
                'hit_type': action.get('hit_type', 'Other'),
                'subtype': action.get('subtype', []),
                'player_actions': action.get('player_actions', []),
                'shot_characteristics': action.get('shot_characteristics', []),
                'ball_area': action.get('ball_area', 'Unknown'),
                'strategies': action.get('strategies', []),
                'quality': action.get('quality', 1)
            }
            labels = temp_ds._map_labels(sample)
            
            for task, idx in labels.items():
                cls_name = temp_ds.classes[task][idx]
                task_counts[task][cls_name] += 1
                
    for task, counts in task_counts.items():
        print(f"\nTask: {task}")
        total = sum(counts.values())
        for cls, count in counts.most_common():
            pct = (count / total) * 100
            print(f"  {cls}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    count_classes()
