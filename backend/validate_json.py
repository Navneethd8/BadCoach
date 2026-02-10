
import json
import sys

def validate_json(file_path):
    print(f"Reading {file_path}...")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("File not found.")
        return
    except json.JSONDecodeError:
        print("Invalid JSON.")
        return

    print(f"Loaded {len(data)} video entries.")
    
    missing_start = 0
    missing_end = 0
    missing_hit_type = 0
    total_hits = 0
    
    for i, video_item in enumerate(data):
        if 'hitting' not in video_item:
            print(f"Video {i} missing 'hitting' key.")
            continue
            
        for j, hit in enumerate(video_item['hitting']):
            total_hits += 1
            if 'start_frame' not in hit:
                missing_start += 1
                if missing_start <= 5:
                    print(f"Missing 'start_frame' in video {video_item.get('video', 'unknown')}, hit {j}: {hit}")
            if 'end_frame' not in hit:
                missing_end += 1
            if 'hit_type' not in hit:
                missing_hit_type += 1
                
    print("\nSummary:")
    print(f"Total Hits Checked: {total_hits}")
    print(f"Missing 'start_frame': {missing_start}")
    print(f"Missing 'end_frame': {missing_end}")
    print(f"Missing 'hit_type': {missing_hit_type}")

if __name__ == "__main__":
    validate_json("/Volumes/NavDisk/BadmintonAiCoach/backend/data/transformed_combined_rounds_output_en_evals_translated.json")
