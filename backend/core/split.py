import os
import torch
from typing import List, Tuple, Dict

SPLIT_SEED = 42
SPLIT_RATIO = 0.8


def video_level_split(samples: List[Dict], seed: int = SPLIT_SEED, ratio: float = SPLIT_RATIO) -> Tuple[List[int], List[int]]:
    """
    Splits dataset indices by video to prevent data leakage.
    Same seed + same samples = same split every time, across all scripts.

    Returns:
        (train_indices, val_indices)
    """
    video_to_indices: Dict[str, List[int]] = {}
    for i, sample in enumerate(samples):
        v_name = os.path.basename(sample['video_path'])
        if v_name not in video_to_indices:
            video_to_indices[v_name] = []
        video_to_indices[v_name].append(i)

    unique_videos = sorted(video_to_indices.keys())
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(unique_videos), generator=g).tolist()

    split_idx = int(ratio * len(unique_videos))
    train_vids = [unique_videos[i] for i in perm[:split_idx]]
    val_vids = [unique_videos[i] for i in perm[split_idx:]]

    train_indices = []
    for v in train_vids:
        train_indices.extend(video_to_indices[v])
    val_indices = []
    for v in val_vids:
        val_indices.extend(video_to_indices[v])

    print(f"Split: {len(train_vids)} train vids ({len(train_indices)} samples) / "
          f"{len(val_vids)} val vids ({len(val_indices)} samples)")

    return train_indices, val_indices
