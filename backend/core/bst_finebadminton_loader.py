"""PyTorch Dataset over BST collated .npy files built for FineBadminton-20K."""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class FineBadmintonBSTCollatedDataset(Dataset):
    """
    Loads ``prepare_bst_finebadminton_collated.py`` output (train/ or val/).

    ``human_pose`` is (N, T, 2, J, 2); model forward expects flat ``in_dim`` last axis.
    """

    def __init__(self, root_dir: str, split: str, pose_style: str):
        super().__init__()
        branch = os.path.join(root_dir, split)
        self.human_pose = np.load(os.path.join(branch, f"{pose_style}.npy"), mmap_mode="r")
        self.pos = np.load(os.path.join(branch, "pos.npy"), mmap_mode="r")
        self.shuttle = np.load(os.path.join(branch, "shuttle.npy"), mmap_mode="r")
        self.videos_len = np.load(os.path.join(branch, "videos_len.npy"), mmap_mode="r")
        self.labels = np.load(os.path.join(branch, "labels.npy"), mmap_mode="r")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, ...]:
        hp = np.asarray(self.human_pose[i], dtype=np.float32)
        pos = np.asarray(self.pos[i], dtype=np.float32)
        shuttle = np.asarray(self.shuttle[i], dtype=np.float32)
        vlen = int(self.videos_len[i])
        y = int(self.labels[i])
        # flat J*, xy -> in_dim (matches BST infer view)
        t, m, j, c = hp.shape
        hp_flat = hp.reshape(t, m, j * c)
        return (
            torch.from_numpy(hp_flat),
            torch.from_numpy(pos),
            torch.from_numpy(shuttle),
            torch.tensor(vlen, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )
