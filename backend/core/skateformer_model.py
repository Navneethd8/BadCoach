"""
SkateFormer multitask wrapper for IsoCourt (MediaPipe pose, T=16, J=33).

Adapts KAIST-VICLab/SkateFormer to pose-only input (B, T, J, 3) and multitask heads
matching ``train_st_tr.py``. Official backbone: ``core.skateformer.official``.

Reference: Do & Kim, "SkateFormer: Skeletal-Temporal Transformer for Human Action
Recognition", ECCV 2024 — https://github.com/KAIST-VICLab/SkateFormer
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from core.skateformer.official import SkateFormer
from core.training_standards import (
    MEDIAPIPE_NUM_FRAMES,
    MEDIAPIPE_NUM_JOINTS,
    SEQUENCE_LENGTH,
)

MEDIAPIPE_NUM_PEOPLE = 1
# Re-export for trainers that import from this module.
NUM_FRAMES = SEQUENCE_LENGTH

# Partition sizes must divide T at every stage (stem T=16 → 8 → 4 → 2 after downsampling).
# Temporal dim uses 2 so the deepest stage (T=2) still partitions cleanly.
DEFAULT_PARTITION_SIZES = {
    "type_1_size": (2, 3),
    "type_2_size": (2, 11),
    "type_3_size": (2, 3),
    "type_4_size": (2, 11),
}


def mediapipe_pose_to_skateformer(pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert IsoCourt pose cache layout to SkateFormer input.

    Args:
        pose: (B, T, J, 3) MediaPipe x, y, z.

    Returns:
        data: (B, C, T, V, M) with M=1
        index_t: (B, T) normalized temporal indices in [-1, 1]
    """
    B, T, _J, _C = pose.shape
    root = pose[:, :, 23:25, :].mean(dim=2, keepdim=True)
    centered = pose - root
    scale = centered.reshape(B, T, -1).norm(dim=-1).mean(dim=1, keepdim=True).clamp(min=1e-6)
    centered = centered / scale.view(B, 1, 1, 1)

    data = centered.permute(0, 3, 1, 2).unsqueeze(-1).contiguous()
    index_t = torch.linspace(-1.0, 1.0, T, device=pose.device, dtype=pose.dtype)
    index_t = index_t.unsqueeze(0).expand(B, -1)
    return data, index_t


def default_skateformer_kwargs(
    *,
    num_frames: int = MEDIAPIPE_NUM_FRAMES,
    num_points: int = MEDIAPIPE_NUM_JOINTS,
    embed_dim: int = 64,
    num_heads: int = 16,
) -> Dict:
    """Lightweight defaults for 16-frame badminton clips (batch_size=4 friendly)."""
    return dict(
        in_channels=3,
        depths=(2, 2, 2, 2),
        channels=(embed_dim, embed_dim * 2, embed_dim * 2, embed_dim * 2),
        embed_dim=embed_dim,
        num_people=MEDIAPIPE_NUM_PEOPLE,
        num_frames=num_frames,
        num_points=num_points,
        kernel_size=7,
        num_heads=num_heads,
        attn_drop=0.2,
        head_drop=0.1,
        rel=True,
        drop_path=0.1,
        index_t=True,
        mlp_ratio=4.0,
        global_pool="avg",
        **DEFAULT_PARTITION_SIZES,
    )


class SkateFormerMultitaskModel(nn.Module):
    """SkateFormer backbone + IsoCourt multitask classification heads."""

    def __init__(
        self,
        task_classes: Dict[str, int],
        dropout: float = 0.1,
        skateformer_kwargs: Dict | None = None,
    ):
        super().__init__()
        kw = default_skateformer_kwargs()
        if skateformer_kwargs:
            kw.update(skateformer_kwargs)

        self.num_frames = kw["num_frames"]
        self.num_joints = kw["num_points"]
        self.feat_dim = kw["channels"][-1]

        self.backbone = SkateFormer(num_classes=1, **kw)
        self.backbone.head = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(self.feat_dim, kw["embed_dim"]),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(kw["embed_dim"], n_cls),
            )
            for task, n_cls in task_classes.items()
        })

    def _embed(self, data: torch.Tensor, index_t: torch.Tensor) -> torch.Tensor:
        B, C, T, V, M = data.shape
        output = data.permute(0, 1, 2, 4, 3).contiguous().view(B, C, T, -1)
        for layer in self.backbone.stem:
            output = layer(output)

        if self.backbone.index_t:
            te = torch.zeros(
                B, T, self.backbone.embed_dim, device=output.device, dtype=output.dtype,
            )
            div_term = torch.exp(
                torch.arange(0, self.backbone.embed_dim, 2, device=output.device, dtype=torch.float32)
                * (-(math.log(10000.0) / self.backbone.embed_dim))
            )
            te[:, :, 0::2] = torch.sin(index_t.unsqueeze(-1).float() * div_term)
            te[:, :, 1::2] = torch.cos(index_t.unsqueeze(-1).float() * div_term)
            output = output + torch.einsum(
                "b t c, c v -> b c t v", te, self.backbone.joint_person_embedding,
            )
        else:
            output = output + self.backbone.joint_person_temporal_embedding
        return output

    def forward(self, pose: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pose: (B, T, J, 3)
        Returns:
            dict of task logits (B, num_classes)
        """
        data, index_t = mediapipe_pose_to_skateformer(pose)
        x = self.backbone.forward_features(self._embed(data, index_t))
        feat = self.dropout(self.backbone.forward_head(x, pre_logits=True))
        return {task: head(feat) for task, head in self.heads.items()}


if __name__ == "__main__":
    tc = {"stroke_type": 9, "position": 10, "intent": 10}
    model = SkateFormerMultitaskModel(tc)
    pose = torch.randn(4, 16, 33, 3)
    out = model(pose)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
