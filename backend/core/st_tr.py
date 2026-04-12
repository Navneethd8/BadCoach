"""
ST-TR: Spatial Transformer – Temporal Transformer for skeleton-based action recognition.

Reference: Plizzari et al., "Spatial Temporal Transformer Network for Skeleton-Based
Action Recognition", arXiv 2008.07404 (ICPR 2021).

Architecture
============
Two *parallel* transformer streams operating on a skeleton sequence (B, T, J, 3):

  Spatial stream  — for each frame, self-attention across joints  (B*T, J, D)
  Temporal stream — for each joint, self-attention across frames  (B*J, T, D)

Stream outputs are fused (concat → projection), globally pooled, and fed to
multitask classification heads.

This differs from STAEformer which applies temporal then spatial *sequentially*
with residual connections.  ST-TR's parallel design lets both streams see the
raw embedding rather than one stream's output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict


class STTRModel(nn.Module):
    def __init__(
        self,
        task_classes: Dict[str, int],
        num_joints: int = 33,
        num_frames: int = 16,
        in_features: int = 3,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,
        fusion: str = "concat",
    ):
        """
        Args:
            task_classes:  {head_name: num_classes} for each multitask output.
            num_joints:    Skeleton joints per frame (MediaPipe = 33).
            num_frames:    Temporal sequence length.
            in_features:   Input features per joint (x, y, z → 3).
            embed_dim:     Transformer hidden size.
            num_heads:     Attention heads per layer.
            num_layers:    Transformer encoder layers per stream.
            ff_mult:       FFN expansion factor.
            dropout:       Dropout rate.
            fusion:        How to merge streams: 'concat' (default) or 'sum'.
        """
        super().__init__()
        assert fusion in ("concat", "sum")
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.fusion = fusion

        # Joint feature projection
        self.joint_proj = nn.Linear(in_features, embed_dim)

        # Learnable positional embeddings
        self.spatial_pos = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

        # Spatial stream: attention across joints per frame
        s_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * ff_mult,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.spatial_stream = nn.TransformerEncoder(s_layer, num_layers=num_layers)

        # Temporal stream: attention across frames per joint
        t_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * ff_mult,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.temporal_stream = nn.TransformerEncoder(t_layer, num_layers=num_layers)

        # Stream fusion
        head_in = embed_dim * 2 if fusion == "concat" else embed_dim
        self.fusion_norm = nn.LayerNorm(head_in)

        # Multitask classification heads (avg+max pool → 2*head_in)
        self.heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(head_in * 2, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, n_cls),
            )
            for task, n_cls in task_classes.items()
        })

    def forward(self, pose: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pose: (B, T, J, 3) skeleton coordinates.
        Returns:
            dict of {task_name: (B, num_classes)} logits.
        """
        B, T, J, _ = pose.shape
        x = self.joint_proj(pose)  # (B, T, J, D)

        # --- Spatial stream: (B*T, J, D) ---
        x_s = x.reshape(B * T, J, self.embed_dim) + self.spatial_pos
        x_s = self.spatial_stream(x_s)           # (B*T, J, D)
        x_s = x_s.reshape(B, T, J, self.embed_dim)
        x_s = x_s.mean(dim=2)                    # pool joints → (B, T, D)

        # --- Temporal stream: (B*J, T, D) ---
        x_t = x.permute(0, 2, 1, 3).reshape(B * J, T, self.embed_dim) + self.temporal_pos
        x_t = self.temporal_stream(x_t)           # (B*J, T, D)
        x_t = x_t.reshape(B, J, T, self.embed_dim)
        x_t = x_t.mean(dim=1)                    # pool joints → (B, T, D)

        # --- Fusion ---
        if self.fusion == "concat":
            fused = torch.cat([x_s, x_t], dim=-1)  # (B, T, 2D)
        else:
            fused = x_s + x_t                       # (B, T, D)
        fused = self.fusion_norm(fused)

        # --- Global temporal pooling: avg + max ---
        avg_pool = fused.mean(dim=1)               # (B, head_in)
        max_pool = fused.max(dim=1).values         # (B, head_in)
        feat = torch.cat([avg_pool, max_pool], dim=-1)  # (B, 2*head_in)

        return {task: head(feat) for task, head in self.heads.items()}


if __name__ == "__main__":
    tc = {"stroke_type": 9, "position": 10, "intent": 10}
    model = STTRModel(tc, embed_dim=64, num_heads=4, num_layers=2)
    pose = torch.randn(2, 16, 33, 3)
    out = model(pose)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
