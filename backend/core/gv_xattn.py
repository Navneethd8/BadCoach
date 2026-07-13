"""Shared graph–vision fusion blocks used by JVC."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.skeleton_streams import build_four_stream_pose


class GraphVisionCrossBlock(nn.Module):
    """Graph nodes attend to visual patch tokens within the same frame."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_g = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, graph: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        g = self.norm_g(graph)
        p = self.norm_v(patches)
        attn_out, _ = self.cross_attn(g, p, p, need_weights=False)
        graph = graph + attn_out
        graph = graph + self.ff(self.norm_ff(graph))
        return graph


def subsample_patch_tokens(patches: torch.Tensor) -> torch.Tensor:
    """``(N, P, D)`` with square P -> 2x2 spatial avg pool (P/4 tokens)."""
    n, p, d = patches.shape
    side = int(round(p**0.5))
    if side * side != p:
        raise ValueError(f"Expected square patch count, got P={p}")
    x = patches.view(n, side, side, d).permute(0, 3, 1, 2)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)
    return x.flatten(2).transpose(1, 2)


def contact_frame_weights(pose: torch.Tensor) -> torch.Tensor:
    """``(B, T)`` softmax weights from joint-motion magnitude."""
    streams = build_four_stream_pose(pose)
    jmotion = streams[..., 6:9]
    per_frame = jmotion.norm(dim=-1).amax(dim=2)
    return F.softmax(per_frame, dim=1)
