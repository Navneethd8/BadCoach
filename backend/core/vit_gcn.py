"""
Per-frame timm ViT (CLS) + fixed skeleton GCN on MediaPipe joints, fused for multitask heads.

Topology matches the body graph used in pose_utils.PoseEstimator (BlazePose-style edges).
Same I/O contract as TimeSformerPoseModel: forward(frames, joint_seq) with joint_seq (B, T, 33, 3).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# Undirected body edges (same connectivity as PoseEstimator.POSE_CONNECTIONS)
MEDIAPIPE_BODY_EDGES: List[Tuple[int, int]] = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
]


def _symmetric_normalized_adjacency(
    num_nodes: int, edges: Sequence[Tuple[int, int]], device, dtype
) -> torch.Tensor:
    a = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)
    for i, j in edges:
        a[i, j] = 1.0
        a[j, i] = 1.0
    a.fill_diagonal_(1.0)
    deg = a.sum(dim=1).clamp(min=1e-6)
    d_inv_sqrt = deg.pow(-0.5)
    return d_inv_sqrt.unsqueeze(1) * a * d_inv_sqrt.unsqueeze(0)


class FixedGCNStack(nn.Module):
    """Fixed A; learned linear maps per layer. Input (B, N, in_dim) -> (B, N, out_dim last)."""

    def __init__(
        self,
        num_nodes: int,
        edges: Sequence[Tuple[int, int]],
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList()
        for li in range(num_layers):
            self.layers.append(nn.Linear(dims[li], dims[li + 1]))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)
        adj0 = _symmetric_normalized_adjacency(
            num_nodes, edges, torch.device("cpu"), torch.float32
        )
        self.register_buffer("_adj", adj0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_dim)
        adj = self._adj.to(device=x.device, dtype=x.dtype)
        for i, lin in enumerate(self.layers):
            x = torch.matmul(adj, x)
            x = lin(x)
            if i < len(self.layers) - 1:
                x = self.act(x)
                x = self.dropout(x)
        return x


class ViTGCNMultitaskModel(nn.Module):
    """
    ViT CLS embedding per frame + GCN over 33 joints, temporal avg/max pools, concat, fuse, heads.
    """

    def __init__(
        self,
        task_classes: Dict[str, int],
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        embed_dim: int = 128,
        gcn_layers: int = 2,
        dropout: float = 0.1,
        vit_model_name: str = "vit_tiny_patch16_224",
        vit_unfreeze_last_n: int = 0,
        pretrained: bool = True,
        use_pose: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.vit_model_name = vit_model_name
        self.vit_unfreeze_last_n = vit_unfreeze_last_n
        self.use_pose = use_pose

        try:
            import timm
        except ImportError as e:
            raise ImportError("ViTGCNMultitaskModel requires timm (pip install timm>=0.9.0)") from e

        self.vit = timm.create_model(vit_model_name, pretrained=pretrained, num_classes=0)
        self.vit_dim = self.vit.embed_dim
        if self.vit.patch_embed.patch_size[0] != patch_size:
            raise ValueError(
                f"vit {vit_model_name} patch_size {self.vit.patch_embed.patch_size} != {patch_size}"
            )
        vis = getattr(self.vit, "img_size", None)
        if vis is not None:
            vis_i = int(vis[0] if isinstance(vis, (tuple, list)) else vis)
            if vis_i != img_size:
                raise ValueError(f"vit {vit_model_name} img_size {vis} != {img_size}")

        self.vit_proj = nn.Linear(self.vit_dim, embed_dim)
        self._freeze_vit()

        if use_pose:
            self.gcn = FixedGCNStack(
                num_nodes=33,
                edges=MEDIAPIPE_BODY_EDGES,
                in_dim=3,
                hidden_dim=embed_dim,
                num_layers=gcn_layers,
                dropout=dropout,
            )
            fuse_in = 4 * embed_dim
        else:
            self.gcn = None
            fuse_in = 2 * embed_dim

        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(embed_dim * 2, embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim, num_c),
                )
                for task, num_c in task_classes.items()
            }
        )

    def _freeze_vit(self) -> None:
        for p in self.vit.parameters():
            p.requires_grad = False
        n = self.vit_unfreeze_last_n
        if n > 0:
            for blk in self.vit.blocks[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self, frames: torch.Tensor, joint_seq: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = frames.shape
        assert T == self.num_frames, f"Expected T={self.num_frames}, got {T}"

        x = frames.view(B * T, C, H, W)
        tok = self.vit.forward_features(x)
        if tok.dim() != 3:
            raise RuntimeError(f"Unexpected ViT feature shape: {tok.shape}")
        cls = tok[:, 0, :]
        vit_bt = self.vit_proj(cls)
        vit_seq = vit_bt.view(B, T, self.embed_dim)
        vit_avg = vit_seq.mean(dim=1)
        vit_max, _ = vit_seq.max(dim=1)
        vit_part = torch.cat([vit_avg, vit_max], dim=1)

        if self.use_pose:
            if joint_seq is None:
                raise ValueError("ViTGCNMultitaskModel(use_pose=True) requires joint_seq")
            assert self.gcn is not None
            j = joint_seq.reshape(B * T, 33, 3)
            gcn_nodes = self.gcn(j)
            gcn_bt = gcn_nodes.mean(dim=1)
            gcn_seq = gcn_bt.view(B, T, self.embed_dim)
            gcn_avg = gcn_seq.mean(dim=1)
            gcn_max, _ = gcn_seq.max(dim=1)
            gcn_part = torch.cat([gcn_avg, gcn_max], dim=1)
            feat = self.fuse(torch.cat([vit_part, gcn_part], dim=1))
        else:
            feat = self.fuse(vit_part)
        return {task: head(feat) for task, head in self.heads.items()}


if __name__ == "__main__":
    tc = {"stroke_type": 9, "position": 10}
    m = ViTGCNMultitaskModel(tc, num_frames=16, gcn_layers=2, embed_dim=128)
    f = torch.rand(2, 16, 3, 224, 224)
    j = torch.rand(2, 16, 33, 3)
    out = m(f, j)
    for k, v in out.items():
        print(k, v.shape)
