"""
TimeSformer-style divided space-time attention on video patches, with a prepended
pose token (MediaPipe 33x3 projected to embed_dim) per frame — same fusion idea as
STAEformer (RGB structure + pose), but spatial reasoning is on patch tokens.

Backbone options:
  - scratch: Conv2d patch embedding (random init).
  - vit: timm ViT patch stem + blocks, pretrained on ImageNet (per-frame tokens).
    We drop the CLS token, prepend a pose token, then run the same divided ST stack.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """(B*T, 3, H, W) -> (B*T, num_patches, dim) via conv, non-overlapping patches."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (N, 3, H, W)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class DividedSTBlock(nn.Module):
    """One divided space-time block: spatial MHSA over patches (+ pose), then temporal MHSA."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.spatial = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.temporal = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

    def forward(self, x):
        # x: (B, T, S, D)  S = 1 pose token + num_patches
        B, T, S, D = x.shape
        xs = x.reshape(B * T, S, D)
        xs = self.spatial(xs)
        xs = xs.reshape(B, T, S, D)
        xt = xs.permute(0, 2, 1, 3).reshape(B * S, T, D)
        xt = self.temporal(xt)
        xt = xt.reshape(B, S, T, D).permute(0, 2, 1, 3)
        return xt


def _vit_patch_count(img_size: int, patch_size: int) -> int:
    return (img_size // patch_size) * (img_size // patch_size)


class TimeSformerPoseModel(nn.Module):
    """
    Divided TimeSformer on RGB patches + one pose token per frame.

    Args:
        joint_seq: (B, T, 33, 3)
        frames:    (B, T, 3, H, W) in [0, 1], expect ImageNet norm applied outside
    """

    def __init__(
        self,
        task_classes: Dict[str, int],
        img_size=224,
        patch_size=16,
        num_frames=16,
        embed_dim=128,
        num_heads=4,
        depth=4,
        dropout=0.1,
        mlp_ratio=4.0,
        backbone: str = "scratch",
        vit_model_name: str = "vit_small_patch16_224",
        vit_unfreeze_last_n: int = 0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.backbone = backbone
        self.vit_model_name = vit_model_name
        self.vit_unfreeze_last_n = vit_unfreeze_last_n

        if backbone == "scratch":
            self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
            num_patches = self.patch_embed.num_patches
            self.vit = None
            self.feat_proj = None
        elif backbone == "vit":
            try:
                import timm
            except ImportError as e:
                raise ImportError(
                    "backbone='vit' requires timm (pip install timm>=0.9.0)"
                ) from e

            self.vit = timm.create_model(vit_model_name, pretrained=True, num_classes=0)
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
            num_patches = _vit_patch_count(img_size, patch_size)
            self.patch_embed = None
            self.feat_proj = nn.Linear(self.vit_dim, embed_dim)
            self._freeze_vit()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.num_spatial_tokens = 1 + num_patches  # pose + patches

        self.pose_proj = nn.Linear(33 * 3, embed_dim)

        # (1, P, D) so (B*T, P, D) + spatial_pos does not broadcast to (1, B*T, P, D)
        self.spatial_pos = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, num_frames, 1, embed_dim))
        self.pose_spatial_bias = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.pose_spatial_bias, std=0.02)

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.blocks = nn.ModuleList(
            [
                DividedSTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

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
        if self.vit is None:
            return
        for p in self.vit.parameters():
            p.requires_grad = False
        n = self.vit_unfreeze_last_n
        if n > 0:
            for blk in self.vit.blocks[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, frames, joint_seq):
        B, T, C, H, W = frames.shape
        assert T == self.num_frames, f"Expected T={self.num_frames}, got {T}"

        x = frames.view(B * T, C, H, W)

        if self.backbone == "scratch":
            assert self.patch_embed is not None
            patches = self.patch_embed(x)
            patches = patches + self.spatial_pos
        else:
            assert self.vit is not None and self.feat_proj is not None
            tok = self.vit.forward_features(x)
            if tok.dim() != 3:
                raise RuntimeError(f"Unexpected ViT feature shape: {tok.shape}")
            # timm ViT: index 0 is CLS; use patch tokens only
            patches = tok[:, 1:, :]
            patches = self.feat_proj(patches)
            patches = patches + self.spatial_pos

        pose_flat = joint_seq.reshape(B, T, -1)
        pose_tok = self.pose_proj(pose_flat).unsqueeze(2)
        pose_tok = pose_tok + self.pose_spatial_bias

        num_p = patches.shape[1]
        x = torch.cat([pose_tok, patches.view(B, T, num_p, self.embed_dim)], dim=2)
        x = x + self.temporal_pos

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        xf = x.mean(dim=2)
        avg_pool = xf.mean(dim=1)
        max_pool, _ = xf.max(dim=1)
        feat = torch.cat([avg_pool, max_pool], dim=1)

        return {task: head(feat) for task, head in self.heads.items()}


if __name__ == "__main__":
    tc = {"stroke_type": 9, "position": 10}
    for bb in ("scratch", "vit"):
        m = TimeSformerPoseModel(tc, depth=2, backbone=bb)
        f = torch.rand(2, 16, 3, 224, 224)
        j = torch.rand(2, 16, 33, 3)
        out = m(f, j)
        for k, v in out.items():
            print(bb, k, v.shape)
