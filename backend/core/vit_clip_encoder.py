"""Per-frame timm ViT clip encoder (CLS pool or patch tokens)."""
from __future__ import annotations

import torch
import torch.nn as nn


class ViTClipEncoder(nn.Module):
    """Per-frame timm ViT CLS tokens, projected and pooled over time (mean + max)."""

    def __init__(
        self,
        *,
        num_frames: int = 16,
        img_size: int = 224,
        patch_size: int = 16,
        vit_model_name: str = "vit_small_patch16_224",
        vit_embed_dim: int = 128,
        vit_unfreeze_last_n: int = 2,
        pretrained: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.vit_model_name = vit_model_name
        self.vit_unfreeze_last_n = vit_unfreeze_last_n
        self.out_dim = vit_embed_dim * 2

        try:
            import timm
        except ImportError as e:
            raise ImportError("ViTClipEncoder requires timm (pip install timm>=0.9.0)") from e

        self.vit = timm.create_model(vit_model_name, pretrained=pretrained, num_classes=0)
        self.vit_dim = int(self.vit.embed_dim)
        if self.vit.patch_embed.patch_size[0] != patch_size:
            raise ValueError(
                f"vit {vit_model_name} patch_size {self.vit.patch_embed.patch_size} != {patch_size}"
            )
        vis = getattr(self.vit, "img_size", None)
        if vis is not None:
            vis_i = int(vis[0] if isinstance(vis, (tuple, list)) else vis)
            if vis_i != img_size:
                raise ValueError(f"vit {vit_model_name} img_size {vis} != {img_size}")

        self.proj = nn.Linear(self.vit_dim, vit_embed_dim)
        self.dropout = nn.Dropout(dropout)
        self._freeze_vit()

    def _freeze_vit(self) -> None:
        for p in self.vit.parameters():
            p.requires_grad = False
        n = self.vit_unfreeze_last_n
        if n > 0:
            for blk in self.vit.blocks[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True

    def _vit_token_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Raw ViT tokens ``(B * T, 1 + P, vit_dim)``."""
        B, T, C, H, W = frames.shape
        if T != self.num_frames:
            raise ValueError(f"Expected T={self.num_frames}, got {T}")
        x = frames.view(B * T, C, H, W)
        tok = self.vit.forward_features(x)
        if tok.dim() != 3:
            raise RuntimeError(f"Unexpected ViT feature shape: {tok.shape}")
        return tok

    def forward_patch_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Patch tokens (no CLS), projected to ``vit_embed_dim``.

        Returns:
            ``(B, T, P, vit_embed_dim)`` where P = (img_size // patch_size) ** 2.
        """
        B, T, C, H, W = frames.shape
        tok = self._vit_token_features(frames)
        patch = int(self.vit.patch_embed.patch_size[0])
        grid = H // patch
        patches = self.dropout(self.proj(tok[:, 1:, :]))
        return patches.view(B, T, grid * grid, -1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: ``(B, T, 3, H, W)`` ImageNet-normalized RGB.
        """
        B, T, C, H, W = frames.shape
        tok = self._vit_token_features(frames)
        cls = self.dropout(self.proj(tok[:, 0, :]))
        seq = cls.view(B, T, -1)
        avg = seq.mean(dim=1)
        mx, _ = seq.max(dim=1)
        return torch.cat([avg, mx], dim=1)
