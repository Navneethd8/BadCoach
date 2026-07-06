"""
ST-TR skeleton trunk + per-frame timm ViT (late fusion) for multitask stroke classification.

Table Y style: MediaPipe pose through upstream ST-TR; RGB clip through a fine-tunable ViT;
concatenate pooled embeddings -> fusion MLP -> task heads.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from core.st_tr_official import IsoCourtOfficialSTTR, StreamMode, build_official_st_tr


def default_st_tr_vit_checkpoint_path(backend_root: str) -> str:
    return os.path.join(os.path.abspath(backend_root), "models", "badminton_model_st_tr_vit.pth")


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


class IsoCourtSTTRViTFusion(nn.Module):
    """Late fusion: ST-TR ``encode_skeleton`` + ``ViTClipEncoder`` -> shared fusion -> heads."""

    def __init__(
        self,
        task_classes: Dict[str, int],
        *,
        window_size: int = 16,
        stream: StreamMode = "both",
        dropout: float = 0.1,
        model_kwargs: Optional[Dict[str, Any]] = None,
        vit_model_name: str = "vit_small_patch16_224",
        vit_embed_dim: int = 128,
        vit_unfreeze_last_n: int = 2,
        vit_pretrained: bool = True,
        fusion_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.task_classes = dict(task_classes)
        self.window_size = window_size
        self.stream = stream

        self.st_tr = build_official_st_tr(
            task_classes,
            window_size=window_size,
            stream=stream,
            dropout=dropout,
            model_kwargs=model_kwargs,
        )
        skel_dim = int(self.st_tr._feat_dim)
        del self.st_tr.heads

        self.vit_encoder = ViTClipEncoder(
            num_frames=window_size,
            vit_model_name=vit_model_name,
            vit_embed_dim=vit_embed_dim,
            vit_unfreeze_last_n=vit_unfreeze_last_n,
            pretrained=vit_pretrained,
            dropout=dropout,
        )
        vit_dim = self.vit_encoder.out_dim
        fuse_in = skel_dim + vit_dim
        hidden = max(skel_dim, vit_embed_dim * 2)

        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, hidden),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(hidden, skel_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
        )
        self._feat_dim = skel_dim

        self.heads = nn.ModuleDict(
            {task: nn.Linear(skel_dim, n_cls) for task, n_cls in task_classes.items()}
        )

    @property
    def skeleton_feat_dim(self) -> int:
        return int(self._feat_dim)

    def encode_skeleton(self, pose: torch.Tensor) -> torch.Tensor:
        return self.st_tr.encode_skeleton(pose)

    def forward(
        self, frames: torch.Tensor, pose: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        skel = self.encode_skeleton(pose)
        vit = self.vit_encoder(frames)
        feat = self.fusion(torch.cat([skel, vit], dim=1))
        return {task: head(feat) for task, head in self.heads.items()}


def build_st_tr_vit_fusion(
    task_classes: Dict[str, int],
    **kwargs: Any,
) -> IsoCourtSTTRViTFusion:
    return IsoCourtSTTRViTFusion(task_classes, **kwargs)


def load_st_tr_skeleton_branch(
    model: IsoCourtSTTRViTFusion,
    checkpoint_path: str,
    *,
    device: torch.device | str = "cpu",
) -> None:
    """Load pose-only ST-TR weights into the skeleton trunk (heads ignored)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "st_tr_vit" in ckpt:
        state = ckpt["st_tr_vit"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded full ST-TR+ViT fusion from {checkpoint_path}")
        return
    key = "st_tr" if "st_tr" in ckpt else None
    if key is None:
        raise KeyError(f"No 'st_tr' or 'st_tr_vit' in checkpoint: {checkpoint_path}")
    trunk_state = model.st_tr.state_dict()
    pretrained = ckpt[key]
    filtered = {k: v for k, v in pretrained.items() if k in trunk_state and v.shape == trunk_state[k].shape}
    trunk_state.update(filtered)
    model.st_tr.load_state_dict(trunk_state, strict=True)
    n_heads = sum(1 for k in pretrained if k.startswith("heads."))
    print(
        f"Loaded ST-TR skeleton trunk from {checkpoint_path} "
        f"({len(filtered)} tensors; skipped {n_heads} head keys)"
    )
