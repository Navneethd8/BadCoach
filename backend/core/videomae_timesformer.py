"""
VideoMAE encoder (frozen / partial) → token reshape → **divided space–time** blocks + pose.

VideoMAE uses **tubelet** embedding: sequence length is
`(num_frames // tubelet_size) * (H/patch) * (W/patch)` — e.g. 8×14×14 = **1568** tokens for
16 frames, tubelet 2, 224², patch 16. That is **8 temporal tubes**, not 16 per-frame grids.

Pose is **MediaPipe (16, 33, 3)**; we average consecutive frame pairs to **8** tube-aligned
pose vectors so spatial + temporal attention matches the video tokens.

This is **not** the ViT path in `timesformer.py`; it **reuses** `DividedSTBlock` on top of
**VideoMAE** patch tokens (same ST idea as your other stack).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from core.timesformer import DividedSTBlock


def _encoder_layers(backbone: nn.Module):
    # Older wrappers: backbone.videomae.encoder.layer; HF VideoMAEModel: backbone.encoder.layer
    if hasattr(backbone, "videomae") and hasattr(backbone.videomae, "encoder"):
        return backbone.videomae.encoder.layer
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        return backbone.encoder.layer
    raise AttributeError(
        "Unexpected VideoMAE structure: expected encoder.layer (or legacy videomae.encoder.layer)"
    )


class VideoMAETimeSformerPoseModel(nn.Module):
    """
    Args:
        frames: (B, T, 3, H, W) with T == num_frames (e.g. 16), ImageNet-normalized.
        joint_seq: (B, T, 33, 3) pose for each frame; internally pooled to tube time.
    """

    def __init__(
        self,
        task_classes: Dict[str, int],
        hf_model_id: str = "MCG-NJU/videomae-base",
        num_frames: int = 16,
        embed_dim: int = 128,
        num_heads: int = 4,
        depth: int = 4,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        freeze_videomae: bool = True,
        videomae_unfreeze_last_n: int = 0,
        use_pose: bool = True,
    ):
        super().__init__()
        try:
            from transformers import VideoMAEModel
        except ImportError as e:
            raise ImportError("pip install transformers") from e

        self.hf_model_id = hf_model_id
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        self.backbone = VideoMAEModel.from_pretrained(hf_model_id)
        cfg = self.backbone.config
        self.tubelet_size = int(cfg.tubelet_size)
        self.hidden_size = cfg.hidden_size
        if num_frames % self.tubelet_size != 0:
            raise ValueError(f"num_frames={num_frames} not divisible by tubelet_size={self.tubelet_size}")

        self.use_pose = use_pose
        self.T_tube = num_frames // self.tubelet_size
        image_size = cfg.image_size
        if isinstance(image_size, (tuple, list)):
            ih, iw = int(image_size[0]), int(image_size[1])
        else:
            ih = iw = int(image_size)
        patch_size = cfg.patch_size
        if isinstance(patch_size, (tuple, list)):
            ph, pw = int(patch_size[0]), int(patch_size[1])
        else:
            ph = pw = int(patch_size)
        self.num_patches_spatial = (ih // ph) * (iw // pw)
        expected_L = self.T_tube * self.num_patches_spatial
        self._expected_seq_len = expected_L

        for p in self.backbone.parameters():
            p.requires_grad = False
        if not freeze_videomae:
            for p in self.backbone.parameters():
                p.requires_grad = True
        elif videomae_unfreeze_last_n > 0:
            layers = _encoder_layers(self.backbone)
            for layer in layers[-videomae_unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True

        self.feat_proj = nn.Linear(self.hidden_size, embed_dim)

        self.spatial_pos = nn.Parameter(torch.zeros(1, self.num_patches_spatial, embed_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, self.T_tube, 1, embed_dim))
        if use_pose:
            self.pose_proj = nn.Linear(33 * 3, embed_dim)
            self.pose_spatial_bias = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
            nn.init.trunc_normal_(self.pose_spatial_bias, std=0.02)
        else:
            self.pose_proj = None
            self.pose_spatial_bias = None
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

        assert embed_dim % num_heads == 0
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

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, frames: torch.Tensor, joint_seq: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = frames.shape
        if T != self.num_frames:
            raise ValueError(f"Expected T={self.num_frames}, got {T}")

        out = self.backbone(pixel_values=frames)
        h = out.last_hidden_state
        if h.shape[1] != self._expected_seq_len:
            raise RuntimeError(
                f"VideoMAE seq_len {h.shape[1]} != expected {self._expected_seq_len} "
                f"(T_tube={self.T_tube}, spatial_patches={self.num_patches_spatial})"
            )

        h = h.view(B, self.T_tube, self.num_patches_spatial, self.hidden_size)
        patches = self.feat_proj(h)
        patches = patches + self.spatial_pos

        if self.use_pose:
            if joint_seq is None:
                raise ValueError("VideoMAETimeSformerPoseModel(use_pose=True) requires joint_seq")
            assert self.pose_proj is not None and self.pose_spatial_bias is not None
            pose_tube = joint_seq.view(B, self.T_tube, self.tubelet_size, 33, 3).mean(dim=2)
            pose_flat = pose_tube.reshape(B, self.T_tube, -1)
            pose_tok = self.pose_proj(pose_flat).unsqueeze(2)
            pose_tok = pose_tok + self.pose_spatial_bias
            x = torch.cat([pose_tok, patches], dim=2)
        else:
            x = patches
        x = x + self.temporal_pos

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        xf = x.mean(dim=2)
        avg_pool = xf.mean(dim=1)
        max_pool, _ = xf.max(dim=1)
        feat = torch.cat([avg_pool, max_pool], dim=1)

        return {task: head(feat) for task, head in self.heads.items()}
