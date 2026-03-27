"""
VideoMAE (Hugging Face) clip encoder + MediaPipe pose fusion + multi-task heads.

This is the **video-native** counterpart to `TimeSformerPoseModel` with `--backbone vit`:
- **ViT path**: per-frame ImageNet ViT tokens → your divided space–time stack (`timesformer.py`).
- **VideoMAE path (here)**: one **spatiotemporal** encoder over the clip → pooled embedding,
  fused with a projected pose vector → same style MLP heads as other IsoCourt trainers.

Same labels, same pose tensor (T, 33, 3), same task heads pattern as CNN/STAEformer/TimeSformer.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


def _encoder_layers(backbone: nn.Module):
    if hasattr(backbone, "videomae") and hasattr(backbone.videomae, "encoder"):
        return backbone.videomae.encoder.layer
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        return backbone.encoder.layer
    raise AttributeError(
        "Unexpected VideoMAE structure: expected encoder.layer (or legacy videomae.encoder.layer)"
    )


class VideoMAEPoseModel(nn.Module):
    """
    Args:
        frames: (B, T, 3, H, W) in [0,1], then ImageNet-normalized in the training script.
        joint_seq: (B, T, 33, 3)
    """

    def __init__(
        self,
        task_classes: Dict[str, int],
        hf_model_id: str = "MCG-NJU/videomae-base",
        num_frames: int = 16,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        unfreeze_last_n: int = 0,
    ):
        super().__init__()
        try:
            from transformers import VideoMAEModel
        except ImportError as e:
            raise ImportError("VideoMAE requires: pip install transformers") from e

        self.hf_model_id = hf_model_id
        self.num_frames = num_frames
        self.backbone = VideoMAEModel.from_pretrained(hf_model_id)
        cfg = self.backbone.config
        cfg_frames = getattr(cfg, "num_frames", None)
        if cfg_frames is not None and int(cfg_frames) != int(num_frames):
            raise ValueError(
                f"Dataset T={num_frames} but VideoMAE config num_frames={cfg_frames} — align sequence_length."
            )

        self.hidden_size = cfg.hidden_size

        for p in self.backbone.parameters():
            p.requires_grad = False
        if not freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = True
        elif unfreeze_last_n > 0:
            layers = _encoder_layers(self.backbone)
            for layer in layers[-unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True

        pose_in = num_frames * 33 * 3
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_in, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(self.hidden_size * 2, self.hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.hidden_size, num_c),
                )
                for task, num_c in task_classes.items()
            }
        )

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, frames: torch.Tensor, joint_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        frames: (B, T, 3, H, W) ImageNet-normalized (mean/std), not raw [0,1].
        """
        B, T, C, H, W = frames.shape
        if T != self.num_frames:
            raise ValueError(f"Expected T={self.num_frames}, got {T}")

        out = self.backbone(pixel_values=frames)
        h = out.last_hidden_state
        video_feat = h.mean(dim=1)

        pose_flat = joint_seq.reshape(B, -1)
        pose_feat = self.pose_proj(pose_flat)
        feat = torch.cat([video_feat, pose_feat], dim=-1)

        return {task: head(feat) for task, head in self.heads.items()}
