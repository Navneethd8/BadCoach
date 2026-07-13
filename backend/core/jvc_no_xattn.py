"""
JVC no-cross-attention ablation: SkateFormer skeleton + Conv3D vision, late concat.

SkateFormerBEncoder (global pool) + R(2+1)D Conv3D (global pool) -> fusion MLP -> multitask heads.
No graph-vision cross-attention, divided ST blocks, or contact-weighted readout.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.conv3d_pose import _build_torchvision_video_backbone
from core.skateformer_b import (
    SkateFormerBEncoder,
    default_skateformer_b_kwargs,
    load_skateformer_b_skeleton_branch,
)
from core.training_standards import MEDIAPIPE_NUM_FRAMES


def default_jvc_no_xattn_checkpoint_path(backend_root: str) -> str:
    return os.path.join(
        os.path.abspath(backend_root), "models", "badminton_model_jvc_no_xattn.pth"
    )


def default_jvc_no_xattn_pose_cache_path(backend_root: str) -> str:
    return os.path.join(
        os.path.abspath(backend_root), "models", "pose_cache_span_linspace.pt"
    )


class Conv3DGlobalEncoder(nn.Module):
    """R(2+1)D trunk with global spatiotemporal pool -> ``(B, feat_dim)``."""

    def __init__(
        self,
        *,
        num_frames: int = MEDIAPIPE_NUM_FRAMES,
        spatial_size: int = 224,
        video_backbone: str = "r2plus1d_18",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        unfreeze_layer4: bool = True,
    ) -> None:
        super().__init__()
        self.num_frames = int(num_frames)
        self.spatial_size = int(spatial_size)
        self.video_backbone_name = str(video_backbone).lower().strip()
        self.freeze_backbone = bool(freeze_backbone)
        self.unfreeze_layer4 = bool(unfreeze_layer4)

        self.backbone, self.feat_dim = _build_torchvision_video_backbone(
            self.video_backbone_name, pretrained=pretrained
        )
        self._configure_backbone_requires_grad()

    def _configure_backbone_requires_grad(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        if self.freeze_backbone:
            if self.unfreeze_layer4 and hasattr(self.backbone, "layer4"):
                for p in self.backbone.layer4.parameters():
                    p.requires_grad = True
        else:
            for p in self.backbone.parameters():
                p.requires_grad = True

    def encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        """``frames``: ``(B, T, 3, H, W)`` ImageNet-normalized -> ``(B, feat_dim)``."""
        B, T, C, H, W = frames.shape
        if T != self.num_frames:
            raise ValueError(f"Expected T={self.num_frames}, got {T}")
        if C != 3:
            raise ValueError(f"Expected 3 RGB channels, got {C}")

        x = frames.permute(0, 2, 1, 3, 4).contiguous()
        if H != self.spatial_size or W != self.spatial_size:
            x = F.interpolate(
                x,
                size=(T, self.spatial_size, self.spatial_size),
                mode="trilinear",
                align_corners=False,
            )
        return self.backbone(x)


class JVCNoCrossAttnFusion(nn.Module):
    """
    SkateFormer-B skeleton embedding + Conv3D clip embedding -> fusion MLP -> multitask heads.
    """

    def __init__(
        self,
        task_classes: Dict[str, int],
        *,
        window_size: int = MEDIAPIPE_NUM_FRAMES,
        embed_dim: int = 64,
        num_heads: int = 16,
        four_stream: bool = True,
        dropout: float = 0.1,
        fusion_dropout: float = 0.2,
        video_backbone: str = "r2plus1d_18",
        spatial_size: int = 224,
        conv_pretrained: bool = True,
        conv_unfreeze_layer4: bool = True,
        skateformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.task_classes = dict(task_classes)
        self.window_size = window_size

        self.skeleton = SkateFormerBEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_frames=window_size,
            four_stream=four_stream,
            dropout=dropout,
            skateformer_kwargs=skateformer_kwargs,
        )
        skel_dim = self.skeleton.feat_dim

        self.vision_encoder = Conv3DGlobalEncoder(
            num_frames=window_size,
            spatial_size=spatial_size,
            video_backbone=video_backbone,
            pretrained=conv_pretrained,
            freeze_backbone=True,
            unfreeze_layer4=conv_unfreeze_layer4,
        )
        vid_dim = self.vision_encoder.feat_dim
        hidden = max(skel_dim, vid_dim)

        self.fusion = nn.Sequential(
            nn.Linear(skel_dim + vid_dim, hidden),
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
        return self.skeleton.encode_skeleton(pose)

    def encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder.encode_video(frames)

    def forward(
        self, frames: torch.Tensor, pose: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        skel = self.encode_skeleton(pose)
        vid = self.encode_video(frames)
        feat = self.fusion(torch.cat([skel, vid], dim=1))
        return {task: head(feat) for task, head in self.heads.items()}


def build_jvc_no_xattn(task_classes: Dict[str, int], **kwargs: Any) -> JVCNoCrossAttnFusion:
    return JVCNoCrossAttnFusion(task_classes, **kwargs)


def _load_conv3d_into_vision(model: JVCNoCrossAttnFusion, state: Dict[str, Any]) -> None:
    bb = {
        k.replace("backbone.", "", 1): v
        for k, v in state.items()
        if k.startswith("backbone.")
    }
    if bb:
        model.vision_encoder.backbone.load_state_dict(bb, strict=False)
        print(f"Loaded Conv3D backbone ({len(bb)} tensors)")


def load_jvc_no_xattn_partial(
    model: JVCNoCrossAttnFusion,
    checkpoint: str | Dict[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> None:
    label = checkpoint if isinstance(checkpoint, str) else "checkpoint"
    ckpt = (
        torch.load(checkpoint, map_location=device, weights_only=False)
        if isinstance(checkpoint, str)
        else checkpoint
    )
    if "jvc_no_xattn" in ckpt:
        model.load_state_dict(ckpt["jvc_no_xattn"], strict=False)
        print(f"Loaded JVC no-xattn from {label}")
        return
    if "k_st_vit" in ckpt:
        state = ckpt["k_st_vit"]
        skel_state = {
            k.replace("skeleton.", "", 1): v
            for k, v in state.items()
            if k.startswith("skeleton.")
        }
        if skel_state:
            model.skeleton.load_state_dict(skel_state, strict=False)
            print(f"Loaded K-STViT skeleton branch from {label}")
        bb = {
            k.replace("vision_encoder.backbone.", "", 1): v
            for k, v in state.items()
            if k.startswith("vision_encoder.backbone.")
        }
        if bb:
            model.vision_encoder.backbone.load_state_dict(bb, strict=False)
            print(f"Loaded K-STViT Conv3D backbone from {label} ({len(bb)} tensors)")
        return
    if "model" in ckpt and ckpt.get("architecture") == "conv3d_pose":
        _load_conv3d_into_vision(model, ckpt["model"])
        print(f"Loaded Conv3D vision encoder from {label}")
        return
    if "skateformer_b" in ckpt:
        state = ckpt["skateformer_b"]
        skel_state = {
            k.replace("skeleton.", "", 1): v
            for k, v in state.items()
            if k.startswith("skeleton.")
        }
        model.skeleton.load_state_dict(skel_state, strict=False)
        print(f"Loaded SkateFormer-B skeleton from {label}")
        return
    raise KeyError(
        f"Expected 'jvc_no_xattn', 'k_st_vit', conv3d_pose 'model', or 'skateformer_b' in {label}, "
        f"got {list(ckpt.keys())}"
    )


def load_jvc_no_xattn_skeleton_branch(
    model: JVCNoCrossAttnFusion,
    checkpoint_path: str,
    *,
    device: torch.device | str = "cpu",
) -> None:
    """Warm-start skeleton from legacy SkateFormer / SkateFormer-B checkpoints."""
    load_skateformer_b_skeleton_branch(model.skeleton, checkpoint_path, device=device)
