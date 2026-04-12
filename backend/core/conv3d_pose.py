"""
R(2+1)D / R3D / MC3 (torchvision **video** models) + explicit MediaPipe skeleton path.

**Literature (SOTA-lite):** Tran et al., *A Closer Look at Spatiotemporal Convolutions for Action Recognition*
(CVPR 2018) — factorized 3D conv (R(2+1)D); Carreira & Zisserman *Quo Vadis, Action Recognition?*
(CVPR 2017) — I3D-style inflated 3D; we use torchvision’s small Kinetics-pretrained variants.

**Interpretability:** two clearly separated streams — (1) spatiotemporal appearance+motion from RGB
voxels ``(B,3,T,H,W)`` (use Grad-CAM / guided backprop on :meth:`grad_cam_target_module`), and
(2) a **human-readable** pose vector ``(B,T,33,3)`` fused only at the classifier (late fusion),
so skeleton contributions can be ablated with ``use_pose=False`` or inspected via pose MLP norms.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Valid torchvision.video constructors (Kinetics400 weights when pretrained=True).
_BACKBONE_FACTORIES = ("r2plus1d_18", "r3d_18", "mc3_18")


def _build_torchvision_video_backbone(
    name: str, pretrained: bool
) -> Tuple[nn.Module, int]:
    name = name.lower().strip()
    if name not in _BACKBONE_FACTORIES:
        raise ValueError(f"video_backbone must be one of {_BACKBONE_FACTORIES}, got {name!r}")

    try:
        from torchvision.models.video import (
            MC3_18_Weights,
            R2Plus1D_18_Weights,
            R3D_18_Weights,
            mc3_18,
            r2plus1d_18,
            r3d_18,
        )
    except ImportError as e:
        raise ImportError("conv3d_pose requires torchvision with video models") from e

    if name == "r2plus1d_18":
        w = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        net = r2plus1d_18(weights=w)
    elif name == "r3d_18":
        w = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        net = r3d_18(weights=w)
    else:
        w = MC3_18_Weights.KINETICS400_V1 if pretrained else None
        net = mc3_18(weights=w)

    feat_dim = net.fc.in_features
    net.fc = nn.Identity()
    return net, int(feat_dim)


class Conv3DPoseMultitaskModel(nn.Module):
    """
    Late-fusion two-stream model: 3D CNN trunk + optional flattened MediaPipe joints.

    ``forward`` expects the same layout as other IsoCourt video models:
    ``frames`` ``(B,T,3,H,W)`` ImageNet-normalized; ``joint_seq`` ``(B,T,33,3)``.
    Internally resizes spatially to ``spatial_size`` (default **112**) before the 3D trunk
    so Kinetics400 pretrained weights apply cleanly while the dataset can stay at 224.
    """

    def __init__(
        self,
        task_classes: Dict[str, int],
        num_frames: int = 16,
        video_backbone: str = "r2plus1d_18",
        spatial_size: int = 112,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        unfreeze_layer4: bool = True,
        dropout: float = 0.1,
        use_pose: bool = True,
    ):
        super().__init__()
        self.num_frames = int(num_frames)
        self.video_backbone_name = str(video_backbone).lower().strip()
        self.spatial_size = int(spatial_size)
        self.pretrained = bool(pretrained)
        self.freeze_backbone = bool(freeze_backbone)
        self.unfreeze_layer4 = bool(unfreeze_layer4)
        self.use_pose = bool(use_pose)

        self.backbone, feat_dim = _build_torchvision_video_backbone(
            self.video_backbone_name, pretrained=self.pretrained
        )
        self.video_feat_dim = feat_dim

        self._configure_backbone_requires_grad()

        if use_pose:
            pose_in = self.num_frames * 33 * 3
            self.pose_proj = nn.Sequential(
                nn.Linear(pose_in, feat_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            head_in = feat_dim * 2
        else:
            self.pose_proj = None
            head_in = feat_dim

        self.heads = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(head_in, feat_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(feat_dim, num_c),
                )
                for task, num_c in task_classes.items()
            }
        )

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

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def grad_cam_target_module(self) -> nn.Module:
        """Last spatiotemporal residual stage — use for Grad-CAM / saliency over RGB voxels."""
        if hasattr(self.backbone, "layer4"):
            return self.backbone.layer4
        return self.backbone

    def forward(self, frames: torch.Tensor, joint_seq: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames: (B, T, 3, H, W) — ImageNet mean/std (not raw [0,1]).
            joint_seq: (B, T, 33, 3) when ``use_pose`` is True.
        """
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

        video_feat = self.backbone(x)

        if self.use_pose:
            if joint_seq is None:
                raise ValueError("Conv3DPoseMultitaskModel(use_pose=True) requires joint_seq")
            assert self.pose_proj is not None
            pose_flat = joint_seq.reshape(B, -1)
            pose_feat = self.pose_proj(pose_flat)
            feat = torch.cat([video_feat, pose_feat], dim=-1)
        else:
            feat = video_feat

        return {task: head(feat) for task, head in self.heads.items()}


def backbone_parameter_groups(model: "Conv3DPoseMultitaskModel") -> Tuple[list, list]:
    """Split 3D trunk vs fusion+heads for differential LR (matches VideoMAE trainer pattern)."""
    bb, other = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone."):
            bb.append(p)
        else:
            other.append(p)
    return bb, other
