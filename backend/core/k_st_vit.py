"""
K-STViT: Kinematic-guided spatiotemporal fusion for badminton stroke recognition.

SkateFormer joint tokens cross-attend to vision patch tokens (Conv3D or ViT),
then a divided space-time transformer mixes joints + patches over the clip.
Contact-weighted temporal readout for hit-span clips.

v2 default: R(2+1)D Conv3D vision encoder (warm-start from conv3d_pose checkpoint).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.conv3d_pose import Conv3DClipTokenEncoder
from core.gv_xattn import (
    GraphVisionCrossBlock,
    contact_frame_weights,
    subsample_patch_tokens,
)
from core.skateformer_b import (
    SkateFormerBEncoder,
    default_skateformer_b_kwargs,
    load_skateformer_b_skeleton_branch,
)
from core.timesformer import DividedSTBlock, _vit_patch_count
from core.training_standards import MEDIAPIPE_NUM_FRAMES, MEDIAPIPE_NUM_JOINTS

VisionBackbone = Literal["conv3d", "vit"]


def default_k_st_vit_checkpoint_path(backend_root: str) -> str:
    return os.path.join(os.path.abspath(backend_root), "models", "badminton_model_k_st_vit.pth")


class KinematicSpatiotemporalViT(nn.Module):
    """
    SkateFormer joint queries + vision patches -> cross-attn -> divided ST -> multitask heads.
    """

    def __init__(
        self,
        task_classes: Dict[str, int],
        *,
        window_size: int = MEDIAPIPE_NUM_FRAMES,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 128,
        skel_embed_dim: int = 64,
        skel_num_heads: int = 16,
        num_heads: int = 4,
        st_depth: int = 4,
        num_cross_layers: int = 2,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        subsample_patches: bool = True,
        contact_pool: bool = True,
        vision_backbone: VisionBackbone = "conv3d",
        video_backbone: str = "r2plus1d_18",
        spatial_size: int = 224,
        conv_pretrained: bool = True,
        conv_freeze_backbone: bool = True,
        conv_unfreeze_layer4: bool = True,
        vit_model_name: str = "vit_small_patch16_224",
        vit_unfreeze_last_n: int = 4,
        vit_pretrained: bool = True,
        use_shuttle: bool = False,
        four_stream: bool = True,
        skateformer_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.task_classes = dict(task_classes)
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.contact_pool = contact_pool
        self.subsample_patches = subsample_patches
        self.num_joints = MEDIAPIPE_NUM_JOINTS
        self.vision_backbone = vision_backbone
        self.use_shuttle = use_shuttle

        sk_kw = default_skateformer_b_kwargs(
            num_frames=window_size,
            embed_dim=skel_embed_dim,
            num_heads=skel_num_heads,
            four_stream=four_stream,
        )
        if skateformer_kwargs:
            sk_kw.update(skateformer_kwargs)

        self.skeleton = SkateFormerBEncoder(
            embed_dim=skel_embed_dim,
            num_heads=skel_num_heads,
            num_frames=window_size,
            four_stream=four_stream,
            dropout=dropout,
            skateformer_kwargs=sk_kw,
        )
        self.joint_proj = nn.Linear(self.skeleton.feat_dim, embed_dim)

        self.vit = None
        self.feat_proj = None
        self.vision_encoder: Conv3DClipTokenEncoder | None = None
        self.vit_model_name = vit_model_name
        self.vit_unfreeze_last_n = vit_unfreeze_last_n

        if vision_backbone == "conv3d":
            self.vision_encoder = Conv3DClipTokenEncoder(
                num_frames=window_size,
                spatial_size=spatial_size,
                embed_dim=embed_dim,
                video_backbone=video_backbone,
                pretrained=conv_pretrained,
                freeze_backbone=conv_freeze_backbone,
                unfreeze_layer4=conv_unfreeze_layer4,
            )
            with torch.no_grad():
                dummy = torch.zeros(1, window_size, 3, spatial_size, spatial_size)
                self.num_patches = int(
                    self.vision_encoder.forward_patch_tokens(dummy).shape[2]
                )
        elif vision_backbone == "vit":
            try:
                import timm
            except ImportError as e:
                raise ImportError("K-STViT vit backbone requires timm") from e

            self.vit = timm.create_model(vit_model_name, pretrained=vit_pretrained, num_classes=0)
            self.vit_dim = int(self.vit.embed_dim)
            if self.vit.patch_embed.patch_size[0] != patch_size:
                raise ValueError(
                    f"vit {vit_model_name} patch_size {self.vit.patch_embed.patch_size} != {patch_size}"
                )
            self.num_patches = _vit_patch_count(img_size, patch_size)
            self.feat_proj = nn.Linear(self.vit_dim, embed_dim)
            self._freeze_vit()
        else:
            raise ValueError(f"vision_backbone must be conv3d|vit, got {vision_backbone!r}")

        self.cross_blocks = nn.ModuleList(
            [
                GraphVisionCrossBlock(embed_dim, num_heads, dropout=dropout)
                for _ in range(num_cross_layers)
            ]
        )

        self.spatial_pos = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.joint_spatial_bias = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, window_size, 1, embed_dim))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.joint_spatial_bias, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

        assert embed_dim % num_heads == 0
        self.st_blocks = nn.ModuleList(
            [
                DividedSTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(st_depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        clip_dim = embed_dim * 2
        self.shuttle_encoder: nn.Module | None
        if use_shuttle:
            self.shuttle_encoder = nn.Sequential(
                nn.Linear(window_size * 2, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            head_in = clip_dim + embed_dim
        else:
            self.shuttle_encoder = None
            head_in = clip_dim

        self.heads = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(head_in, embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim, n_cls),
                )
                for task, n_cls in task_classes.items()
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

    def _vision_patch_tokens(self, frames: torch.Tensor) -> torch.Tensor:
        """``(B, T, P, embed_dim)`` from ImageNet-normalized RGB."""
        if self.vision_encoder is not None:
            return self.vision_encoder.forward_patch_tokens(frames)
        assert self.vit is not None and self.feat_proj is not None
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)
        tok = self.vit.forward_features(x)
        if tok.dim() != 3:
            raise RuntimeError(f"Unexpected ViT feature shape: {tok.shape}")
        patches = self.feat_proj(tok[:, 1:, :])
        return patches.view(B, T, self.num_patches, self.embed_dim)

    def _align_joint_time(
        self, joints: torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        """SkateFormer stages downsample T; resample joints to match RGB frame count."""
        Tj = joints.shape[1]
        if Tj == num_frames:
            return joints
        x = joints.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(num_frames, joints.shape[2]), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1).contiguous()

    def _fuse_frame_tokens(
        self, joints: torch.Tensor, patches: torch.Tensor
    ) -> torch.Tensor:
        """Joint cross-attn to patches, concat -> ``(B, T, S, D)``."""
        B, T, J, D = joints.shape
        P = patches.shape[2]
        joint_bt = joints.reshape(B * T, J, D) + self.joint_spatial_bias.squeeze(1)
        patch_bt = patches.reshape(B * T, P, D) + self.spatial_pos
        if self.subsample_patches:
            patch_bt = subsample_patch_tokens(patch_bt)
        for block in self.cross_blocks:
            joint_bt = block(joint_bt, patch_bt)
        tokens = torch.cat([joint_bt, patch_bt], dim=1)
        return tokens.view(B, T, tokens.shape[1], D)

    def _clip_features(self, tokens: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """``(B, 2*embed_dim)`` contact + max temporal pool over frame summaries."""
        x = tokens + self.temporal_pos
        for blk in self.st_blocks:
            x = blk(x)
        x = self.norm(x)
        frame_feat = x.mean(dim=2)
        if self.contact_pool:
            w = contact_frame_weights(pose).unsqueeze(-1)
            weighted = (frame_feat * w).sum(dim=1)
        else:
            weighted = frame_feat.mean(dim=1)
        max_pool, _ = frame_feat.max(dim=1)
        return torch.cat([weighted, max_pool], dim=1)

    def forward(
        self,
        frames: torch.Tensor,
        pose: torch.Tensor,
        shuttle: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, _, _, _ = frames.shape
        joints = self.joint_proj(self.skeleton.encode_joint_tokens(pose))
        joints = self._align_joint_time(joints, T)
        patches = self._vision_patch_tokens(frames)
        tokens = self._fuse_frame_tokens(joints, patches)
        feat = self._clip_features(tokens, pose)
        if self.use_shuttle and shuttle is not None and self.shuttle_encoder is not None:
            sh = self.shuttle_encoder(shuttle.reshape(B, -1))
            feat = torch.cat([feat, sh], dim=1)
        return {task: head(feat) for task, head in self.heads.items()}


def build_k_st_vit(task_classes: Dict[str, int], **kwargs: Any) -> KinematicSpatiotemporalViT:
    return KinematicSpatiotemporalViT(task_classes, **kwargs)


def _load_conv3d_into_vision(model: KinematicSpatiotemporalViT, state: Dict[str, Any]) -> None:
    if model.vision_encoder is None:
        print("Skipping Conv3D load: model vision_backbone is not conv3d")
        return
    bb = {
        k.replace("backbone.", "", 1): v
        for k, v in state.items()
        if k.startswith("backbone.")
    }
    if bb:
        model.vision_encoder.backbone.load_state_dict(bb, strict=False)
        print(f"Loaded Conv3D backbone ({len(bb)} tensors)")
    proj = {k: v for k, v in state.items() if k.startswith("pose_proj.")}
    # pose_proj not used in vision encoder — ignore


def load_k_st_vit_partial(
    model: KinematicSpatiotemporalViT,
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
    if "k_st_vit" in ckpt:
        model.load_state_dict(ckpt["k_st_vit"], strict=False)
        print(f"Loaded K-STViT from {label}")
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
        if model.vit is not None:
            vit_state = {
                k.replace("vit_encoder.", "", 1): v
                for k, v in state.items()
                if k.startswith("vit_encoder.")
            }
            vit_only = {k: v for k, v in vit_state.items() if k.startswith("vit.")}
            model.vit.load_state_dict(vit_only, strict=False)
            proj_state = {k: v for k, v in vit_state.items() if k.startswith("proj.")}
            if model.feat_proj is not None and proj_state:
                model.feat_proj.load_state_dict(proj_state, strict=False)
        print(f"Loaded SkateFormer-B skeleton from {label}")
        return
    if "gv_xattn" in ckpt and model.vit is not None:
        state = ckpt["gv_xattn"]
        vit_state = {
            k.replace("vit_encoder.", "", 1): v
            for k, v in state.items()
            if k.startswith("vit_encoder.")
        }
        model.vit.load_state_dict(
            {k: v for k, v in vit_state.items() if k.startswith("vit.")}, strict=False
        )
        proj_state = {k: v for k, v in vit_state.items() if k.startswith("proj.")}
        if model.feat_proj is not None and proj_state:
            model.feat_proj.load_state_dict(proj_state, strict=False)
        print(f"Loaded ViT branch from GV-XAttn: {label}")
        return
    raise KeyError(
        f"Expected 'k_st_vit', conv3d_pose 'model', 'skateformer_b', or 'gv_xattn' in {label}, "
        f"got {list(ckpt.keys())}"
    )


def load_k_st_vit_skeleton_branch(
    model: KinematicSpatiotemporalViT,
    checkpoint_path: str,
    *,
    device: torch.device | str = "cpu",
) -> None:
    """Warm-start skeleton from pose-only or SkateFormer-B checkpoints."""
    from core.skateformer_b import SkateFormerBFusion

    proxy = SkateFormerBFusion(
        model.task_classes,
        window_size=model.window_size,
        four_stream=model.skeleton.four_stream,
    )
    load_skateformer_b_skeleton_branch(proxy, checkpoint_path, device=device)
    tgt = model.skeleton.state_dict()
    src = proxy.skeleton.state_dict()
    filtered = {k: v for k, v in src.items() if k in tgt and v.shape == tgt[k].shape}
    tgt.update(filtered)
    model.skeleton.load_state_dict(tgt, strict=False)
    print(
        f"Loaded K-STViT skeleton from {checkpoint_path} "
        f"({len(filtered)}/{len(src)} tensors)"
    )
