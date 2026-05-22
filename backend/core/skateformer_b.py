"""
SkateFormer-B: badminton-oriented skeletal–temporal model + optional ViT fusion.

Novelty (vs generic SkateFormer / ST-TR on the same data):
  - Four-stream skeleton input (joint, bone, joint-motion, bone-motion) from MediaPipe-33.
  - Skate-MSA partitions tuned for T=16, V=33 (broadcast badminton clips).
  - Late fusion with per-frame ViT image context (skeleton embedding + visual embedding).

Reference backbone: KAIST-VICLab/SkateFormer (ECCV 2024).
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from core.skateformer.official import SkateFormer
from core.skeleton_streams import (
    NUM_SKELETON_STREAMS,
    STREAMS_PER_COORD,
    four_stream_to_skateformer_input,
)
from core.skateformer_model import DEFAULT_PARTITION_SIZES
from core.st_tr_vit_fusion import ViTClipEncoder
from core.training_standards import MEDIAPIPE_NUM_FRAMES, MEDIAPIPE_NUM_JOINTS, SEQUENCE_LENGTH

IN_CHANNELS_FOUR_STREAM = NUM_SKELETON_STREAMS * STREAMS_PER_COORD  # 12


def default_skateformer_b_kwargs(
    *,
    num_frames: int = MEDIAPIPE_NUM_FRAMES,
    num_points: int = MEDIAPIPE_NUM_JOINTS,
    embed_dim: int = 64,
    num_heads: int = 16,
    four_stream: bool = True,
) -> Dict[str, Any]:
    in_ch = IN_CHANNELS_FOUR_STREAM if four_stream else 3
    return dict(
        in_channels=in_ch,
        depths=(2, 2, 2, 2),
        channels=(embed_dim, embed_dim * 2, embed_dim * 2, embed_dim * 2),
        embed_dim=embed_dim,
        num_people=1,
        num_frames=num_frames,
        num_points=num_points,
        kernel_size=7,
        num_heads=num_heads,
        attn_drop=0.2,
        head_drop=0.1,
        rel=True,
        drop_path=0.1,
        index_t=True,
        mlp_ratio=4.0,
        global_pool="avg",
        **DEFAULT_PARTITION_SIZES,
    )


def default_skateformer_b_checkpoint_path(backend_root: str) -> str:
    return os.path.join(
        os.path.abspath(backend_root), "models", "badminton_model_skateformer_b.pth"
    )


class SkateFormerBEncoder(nn.Module):
    """SkateFormer trunk with joint+bone+motion streams (pose-only embedding)."""

    def __init__(
        self,
        *,
        embed_dim: int = 64,
        num_heads: int = 16,
        num_frames: int = MEDIAPIPE_NUM_FRAMES,
        num_points: int = MEDIAPIPE_NUM_JOINTS,
        four_stream: bool = True,
        dropout: float = 0.1,
        skateformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.four_stream = four_stream
        kw = default_skateformer_b_kwargs(
            num_frames=num_frames,
            num_points=num_points,
            embed_dim=embed_dim,
            num_heads=num_heads,
            four_stream=four_stream,
        )
        if skateformer_kwargs:
            kw.update(skateformer_kwargs)

        self.backbone = SkateFormer(num_classes=1, **kw)
        self.backbone.head = nn.Identity()
        self.feat_dim = int(kw["channels"][-1])
        self.embed_dim = int(kw["embed_dim"])
        self.dropout = nn.Dropout(dropout)

    def _to_input(self, pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.four_stream:
            return four_stream_to_skateformer_input(pose)
        from core.skateformer_model import mediapipe_pose_to_skateformer

        return mediapipe_pose_to_skateformer(pose)

    def _embed(self, data: torch.Tensor, index_t: torch.Tensor) -> torch.Tensor:
        B, C, T, V, M = data.shape
        output = data.permute(0, 1, 2, 4, 3).contiguous().view(B, C, T, -1)
        for layer in self.backbone.stem:
            output = layer(output)

        if self.backbone.index_t:
            te = torch.zeros(
                B, T, self.backbone.embed_dim, device=output.device, dtype=output.dtype,
            )
            div_term = torch.exp(
                torch.arange(0, self.backbone.embed_dim, 2, device=output.device, dtype=torch.float32)
                * (-(math.log(10000.0) / self.backbone.embed_dim))
            )
            te[:, :, 0::2] = torch.sin(index_t.unsqueeze(-1).float() * div_term)
            te[:, :, 1::2] = torch.cos(index_t.unsqueeze(-1).float() * div_term)
            output = output + torch.einsum(
                "b t c, c v -> b c t v", te, self.backbone.joint_person_embedding,
            )
        else:
            output = output + self.backbone.joint_person_temporal_embedding
        return output

    def encode_skeleton(self, pose: torch.Tensor) -> torch.Tensor:
        """Pooled skeleton embedding ``(B, feat_dim)``."""
        data, index_t = self._to_input(pose)
        x = self.backbone.forward_features(self._embed(data, index_t))
        return self.dropout(self.backbone.forward_head(x, pre_logits=True))

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        return self.encode_skeleton(pose)


class SkateFormerBFusion(nn.Module):
    """
    SkateFormer-B skeleton embedding + ViT clip context -> fusion MLP -> multitask heads.
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
        vit_model_name: str = "vit_small_patch16_224",
        vit_embed_dim: int = 128,
        vit_unfreeze_last_n: int = 2,
        vit_pretrained: bool = True,
        fusion_dropout: float = 0.2,
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

        self.vit_encoder = ViTClipEncoder(
            num_frames=window_size,
            vit_model_name=vit_model_name,
            vit_embed_dim=vit_embed_dim,
            vit_unfreeze_last_n=vit_unfreeze_last_n,
            pretrained=vit_pretrained,
            dropout=dropout,
        )
        vit_dim = self.vit_encoder.out_dim
        hidden = max(skel_dim, vit_embed_dim * 2)

        self.fusion = nn.Sequential(
            nn.Linear(skel_dim + vit_dim, hidden),
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

    def forward(
        self, frames: torch.Tensor, pose: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        skel = self.encode_skeleton(pose)
        vit = self.vit_encoder(frames)
        feat = self.fusion(torch.cat([skel, vit], dim=1))
        return {task: head(feat) for task, head in self.heads.items()}


def build_skateformer_b_fusion(task_classes: Dict[str, int], **kwargs: Any) -> SkateFormerBFusion:
    return SkateFormerBFusion(task_classes, **kwargs)


def load_skateformer_b_skeleton_branch(
    model: SkateFormerBFusion,
    checkpoint_path: str,
    *,
    device: torch.device | str = "cpu",
) -> None:
    """Load pose-only SkateFormer or SkateFormer-B weights into the skeleton trunk."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "skateformer_b" in ckpt:
        state = ckpt["skateformer_b"]
        skel_state = {
            k.replace("skeleton.", "", 1): v
            for k, v in state.items()
            if k.startswith("skeleton.")
        }
        model.skeleton.load_state_dict(skel_state, strict=False)
        print(f"Loaded SkateFormer-B skeleton from {checkpoint_path}")
        return
    if "skateformer" in ckpt:
        trunk = model.skeleton.backbone.state_dict()
        pretrained = ckpt["skateformer"]
        filtered = {
            k: v for k, v in pretrained.items()
            if k in trunk and v.shape == trunk[k].shape
        }
        trunk.update(filtered)
        model.skeleton.backbone.load_state_dict(trunk, strict=False)
        print(
            f"Loaded SkateFormer backbone (partial) from {checkpoint_path} "
            f"({len(filtered)} tensors; 12-ch stem may be random)"
        )
        return
    raise KeyError(
        f"Expected 'skateformer_b' or 'skateformer' in {checkpoint_path}, "
        f"got {list(ckpt.keys())}"
    )
