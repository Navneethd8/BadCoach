"""
SkateFormer skeleton encoder for IsoCourt (four-stream MediaPipe-33, T=16).

Shared backbone used by K-STViT and JVC no-xattn. Vendored implementation:
``core.skateformer.official`` (KAIST-VICLab/SkateFormer, ECCV 2024).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from core.skateformer.official import SkateFormer
from core.skeleton_streams import (
    NUM_SKELETON_STREAMS,
    STREAMS_PER_COORD,
    four_stream_to_skateformer_input,
)
from core.training_standards import MEDIAPIPE_NUM_FRAMES, MEDIAPIPE_NUM_JOINTS

IN_CHANNELS_FOUR_STREAM = NUM_SKELETON_STREAMS * STREAMS_PER_COORD  # 12

# Partition sizes must divide T at every stage (stem T=16 → 8 → 4 → 2).
DEFAULT_PARTITION_SIZES = {
    "type_1_size": (2, 3),
    "type_2_size": (2, 11),
    "type_3_size": (2, 3),
    "type_4_size": (2, 11),
}


def mediapipe_pose_to_skateformer(pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert ``(B, T, J, 3)`` MediaPipe pose to SkateFormer ``(B, C, T, V, M)`` input."""
    B, T, _J, _C = pose.shape
    root = pose[:, :, 23:25, :].mean(dim=2, keepdim=True)
    centered = pose - root
    scale = centered.reshape(B, T, -1).norm(dim=-1).mean(dim=1, keepdim=True).clamp(min=1e-6)
    centered = centered / scale.view(B, 1, 1, 1)

    data = centered.permute(0, 3, 1, 2).unsqueeze(-1).contiguous()
    index_t = torch.linspace(-1.0, 1.0, T, device=pose.device, dtype=pose.dtype)
    index_t = index_t.unsqueeze(0).expand(B, -1)
    return data, index_t


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

    def encode_joint_tokens(self, pose: torch.Tensor) -> torch.Tensor:
        """Per-frame joint tokens ``(B, T, V, feat_dim)`` before global pool."""
        data, index_t = self._to_input(pose)
        x = self.backbone.forward_features(self._embed(data, index_t))
        return self.dropout(x.permute(0, 2, 3, 1).contiguous())

    def encode_skeleton(self, pose: torch.Tensor) -> torch.Tensor:
        """Pooled skeleton embedding ``(B, feat_dim)``."""
        data, index_t = self._to_input(pose)
        x = self.backbone.forward_features(self._embed(data, index_t))
        return self.dropout(self.backbone.forward_head(x, pre_logits=True))

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        return self.encode_skeleton(pose)


def load_skateformer_b_skeleton_branch(
    encoder: SkateFormerBEncoder,
    checkpoint_path: str,
    *,
    device: torch.device | str = "cpu",
) -> None:
    """Load legacy SkateFormer / SkateFormer-B skeleton weights into ``encoder``."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "skateformer_b" in ckpt:
        state = ckpt["skateformer_b"]
        skel_state = {
            k.replace("skeleton.", "", 1): v
            for k, v in state.items()
            if k.startswith("skeleton.")
        }
        tgt = encoder.state_dict()
        filtered = {k: v for k, v in skel_state.items() if k in tgt and v.shape == tgt[k].shape}
        tgt.update(filtered)
        encoder.load_state_dict(tgt, strict=False)
        print(
            f"Loaded SkateFormer-B skeleton from {checkpoint_path} "
            f"({len(filtered)}/{len(skel_state)} tensors)"
        )
        return
    if "skateformer" in ckpt:
        trunk = encoder.backbone.state_dict()
        pretrained = ckpt["skateformer"]
        filtered = {
            k: v for k, v in pretrained.items()
            if k in trunk and v.shape == trunk[k].shape
        }
        trunk.update(filtered)
        encoder.backbone.load_state_dict(trunk, strict=False)
        print(
            f"Loaded SkateFormer backbone (partial) from {checkpoint_path} "
            f"({len(filtered)} tensors; 12-ch stem may be random)"
        )
        return
    raise KeyError(
        f"Expected 'skateformer_b' or 'skateformer' in {checkpoint_path}, "
        f"got {list(ckpt.keys())}"
    )
