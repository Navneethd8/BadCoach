"""
GV-XAttn: graph joint tokens cross-attend to ViT patch tokens for badminton stroke recognition.

Four-stream MediaPipe pose -> per-frame graph node tokens -> cross-attention (Q=graph, K,V=patches)
-> spatial pool -> contact-weighted temporal pool -> multitask heads.

See docs/plan.md.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.skeleton_streams import NUM_SKELETON_STREAMS, STREAMS_PER_COORD, build_four_stream_pose
from core.st_tr_vit_fusion import ViTClipEncoder
from core.training_standards import MEDIAPIPE_NUM_FRAMES, MEDIAPIPE_NUM_JOINTS
from core.vit_gcn import MEDIAPIPE_BODY_EDGES, FixedGCNStack

GRAPH_IN_DIM = NUM_SKELETON_STREAMS * STREAMS_PER_COORD  # 12
FusionMode = Literal["cross", "late"]


def default_gv_xattn_checkpoint_path(backend_root: str) -> str:
    return os.path.join(os.path.abspath(backend_root), "models", "badminton_model_gv_xattn.pth")


class GraphVisionCrossBlock(nn.Module):
    """Graph nodes attend to visual patch tokens within the same frame."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_g = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, graph: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        g = self.norm_g(graph)
        p = self.norm_v(patches)
        attn_out, _ = self.cross_attn(g, p, p, need_weights=False)
        graph = graph + attn_out
        graph = graph + self.ff(self.norm_ff(graph))
        return graph


class GraphTokenEncoder(nn.Module):
    """Four-stream pose -> GCN node tokens ``(B, T, V, D)``."""

    def __init__(
        self,
        *,
        num_joints: int = MEDIAPIPE_NUM_JOINTS,
        hidden_dim: int = 128,
        gcn_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gcn = FixedGCNStack(
            num_nodes=num_joints,
            edges=MEDIAPIPE_BODY_EDGES,
            in_dim=GRAPH_IN_DIM,
            hidden_dim=hidden_dim,
            num_layers=gcn_layers,
            dropout=dropout,
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        B, T, J, _ = pose.shape
        streams = build_four_stream_pose(pose)
        nodes = self.gcn(streams.view(B * T, J, GRAPH_IN_DIM))
        return nodes.view(B, T, J, -1)


def subsample_patch_tokens(patches: torch.Tensor) -> torch.Tensor:
    """``(N, P, D)`` with square P -> 2x2 spatial avg pool (P/4 tokens)."""
    n, p, d = patches.shape
    side = int(round(p**0.5))
    if side * side != p:
        raise ValueError(f"Expected square patch count, got P={p}")
    x = patches.view(n, side, side, d).permute(0, 3, 1, 2)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)
    return x.flatten(2).transpose(1, 2)


def contact_frame_weights(pose: torch.Tensor) -> torch.Tensor:
    """``(B, T)`` softmax weights from joint-motion magnitude."""
    streams = build_four_stream_pose(pose)
    jmotion = streams[..., 6:9]
    per_frame = jmotion.norm(dim=-1).amax(dim=2)
    return F.softmax(per_frame, dim=1)


class GraphVisionCrossAttnModel(nn.Module):
    """GV-XAttn: cross-modal fusion or late-fusion ablation."""

    def __init__(
        self,
        task_classes: Dict[str, int],
        *,
        window_size: int = MEDIAPIPE_NUM_FRAMES,
        embed_dim: int = 128,
        gcn_layers: int = 2,
        num_cross_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        vit_model_name: str = "vit_small_patch16_224",
        vit_embed_dim: int = 128,
        vit_unfreeze_last_n: int = 2,
        vit_pretrained: bool = True,
        contact_pool: bool = True,
        fusion_mode: FusionMode = "cross",
    ) -> None:
        super().__init__()
        self.task_classes = dict(task_classes)
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.contact_pool = contact_pool
        self.fusion_mode = fusion_mode

        self.graph_encoder = GraphTokenEncoder(
            hidden_dim=embed_dim,
            gcn_layers=gcn_layers,
            dropout=dropout,
        )
        self.vit_encoder = ViTClipEncoder(
            num_frames=window_size,
            vit_model_name=vit_model_name,
            vit_embed_dim=vit_embed_dim,
            vit_unfreeze_last_n=vit_unfreeze_last_n,
            pretrained=vit_pretrained,
            dropout=dropout,
        )
        self.cross_blocks = nn.ModuleList(
            [
                GraphVisionCrossBlock(embed_dim, num_heads, dropout=dropout)
                for _ in range(num_cross_layers)
            ]
        )
        self.temporal_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=1,
        )

        self.late_fusion: nn.Module | None
        if fusion_mode == "late":
            fuse_in = embed_dim + self.vit_encoder.out_dim
            hidden = max(embed_dim, vit_embed_dim * 2)
            self.late_fusion = nn.Sequential(
                nn.Linear(fuse_in, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.late_fusion = None

        self.heads = nn.ModuleDict(
            {task: nn.Linear(embed_dim, n_cls) for task, n_cls in task_classes.items()}
        )

    def _pool_temporal(self, frame_feat: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        if self.contact_pool:
            w = contact_frame_weights(pose).unsqueeze(-1)
            return (frame_feat * w).sum(dim=1)
        avg = frame_feat.mean(dim=1)
        mx, _ = frame_feat.max(dim=1)
        return 0.5 * (avg + mx)

    def _encode_cross(self, frames: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        B, T, J, _ = pose.shape
        graph = self.graph_encoder(pose)
        patches = self.vit_encoder.forward_patch_tokens(frames)

        graph_bt = graph.view(B * T, J, self.embed_dim)
        patch_bt = subsample_patch_tokens(patches.view(B * T, -1, self.embed_dim))
        for block in self.cross_blocks:
            graph_bt = block(graph_bt, patch_bt)

        frame_feat = graph_bt.view(B, T, J, self.embed_dim).mean(dim=2)
        frame_feat = self.temporal_attn(frame_feat)
        return self._pool_temporal(frame_feat, pose)

    def _encode_late(self, frames: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        graph = self.graph_encoder(pose)
        frame_feat = graph.mean(dim=2)
        skel = self._pool_temporal(frame_feat, pose)
        vit = self.vit_encoder(frames)
        assert self.late_fusion is not None
        return self.late_fusion(torch.cat([skel, vit], dim=1))

    def forward(
        self, frames: torch.Tensor, pose: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        feat = self._encode_late(frames, pose) if self.fusion_mode == "late" else self._encode_cross(
            frames, pose
        )
        return {task: head(feat) for task, head in self.heads.items()}


def build_gv_xattn(task_classes: Dict[str, int], **kwargs: Any) -> GraphVisionCrossAttnModel:
    return GraphVisionCrossAttnModel(task_classes, **kwargs)


def load_gv_xattn_partial(
    model: GraphVisionCrossAttnModel,
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
    if "gv_xattn" in ckpt:
        model.load_state_dict(ckpt["gv_xattn"], strict=False)
        print(f"Loaded GV-XAttn from {label}")
        return
    if "skateformer_b" in ckpt:
        vit_state = {
            k.replace("vit_encoder.", "", 1): v
            for k, v in ckpt["skateformer_b"].items()
            if k.startswith("vit_encoder.")
        }
        model.vit_encoder.load_state_dict(vit_state, strict=False)
        print(f"Loaded ViT branch from SkateFormer-B: {label}")
        return
    raise KeyError(
        f"Expected 'gv_xattn' or 'skateformer_b' in {label}, got {list(ckpt.keys())}"
    )
