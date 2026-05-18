"""
Official ST-TR / ST-GCN ``Model`` from Chiaraplizz/ST-TR (MIT), vendored under
``backend/third_party/ST-TR``, adapted for IsoCourt MediaPipe clips.

Upstream: https://github.com/Chiaraplizz/ST-TR

``IsoCourtOfficialSTTR`` subclasses ``st_gcn.net.st_gcn.Model``, overrides
``forward`` to emit multitask logits (upstream is single-task).

Input pose layout matches the rest of IsoCourt: ``(B, T, V, C)`` with ``V=33``,
``C=3`` (MediaPipe x,y,z in [0,1]-ish normalized space from the pose cache).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_ST_TR_CODE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "third_party", "ST-TR", "code")
)
if _ST_TR_CODE not in sys.path:
    sys.path.insert(0, _ST_TR_CODE)

from st_gcn.net.st_gcn import Model  # noqa: E402


StreamMode = Literal["spatial", "temporal", "both"]


def default_upstream_model_kwargs(
    *,
    window_size: int,
    stream: StreamMode = "both",
    dropout: float = 0.1,
) -> Dict[str, Any]:
    """Hyperparameters aligned with ``code/config/st_gcn/nturgbd/train.yaml`` ST-TR-style runs,
    simplified (no ``concat_original``) for short badminton clips."""
    att = stream in ("spatial", "both")
    tcn_att = stream in ("temporal", "both")
    return dict(
        channel=3,
        num_class=1,
        window_size=window_size,
        num_point=33,
        attention=att,
        only_attention=True,
        tcn_attention=tcn_att,
        only_temporal_attention=True,
        attention_3=False,
        relative=False,
        kernel_temporal=9,
        double_channel=False,
        drop_connect=True,
        concat_original=False,
        dv=0.25,
        dk=0.25,
        Nh=8,
        dim_block1=10,
        dim_block2=30,
        dim_block3=75,
        all_layers=False,
        data_normalization=True,
        visualization=False,
        skip_conn=True,
        adjacency=False,
        bn_flag=True,
        weight_matrix=2,
        device=0,
        n=4,
        more_channels=False,
        num_person=1,
        use_data_bn=True,
        graph="st_gcn.graph.mediapipe_blazepose.Graph",
        graph_args={"labeling_mode": "spatial"},
        mask_learning=False,
        use_local_bn=False,
        multiscale=False,
        temporal_kernel_size=9,
        dropout=dropout,
        agcn=False,
    )


class IsoCourtOfficialSTTR(Model):
    """Multitask wrapper around upstream ``Model`` (single skeleton stream)."""

    def __init__(
        self,
        task_classes: Dict[str, int],
        *,
        window_size: int = 16,
        stream: StreamMode = "both",
        dropout: float = 0.1,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        kw = default_upstream_model_kwargs(window_size=window_size, stream=stream, dropout=dropout)
        if model_kwargs:
            kw.update(model_kwargs)
        super().__init__(**kw)
        feat_dim = int(self.fcn.in_channels)
        del self.fcn
        self._feat_dim = feat_dim
        self.task_classes = dict(task_classes)
        self.heads = nn.ModuleDict(
            {task: nn.Linear(feat_dim, n_cls) for task, n_cls in task_classes.items()}
        )

    def encode_skeleton(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Upstream ST-TR trunk: pose tensor -> clip embedding.

        Args:
            pose: ``(B, T, V, C)`` skeleton sequence (e.g. MediaPipe cache).

        Returns:
            ``(B, feat_dim)`` before task heads.
        """
        x = pose.permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        label = torch.zeros(N, dtype=torch.long, device=x.device)
        name = [""] * N

        if self.concat_original:
            x_coord = x
            x_coord = x_coord.permute(0, 4, 1, 2, 3).reshape(N * M, C, T, V)

        if self.use_data_bn:
            if self.M_dim_bn:
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
            else:
                x = x.permute(0, 4, 3, 1, 2).contiguous().view(N * M, V * C, T)
            x = self.data_bn(x)
            x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(
                N * M, C, T, V
            )
        else:
            x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        if not self.all_layers:
            x = self.gcn0(x, label, name)
            x = self.tcn0(x)

        for i, m in enumerate(self.backbone):
            if i == 3 and self.concat_original:
                x = m(torch.cat((x, x_coord), dim=1), label, name)
            else:
                x = m(x, label, name)

        x = F.avg_pool2d(x, kernel_size=(1, V))
        c = x.size(1)
        t = x.size(2)
        x = x.view(N, M, c, t).mean(dim=1).view(N, c, t)
        x = F.avg_pool1d(x, kernel_size=x.size()[2])
        return x.squeeze(-1)

    def forward(self, pose: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pose: ``(B, T, V, C)`` skeleton sequence (e.g. MediaPipe cache).
        """
        x = self.encode_skeleton(pose)
        return {task: head(x) for task, head in self.heads.items()}


def build_official_st_tr(
    task_classes: Dict[str, int],
    *,
    window_size: int = 16,
    stream: StreamMode = "both",
    dropout: float = 0.1,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> IsoCourtOfficialSTTR:
    return IsoCourtOfficialSTTR(
        task_classes,
        window_size=window_size,
        stream=stream,
        dropout=dropout,
        model_kwargs=model_kwargs,
    )
