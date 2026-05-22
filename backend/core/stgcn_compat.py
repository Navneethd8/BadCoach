"""
Python 3.9-compatible ST-GCN classes adapted from the BST third-party repo.

Original: Yan et al., "Spatial Temporal Graph Convolutional Networks for
Skeleton-Based Action Recognition" (2018). Modified by Jing-Yuan Chang for BST.

The upstream ``model/stgcn.py`` uses ``match/case`` (Python 3.10+).  This
module replaces that with ``if/elif`` so it works on Amazon Linux 2023 (3.9).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _hop_distance(num_node: int, edge, max_hop: int = 1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    hop_dis = np.full((num_node, num_node), np.inf)
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def _normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    return Dn @ A @ Dn


class Graph:
    """Skeleton graph with if/elif layout selection (Python 3.9 compatible)."""

    def __init__(self, layout: str = "coco", strategy: str = "spatial",
                 max_hop: int = 1, dilation: int = 1):
        self.max_hop = max_hop
        self.dilation = dilation
        self._get_edge(layout)
        self.hop_dis = _hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self._get_adjacency(strategy)

    def _get_edge(self, layout: str):
        if layout == "coco":
            self.num_node = 17
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [16, 14], [14, 12], [17, 15], [15, 13],
                [12, 13], [6, 12], [7, 13], [6, 7],
                [8, 6], [10, 8], [9, 7], [11, 9],
                [2, 3],
                [2, 1], [3, 1], [4, 2], [5, 3],
                [4, 6], [5, 7],
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == "openpose":
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (4, 3), (3, 2), (7, 6), (6, 5),
                (13, 12), (12, 11), (10, 9), (9, 8), (11, 5),
                (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0),
                (17, 15), (16, 14),
            ]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == "mediapipe":
            self.num_node = 23
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                (6, 12), (6, 10), (6, 8), (8, 10),
                (7, 13), (7, 11), (7, 9), (9, 11),
                (2, 4), (4, 6), (3, 5), (5, 7),
                (2, 3), (2, 14), (3, 15), (14, 15),
                (14, 16), (15, 17), (16, 18), (17, 19),
                (18, 20), (18, 22), (20, 22),
                (19, 21), (19, 23), (21, 23),
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 0
        else:
            raise ValueError(f"Unknown layout: {layout}")

    def _get_adjacency(self, strategy: str):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        norm_adj = _normalize_undigraph(adjacency)

        if strategy == "uniform":
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = norm_adj
            self.A = A
        elif strategy == "distance":
            A = np.zeros((len(list(valid_hop)), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = norm_adj[self.hop_dis == hop]
            self.A = A
        elif strategy == "spatial":
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for j in range(self.num_node):
                    for i in range(self.num_node):
                        if self.hop_dis[i, j] == hop:
                            if self.hop_dis[self.center, i] == self.hop_dis[self.center, j]:
                                a_root[i, j] = norm_adj[i, j]
                            elif self.hop_dis[self.center, i] < self.hop_dis[self.center, j]:
                                a_close[i, j] = norm_adj[i, j]
                            else:
                                a_further[i, j] = norm_adj[i, j]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            self.A = np.stack(A)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


class GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels, out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x: Tensor, A: Tensor):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("n k c t v , k v w -> n c t w", x, A)
        return x.contiguous(), A


class ST_GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dropout=0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = GCN_Block(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


class ST_GCN_18(nn.Module):
    """ST-GCN with 10 blocks (18-layer equivalent). Input: (N, C, T, V, M)."""

    def __init__(self, in_channels, num_class, graph_cfg,
                 edge_importance_weighting=True, data_bn=True,
                 tem_kernel_size=9, **kwargs):
        super().__init__()
        self.graph = Graph(**graph_cfg)
        A = torch.from_numpy(self.graph.A).float()
        self.register_buffer("A", A)

        spatial_kernel_size = A.size(0)
        kernel_size = (tem_kernel_size, spatial_kernel_size)

        self.data_bn = (nn.BatchNorm1d(in_channels * A.size(1))
                        if data_bn else nn.Identity())

        kwargs0 = {k: v for k, v in kwargs.items() if k != "dropout"}
        self.st_gcn_networks = nn.ModuleList([
            ST_GCN_Block(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            ST_GCN_Block(64, 64, kernel_size, 1, **kwargs),
            ST_GCN_Block(64, 64, kernel_size, 1, **kwargs),
            ST_GCN_Block(64, 64, kernel_size, 1, **kwargs),
            ST_GCN_Block(64, 128, kernel_size, 2, **kwargs),
            ST_GCN_Block(128, 128, kernel_size, 1, **kwargs),
            ST_GCN_Block(128, 128, kernel_size, 1, **kwargs),
            ST_GCN_Block(128, 256, kernel_size, 2, **kwargs),
            ST_GCN_Block(256, 256, kernel_size, 1, **kwargs),
            ST_GCN_Block(256, 256, kernel_size, 1, **kwargs),
        ])

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(A.size())) for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x: Tensor):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        x = self.fcn(x)
        return x.view(x.size(0), -1)
