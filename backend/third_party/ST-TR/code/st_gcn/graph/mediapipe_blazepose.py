"""
BlazePose / MediaPipe 33-landmark graph for ST-GCN / ST-TR (IsoCourt).

Topology matches IsoCourt ``_BODY_EDGES`` in this module (BlazePose / MediaPipe).
Root for spatial partitioning: left hip (23), BFS tree → inward / outward.
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from . import tools

# Undirected body edges (same connectivity as legacy IsoCourt ST-TR module).
_BODY_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 11),
    (0, 12),
    (0, 1),
    (0, 4),
    (1, 2),
    (2, 3),
    (3, 7),
    (4, 5),
    (5, 6),
    (6, 8),
    (7, 8),
    (9, 10),
    (7, 9),
    (8, 10),
    (3, 5),
    (15, 17),
    (15, 19),
    (15, 21),
    (16, 18),
    (16, 20),
    (16, 22),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)

num_node = 33
self_link = [(i, i) for i in range(num_node)]


def _bfs_inward(root: int = 23) -> List[Tuple[int, int]]:
    adj: List[List[int]] = [[] for _ in range(num_node)]
    for a, b in _BODY_EDGES:
        adj[a].append(b)
        adj[b].append(a)
    parent = [-1] * num_node
    q = deque([root])
    parent[root] = root
    while q:
        u = q.popleft()
        for v in adj[u]:
            if parent[v] == -1:
                parent[v] = u
                q.append(v)
    for i in range(num_node):
        if parent[i] == -1:
            parent[i] = i
    return [(i, parent[i]) for i in range(num_node) if i != parent[i]]


inward = _bfs_inward(23)
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    """MediaPipe 33-node skeleton graph for upstream ST-TR / ST-GCN."""

    def __init__(self, labeling_mode: str = "spatial") -> None:
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode: Optional[str] = None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == "uniform":
            return tools.get_uniform_graph(num_node, self_link, neighbor)
        if labeling_mode == "distance*":
            return tools.get_uniform_distance_graph(num_node, self_link, neighbor)
        if labeling_mode == "distance":
            return tools.get_distance_graph(num_node, self_link, neighbor)
        if labeling_mode == "spatial":
            return tools.get_spatial_graph(num_node, self_link, inward, outward)
        if labeling_mode == "DAD":
            return tools.get_DAD_graph(num_node, self_link, neighbor)
        if labeling_mode == "DLD":
            return tools.get_DLD_graph(num_node, self_link, neighbor)
        raise ValueError(f"Unknown labeling_mode: {labeling_mode!r}")
