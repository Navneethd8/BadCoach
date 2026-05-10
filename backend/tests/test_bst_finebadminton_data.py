"""Unit checks for BST FineBadminton preprocessing helpers."""
from __future__ import annotations

import numpy as np

from core.bst_finebadminton_data import (
    create_bones,
    get_bone_pairs_coco,
    normalize_joints_bst,
)


def test_bone_pair_count_matches_bst_shuttleset():
    assert len(get_bone_pairs_coco()) == 19


def test_jnb_in_dim_matches_bst_formula():
    n_j, n_b = 17, 19
    in_dim = (n_j + n_b) * 2
    assert in_dim == 72


def test_normalize_joints_identity_bbox():
    arr = np.array([[[10.0, 10.0], [20.0, 10.0]]], dtype=np.float32)
    bbox = np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32)
    out = normalize_joints_bst(arr, bbox, center_align=True)
    assert out.shape == (1, 2, 2)


def test_create_bones_shape():
    t, m, j = 4, 2, 17
    joints = np.random.randn(t, m, j, 2).astype(np.float32)
    bones = create_bones(joints, get_bone_pairs_coco())
    assert bones.shape == (t, m, 19, 2)
