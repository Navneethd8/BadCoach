"""Shuttle trajectory cache aligned with pose cache sample indices."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

DEFAULT_SHUTTLE_CACHE_FILENAME = "shuttle_cache.pt"


def default_shuttle_cache_path(backend_root: str) -> str:
    return os.path.join(
        os.path.abspath(backend_root), "models", DEFAULT_SHUTTLE_CACHE_FILENAME
    )


def load_shuttle_cache_bundle(cache_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(cache_path):
        return None
    print(f"Loading shuttle cache from {cache_path}...")
    return torch.load(cache_path, map_location="cpu", weights_only=False)


def save_shuttle_cache(cache_path: str, shuttle_cache: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save({"shuttle_cache": shuttle_cache}, cache_path)
    print(f"Saved shuttle cache to {cache_path} shape={tuple(shuttle_cache.shape)}")
