import torch

from core.pose_cache_build import (
    DEFAULT_POSE_CACHE_FILENAME,
    LEGACY_POSE_CACHE_FILENAME,
    load_pose_cache_bundle,
    media_pipe_fill_pose_cache,
)


class _TinyDataset:
    sequence_length = 2

    def __init__(self, n: int):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        # (T, C, H, W) RGB in [0,1] — values unused by mock estimator
        return torch.zeros(self.sequence_length, 3, 4, 4), {}


class _MockPoseEst:
    def extract_tensor_poses(self, frames_tensor):
        T = frames_tensor.shape[0]
        return torch.arange(float(T * 99)).reshape(T, 99)


def test_load_pose_cache_bundle_legacy_fallback(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    legacy = models / LEGACY_POSE_CACHE_FILENAME
    torch.save({"pose_cache": torch.zeros(2, 3, 33, 3)}, legacy)
    want = models / DEFAULT_POSE_CACHE_FILENAME
    bundle = load_pose_cache_bundle(str(want))
    assert bundle is not None
    assert bundle["pose_cache"].shape == (2, 3, 33, 3)


def test_media_pipe_fill_pose_cache_preallocated_shape_and_order():
    ds = _TinyDataset(3)
    out = media_pipe_fill_pose_cache(ds, _MockPoseEst())
    assert out.shape == (3, 2, 33, 3)
    assert out.dtype == torch.float32
    # Row 0 should match first clip's flat range reshaped
    row0 = torch.arange(198.0).reshape(2, 33, 3)
    assert torch.allclose(out[0], row0)
