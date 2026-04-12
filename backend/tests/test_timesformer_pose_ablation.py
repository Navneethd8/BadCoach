"""TimeSformer use_pose flag — CPU, scratch backbone only."""
import torch

from core.timesformer import TimeSformerPoseModel

TASK_CLASSES = {"stroke_type": 9, "position": 10}


def test_timesformer_use_pose_true_requires_joints():
    m = TimeSformerPoseModel(TASK_CLASSES, depth=2, backbone="scratch", num_frames=16, use_pose=True)
    f = torch.randn(1, 16, 3, 224, 224)
    with torch.no_grad():
        j = torch.randn(1, 16, 33, 3)
        out = m(f, j)
    assert set(out) == set(TASK_CLASSES)


def test_timesformer_use_pose_false_patch_only():
    m = TimeSformerPoseModel(TASK_CLASSES, depth=2, backbone="scratch", num_frames=16, use_pose=False)
    f = torch.randn(1, 16, 3, 224, 224)
    with torch.no_grad():
        out = m(f, None)
    assert set(out) == set(TASK_CLASSES)
