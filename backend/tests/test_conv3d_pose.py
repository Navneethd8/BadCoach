"""Conv3D + pose multitask — CPU, random 3D trunk (no Kinetics weight download)."""
import pytest
import torch

from core.conv3d_pose import Conv3DPoseMultitaskModel, backbone_parameter_groups

pytest.importorskip("torchvision.models.video")

TASK_CLASSES = {
    "stroke_type": 9,
    "technique": 3,
    "placement": 10,
    "position": 10,
    "intent": 10,
    "quality": 7,
}


def test_forward_with_pose():
    m = Conv3DPoseMultitaskModel(
        TASK_CLASSES,
        num_frames=16,
        video_backbone="r2plus1d_18",
        spatial_size=112,
        pretrained=False,
        freeze_backbone=False,
        use_pose=True,
    )
    m.eval()
    b = 2
    f = torch.randn(b, 16, 3, 224, 224)
    j = torch.randn(b, 16, 33, 3)
    with torch.no_grad():
        out = m(f, j)
    assert set(out) == set(TASK_CLASSES)
    for task, n in TASK_CLASSES.items():
        assert out[task].shape == (b, n)


def test_forward_no_pose():
    m = Conv3DPoseMultitaskModel(
        TASK_CLASSES,
        num_frames=16,
        pretrained=False,
        freeze_backbone=False,
        use_pose=False,
    )
    f = torch.randn(1, 16, 3, 224, 224)
    with torch.no_grad():
        out = m(f, None)
    assert set(out) == set(TASK_CLASSES)


def test_grad_cam_target_is_module():
    m = Conv3DPoseMultitaskModel(TASK_CLASSES, num_frames=8, pretrained=False, freeze_backbone=False)
    g = m.grad_cam_target_module()
    assert isinstance(g, torch.nn.Module)


def test_backbone_parameter_groups_splits():
    m = Conv3DPoseMultitaskModel(
        TASK_CLASSES,
        num_frames=16,
        pretrained=False,
        freeze_backbone=True,
        unfreeze_layer4=True,
        use_pose=True,
    )
    bb, other = backbone_parameter_groups(m)
    assert len(bb) > 0 and len(other) > 0
