import torch

from core.staeformer import STAEformerModel

TC = {"stroke_type": 9, "position": 10}


def test_staeformer_pose_and_cnn():
    m = STAEformerModel(TC, embed_dim=32, num_layers=1, use_cnn=True, use_pose=True)
    j = torch.randn(2, 16, 33, 3)
    c = torch.randn(2, 16, 2048)
    with torch.no_grad():
        out = m(j, c)
    assert set(out) == set(TC)


def test_staeformer_cnn_only():
    m = STAEformerModel(TC, embed_dim=32, num_layers=1, use_cnn=True, use_pose=False)
    c = torch.randn(2, 16, 2048)
    with torch.no_grad():
        out = m(cnn_seq=c)
    assert set(out) == set(TC)
