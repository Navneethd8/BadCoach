"""ViT + fixed GCN multitask model — CPU, random ViT weights (no checkpoint download)."""
import pytest
import torch

from core.vit_gcn import ViTGCNMultitaskModel, _symmetric_normalized_adjacency, MEDIAPIPE_BODY_EDGES


# Matches train_vit_gcn / train_timesformer task head set (stroke_subtype omitted)
TASK_CLASSES = {
    "stroke_type": 9,
    "technique": 3,
    "placement": 10,
    "position": 10,
    "intent": 10,
    "quality": 7,
}


@pytest.fixture(scope="module")
def model_cpu():
    m = ViTGCNMultitaskModel(
        TASK_CLASSES,
        num_frames=16,
        embed_dim=64,
        gcn_layers=2,
        pretrained=False,
    )
    m.eval()
    return m


class TestViTGCN:

    def test_forward_keys_and_shapes(self, model_cpu):
        b = 2
        f = torch.randn(b, 16, 3, 224, 224)
        j = torch.randn(b, 16, 33, 3)
        with torch.no_grad():
            out = model_cpu(f, j)
        assert set(out.keys()) == set(TASK_CLASSES.keys())
        for task, n in TASK_CLASSES.items():
            assert out[task].shape == (b, n)

    def test_output_finite(self, model_cpu):
        f = torch.randn(1, 16, 3, 224, 224)
        j = torch.randn(1, 16, 33, 3)
        with torch.no_grad():
            out = model_cpu(f, j)
        for logits in out.values():
            assert torch.isfinite(logits).all()

    def test_forward_no_pose(self):
        m = ViTGCNMultitaskModel(
            TASK_CLASSES,
            num_frames=16,
            embed_dim=64,
            gcn_layers=2,
            pretrained=False,
            use_pose=False,
        )
        m.eval()
        f = torch.randn(2, 16, 3, 224, 224)
        with torch.no_grad():
            out = m(f, None)
        assert set(out.keys()) == set(TASK_CLASSES.keys())
        for task, n in TASK_CLASSES.items():
            assert out[task].shape == (2, n)

    def test_adjacency_symmetric_normalized(self):
        adj = _symmetric_normalized_adjacency(33, MEDIAPIPE_BODY_EDGES, torch.device("cpu"), torch.float32)
        assert adj.shape == (33, 33)
        assert torch.allclose(adj, adj.t())
        assert (adj.diagonal() > 0).all()
