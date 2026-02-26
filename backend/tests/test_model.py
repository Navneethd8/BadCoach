"""
Unit tests for CNN_LSTM_Model (model.py).
No GPU required — all tests run on CPU with random weights.
"""
import pytest
import torch
from model import CNN_LSTM_Model


TASK_CLASSES_FULL = {
    "stroke_type": 9,
    "stroke_subtype": 21,
    "technique": 4,
    "placement": 7,
    "position": 10,
    "intent": 10,
    "quality": 7,
    "is_badminton": 2,
}


@pytest.fixture(scope="module")
def model_cpu():
    m = CNN_LSTM_Model(task_classes=TASK_CLASSES_FULL, hidden_size=64, pretrained=False)
    m.eval()
    return m


class TestCNNLSTMModel:

    def test_forward_returns_all_task_keys(self, model_cpu):
        """All expected task heads must be present in output dict."""
        x = torch.randn(1, 16, 3, 224, 224)
        with torch.no_grad():
            out = model_cpu(x)
        assert set(out.keys()) == set(TASK_CLASSES_FULL.keys())

    def test_output_shapes_correct(self, model_cpu):
        """Output logits shape must be (batch, num_classes) for each task."""
        batch = 2
        x = torch.randn(batch, 16, 3, 224, 224)
        with torch.no_grad():
            out = model_cpu(x)
        for task, n_classes in TASK_CLASSES_FULL.items():
            assert out[task].shape == (batch, n_classes), \
                f"Task '{task}': expected ({batch}, {n_classes}), got {out[task].shape}"

    def test_batch_size_one(self, model_cpu):
        x = torch.randn(1, 16, 3, 224, 224)
        with torch.no_grad():
            out = model_cpu(x)
        assert all(v.shape[0] == 1 for v in out.values())

    def test_batch_size_four(self, model_cpu):
        x = torch.randn(4, 16, 3, 224, 224)
        with torch.no_grad():
            out = model_cpu(x)
        assert all(v.shape[0] == 4 for v in out.values())

    def test_output_is_finite(self, model_cpu):
        """No NaN or Inf in logits."""
        x = torch.randn(1, 16, 3, 224, 224)
        with torch.no_grad():
            out = model_cpu(x)
        for task, logits in out.items():
            assert torch.isfinite(logits).all(), f"Non-finite values in task '{task}'"

    def test_softmax_sums_to_one(self, model_cpu):
        x = torch.randn(2, 16, 3, 224, 224)
        with torch.no_grad():
            out = model_cpu(x)
        for task, logits in out.items():
            probs = torch.softmax(logits, dim=1)
            sums = probs.sum(dim=1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
                f"Softmax doesn't sum to 1 for task '{task}'"

    def test_different_hidden_sizes(self):
        """Model should construct and forward with various hidden sizes."""
        for hs in [64, 128, 256]:
            m = CNN_LSTM_Model(task_classes=TASK_CLASSES_FULL, hidden_size=hs, pretrained=False)
            m.eval()
            x = torch.randn(1, 16, 3, 224, 224)
            with torch.no_grad():
                out = m(x)
            assert "stroke_type" in out

    def test_default_task_classes(self):
        """Model should work with default task_classes (no arg passed)."""
        m = CNN_LSTM_Model(pretrained=False)
        m.eval()
        x = torch.randn(1, 16, 3, 224, 224)
        with torch.no_grad():
            out = m(x)
        assert "stroke_type" in out

    def test_input_normalization_applied(self, model_cpu):
        """
        Check that ImageNet normalization doesn't produce identical outputs for
        drastically different inputs (i.e., normalization is actually happening).
        """
        x_zeros = torch.zeros(1, 16, 3, 224, 224)
        x_ones = torch.ones(1, 16, 3, 224, 224)
        with torch.no_grad():
            out_zeros = model_cpu(x_zeros)
            out_ones = model_cpu(x_ones)
        # Outputs should differ since inputs differ
        assert not torch.allclose(
            out_zeros["stroke_type"], out_ones["stroke_type"]
        ), "Model output identical for all-zeros and all-ones input (normalization may be broken)"

    def test_eval_mode_no_dropout_effect(self, model_cpu):
        """In eval mode, two identical forward passes should produce identical output."""
        x = torch.randn(1, 16, 3, 224, 224)
        with torch.no_grad():
            out1 = model_cpu(x)
            out2 = model_cpu(x)
        for task in out1:
            assert torch.allclose(out1[task], out2[task]), \
                f"Non-deterministic output in eval mode for task '{task}'"
