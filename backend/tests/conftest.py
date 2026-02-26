"""
Shared pytest fixtures for the IsoCourt backend test suite.
All heavy dependencies (model weights, MediaPipe) are mocked so tests
run in CI without GPU or downloaded model files.
"""
import io
import os
import sys
import importlib
import struct
import hashlib
import tempfile
import cv2
import torch
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from core.model import CNN_LSTM_Model

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_video(path: str, n_frames: int = 30, fps: int = 30,
                           width: int = 128, height: int = 128) -> str:
    """Write a minimal valid MP4-like video using OpenCV VideoWriter."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    rng = np.random.default_rng(42)
    for _ in range(n_frames):
        frame = (rng.random((height, width, 3)) * 255).astype(np.uint8)
        out.write(frame)
    out.release()
    return path


def _make_corrupt_video(path: str) -> str:
    """Write nonsense bytes that are not a valid video."""
    with open(path, "wb") as f:
        f.write(b"\xff\xfe" * 512)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def synthetic_video_path(tmp_path_factory):
    """Short synthetic 1-second badminton-ish video (session-scoped, created once)."""
    p = tmp_path_factory.mktemp("videos") / "synthetic.mp4"
    _make_synthetic_video(str(p), n_frames=30, fps=30)
    return str(p)


@pytest.fixture(scope="session")
def long_video_path(tmp_path_factory):
    """Synthetic 25-minute equivalent stub — only 10 frames but FPS set so duration > 1200s."""
    p = tmp_path_factory.mktemp("videos") / "long.mp4"
    # 10 frames at 1/120 fps => duration = 1200s, triggering the guard
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(p), fourcc, 1, (64, 64))
    for _ in range(10):
        out.write(np.zeros((64, 64, 3), dtype=np.uint8))
    out.release()
    return str(p)


@pytest.fixture(scope="session")
def corrupt_video_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("videos") / "corrupt.bin"
    _make_corrupt_video(str(p))
    return str(p)


@pytest.fixture(scope="session")
def mock_task_classes():
    return {
        "stroke_type": 9,
        "stroke_subtype": 21,
        "technique": 4,
        "placement": 7,
        "position": 10,
        "intent": 10,
        "quality": 7,
        "is_badminton": 2,
    }


@pytest.fixture(scope="session")
def mock_model(mock_task_classes):
    """CNN-LSTM with random weights — no file I/O required."""
    m = CNN_LSTM_Model(task_classes=mock_task_classes, hidden_size=64, pretrained=False)
    m.eval()
    return m


@pytest.fixture
def dummy_landmarks_overhead():
    """Synthetic landmarks where wrists are clearly above shoulders (overhead pose)."""
    lms = []
    for i in range(33):
        lms.append({"id": i, "x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9})
    # Shoulders at y=0.5, wrists high at y=0.1 (remember: smaller y = higher in image)
    lms[11]["y"] = 0.5  # left shoulder
    lms[12]["y"] = 0.5  # right shoulder
    lms[15]["y"] = 0.1  # left wrist (raised high)
    lms[16]["y"] = 0.1  # right wrist (raised high)
    return lms


@pytest.fixture
def dummy_landmarks_neutral():
    """Landmarks where wrists are below shoulders (neutral/resting pose)."""
    lms = []
    for i in range(33):
        lms.append({"id": i, "x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9})
    lms[11]["y"] = 0.4  # left shoulder
    lms[12]["y"] = 0.4  # right shoulder
    lms[15]["y"] = 0.8  # left wrist (below shoulder)
    lms[16]["y"] = 0.8  # right wrist (below shoulder)
    return lms


@pytest.fixture
def dummy_landmarks_wide_stance():
    """Landmarks with a wide badminton stance (feet far apart)."""
    lms = []
    for i in range(33):
        lms.append({"id": i, "x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9})
    # Hips close together
    lms[23]["x"] = 0.45  # left hip
    lms[24]["x"] = 0.55  # right hip
    # Ankles far apart (>1.5x hip width)
    lms[27]["x"] = 0.1   # left ankle
    lms[28]["x"] = 0.9   # right ankle
    return lms


@pytest.fixture
def dummy_landmarks_asymmetric_arms():
    """Landmarks with highly asymmetric arm extension (racket-holding pose)."""
    lms = []
    for i in range(33):
        lms.append({"id": i, "x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9})
    # Left arm folded (elbow and wrist close)
    lms[13]["x"] = 0.48  # left elbow
    lms[13]["y"] = 0.5
    lms[15]["x"] = 0.49  # left wrist
    lms[15]["y"] = 0.51
    # Right arm fully extended
    lms[14]["x"] = 0.52  # right elbow
    lms[14]["y"] = 0.5
    lms[16]["x"] = 0.85  # right wrist (far out)
    lms[16]["y"] = 0.2
    return lms


@pytest.fixture
def app_client(monkeypatch):
    """
    FastAPI TestClient with model, pose estimator, and Gemini mocked out.
    This avoids needing GPU, model weights, or MediaPipe model files.
    """
    # We must patch BEFORE importing server to avoid startup loading the real model
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
        # Patch model loading
        with patch("api.server.CNN_LSTM_Model") as MockModel, \
             patch("api.server.PoseEstimator") as MockPose, \
             patch("api.server.BadmintonPoseDetector") as MockDetector, \
             patch("torch.load", return_value={}):

            # Setup mock model forward pass
            mock_model_instance = MagicMock()
            def _fake_forward(tensor):
                b = tensor.shape[0]
                return {
                    "stroke_type": torch.zeros(b, 9),
                    "stroke_subtype": torch.zeros(b, 21),
                    "technique": torch.zeros(b, 4),
                    "placement": torch.zeros(b, 7),
                    "position": torch.zeros(b, 10),
                    "intent": torch.zeros(b, 10),
                    "quality": torch.zeros(b, 7),
                    "is_badminton": torch.tensor([[0.2, 0.8]] * b),
                }
            mock_model_instance.side_effect = _fake_forward  # side_effect calls the fn; return_value would return the fn itself
            MockModel.return_value = mock_model_instance

            # Setup mock pose estimator
            mock_pose = MagicMock()
            mock_pose.process_frame.return_value = MagicMock(pose_landmarks=[])
            mock_pose.draw_landmarks.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_pose.get_landmarks_as_list.return_value = []
            MockPose.return_value = mock_pose

            # Setup mock badminton detector
            mock_detector = MagicMock()
            mock_detector.is_badminton_video.return_value = (
                True, 0.75,
                {"overhead_score": 0.7, "stance_score": 0.6, "racket_score": 0.5}
            )
            MockDetector.return_value = mock_detector

            from fastapi.testclient import TestClient
            # Import server fresh in this patched context
            import importlib
            import api.server as server_module
            importlib.reload(server_module)

            # Inject mocks into the reloaded module
            server_module.model = mock_model_instance
            server_module.pose_estimator = mock_pose
            server_module.badminton_detector = mock_detector
            server_module.gemini_enabled = False
            server_module.dataset_metadata = {
                "stroke_type": ["Serve", "Clear", "Smash", "Drop", "Drive",
                                "Net_Shot", "Lob", "Defensive_Shot", "Other"],
                "stroke_subtype": ["None"] * 21,
                "technique": ["Forehand", "Backhand", "Turnaround", "Unknown"],
                "placement": ["Straight", "Cross-court", "Body_Hit", "Over_Head",
                              "Passing_Shot", "Wide", "Unknown"],
                "position": ["Mid_Front", "Mid_Court", "Mid_Back", "Left_Front",
                             "Left_Mid", "Left_Back", "Right_Front", "Right_Mid",
                             "Right_Back", "Unknown"],
                "intent": ["Intercept", "Passive", "Defensive", "To_Create_Depth",
                           "Move_To_Net", "Early_Net_Shot", "Deception",
                           "Hesitation", "Seamlessly", "None"],
            }

            client = TestClient(server_module.app, raise_server_exceptions=False)
            yield client
