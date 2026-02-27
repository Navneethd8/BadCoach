"""
Unit tests for PoseEstimator (pose_utils.py).
Uses real OpenCV frames but mocks the MediaPipe detector to avoid
needing the pose_landmarker_lite.task model file in CI.
"""
import os
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from core.pose_utils import PoseEstimator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def mock_detector_no_landmarks():
    d = MagicMock()
    d.detect.return_value = MagicMock(pose_landmarks=[])
    return d


@pytest.fixture
def mock_landmarks_result():
    """Simulate MediaPipe result with one person detected."""
    lm = MagicMock()
    lm.x = 0.5
    lm.y = 0.5
    lm.z = 0.0
    lm.visibility = 0.9
    result = MagicMock()
    result.pose_landmarks = [[lm] * 33]
    return result


@pytest.fixture
def black_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def random_frame():
    rng = np.random.default_rng(0)
    return (rng.random((480, 640, 3)) * 255).astype(np.uint8)


class TestPoseEstimatorProcessFrame:

    def test_process_frame_returns_result_object(self, black_frame, mock_detector_no_landmarks):
        with patch("mediapipe.tasks.python.vision.PoseLandmarker.create_from_options",
                   return_value=mock_detector_no_landmarks):
            estimator = PoseEstimator.__new__(PoseEstimator)
            estimator.detector = mock_detector_no_landmarks
            estimator.POSE_CONNECTIONS = []
            result = estimator.process_frame(black_frame)
            assert result is not None
            mock_detector_no_landmarks.detect.assert_called_once()

    def test_process_frame_handles_random_frame(self, random_frame, mock_detector_no_landmarks):
        estimator = PoseEstimator.__new__(PoseEstimator)
        estimator.detector = mock_detector_no_landmarks
        estimator.POSE_CONNECTIONS = []
        result = estimator.process_frame(random_frame)
        assert result is not None


class TestPoseEstimatorDrawLandmarks:

    def test_draw_with_no_landmarks_returns_copy(self, black_frame, mock_detector_no_landmarks):
        estimator = PoseEstimator.__new__(PoseEstimator)
        estimator.detector = mock_detector_no_landmarks
        estimator.POSE_CONNECTIONS = []

        empty_result = MagicMock()
        empty_result.pose_landmarks = []

        out = estimator.draw_landmarks(black_frame, empty_result)
        assert out.shape == black_frame.shape
        assert np.array_equal(out, black_frame)

    def test_draw_with_landmarks_doesnt_crash(self, random_frame, mock_landmarks_result):
        estimator = PoseEstimator.__new__(PoseEstimator)
        estimator.detector = MagicMock()
        estimator.POSE_CONNECTIONS = [(11, 12), (11, 13)]

        out = estimator.draw_landmarks(random_frame, mock_landmarks_result)
        assert out.shape == random_frame.shape

    def test_draw_does_not_mutate_original(self, black_frame, mock_landmarks_result):
        estimator = PoseEstimator.__new__(PoseEstimator)
        estimator.detector = MagicMock()
        estimator.POSE_CONNECTIONS = []

        original = black_frame.copy()
        estimator.draw_landmarks(black_frame, mock_landmarks_result)
        assert np.array_equal(black_frame, original)


class TestPoseEstimatorGetLandmarksAsList:

    def test_empty_result_returns_empty_list(self):
        estimator = PoseEstimator.__new__(PoseEstimator)
        empty_result = MagicMock()
        empty_result.pose_landmarks = []
        result = estimator.get_landmarks_as_list(empty_result)
        assert result == []

    def test_none_pose_landmarks_returns_empty_list(self):
        estimator = PoseEstimator.__new__(PoseEstimator)
        empty_result = MagicMock()
        empty_result.pose_landmarks = None
        result = estimator.get_landmarks_as_list(empty_result)
        assert result == []

    def test_landmarks_returned_as_dicts(self, mock_landmarks_result):
        estimator = PoseEstimator.__new__(PoseEstimator)
        result = estimator.get_landmarks_as_list(mock_landmarks_result)
        assert len(result) == 1  # one person
        person = result[0]
        assert len(person) == 33
        first = person[0]
        assert "x" in first
        assert "y" in first
        assert "z" in first
        assert "visibility" in first
        assert "id" in first

    def test_landmark_values_are_floats(self, mock_landmarks_result):
        estimator = PoseEstimator.__new__(PoseEstimator)
        result = estimator.get_landmarks_as_list(mock_landmarks_result)
        for lm in result[0]:
            assert isinstance(lm["x"], float)
            assert isinstance(lm["y"], float)


class TestPoseEstimatorInitialization:

    @patch("core.pose_utils.vision.PoseLandmarker.create_from_options")
    def test_default_model_path_resolution_not_local_to_core(self, mock_create):
        """
        Verify that the default model path correctly points to the `models` folder
        alongside `core` and not a local `core/models` folder.
        """
        estimator = PoseEstimator()
        
        mock_create.assert_called_once()
        args, _ = mock_create.call_args
        options = args[0]
        asset_path = options.base_options.model_asset_path
        
        # It must resolve to the models directory, independent of where the script runs from
        expected_suffix = os.path.join("models", "pose_landmarker_lite.task")
        assert asset_path.endswith(expected_suffix)
        
        # Crucially, it should NOT try to look inside 'core/models'
        assert os.path.join("core", "models") not in asset_path
