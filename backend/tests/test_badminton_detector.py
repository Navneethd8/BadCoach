"""
Unit tests for BadmintonPoseDetector (badminton_detector.py).
Pure Python — no MediaPipe or OpenCV needed.
"""
import pytest
import numpy as np
from core.badminton_detector import BadmintonPoseDetector


@pytest.fixture
def detector():
    return BadmintonPoseDetector(
        overhead_threshold=0.15,
        stance_width_threshold=0.25,
        arm_asymmetry_threshold=0.20,
    )


def _make_landmarks(n=33, x=0.5, y=0.5, z=0.0, vis=0.9):
    """Generate a list of n landmarks with uniform values."""
    return [{"id": i, "x": x, "y": y, "z": z, "visibility": vis} for i in range(n)]


class TestDetectOverheadMotion:

    def test_wrists_above_shoulders_returns_true(self, detector, dummy_landmarks_overhead):
        detected, conf = detector.detect_overhead_motion(dummy_landmarks_overhead)
        assert detected is True
        assert conf > 0.0

    def test_wrists_below_shoulders_returns_false(self, detector, dummy_landmarks_neutral):
        detected, conf = detector.detect_overhead_motion(dummy_landmarks_neutral)
        assert detected is False
        assert conf == 0.0

    def test_empty_landmarks_returns_false(self, detector):
        detected, conf = detector.detect_overhead_motion([])
        assert detected is False
        assert conf == 0.0

    def test_too_few_landmarks_returns_false(self, detector):
        lms = _make_landmarks(n=10)
        detected, conf = detector.detect_overhead_motion(lms)
        assert detected is False
        assert conf == 0.0

    def test_confidence_increases_with_height(self, detector):
        """Higher wrists → higher confidence score."""
        lms_low = _make_landmarks(n=33)
        lms_low[11]["y"] = 0.5
        lms_low[12]["y"] = 0.5
        lms_low[15]["y"] = 0.32    # just barely above shoulder
        lms_low[16]["y"] = 0.32

        lms_high = _make_landmarks(n=33)
        lms_high[11]["y"] = 0.5
        lms_high[12]["y"] = 0.5
        lms_high[15]["y"] = 0.05   # very high
        lms_high[16]["y"] = 0.05

        _, conf_low = detector.detect_overhead_motion(lms_low)
        _, conf_high = detector.detect_overhead_motion(lms_high)
        assert conf_high > conf_low


class TestDetectWideStance:

    def test_wide_stance_detected(self, detector, dummy_landmarks_wide_stance):
        detected, conf = detector.detect_wide_stance(dummy_landmarks_wide_stance)
        assert detected is True
        assert conf > 0.0

    def test_narrow_stance_returns_false(self, detector):
        lms = _make_landmarks(n=33)
        # Hips and ankles at same narrow width
        lms[23]["x"] = 0.45
        lms[24]["x"] = 0.55
        lms[27]["x"] = 0.46  # ankles barely wider than hips (ratio < 1.5)
        lms[28]["x"] = 0.54
        detected, conf = detector.detect_wide_stance(lms)
        assert detected is False

    def test_empty_landmarks_returns_false(self, detector):
        detected, conf = detector.detect_wide_stance([])
        assert detected is False
        assert conf == 0.0

    def test_too_few_landmarks_returns_false(self, detector):
        lms = _make_landmarks(n=20)  # need 29+
        detected, conf = detector.detect_wide_stance(lms)
        assert detected is False

    def test_zero_hip_width_handled(self, detector):
        """Hip width of zero should not raise ZeroDivisionError."""
        lms = _make_landmarks(n=33)
        lms[23]["x"] = 0.5   # same x as right hip → hip_width=0
        lms[24]["x"] = 0.5
        lms[27]["x"] = 0.2
        lms[28]["x"] = 0.8
        # Should not raise
        detected, conf = detector.detect_wide_stance(lms)
        assert isinstance(detected, bool)


class TestDetectRacketHoldingPose:

    def test_asymmetric_arms_detected(self, detector, dummy_landmarks_asymmetric_arms):
        detected, conf = detector.detect_racket_holding_pose(dummy_landmarks_asymmetric_arms)
        assert detected is True
        assert conf > 0.0

    def test_symmetric_arms_not_detected(self, detector):
        lms = _make_landmarks(n=33)
        # Both arms fully symmetric
        lms[13]["x"] = 0.4; lms[13]["y"] = 0.6  # left elbow
        lms[15]["x"] = 0.3; lms[15]["y"] = 0.7  # left wrist
        lms[14]["x"] = 0.6; lms[14]["y"] = 0.6  # right elbow
        lms[16]["x"] = 0.7; lms[16]["y"] = 0.7  # right wrist
        detected, conf = detector.detect_racket_holding_pose(lms)
        assert detected is False

    def test_empty_landmarks_returns_false(self, detector):
        detected, conf = detector.detect_racket_holding_pose([])
        assert detected is False
        assert conf == 0.0

    def test_too_few_landmarks_returns_false(self, detector):
        lms = _make_landmarks(n=10)
        detected, conf = detector.detect_racket_holding_pose(lms)
        assert detected is False


class TestCalculateBadmintonScore:

    def test_empty_returns_zero(self, detector):
        score = detector.calculate_badminton_score([])
        assert score == 0.0

    def test_all_empty_sublists_returns_zero(self, detector):
        score = detector.calculate_badminton_score([[], [], []])
        assert score == 0.0

    def test_strong_overhead_gives_high_score(self, detector, dummy_landmarks_overhead):
        score = detector.calculate_badminton_score([dummy_landmarks_overhead] * 10)
        assert score > 0.3

    def test_score_capped_at_one(self, detector, dummy_landmarks_overhead):
        score = detector.calculate_badminton_score([dummy_landmarks_overhead] * 100)
        assert score <= 1.0

    def test_score_non_negative(self, detector, dummy_landmarks_neutral):
        score = detector.calculate_badminton_score([dummy_landmarks_neutral] * 5)
        assert score >= 0.0


class TestIsBadmintonVideo:

    def test_badminton_video_passes_threshold(self, detector, dummy_landmarks_overhead):
        is_badminton, score, details = detector.is_badminton_video(
            [dummy_landmarks_overhead] * 15, threshold=0.05
        )
        assert is_badminton is True
        assert score > 0.0
        assert "overhead_score" in details
        assert "stance_score" in details
        assert "racket_score" in details
        assert "frames_analyzed" in details

    def test_empty_input_fails(self, detector):
        is_badminton, score, details = detector.is_badminton_video([])
        assert is_badminton is False
        assert score == 0.0

    def test_details_contain_frame_count(self, detector, dummy_landmarks_overhead):
        n = 8
        _, _, details = detector.is_badminton_video(
            [dummy_landmarks_overhead] * n, threshold=0.0
        )
        assert details["frames_analyzed"] == n

    def test_non_badminton_fails_high_threshold(self, detector, dummy_landmarks_neutral):
        is_badminton, score, _ = detector.is_badminton_video(
            [dummy_landmarks_neutral] * 15, threshold=0.99
        )
        assert is_badminton is False

    def test_mixed_frames(self, detector, dummy_landmarks_overhead, dummy_landmarks_neutral):
        """Mix of good and neutral frames — score should be intermediate."""
        frames = [dummy_landmarks_overhead] * 5 + [dummy_landmarks_neutral] * 5
        _, score, _ = detector.is_badminton_video(frames, threshold=0.0)
        assert 0.0 < score <= 1.0
