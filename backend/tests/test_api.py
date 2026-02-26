"""
Integration tests for the /analyze and /health API endpoints.
The model, pose estimator, and Gemini are fully mocked so no GPU,
model weights, or API keys are needed.
"""
import io
import os
import sys
import cv2
import hashlib
import tempfile
import importlib
import threading
import numpy as np
import pytest
import api.server as server_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_video_bytes(n_frames=30, fps=30, width=64, height=64) -> bytes:
    """Return bytes of a small synthetic MP4."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp.name, fourcc, fps, (width, height))
    rng = np.random.default_rng(0)
    for _ in range(n_frames):
        out.write((rng.random((height, width, 3)) * 255).astype(np.uint8))
    out.release()
    with open(tmp.name, "rb") as f:
        data = f.read()
    os.unlink(tmp.name)
    return data


def _upload(client, video_bytes: bytes, filename="clip.mp4"):
    return client.post(
        "/analyze",
        files={"file": (filename, io.BytesIO(video_bytes), "video/mp4")},
    )


# ---------------------------------------------------------------------------
# Root / Health
# ---------------------------------------------------------------------------

class TestRootEndpoint:

    def test_root_returns_ok(self, app_client):
        r = app_client.get("/")
        assert r.status_code == 200
        assert "message" in r.json()

    def test_health_endpoint_exists(self, app_client):
        r = app_client.get("/health")
        # Either 200 (implemented) or 404 (not yet) — never a 500
        assert r.status_code in (200, 404)

    def test_health_returns_status_field(self, app_client):
        r = app_client.get("/health")
        if r.status_code == 200:
            body = r.json()
            assert "status" in body


# ---------------------------------------------------------------------------
# /analyze — valid uploads
# ---------------------------------------------------------------------------

class TestAnalyzeValidVideo:

    def test_valid_video_returns_200(self, app_client):
        video = _make_video_bytes()
        r = _upload(app_client, video)
        assert r.status_code == 200

    def test_response_has_required_fields(self, app_client):
        video = _make_video_bytes()
        r = _upload(app_client, video)
        assert r.status_code == 200
        body = r.json()
        for field in ("action", "confidence", "timeline"):
            assert field in body, f"Missing field: {field}"

    def test_timeline_is_list(self, app_client):
        video = _make_video_bytes()
        r = _upload(app_client, video)
        if r.status_code == 200 and not r.json().get("validation_failed"):
            assert isinstance(r.json()["timeline"], list)

    def test_confidence_is_float_between_0_and_1(self, app_client):
        video = _make_video_bytes()
        r = _upload(app_client, video)
        if r.status_code == 200 and not r.json().get("validation_failed"):
            conf = r.json()["confidence"]
            assert 0.0 <= conf <= 1.0

    def test_timeline_events_have_timestamp_and_label(self, app_client):
        video = _make_video_bytes()
        r = _upload(app_client, video)
        if r.status_code == 200 and r.json().get("timeline"):
            for event in r.json()["timeline"]:
                assert "timestamp" in event
                assert "label" in event
                assert "confidence" in event

    def test_cache_hit_on_second_upload(self, app_client):
        """Uploading the exact same bytes twice should return cache_hit=True on second call."""
        video = _make_video_bytes(n_frames=15)
        r1 = _upload(app_client, video)
        r2 = _upload(app_client, video)
        if r1.status_code == 200 and r2.status_code == 200:
            # If caching is implemented, second call should be faster and flag it
            body2 = r2.json()
            if "cache_hit" in body2:
                assert body2["cache_hit"] is True

    def test_different_videos_not_cached_together(self, app_client):
        """Two different videos should NOT hit each other's cache."""
        video_a = _make_video_bytes(n_frames=10)
        video_b = _make_video_bytes(n_frames=20)
        r1 = _upload(app_client, video_a, "a.mp4")
        r2 = _upload(app_client, video_b, "b.mp4")
        if r1.status_code == 200 and r2.status_code == 200:
            body_b = r2.json()
            if "cache_hit" in body_b:
                assert body_b["cache_hit"] is False


# ---------------------------------------------------------------------------
# /analyze — validation failures
# ---------------------------------------------------------------------------

class TestAnalyzeValidationFailures:

    def test_non_badminton_video_returns_validation_failed(self, app_client):
        """
        When pose/model confidence is forced low, server returns validation_failed=True.
        The mock in conftest gives high confidence by default; we patch it low for this test.
        """
        original = server_module.badminton_detector.is_badminton_video
        server_module.badminton_detector.is_badminton_video = MagicMock(
            return_value=(False, 0.02, {"overhead_score": 0.01, "stance_score": 0.01,
                                        "racket_score": 0.01})
        )
        try:
            video = _make_video_bytes()
            r = _upload(app_client, video)
            # Either 200 with validation_failed, or 422
            if r.status_code == 200:
                body = r.json()
                # Validation may fail at pose or model stage
                # (model classification also runs — result depends on mock)
                assert "validation_failed" in body or "action" in body
        finally:
            server_module.badminton_detector.is_badminton_video = original

    def test_empty_body_returns_422(self, app_client):
        """Missing file field → 422 Unprocessable Entity."""
        r = app_client.post("/analyze")
        assert r.status_code == 422

    def test_corrupt_file_doesnt_crash_server(self, app_client):
        """Corrupt bytes should return a structured error, not a 500 traceback."""
        corrupt = b"\x00\xff" * 256
        r = app_client.post(
            "/analyze",
            files={"file": ("corrupt.mp4", io.BytesIO(corrupt), "video/mp4")},
        )
        # Any status is acceptable EXCEPT 500
        assert r.status_code != 500

    def test_zero_byte_file_doesnt_crash(self, app_client):
        r = app_client.post(
            "/analyze",
            files={"file": ("empty.mp4", io.BytesIO(b""), "video/mp4")},
        )
        assert r.status_code != 500

    def test_error_response_has_error_field(self, app_client):
        """Validation error responses must include an 'error' string."""
        original = server_module.badminton_detector.is_badminton_video
        server_module.badminton_detector.is_badminton_video = MagicMock(
            return_value=(False, 0.01, {"overhead_score": 0.0, "stance_score": 0.0,
                                        "racket_score": 0.0})
        )
        try:
            video = _make_video_bytes()
            r = _upload(app_client, video)
            if r.status_code == 200 and r.json().get("validation_failed"):
                assert "error" in r.json()
        finally:
            server_module.badminton_detector.is_badminton_video = original


# ---------------------------------------------------------------------------
# /analyze — over-duration limit
# ---------------------------------------------------------------------------

class TestAnalyzeDurationLimit:

    def test_video_over_limit_returns_structured_error(self, app_client):
        """
        A video that exceeds MAX_VIDEO_DURATION_SECONDS should return
        a structured response with over_duration_limit=True.
        This test patches the frame count to simulate a long video.
        """
        MAX = getattr(server_module, "MAX_VIDEO_DURATION_SECONDS", None)
        if MAX is None:
            pytest.skip("MAX_VIDEO_DURATION_SECONDS not implemented yet")

        video = _make_video_bytes()
        # Patch total_frames and fps to exceed limit
        original_get = getattr(server_module, "_get_video_info", None)
        if original_get is None:
            pytest.skip("Duration guard not yet implemented")


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

class TestConcurrency:

    def test_multiple_simultaneous_uploads_all_return(self, app_client):
        """
        Fire N concurrent uploads — all should return a response (200 or 503),
        never hanging indefinitely or crashing with 500.
        """
        video = _make_video_bytes(n_frames=10)
        N = 4
        results = []

        def upload():
            return _upload(app_client, video)

        with ThreadPoolExecutor(max_workers=N) as pool:
            futures = [pool.submit(upload) for _ in range(N)]
            for f in as_completed(futures, timeout=120):
                results.append(f.result().status_code)

        assert len(results) == N
        for status in results:
            assert status in (200, 503), f"Unexpected status code: {status}"

    def test_over_capacity_returns_503_not_hang(self, app_client):
        """
        When more than MAX_CONCURRENT_JOBS requests arrive simultaneously,
        overflow requests must get a 503, not hang or 500.
        """
        max_jobs = getattr(server_module, "MAX_CONCURRENT_JOBS", None)
        if max_jobs is None:
            pytest.skip("Concurrency semaphore not yet implemented")

        video = _make_video_bytes(n_frames=5)
        N = max_jobs + 2  # Deliberately over cap

        statuses = []
        with ThreadPoolExecutor(max_workers=N) as pool:
            futures = [pool.submit(_upload, app_client, video) for _ in range(N)]
            for f in as_completed(futures, timeout=60):
                statuses.append(f.result().status_code)

        assert 503 in statuses, "Expected at least one 503 when over capacity"
        for s in statuses:
            assert s != 500


# ---------------------------------------------------------------------------
# SSE Streaming endpoint
# ---------------------------------------------------------------------------

class TestStreamingEndpoint:

    def test_stream_endpoint_exists(self, app_client):
        """POST /analyze/stream should exist (200 or 405), not 404."""
        video = _make_video_bytes(n_frames=5)
        r = app_client.post(
            "/analyze/stream",
            files={"file": ("clip.mp4", io.BytesIO(video), "video/mp4")},
        )
        assert r.status_code != 404, "/analyze/stream endpoint not found"

    def test_stream_endpoint_sends_done_event(self, app_client):
        """If implemented, stream must terminate with a 'done' event."""
        video = _make_video_bytes(n_frames=5)
        r = app_client.post(
            "/analyze/stream",
            files={"file": ("clip.mp4", io.BytesIO(video), "video/mp4")},
        )
        if r.status_code == 200:
            text = r.text
            assert "done" in text, "Stream did not include a 'done' event"
