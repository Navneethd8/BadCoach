"""
Backward-compatible module: `uvicorn api.server:app` redirects to the split app.
Tests and tooling may import MAX_* from here.
"""
from api.config import (
    MAX_CONCURRENT_JOBS,
    MAX_CONCURRENT_LIVE_SESSIONS,
    MAX_QUEUED_JOBS,
    MAX_VIDEO_DURATION_SECONDS,
)
from api.main import app

__all__ = [
    "app",
    "MAX_CONCURRENT_JOBS",
    "MAX_QUEUED_JOBS",
    "MAX_CONCURRENT_LIVE_SESSIONS",
    "MAX_VIDEO_DURATION_SECONDS",
]
