"""Central env-backed settings for the API."""
import os

MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "4"))
# Max jobs waiting in FIFO (not counting currently running workers)
MAX_QUEUED_JOBS = int(os.getenv("MAX_QUEUED_JOBS", "16"))
MAX_CONCURRENT_LIVE_SESSIONS = int(os.getenv("MAX_CONCURRENT_LIVE_SESSIONS", "1"))
MAX_VIDEO_DURATION_SECONDS = int(os.getenv("MAX_VIDEO_DURATION_SECONDS", "1200"))
