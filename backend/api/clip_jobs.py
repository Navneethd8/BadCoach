"""In-memory FIFO clip job queue with worker pool (one Space replica)."""
from __future__ import annotations

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from api import state
from api.clip_analysis import run_analyze_stream_async, sse_line
from api.config import MAX_CONCURRENT_JOBS, MAX_QUEUED_JOBS
from api.temp_video import safe_unlink


class ClipQueueFull(Exception):
    """Raised when MAX_QUEUED_JOBS waiting jobs is reached."""


@dataclass
class ClipJob:
    id: str
    temp_path: str
    video_hash: str
    filename: str
    status: str  # queued | running | done | failed | completed_cache
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    error: Optional[str] = None
    summary: Optional[dict] = None


class ClipJobManager:
    """
    FIFO work queue + up to MAX_CONCURRENT_JOBS concurrent workers.
    Rejects new jobs when len(waiting) >= MAX_QUEUED_JOBS.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, ClipJob] = {}
        self._waiting: deque[str] = deque()
        self._work_queue: asyncio.Queue[str] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._workers: list[asyncio.Task] = []
        self._started = False

    def start_workers(self) -> None:
        if self._started:
            return
        self._started = True
        for _ in range(MAX_CONCURRENT_JOBS):
            self._workers.append(asyncio.create_task(self._worker_loop()))

    async def _worker_loop(self) -> None:
        while True:
            job_id = await self._work_queue.get()
            try:
                job = self._jobs.get(job_id)
                if not job:
                    continue
                async with self._lock:
                    if job_id in self._waiting:
                        self._waiting.remove(job_id)
                job.status = "running"
                try:
                    async for ev in run_analyze_stream_async(job.temp_path, job.video_hash):
                        await job.event_queue.put(ev)
                        if ev.get("event") == "error":
                            job.status = "failed"
                            job.error = ev.get("error", "Unknown error")
                        elif ev.get("event") == "done":
                            job.status = "done"
                            job.summary = ev.get("summary")
                except Exception as e:
                    job.status = "failed"
                    job.error = str(e)
                    await job.event_queue.put({"event": "error", "error": str(e)})
            finally:
                self._work_queue.task_done()
                j = self._jobs.get(job_id)
                if j and j.temp_path:
                    safe_unlink(j.temp_path)
                    j.temp_path = ""

    def ahead_of(self, job_id: str) -> int:
        try:
            return list(self._waiting).index(job_id)
        except ValueError:
            return 0

    async def create_job_from_upload(
        self,
        temp_path: str,
        video_hash: str,
        filename: str,
    ) -> ClipJob:
        """Enqueue a new clip job or raise QueueFull."""
        job_id = str(uuid.uuid4())
        job = ClipJob(
            id=job_id,
            temp_path=temp_path,
            video_hash=video_hash,
            filename=filename,
            status="queued",
        )
        async with self._lock:
            if len(self._waiting) >= MAX_QUEUED_JOBS:
                raise ClipQueueFull()
            self._jobs[job_id] = job
            self._waiting.append(job_id)
        await self._work_queue.put(job_id)
        return job

    def create_cached_job(self, video_hash: str, summary: dict[str, Any]) -> ClipJob:
        """Immediate completed job (cache hit) — no temp file."""
        job_id = str(uuid.uuid4())
        job = ClipJob(
            id=job_id,
            temp_path="",
            video_hash=video_hash,
            filename="cached",
            status="completed_cache",
            summary=summary,
        )
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[ClipJob]:
        return self._jobs.get(job_id)

    def clip_queue_depth(self) -> int:
        return len(self._waiting)

    def workers_busy(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == "running")


clip_manager = ClipJobManager()
