"""Queued clip analysis API: POST job, SSE stream, GET result."""
from __future__ import annotations

import asyncio
import hashlib

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from api import state
from api.clip_analysis import sse_line
from api.clip_jobs import ClipQueueFull, clip_manager
from api.temp_video import safe_unlink, write_video_bytes_to_tempfile

router = APIRouter(tags=["clips"])


@router.post("/jobs")
async def create_clip_job(file: UploadFile = File(...)):
    """
    Upload a clip. Returns job_id and queue position.
    Cache hit: status completed immediately (200).
    """
    content = await file.read()
    video_hash = hashlib.sha256(content).hexdigest()

    if video_hash in state._result_cache:
        cached = dict(state._result_cache[video_hash])
        job = clip_manager.create_cached_job(video_hash, cached)
        return JSONResponse(
            status_code=200,
            content={
                "job_id": job.id,
                "status": "completed",
                "queue_ahead": 0,
                "cache_hit": True,
            },
        )

    try:
        temp_file = write_video_bytes_to_tempfile(content)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to store upload: {e}") from e

    try:
        job = await clip_manager.create_job_from_upload(temp_file, video_hash, file.filename or "clip.mp4")
    except ClipQueueFull:
        safe_unlink(temp_file)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Too many videos are waiting right now. Please try again later.",
                "retry_after": 60,
            },
        ) from None

    ahead = clip_manager.ahead_of(job.id)
    return JSONResponse(
        status_code=202,
        content={
            "job_id": job.id,
            "status": "queued",
            "queue_ahead": ahead,
            "cache_hit": False,
        },
    )


@router.get("/jobs/{job_id}/status")
async def clip_job_status(job_id: str):
    job = clip_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    ahead = clip_manager.ahead_of(job_id) if job.status == "queued" else 0
    return {
        "job_id": job.id,
        "status": job.status,
        "queue_ahead": ahead,
        "error": job.error,
    }


@router.get("/jobs/{job_id}/result")
async def clip_job_result(job_id: str):
    job = clip_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in ("completed_cache", "done") and job.summary:
        body = dict(job.summary)
        body["cache_hit"] = job.status == "completed_cache"
        return body

    if job.status == "failed":
        raise HTTPException(
            status_code=422,
            detail={"error": job.error or "Analysis failed", "validation_failed": True},
        )

    return {
        "job_id": job.id,
        "status": job.status,
        "queue_ahead": clip_manager.ahead_of(job_id) if job.status == "queued" else 0,
        "message": "Result not ready yet. Use GET .../stream or poll status.",
    }


async def _sse_clip_stream(job_id: str):
    job = clip_manager.get_job(job_id)
    if not job:
        yield sse_line({"event": "error", "error": "Job not found"})
        return

    if job.status == "completed_cache" and job.summary:
        summary = job.summary
        timeline = summary.get("timeline") or []
        for i, seg in enumerate(timeline):
            ev = {"event": "progress", "window": i, **seg}
            yield sse_line(ev)
            await asyncio.sleep(0)
        summary_out = {k: v for k, v in summary.items() if k != "timeline"}
        summary_out["cache_hit"] = True
        yield sse_line({"event": "done", "summary": summary_out})
        return

    while job.status == "queued":
        ahead = clip_manager.ahead_of(job_id)
        yield sse_line({"event": "queue", "ahead": ahead})
        await asyncio.sleep(1.5)

    if job.status == "failed":
        yield sse_line({"event": "error", "error": job.error or "Unknown error"})
        return

    while True:
        try:
            ev = await asyncio.wait_for(job.event_queue.get(), timeout=600.0)
        except asyncio.TimeoutError:
            yield sse_line({"event": "error", "error": "Stream timed out"})
            return
        yield sse_line(ev)
        if ev.get("event") in ("done", "error"):
            break


@router.get("/jobs/{job_id}/stream")
async def clip_job_stream(job_id: str):
    job = clip_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return StreamingResponse(_sse_clip_stream(job_id), media_type="text/event-stream")
