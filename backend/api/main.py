"""FastAPI application entry: one process, clip + live + legacy routes."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.clip_jobs import clip_manager
from api.lifespan import app_lifespan
from api.routes_clips import router as clips_router
from api.routes_live import router as live_router
from api.routes_legacy import router as legacy_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with app_lifespan(app):
        clip_manager.start_workers()
        yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(legacy_router)
app.include_router(clips_router, prefix="/clips")
app.include_router(live_router, prefix="/live")
