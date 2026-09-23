import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import auth
from app.config import get_settings
from app.routers import (
    diagnoses,
    local_storage,
    models_registry,
    outcomes,
    sessions,
    swings,
    training,
    videos,
)
from app.storage import S3Storage, get_storage

log = logging.getLogger("hitfar")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    try:
        from app.db import get_sessionmaker
        from app.diagnosis.catalog import upsert_catalog

        with get_sessionmaker()() as db:
            upsert_catalog(db)
    except Exception:
        log.exception("could not sync the fault catalog")
    if s.auto_start_jobs:
        try:
            from app.automation import queue_startup_jobs
            from app.db import get_sessionmaker

            with get_sessionmaker()() as db:
                queue_startup_jobs(db)
        except Exception:
            log.exception("could not queue start-up jobs")
    if s.app_password == "changeme":
        log.warning("APP_PASSWORD is the default; set it before deploying")
    if s.storage_backend == "s3" and s.public_origin:
        try:
            st = get_storage()
            assert isinstance(st, S3Storage)
            st.set_cors([s.public_origin])
            log.info("bucket CORS set for %s", s.public_origin)
        except Exception:
            log.exception("could not set bucket CORS; browser uploads may fail preflight")
    yield


app = FastAPI(title="hit-far", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


app.include_router(auth.router)
app.include_router(sessions.router)
app.include_router(videos.router)
app.include_router(swings.router)
app.include_router(models_registry.router)
app.include_router(diagnoses.router)
app.include_router(training.router)
app.include_router(outcomes.router)
app.include_router(local_storage.router)


# --- SPA ----------------------------------------------------------------------------------------
_dist = get_settings().web_dist_dir
if (_dist / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def spa(path: str):
    if path.startswith("api/"):
        raise HTTPException(404)
    f = (_dist / path).resolve()
    if path and f.is_file() and f.is_relative_to(_dist.resolve()):
        return FileResponse(f)
    index = _dist / "index.html"
    if not index.is_file():
        raise HTTPException(404, "web app not built (run `npm run build` in web/)")
    return FileResponse(index)
