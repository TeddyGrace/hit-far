"""Background worker: `python -m app.worker`. Runs as its own Railway service (same image as the API)."""

import logging
import signal
import time
import traceback
import uuid

from app import jobs
from app.config import get_settings
from app.db import get_sessionmaker
from app.pipeline.run import PermanentError, process_video

log = logging.getLogger("hitfar.worker")
_stop = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global _stop
    _stop = True
    log.info("shutdown requested; finishing current job")


def run_job(job_id: uuid.UUID, type_: str, payload: dict) -> None:
    Session = get_sessionmaker()
    with Session() as db, Session() as status_db:
        def on_stage(stage: str) -> None:
            jobs.set_stage(status_db, job_id, stage)

        if type_ == jobs.JOB_PROCESS_VIDEO:
            process_video(db, uuid.UUID(payload["video_id"]), force=payload.get("force", False), on_stage=on_stage)
        else:
            raise PermanentError(f"unknown job type {type_!r}")


def run_once() -> bool:
    """Claim and run a single job. Returns False if the queue was empty."""
    s = get_settings()
    Session = get_sessionmaker()
    with Session() as db:
        job = jobs.claim(db, s.job_lock_timeout_s)
        if job is None:
            return False
        job_id, type_, payload = job.id, job.type, dict(job.payload)
    log.info("running job %s %s %s", job_id, type_, payload)
    try:
        run_job(job_id, type_, payload)
    except PermanentError as e:
        log.warning("job %s failed permanently: %s", job_id, e)
        with Session() as db:
            jobs.fail(db, job_id, str(e), retryable=False)
    except Exception:
        tb = traceback.format_exc()
        log.error("job %s failed: %s", job_id, tb)
        with Session() as db:
            jobs.fail(db, job_id, tb, retryable=True)
    else:
        with Session() as db:
            jobs.finish(db, job_id)
        log.info("job %s done", job_id)
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    from app.pipeline.pose import ensure_pose_model

    ensure_pose_model()
    log.info("worker started")
    interval = get_settings().worker_poll_interval_s
    while not _stop:
        try:
            if not run_once():
                time.sleep(interval)
        except Exception:
            log.exception("worker loop error")
            time.sleep(interval)


if __name__ == "__main__":
    main()
