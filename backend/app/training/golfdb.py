"""GolfDB acquisition + pose extraction, cached in the bucket so reruns are cheap.

GolfDB (McNally et al., CVPR-W 2019): 1400 swing clips with 8 labelled events, CC BY-NC 4.0 -
fine for this personal, non-commercial project. Labels come from the GolfDB GitHub repo; the
160x160 preprocessed clips from the authors' Google Drive link (override with GOLFDB_VIDEOS_URL,
or drop the zip at `datasets/golfdb/videos_160.zip` in the bucket).

Clip frame i corresponds to original frame events[0] + i, so event frames within a clip are
events[1:9] - events[0].
"""

import io
import json
import logging
import os
import re
import signal
import tempfile
import zipfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.pipeline.posedata import PoseData
from app.resources import pose_workers
from app.storage import get_storage

log = logging.getLogger(__name__)

PREFIX = "datasets/golfdb"
LABELS_KEY = f"{PREFIX}/labels.json"
ZIP_KEY = f"{PREFIX}/videos_160.zip"


def pose_key(clip_id: int) -> str:
    return f"{PREFIX}/pose/{clip_id}.npz"


@dataclass
class Clip:
    id: int
    view: str  # face-on | down-the-line | other
    slow: bool
    split: int  # GolfDB's 4-fold split (1-4)
    events: list[int]  # 8 event frames, relative to the clip start


def load_labels() -> list[Clip]:
    st = get_storage()
    if st.exists(LABELS_KEY):
        rows = json.loads(st.get_bytes(LABELS_KEY))
    else:
        import urllib.request

        import pandas as pd

        with urllib.request.urlopen(get_settings().golfdb_labels_url, timeout=60) as r:
            df = pd.read_pickle(io.BytesIO(r.read()))
        rows = []
        for rec in df.to_dict("records"):
            ev = [int(x) for x in rec["events"]]
            rows.append({"id": int(rec["id"]), "view": str(rec["view"]), "slow": bool(rec["slow"]),
                         "split": int(rec["split"]), "events": [e - ev[0] for e in ev[1:9]]})
        st.put_bytes(LABELS_KEY, json.dumps(rows).encode(), "application/json")
    return [Clip(**r) for r in rows]


def drive_file_id(url: str) -> str | None:
    """File id from a Google Drive share link (/file/d/<id>/... or ?id=<id>)."""
    m = re.search(r"/file/d/([\w-]+)", url) or re.search(r"[?&]id=([\w-]+)", url)
    return m.group(1) if m else None


def _download_videos(workdir: Path, progress: Callable[[str], None] | None = None) -> Path:
    """Returns the directory containing <id>.mp4 clips."""
    st = get_storage()
    zpath = workdir / "videos_160.zip"
    if st.exists(ZIP_KEY):
        log.info("golfdb: using cached zip from bucket")
        st.download_file(ZIP_KEY, zpath)
    else:
        import gdown

        url = get_settings().golfdb_videos_url
        log.info("golfdb: downloading clips from %s", url)
        last = [-1]

        def report(done: int, total: int | None) -> None:
            pct = int(done * 100 / total) if total else done // (1 << 20)
            if progress and pct != last[0]:
                last[0] = pct
                progress(f"download {pct}%" if total else f"download {pct} MB")

        # Pass the file id explicitly: gdown 6 parses share links differently from 5.x.
        fid = drive_file_id(url)
        kwargs = {"id": fid} if fid else {"url": url}
        out = gdown.download(output=str(zpath), quiet=True, progress=report, retries=3, **kwargs)
        if not out or not zpath.exists() or zpath.stat().st_size < 1_000_000:
            raise RuntimeError(
                "Could not download the GolfDB clips (Google Drive may be rate-limiting this file). "
                "Download videos_160.zip yourself and upload it to the bucket at "
                f"'{ZIP_KEY}', or set GOLFDB_VIDEOS_URL to a direct link, then retry."
            )
        st.upload_file(ZIP_KEY, zpath, "application/zip")  # cache for future runs
    with zipfile.ZipFile(zpath) as z:
        z.extractall(workdir)
    zpath.unlink()
    hits = list(workdir.rglob("*.mp4"))
    if not hits:
        raise RuntimeError("GolfDB zip contained no .mp4 clips")
    return hits[0].parent


def _init_worker() -> None:
    """Pool workers are forked from the job runner and inherit its "finish the current job" SIGTERM
    handler, which makes them ignore SIGTERM. Restore the default, so that when the pool breaks (a
    worker is OOM-killed and the pool terminates the rest) or the service stops, they exit instead
    of leaving the job hung."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def run_pool(tasks: list[tuple[int, str]], workers: int,
             on_result: Callable[[tuple[int, bytes | None, str | None]], None],
             fn: Callable[[tuple[int, str]], tuple[int, bytes | None, str | None]] | None = None) -> None:
    """Run `fn` over the tasks in `workers` processes. If a worker dies (almost always out of
    memory), finish the remaining tasks with half as many workers; give up only if one worker
    can't manage."""
    fn = fn or _extract_one
    pending = dict(tasks)
    while pending:
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
                futures = [pool.submit(fn, t) for t in list(pending.items())]
                for fut in as_completed(futures):
                    result = fut.result()
                    pending.pop(result[0], None)
                    on_result(result)
        except BrokenProcessPool:
            if workers == 1:
                raise RuntimeError("pose extraction worker died even with a single worker "
                                   "(out of memory?); give the trainer more memory") from None
            workers = max(1, workers // 2)
            log.warning("pose worker died (likely out of memory); retrying %d clips with %d workers",
                        len(pending), workers)


def _extract_one(args: tuple[int, str]) -> tuple[int, bytes | None, str | None]:
    """Worker-process entry: run pose on one clip. Returns npz bytes (or an error)."""
    clip_id, path = args
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        from app.pipeline.pose import run_pose

        pose = run_pose(Path(path))
        return clip_id, pose.to_npz(), None
    except Exception as e:  # keep going; a few bad clips shouldn't sink the run
        return clip_id, None, f"{type(e).__name__}: {e}"


def ensure_pose(clips: list[Clip], progress: Callable[[str], None] = lambda s: None,
                workers: int | None = None) -> dict[int, str]:
    """Make sure every clip has a cached pose npz. Returns {clip_id: error} for failures."""
    st = get_storage()
    missing = [c for c in clips if not st.exists(pose_key(c.id))]
    if not missing:
        return {}
    progress(f"golfdb: downloading clips ({len(missing)} need pose)")
    errors: dict[int, str] = {}
    with tempfile.TemporaryDirectory(prefix="golfdb-") as tmp:
        vdir = _download_videos(Path(tmp), progress)
        tasks = []
        for c in missing:
            p = vdir / f"{c.id}.mp4"
            if p.exists():
                tasks.append((c.id, str(p)))
            else:
                errors[c.id] = "clip missing from zip"
        workers = workers or get_settings().train_workers or pose_workers()
        done = 0

        def on_result(result: tuple[int, bytes | None, str | None]) -> None:
            nonlocal done
            clip_id, data, err = result
            if data is not None:
                st.put_bytes(pose_key(clip_id), data)
            else:
                errors[clip_id] = err or "unknown error"
            done += 1
            if done % 10 == 0 or done == len(tasks):
                progress(f"golfdb: pose {done}/{len(tasks)}")

        run_pool(tasks, workers, on_result)
    if errors:
        log.warning("golfdb: %d clips failed pose extraction", len(errors))
    return errors


def load_pose(clip_id: int) -> PoseData | None:
    st = get_storage()
    key = pose_key(clip_id)
    return PoseData.from_npz(st.get_bytes(key)) if st.exists(key) else None
