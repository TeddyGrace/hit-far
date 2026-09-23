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
import tempfile
import zipfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.resources import pose_workers
from app.pipeline.posedata import PoseData
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


def _download_videos(workdir: Path) -> Path:
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
        out = gdown.download(url, str(zpath), quiet=True, fuzzy=True)
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
        vdir = _download_videos(Path(tmp))
        tasks = []
        for c in missing:
            p = vdir / f"{c.id}.mp4"
            if p.exists():
                tasks.append((c.id, str(p)))
            else:
                errors[c.id] = "clip missing from zip"
        workers = workers or get_settings().train_workers or pose_workers()
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_extract_one, t) for t in tasks]
            for fut in as_completed(futures):
                clip_id, data, err = fut.result()
                if data is not None:
                    st.put_bytes(pose_key(clip_id), data)
                else:
                    errors[clip_id] = err or "unknown error"
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    progress(f"golfdb: pose {done}/{len(tasks)}")
    if errors:
        log.warning("golfdb: %d clips failed pose extraction", len(errors))
    return errors


def load_pose(clip_id: int) -> PoseData | None:
    st = get_storage()
    key = pose_key(clip_id)
    return PoseData.from_npz(st.get_bytes(key)) if st.exists(key) else None
