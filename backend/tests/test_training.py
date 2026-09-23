import io
import itertools
import zipfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app import jobs, worker  # noqa: E402
from app.training import golfdb  # noqa: E402
from app.training.features import mirror, pose_features, resample  # noqa: E402
from app.training.model import decode_ordered, golfdb_tolerance, pce  # noqa: E402
from tests.synthetic import random_swing  # noqa: E402
from tests.test_api import _upload, fake_pose, sample_video  # noqa: E402, F401  (fixtures)

# --- Pure pieces -------------------------------------------------------------------------------


def test_decode_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        T, E = 7, 3
        lp = np.log(rng.dirichlet(np.ones(E), size=T))
        best = max(itertools.combinations(range(T), E), key=lambda c: sum(lp[t, e] for e, t in enumerate(c)))
        assert tuple(decode_ordered(lp)) == best


def test_mirror_is_involution_and_resample_shapes():
    pose, _ = random_swing(np.random.default_rng(1))
    xy, vis = pose_features(pose)
    xy2, vis2 = mirror(*mirror(xy, vis))
    assert np.allclose(xy, xy2) and np.allclose(vis, vis2)
    assert resample(xy, 2.0).shape[0] == 2 * len(xy)
    assert resample(xy, 0.25).shape[1:] == xy.shape[1:]


def test_pce_uses_golfdb_tolerance():
    true = [10, 20, 30, 40, 50, 130, 140, 150]  # impact - address = 120 -> tol 4
    assert golfdb_tolerance(true) == 4
    assert pce([14, 20, 30, 40, 50, 130, 140, 155], true).tolist() == [True] * 7 + [False]


# --- Fake GolfDB (synthetic swings in place of the real clips) ---------------------------------


@pytest.fixture
def fake_golfdb(monkeypatch):
    rng = np.random.default_rng(3)
    clips, poses = [], {}
    for i in range(48):
        pose, ev = random_swing(rng, fps=30)
        split = 1 if i % 4 == 0 else 2
        clips.append(golfdb.Clip(id=i, view="face-on", slow=False, split=split, events=ev))
        poses[i] = pose
    monkeypatch.setattr(golfdb, "load_labels", lambda: clips)
    monkeypatch.setattr(golfdb, "ensure_pose", lambda clips, progress=None, workers=None: {})
    monkeypatch.setattr(golfdb, "load_pose", lambda cid: poses[cid])
    return clips


SMALL = {"epochs": 12, "hidden": 32, "patience": 12}


def _train(authed, **cfg):
    r = authed.post("/api/training/events", json={"epochs": cfg.pop("epochs", 12)})
    assert r.status_code == 201, r.text
    from app.db import get_sessionmaker
    from app.models import Job

    with get_sessionmaker()() as db:  # smaller net for test speed
        job = db.get(Job, r.json()["id"])
        job.payload = {"config": {**job.payload["config"], **SMALL, **cfg}}
        db.commit()
    return r.json()


def test_train_register_promote_redetect(authed, sample_video, fake_pose, fake_golfdb):  # noqa: F811
    sess, video = _upload(authed, sample_video)
    assert worker.run_once("worker")
    swing_id = authed.get(f"/api/sessions/{sess['id']}").json()["videos"][0]["swing_id"]

    job = _train(authed)
    assert authed.post("/api/training/events", json={}).status_code == 409  # one run at a time
    assert worker.run_once("worker") is False  # the processing worker never takes training jobs
    assert worker.run_once("trainer") is True

    runs = authed.get("/api/training/jobs").json()
    assert runs[0]["id"] == job["id"] and runs[0]["status"] == "done", runs[0]
    models = {m["name"]: m for m in authed.get("/api/models").json()}
    learned = models["event-bilstm"]
    em = learned["eval_metrics"]
    assert em["golfdb_test"]["n"] == 12 and 0 <= em["golfdb_test"]["pce"] <= 1
    assert "rules_golfdb_test_face_on" in em and em["n_train_golfdb"] > 20
    # Auto-promoted only when it beats the rules on held-out face-on swings.
    beats = em["golfdb_test_face_on"]["pce"] > em["rules_golfdb_test_face_on"]["pce"]
    assert learned["status"] == ("active" if beats else "experimental")
    assert em.get("auto_promoted", False) is beats
    while worker.run_once("worker"):  # drain re-detections queued by an auto-promotion
        pass

    # Promote -> becomes the active event model; the rule baseline is kept (deprecated).
    after = {m["name"]: m for m in authed.post(f"/api/models/{learned['id']}/promote").json()}
    assert after["event-bilstm"]["status"] == "active" and after["rule-events"]["status"] == "deprecated"

    # Re-run events on existing swings with the promoted model.
    assert authed.post("/api/swings/redetect-events").json() == {"queued": 1}
    assert worker.run_once("worker")
    swing = authed.get(f"/api/swings/{swing_id}").json()
    assert {e["model"]["name"] for e in swing["events"]} == {"event-bilstm"}
    frames = [e["frame_index"] for e in swing["events"]]
    assert frames == sorted(frames) and len(frames) == 8
    assert all(0 <= e["confidence"] <= 1 for e in swing["events"])

    # Roll back: promoting the rules again makes new detections use rules.
    authed.post(f"/api/models/{models['rule-events']['id']}/promote")
    authed.post("/api/swings/redetect-events")
    worker.run_once("worker")
    swing = authed.get(f"/api/swings/{swing_id}").json()
    assert {e["model"]["name"] for e in swing["events"]} == {"rule-events"}


def test_broken_checkpoint_falls_back_to_rules(authed, sample_video, fake_pose, fake_golfdb):  # noqa: F811
    _train(authed, epochs=1)
    worker.run_once("trainer")
    learned = next(m for m in authed.get("/api/models").json() if m["name"] == "event-bilstm")
    authed.post(f"/api/models/{learned['id']}/promote")
    from app.storage import get_storage

    get_storage().put_bytes(learned["checkpoint_uri"], b"corrupt")
    from app.training.inference import load_checkpoint

    load_checkpoint.cache_clear()
    sess, _ = _upload(authed, sample_video)
    assert worker.run_once("worker")
    v = authed.get(f"/api/sessions/{sess['id']}").json()["videos"][0]
    assert v["job"]["status"] == "done"
    swing = authed.get(f"/api/swings/{v['swing_id']}").json()
    assert {e["model"]["name"] for e in swing["events"]} == {"rule-events"}


def test_reviewed_swings_become_training_data(authed, sample_video, fake_pose):  # noqa: F811
    sess, _ = _upload(authed, sample_video)
    worker.run_once("worker")
    swing_id = authed.get(f"/api/sessions/{sess['id']}").json()["videos"][0]["swing_id"]
    assert authed.get("/api/training/status").json() == {"reviewed_swings": 0}
    r = authed.put(f"/api/swings/{swing_id}/review", json={"reviewed": True})
    assert r.json()["events_reviewed"] is True
    assert authed.get("/api/training/status").json() == {"reviewed_swings": 1}

    from app.db import get_sessionmaker
    from app.training.train_events import user_samples

    with get_sessionmaker()() as db:
        samples = user_samples(db, "right")
    assert len(samples) == 1 and samples[0].source == "self" and len(samples[0].events) == 8

    authed.put(f"/api/swings/{swing_id}/review", json={"reviewed": False})
    assert authed.get("/api/training/status").json() == {"reviewed_swings": 0}


def test_auto_promote_rules(engine):
    from app.db import get_sessionmaker
    from app.models import Model, ModelStatus
    from app.pipeline.registry import TASK_EVENTS
    from app.training.train_events import auto_promote

    def em(mine, rules, self_m=None, self_r=None):
        d = {"golfdb_test_face_on": {"n": 50, "pce": mine}, "rules_golfdb_test_face_on": {"n": 50, "pce": rules}}
        if self_m is not None:
            d |= {"self_holdout": {"n": 3, "pce": self_m}, "rules_self_holdout": {"n": 3, "pce": self_r}}
        return d

    with get_sessionmaker()() as db:
        rules = Model(name="rule-events", version="t", task=TASK_EVENTS, status=ModelStatus.active)
        worse = Model(name="m", version="1", task=TASK_EVENTS, checkpoint_uri="x", eval_metrics=em(0.7, 0.8))
        worse_on_mine = Model(name="m", version="2", task=TASK_EVENTS, checkpoint_uri="x",
                              eval_metrics=em(0.9, 0.8, 0.5, 0.7))
        good = Model(name="m", version="3", task=TASK_EVENTS, checkpoint_uri="x", eval_metrics=em(0.9, 0.8))
        not_better_than_active = Model(name="m", version="4", task=TASK_EVENTS, checkpoint_uri="x",
                                       eval_metrics=em(0.85, 0.8))
        db.add_all([rules, worse, worse_on_mine, good, not_better_than_active])
        db.flush()
        assert not auto_promote(db, worse) and not auto_promote(db, worse_on_mine)
        assert auto_promote(db, good) and good.status == ModelStatus.active and rules.status == ModelStatus.deprecated
        assert not auto_promote(db, not_better_than_active) and good.status == ModelStatus.active
        for m in (rules, worse, worse_on_mine, good, not_better_than_active):
            db.delete(m)
        db.commit()


def test_heartbeat_refreshes_lock(engine):
    from app.db import get_sessionmaker
    from app.models import Job

    with get_sessionmaker()() as db:
        job = jobs.enqueue(db, jobs.JOB_TRAIN_EVENTS, {})
        db.commit()
        claimed = jobs.claim(db, 1800, jobs.ROLE_JOB_TYPES["trainer"])
        first = claimed.locked_at
        jobs.set_stage(db, claimed.id, "train epoch 1/40")
        db.expire_all()
        assert db.get(Job, job.id).locked_at > first
        db.delete(db.get(Job, job.id))
        db.commit()


# --- Real MediaPipe on a GolfDB-style zip (skipped if the pose model isn't available) ----------


def test_ensure_pose_extracts_and_caches(engine, monkeypatch):
    from app.config import get_settings

    model_path = get_settings().pose_model_path
    person = Path("/tmp/claude-0/person.mp4")
    if not model_path.exists() or not person.exists():
        pytest.skip("needs the pose model and a real person clip")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.write(person, "videos_160/7.mp4")
    from app.storage import get_storage

    st = get_storage()
    st.put_bytes(golfdb.ZIP_KEY, buf.getvalue())
    st.delete(golfdb.pose_key(7))
    clips = [golfdb.Clip(7, "face-on", False, 2, list(range(8))), golfdb.Clip(8, "face-on", False, 2, list(range(8)))]
    errors = golfdb.ensure_pose(clips, workers=2)
    assert errors == {8: "clip missing from zip"}
    pose = golfdb.load_pose(7)
    assert pose is not None and pose.detected.sum() > 50
    assert golfdb.ensure_pose(clips[:1]) == {}  # cached: nothing to do


def test_golfdb_download_matches_installed_gdown(engine, monkeypatch, tmp_path):
    """Calls gdown with arguments validated against the installed gdown's real signature."""
    import inspect

    gdown = pytest.importorskip("gdown")
    real_sig = inspect.signature(gdown.download)
    calls = []

    def fake(*args, **kwargs):
        bound = real_sig.bind(*args, **kwargs)  # TypeError here = the production bug
        calls.append(bound.arguments)
        with zipfile.ZipFile(bound.arguments["output"], "w") as z:
            z.writestr("videos_160/1.mp4", b"x" * 2_000_000)
        kwargs.get("progress", lambda d, t: None)(2_000_000, 2_000_000)
        return bound.arguments["output"]

    monkeypatch.setattr(gdown, "download", fake)
    from app.storage import get_storage

    get_storage().delete(golfdb.ZIP_KEY)
    stages = []
    vdir = golfdb._download_videos(tmp_path, stages.append)
    assert (vdir / "1.mp4").exists() and calls[0]["id"] == "1uBwRxFxW04EqG87VCoX3l6vXeV5T5JYJ"
    assert stages[-1] == "download 100%"
    assert get_storage().exists(golfdb.ZIP_KEY)  # cached for the next run
    get_storage().delete(golfdb.ZIP_KEY)
    assert golfdb.drive_file_id("https://drive.google.com/open?id=abc-123") == "abc-123"
