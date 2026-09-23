import numpy as np
import pytest

from app import jobs, worker
from app.models import EventType
from app.outcomes import service
from app.outcomes.train import CVSettings, eligibility, train_problem
from app.pipeline.events import detect_events
from app.pipeline.metrics import PIPELINE_VERSION, compute_metrics
from tests.synthetic import make_swing
from tests.test_api import _upload, fake_pose, sample_video  # noqa: F401  (fixtures)

FAST = CVSettings(n_repeats=3, n_boot=200, n_perm=8, importance_repeats=2, importance_shuffles=2)


# --- v0.2.0 metrics ----------------------------------------------------------------------------


@pytest.mark.parametrize("target_dir", [1, -1])
def test_open_rotation_is_signed_and_camera_independent(target_dir):
    def metrics(**kw):
        pose = make_swing(target_dir=target_dir, **kw)
        ev = {k: v.frame for k, v in detect_events(pose, "right").events.items()}
        return {(m.metric_name, m.event_ref): m for m in compute_metrics(pose, ev)}

    square = metrics()
    opened = metrics(shoulders_open_deg=25, hips_open_deg=40)
    assert abs(square[("shoulders_open", EventType.impact)].value) < 3
    assert opened[("shoulders_open", EventType.impact)].value == pytest.approx(25, abs=3)
    assert opened[("hips_open", EventType.impact)].value == pytest.approx(40, abs=3)
    assert square[("shoulders_open", EventType.mid_downswing)].value < 0  # still closed mid-downswing
    assert opened[("shoulders_open", EventType.impact)].is_estimate
    assert square[("pelvis_toward_lead_foot", EventType.impact)].value == pytest.approx(50, abs=1)
    assert not square[("hands_ahead", EventType.impact)].is_estimate
    for name in ("transition_time", "sequencing_hip_lead"):
        assert (name, None) in square


# --- Model training on synthetic tables --------------------------------------------------------


def _synthetic(signal: float, n=60, d=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X[rng.random((n, d)) < 0.05] = np.nan
    y = (rng.random(n) < 1 / (1 + np.exp(-signal * np.nan_to_num(X[:, 2])))).astype(int)
    return X, y, [f"f{j}" for j in range(d)]


def test_signal_is_found_and_ranked_first():
    X, y, names = _synthetic(signal=4.0)
    r = train_problem(X, y, names, [str(i) for i in range(len(y))], FAST)
    assert r.eval["reliable"] and r.eval["cv_auc"] > 0.8
    top = r.eval["factors"][0]
    assert top["feature"] == "f2" and top["supported"] and top["direction"] == "higher"
    assert top["bad_median"] > top["good_median"]


def test_noise_is_not_reported_as_a_pattern():
    X, y, names = _synthetic(signal=0.0, seed=1)
    r = train_problem(X, y, names, [str(i) for i in range(len(y))], FAST)
    assert not r.eval["reliable"]
    assert r.eval["cv_auc_ci"][0] <= 0.60
    assert r.eval["null_auc_95"] > 0.5  # what chance looks like at this sample size


def test_minimums_gate_training():
    cfg = CVSettings()
    assert eligibility(np.array([1] * 5 + [0] * 30), cfg)["need"] == {"swings": 0, "positive": 1, "negative": 0}
    assert eligibility(np.array([1] * 6 + [0] * 13), cfg)["need"]["swings"] == 1
    assert eligibility(np.array([1] * 6 + [0] * 14), cfg)["eligible"]


# --- End to end: tag -> automatic training -> promotion -> analysis ----------------------------


def _make_swings(n, seed=0):
    """Swings with metrics but no video processing; 'shoulders_open@impact' drives slicing."""
    from datetime import datetime, timezone

    from app.db import get_sessionmaker
    from app.models import Metric, RecordingSession, Swing, Video, VideoStatus

    rng = np.random.default_rng(seed)
    out = []
    with get_sessionmaker()() as db:
        sess = RecordingSession(recorded_at=datetime.now(timezone.utc))
        db.add(sess)
        db.flush()
        for i in range(n):
            v = Video(session_id=sess.id, file_uri=f"videos/{seed}-{i}.mp4", status=VideoStatus.preprocessed,
                      fps=120.0, original_filename=f"s{i}.mp4")
            db.add(v)
            db.flush()
            sw = Swing(session_id=sess.id, video_ids=[v.id])
            db.add(sw)
            db.flush()
            opened = rng.normal(10, 12)
            slice_ = bool(rng.random() < 1 / (1 + np.exp(-(opened - 10) / 3)))
            vals = {("shoulders_open", EventType.impact): opened, ("tempo_ratio", None): rng.normal(3, 0.3),
                    ("head_rise", EventType.impact): rng.normal(0, 5), ("hip_turn", EventType.top): rng.normal(45, 8)}
            for (name, ev), val in vals.items():
                db.add(Metric(swing_id=sw.id, pipeline_version=PIPELINE_VERSION, metric_name=name, event_ref=ev,
                              value=float(val), unit="deg", is_estimate=name != "tempo_ratio"))
            out.append((sw.id, "slice" if slice_ else "straight"))
        db.commit()
    return out


@pytest.fixture
def fast_cv(monkeypatch):
    monkeypatch.setattr(service, "_cv_settings", lambda overrides: FAST)


def test_tagging_trains_and_promotes_automatically(authed, fast_cv):
    swings = _make_swings(40)
    for i, (sid, shape) in enumerate(swings):
        r = authed.put(f"/api/swings/{sid}/outcome", json={"shape": shape, "contact": "solid"})
        assert r.status_code == 200 and r.json()["shape"] == shape
        if i == 3:  # 4 changes: below the retrain threshold
            s = authed.get("/api/outcomes").json()
            assert s["changes_since_training"] == 4 and s["training"] is None
    assert worker.run_once("worker") is True  # the queued train_outcomes job
    assert worker.run_once("worker") is False

    summary = authed.get("/api/outcomes").json()
    probs = {p["key"]: p for p in summary["problems"]}
    assert summary["tagged"] == 40 and summary["changes_since_training"] == 0
    assert probs["slice"]["model"]["reliable"] is True
    assert probs["hook"]["model"] is None and probs["hook"]["need"]["positive"] == 6  # no hooks tagged
    assert probs["fat"]["eligible"] is False

    a = authed.get("/api/outcomes/slice").json()
    top = a["factors"][0]
    assert top["feature"] == "shoulders_open@impact" and top["supported"] and top["direction"] == "higher"
    assert top["unit"] == "deg" and top["is_estimate"] is True and len(top["points"]) == 40
    assert a["references"]["good"]["outcome"]["shape"] == "straight"
    assert a["references"]["bad"]["outcome"]["shape"] == "slice"
    assert a["what_if"]["items"] and all("delta" in w for w in a["what_if"]["items"])

    preds = authed.get(f"/api/swings/{swings[0][0]}/predictions").json()
    assert [p["problem"] for p in preds] == ["slice"] and 0 <= preds[0]["probability"] <= 1

    models = [m for m in authed.get("/api/models").json() if m["name"] == "outcome-slice"]
    assert len(models) == 1 and models[0]["status"] == "active" and models[0]["task"] == "outcome_slice"

    # A retrain whose recipe scores worse on the same swings than the active one's is not promoted.
    from app.db import get_sessionmaker
    from app.models import Model

    with get_sessionmaker()() as db:
        active = db.get(Model, models[0]["id"])
        active.eval_metrics = {**active.eval_metrics, "kind": "logreg", "features": ["shoulders_open@impact"]}
        db.commit()
    for sid, shape in swings[:5]:  # 5 edits -> retrain
        authed.put(f"/api/swings/{sid}/outcome", json={"start_line": "left"})
    assert worker.run_once("worker") is True
    versions = {m["version"]: m for m in authed.get("/api/models").json() if m["name"] == "outcome-slice"}
    assert len(versions) == 2
    new = next(m for v, m in versions.items() if v != models[0]["version"])
    cmp = new["eval_metrics"]["compared_to"]
    assert cmp["version"] == models[0]["version"]
    # The one-feature recipe scores better on these swings than the all-features retrain.
    assert new["eval_metrics"]["cv_auc"] < cmp["cv_auc_same_data"]
    assert new["status"] == "experimental" and new["eval_metrics"]["auto_promoted"] is False
    assert versions[models[0]["version"]]["status"] == "active"


def test_outcome_tag_audit_and_clear(authed, sample_video, fake_pose):  # noqa: F811
    sess, _ = _upload(authed, sample_video)
    worker.run_once("worker")
    sid = authed.get(f"/api/sessions/{sess['id']}").json()["videos"][0]["swing_id"]
    assert authed.get(f"/api/swings/{sid}").json()["outcome"] is None
    assert authed.put(f"/api/swings/{sid}/outcome", json={"shape": "banana"}).status_code == 422

    authed.put(f"/api/swings/{sid}/outcome", json={"shape": "slice"})
    authed.put(f"/api/swings/{sid}/outcome", json={"contact": "thin"})
    detail = authed.get(f"/api/sessions/{sess['id']}").json()["videos"][0]
    assert detail["outcome"]["shape"] == "slice" and detail["outcome"]["contact"] == "thin"
    assert authed.get(f"/api/swings/{sid}").json()["outcome"]["contact"] == "thin"

    # Clearing every field removes the tag; each change is kept as an audit label.
    r = authed.put(f"/api/swings/{sid}/outcome", json={"shape": None, "contact": None})
    assert r.status_code == 200 and r.json() is None
    assert authed.get(f"/api/swings/{sid}").json()["outcome"] is None
    from app.db import get_sessionmaker
    from app.models import Label

    with get_sessionmaker()() as db:
        labels = db.query(Label).filter(Label.task == "outcome").order_by(Label.created_at).all()
    assert [lb.corrected_value["shape"] for lb in labels] == ["slice", "slice", None]


def test_recompute_job_after_formula_change(authed, sample_video, fake_pose):  # noqa: F811
    from app.automation import queue_startup_jobs
    from app.db import get_sessionmaker
    from app.models import Metric
    from app.pipeline.run import swings_needing_metrics

    sess, _ = _upload(authed, sample_video)
    worker.run_once("worker")
    sid = authed.get(f"/api/sessions/{sess['id']}").json()["videos"][0]["swing_id"]
    authed.put(f"/api/swings/{sid}/outcome", json={"shape": "fade"})
    with get_sessionmaker()() as db:  # pretend the metrics came from an older pipeline version
        db.query(Metric).update({Metric.pipeline_version: "0.0.1"})
        db.commit()
        assert len(swings_needing_metrics(db)) == 1
        assert jobs.JOB_RECOMPUTE_METRICS in queue_startup_jobs(db)
        assert jobs.JOB_RECOMPUTE_METRICS not in queue_startup_jobs(db)  # already queued
    assert worker.run_once("worker") is True
    swing = authed.get(f"/api/swings/{sid}").json()
    assert swing["pipeline_version"] == PIPELINE_VERSION
    assert any(m["metric_name"] == "shoulders_open" for m in swing["metrics"])
    with get_sessionmaker()() as db:
        assert swings_needing_metrics(db) == []
        assert jobs.pending(db, jobs.JOB_TRAIN_OUTCOMES) is not None  # features changed -> retrain


def test_first_event_training_is_queued_once(engine):
    from app.automation import queue_startup_jobs
    from app.db import get_sessionmaker
    from app.models import Job

    with get_sessionmaker()() as db:
        assert jobs.JOB_TRAIN_EVENTS in queue_startup_jobs(db)
        assert queue_startup_jobs(db) == []
        db.query(Job).delete()
        db.commit()


def test_llm_explain_is_off_by_default_and_only_sees_model_output(authed, fast_cv, monkeypatch):
    assert authed.post("/api/outcomes/slice/explain").status_code == 403
    from app.config import get_settings
    from app.diagnosis.claude import ClaudeResult
    from app.outcomes import explain
    from app.outcomes.explain import Explanation

    monkeypatch.setattr(get_settings(), "outcome_llm_explain", True)
    assert authed.post("/api/outcomes/slice/explain").status_code == 409  # nothing trained yet
    for sid, shape in _make_swings(30, seed=2):
        authed.put(f"/api/swings/{sid}/outcome", json={"shape": shape})
    worker.run_once("worker")
    seen = {}

    def fake(system, content, schema, client=None, output_model=None):
        seen["content"] = content
        return ClaudeResult(output=Explanation(explanation="words"), model="m", usage={})

    monkeypatch.setattr(explain, "call_claude", fake)
    r = authed.post("/api/outcomes/slice/explain").json()
    assert r["explanation"] == "words"
    assert [c["type"] for c in seen["content"]] == ["text"]  # no images, just the model's JSON
    assert '"points"' not in seen["content"][0]["text"]
    assert authed.get("/api/outcomes/nope").status_code == 404
