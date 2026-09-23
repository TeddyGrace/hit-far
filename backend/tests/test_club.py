import numpy as np
import pytest

from app import jobs, worker
from app.models import EventType as E
from app.pipeline import pose as pose_mod
from app.pipeline.club import ShaftTrack, track_shaft, window_from_events
from app.pipeline.events import detect_events
from app.pipeline.metrics import compute_metrics
from tests.synthetic import make_swing, render_video, shaft_path


def _render(tmp_path, fps=120, **kw):
    pose = make_swing(target_dir=kw.pop("target_dir", 1), fps=fps)
    truth = shaft_path(pose, lean_impact_deg=kw.pop("lean", 12.0))
    path = tmp_path / f"club-{fps}.mp4"
    scaled = render_video(path, pose, truth, **kw)
    ev = {k: v.frame for k, v in detect_events(scaled, "right").events.items()}
    return path, scaled, truth, ev


def _err(track, truth):
    ok = ~np.isnan(track.angle)
    return ok, np.degrees(np.abs((track.angle - truth + np.pi) % (2 * np.pi) - np.pi))


@pytest.mark.parametrize("target_dir", [1, -1])
def test_tracks_known_shaft_through_clutter_and_blur(tmp_path, target_dir):
    path, pose, truth, ev = _render(tmp_path, target_dir=target_dir, clutter=25, motion_blur=True)
    track = track_shaft(path, pose, window_from_events(ev, pose.num_frames, pose.fps))
    ok, err = _err(track, truth)
    confident = ok & (track.confidence >= 0.5)
    assert ok.sum() > 150 and confident.mean() > 0.5
    assert np.median(err[ok]) < 5
    assert np.percentile(err[confident], 95) < 12  # confident frames are right
    assert np.isnan(track.angle[: ev[E.address] - 30]).all()  # outside the swing: not tracked


def test_low_frame_rate_blur_is_flagged_not_guessed(tmp_path):
    path, pose, truth, ev = _render(tmp_path, fps=30, clutter=25, motion_blur=True)
    track = track_shaft(path, pose, window_from_events(ev, pose.num_frames, pose.fps))
    ok, err = _err(track, truth)
    assert track.confidence[ev[E.impact]] < 0.5
    assert (err[ok & (err > 45)] >= 0).all() and not (ok & (err > 45) & (track.confidence >= 0.5)).any()


def test_club_metric_signs():
    pose = make_swing(target_dir=1)
    ev = {k: v.frame for k, v in detect_events(pose, "right").events.items()}
    T = pose.num_frames
    truth = shaft_path(pose, hinge_top_deg=90, lean_impact_deg=10)
    track = ShaftTrack(angle=truth, confidence=np.ones(T), grip=pose.kp2d[:, 15], length_px=380, fps=pose.fps)
    m = {(x.metric_name, x.event_ref): x.value for x in compute_metrics(pose, ev, club=track)}
    assert m[("shaft_lean", E.impact)] > 5  # hands ahead
    assert abs(m[("shaft_lean", E.address)]) < 3
    assert 60 < m[("wrist_hinge_shaft", E.top)] < 130  # cocked (synthetic arm geometry is approximate)
    assert m[("shaft_release_speed", E.impact)] > 0
    assert ("shaft_past_parallel", E.top) in m and ("lag_angle", E.mid_downswing) in m
    # Unconfident frames produce no club metrics.
    track.confidence[:] = 0.2
    names = {x.metric_name for x in compute_metrics(pose, ev, club=track)}
    assert not names & {"shaft_lean", "wrist_hinge_shaft", "lag_angle", "shaft_release_speed", "shaft_past_parallel"}


@pytest.fixture
def rendered(tmp_path_factory, monkeypatch):
    tmp = tmp_path_factory.mktemp("club")
    path, pose, truth, ev = _render(tmp, clutter=10)
    monkeypatch.setattr(pose_mod, "run_pose", lambda p, progress=None: pose)
    return path.read_bytes()


def test_pipeline_tracks_shaft_and_accepts_corrections(authed, rendered):
    from tests.test_api import _upload

    sess, _ = _upload(authed, rendered)
    assert worker.run_once("worker")
    sid = authed.get(f"/api/sessions/{sess['id']}").json()["videos"][0]["swing_id"]
    club = authed.get(f"/api/swings/{sid}/club").json()
    assert sum(a is not None for a in club["angle_deg"]) > 150 and club["corrected"] == []
    swing = authed.get(f"/api/swings/{sid}").json()
    metrics = {(m["metric_name"], m["event_ref"]): m["value"] for m in swing["metrics"]}
    assert metrics[("shaft_lean", "impact")] > 0

    impact = next(e["frame_index"] for e in swing["events"] if e["event_type"] == "impact")
    target_sign = 1  # make_swing(target_dir=1)
    lean_back = (90 - 20 * target_sign) % 360  # clubhead ahead of the hands: 20 deg negative lean
    swing2 = authed.put(f"/api/swings/{sid}/club/{impact}", json={"angle_deg": lean_back}).json()
    m2 = {(m["metric_name"], m["event_ref"]): m["value"] for m in swing2["metrics"]}
    assert m2[("shaft_lean", "impact")] == pytest.approx(-20, abs=0.5)
    assert authed.get(f"/api/swings/{sid}/club").json()["corrected"] == [impact]

    swing3 = authed.put(f"/api/swings/{sid}/club/{impact}", json={"angle_deg": None}).json()
    m3 = {(m["metric_name"], m["event_ref"]): m["value"] for m in swing3["metrics"]}
    assert m3[("shaft_lean", "impact")] == pytest.approx(metrics[("shaft_lean", "impact")])
    assert authed.put(f"/api/swings/{sid}/club/99999", json={"angle_deg": 1}).status_code == 400

    # Swings from before the tracker existed get tracked by start-up jobs.
    from app.automation import queue_startup_jobs
    from app.db import get_sessionmaker
    from app.models import ClubTrack

    with get_sessionmaker()() as db:
        db.query(ClubTrack).delete()
        db.commit()
        assert any(q.startswith(jobs.JOB_TRACK_CLUB) for q in queue_startup_jobs(db))
        assert not any(q.startswith(jobs.JOB_TRACK_CLUB) for q in queue_startup_jobs(db))
    while worker.run_once("worker"):
        pass
    assert authed.get(f"/api/swings/{sid}/club").status_code == 200
