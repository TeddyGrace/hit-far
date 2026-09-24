import subprocess
from pathlib import Path

import pytest

from app import worker
from app.pipeline import pose as pose_mod
from app.pipeline.ingest import ffmpeg_bin
from tests.synthetic import END_S, FPS, expected_frames, make_swing


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory) -> bytes:
    out = tmp_path_factory.mktemp("vid") / "swing.mp4"
    subprocess.run(
        [ffmpeg_bin(), "-y", "-v", "error", "-f", "lavfi", "-i",
         f"testsrc=duration={END_S}:size=320x240:rate={int(FPS)}", "-pix_fmt", "yuv420p", str(out)],
        check=True,
    )
    return out.read_bytes()


@pytest.fixture
def fake_pose(monkeypatch):
    """The real MediaPipe model needs a real person; swap in the synthetic swing."""
    monkeypatch.setattr(pose_mod, "run_pose", lambda path, progress=None: make_swing())


def test_requires_auth(client):
    assert client.get("/api/sessions").status_code == 401
    assert client.post("/api/auth/login", json={"username": "owner", "password": "nope"}).status_code == 401
    assert client.get("/api/health").status_code == 200


def _upload(authed, sample_video, name="swing.mp4"):
    sess = authed.post("/api/sessions", json={"location": "range", "club_used": "7i"}).json()
    up = authed.post(f"/api/sessions/{sess['id']}/videos",
                     json={"filename": name, "content_type": "video/mp4"}).json()
    assert authed.put(up["upload_url"], content=sample_video, headers=up["upload_headers"]).status_code == 200
    job = authed.post(f"/api/videos/{up['video']['id']}/complete").json()
    assert job["status"] == "queued"
    return sess, up["video"]


def test_full_pipeline_and_event_correction(authed, sample_video, fake_pose):
    sess, video = _upload(authed, sample_video)
    assert worker.run_once() is True
    assert worker.run_once() is False  # queue drained

    detail = authed.get(f"/api/sessions/{sess['id']}").json()
    v = detail["videos"][0]
    assert v["status"] == "preprocessed", v
    assert v["job"]["status"] == "done", v["job"]
    assert v["num_frames"] == int(END_S * FPS)
    assert v["fps"] == pytest.approx(FPS)
    swing_id = v["swing_id"]

    swing = authed.get(f"/api/swings/{swing_id}").json()
    assert swing["video"]["playback_url"]
    assert len(swing["events"]) == 8
    assert swing["pose"]["model"]["name"] == pose_mod.MODEL_NAME
    impact = next(e for e in swing["events"] if e["event_type"] == "impact")
    assert abs(impact["frame_index"] - expected_frames()["impact"]) <= 6
    metrics = {(m["metric_name"], m["event_ref"]): m for m in swing["metrics"]}
    tempo_before = metrics[("tempo_ratio", None)]["value"]
    assert metrics[("shoulder_turn", "top")]["is_estimate"] is True

    # The playback URL serves the proxy with range support.
    r = authed.get(swing["video"]["playback_url"], headers={"Range": "bytes=0-99"})
    assert r.status_code == 206

    pose = authed.get(f"/api/swings/{swing_id}/pose").json()
    assert len(pose["frames"]) == int(END_S * FPS)
    assert len(pose["frames"][0]) == 33 * 3

    # Correct the top -> label written, metrics recomputed from the corrected frame.
    top = next(e for e in swing["events"] if e["event_type"] == "top")
    new_top = top["frame_index"] - 12
    swing2 = authed.put(f"/api/swings/{swing_id}/events/top", json={"frame_index": new_top}).json()
    top2 = next(e for e in swing2["events"] if e["event_type"] == "top")
    assert top2["corrected"] and top2["frame_index"] == new_top
    assert top2["predicted_frame_index"] == top["frame_index"]
    tempo_after = next(m for m in swing2["metrics"] if m["metric_name"] == "tempo_ratio")["value"]
    assert tempo_after < tempo_before

    # Revert.
    swing3 = authed.put(f"/api/swings/{swing_id}/events/top", json={"frame_index": None}).json()
    top3 = next(e for e in swing3["events"] if e["event_type"] == "top")
    assert not top3["corrected"] and top3["frame_index"] == top["frame_index"]

    models = authed.get("/api/models").json()
    assert {m["name"] for m in models} == {"mediapipe-pose-landmarker-heavy", "rule-events", "shaft-line-tracker"}


def test_duplicate_upload_is_rejected(authed, sample_video, fake_pose):
    _upload(authed, sample_video)
    worker.run_once()
    sess2, video2 = _upload(authed, sample_video, name="again.mp4")
    worker.run_once()
    v = authed.get(f"/api/sessions/{sess2['id']}").json()["videos"][0]
    assert v["status"] == "failed"
    assert "duplicate" in v["error"]
    assert v["job"]["status"] == "failed"


def test_reprocess_and_delete(authed, sample_video, fake_pose):
    sess, video = _upload(authed, sample_video)
    worker.run_once()

    # Renaming (no extension) changes the display name and doesn't break reprocessing.
    swing_id = authed.get(f"/api/sessions/{sess['id']}").json()["videos"][0]["swing_id"]
    renamed = authed.patch(f"/api/swings/{swing_id}", json={"name": "  Driver, good one "}).json()
    assert renamed["video"]["original_filename"] == "Driver, good one"
    assert authed.get(f"/api/sessions/{sess['id']}").json()["videos"][0]["original_filename"] == "Driver, good one"
    assert authed.patch(f"/api/swings/{swing_id}", json={"name": "   "}).status_code == 422

    r = authed.post(f"/api/videos/{video['id']}/reprocess", json={"force": True})
    assert r.status_code == 200
    assert authed.post(f"/api/videos/{video['id']}/reprocess", json={"force": True}).status_code == 409
    worker.run_once()
    assert authed.get(f"/api/sessions/{sess['id']}").json()["videos"][0]["status"] == "preprocessed"
    assert authed.delete(f"/api/sessions/{sess['id']}").status_code == 204
    assert authed.get(f"/api/sessions/{sess['id']}").status_code == 404


def test_rejects_bad_extension(authed):
    sess = authed.post("/api/sessions", json={}).json()
    r = authed.post(f"/api/sessions/{sess['id']}/videos", json={"filename": "x.gif"})
    assert r.status_code == 400


def test_storage_token_cannot_escape_root(authed):
    from app.storage import get_storage

    with pytest.raises(ValueError):
        get_storage().path_for("../../etc/passwd")
    assert Path(get_storage().path_for("a/b.mp4")).name == "b.mp4"
