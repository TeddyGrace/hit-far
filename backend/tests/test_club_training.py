"""Stage-2 club detector: pseudo-labels, crops, scoring, the data gate, training, promotion."""

import math
import random
import uuid

import numpy as np
import pytest

from app import jobs, worker
from app.db import get_sessionmaker
from app.models import ClubTrack, Job, Model, ModelStatus, Swing
from app.pipeline import club as club_mod
from app.pipeline import landmarks as L
from app.pipeline import pose as pose_mod
from app.pipeline.club import CONF_Z, N_BINS, shaft_end, track_shaft, window_from_events
from app.pipeline.events import detect_events
from app.pipeline.registry import TASK_CLUB
from app.training import club_model as cm
from app.training import train_club as tc
from tests.synthetic import make_swing, render_video, shaft_path


def _render(tmp_path, seed=0, fps=120, target_dir=1, **kw):
    pose = make_swing(target_dir=target_dir, fps=fps)
    truth = shaft_path(pose, lean_impact_deg=12.0)
    path = tmp_path / f"club-{seed}-{fps}.mp4"
    scaled = render_video(path, pose, truth, seed=seed, **kw)
    ev = {k: v.frame for k, v in detect_events(scaled, "right").events.items()}
    return path, scaled, truth, ev


def _tip(pose, truth):
    kp = pose.kp2d
    torso = np.linalg.norm((kp[0, L.L_SHOULDER] + kp[0, L.R_SHOULDER]) / 2 - (kp[0, L.L_HIP] + kp[0, L.R_HIP]) / 2)
    return kp[:, L.L_WRIST] + 1.9 * torso * np.stack([np.cos(truth), np.sin(truth)], 1), 1.9 * torso


def test_shaft_end_finds_the_clubhead(tmp_path):
    path, pose, truth, ev = _render(tmp_path, clutter=10)
    w = window_from_events(ev, pose.num_frames, pose.fps)
    track = track_shaft(path, pose, w)
    frames, bg, W, H = club_mod.read_frames(path, *w)
    torso = club_mod.torso_px(pose, w[0])
    tip, length = _tip(pose, truth)
    sweep = tc._sweep_deg(track.angle)
    errs, found, tried = [], 0, 0
    for i, gray in enumerate(frames):
        f = w[0] + i
        inside = 0 <= tip[f, 0] < W and 0 <= tip[f, 1] < H
        if track.confidence[f] < tc.PSEUDO_CONF or sweep[f] > tc.CLUBHEAD_MAX_SWEEP_DEG or not inside:
            continue
        tried += 1
        end = shaft_end(gray.astype(np.float32) - bg, track.grip[f], track.angle[f], torso)
        if end is not None:
            found += 1
            errs.append(np.linalg.norm(end - tip[f]) / length)
    assert tried > 30 and found / tried > 0.8
    assert np.median(errs) < 0.03 and np.percentile(errs, 90) < 0.08


def test_crop_and_augmentation_geometry_agree():
    g, torso = np.array([200.0, 150.0]), 80.0
    a = math.radians(125)
    p = g + 1.5 * torso * np.array([math.cos(a), math.sin(a)])
    M = cm.crop_matrix(g, torso)
    c = cm.apply_affine(M, np.stack([g, p]))
    assert np.allclose(c[0], [cm.CROP / 2, cm.CROP / 2])
    assert np.linalg.norm(c[1] - c[0]) == pytest.approx(1.5 * cm.TORSO_CROP_PX)
    for rot, mirror in [(0.7, False), (-2.0, True), (0.0, True)]:
        A = cm.crop_matrix(np.array([cm.CROP / 2, cm.CROP / 2]), cm.TORSO_CROP_PX, rot, 1.1, mirror)
        q = cm.apply_affine(A, c)
        got = math.atan2(q[1, 1] - q[0, 1], q[1, 0] - q[0, 0]) % (2 * math.pi)
        assert got == pytest.approx(cm.transform_angle(a, rot, mirror), abs=1e-6)
        assert np.allclose(q[0], [cm.CROP / 2, cm.CROP / 2])


def test_direction_scores_pick_the_heatmap_direction_and_calibrate():
    a = math.radians(40)
    head = np.array([cm.CROP / 2, cm.CROP / 2]) + 1.6 * cm.TORSO_CROP_PX * np.array([math.cos(a), math.sin(a)])
    prob = cm.gaussian_target(head)[None]
    z = cm.direction_scores(prob)[0]
    best = int(z.argmax())
    assert abs(best * 360 / N_BINS - 40) <= 2
    assert 1 / (1 + math.exp(-(z[best] - CONF_Z) / 0.75)) > 0.6  # confident when the mass is in one wedge
    spread = np.full((1, cm.HM, cm.HM), 1 / cm.HM ** 2, np.float32)
    zs = cm.direction_scores(spread)[0]
    assert 1 / (1 + math.exp(-(zs.max() - CONF_Z) / 0.75)) < 0.1  # not when it's everywhere
    pt, mass = cm.clubhead_along(prob[0], a)
    assert np.linalg.norm(pt - head) < 2 and mass > 0.6


def test_gate_blocks_training_until_there_is_data(authed):
    st = authed.get("/api/training/club/status").json()
    assert st["ready"] is False and st["swings"] == 0 and st["needs"]
    r = authed.post("/api/training/club")
    assert r.status_code == 400 and "not enough data" in r.json()["detail"]
    with get_sessionmaker()() as db:
        assert tc.maybe_queue_training(db) is None
        job = jobs.enqueue(db, jobs.JOB_TRAIN_CLUB, {"config": {}})
        db.commit()
    assert worker.run_once("trainer")
    with get_sessionmaker()() as db:
        job = db.get(Job, job.id)
        assert job.status.value == "done" and job.stage == "done"
        assert db.query(Model).filter(Model.name == tc.MODEL_NAME).count() == 0


def _fake_model(db, name_version: str, metrics: dict, checkpoint="models/x/model.pt") -> Model:
    m = Model(name=tc.MODEL_NAME, version=name_version, task=TASK_CLUB, checkpoint_uri=checkpoint,
              eval_metrics=metrics, status=ModelStatus.experimental)
    db.add(m)
    db.commit()
    return m


def _em(fix, within, agree=2.0, current=None):
    """fix: (learned, line) median error on your fixes; within: (learned, line) within-10 rate
    over all your checked frames; current: (fix median, within) of the active detector."""
    def h(med, w):
        return {"n": 30, "median_err_deg": med, "within_10": w}
    em = {"learned": {"human": h(1.0, within[0]), "fixes": h(fix[0], 0.5),
                      "agreement": {"n": 100, "median_diff_deg": agree}},
          "line": {"human": h(0.0, within[1]), "fixes": h(fix[1], 0.2)}}
    if current:
        em["current"] = {"human": h(0.5, current[1]), "fixes": h(current[0], 0.5)}
    return em


def test_auto_promotion_rules(engine):
    with get_sessionmaker()() as db:
        assert not tc.auto_promote_club(db, _fake_model(db, "worse", _em((25, 20), (0.9, 0.8))))
        assert not tc.auto_promote_club(db, _fake_model(db, "loses-confirmed", _em((5, 20), (0.7, 0.8))))
        assert not tc.auto_promote_club(db, _fake_model(db, "disagrees", _em((5, 20), (0.9, 0.8), agree=9)))
        few = _em((5, 20), (0.9, 0.8))
        few["learned"]["human"]["n"] = 5
        assert not tc.auto_promote_club(db, _fake_model(db, "few", few))
        no_fixes = _em((5, 20), (0.9, 0.8))
        no_fixes["learned"]["fixes"] = no_fixes["line"]["fixes"] = {"n": 0}
        assert not tc.auto_promote_club(db, _fake_model(db, "confirmations-only", no_fixes))
        # Confirmations of the line tracker give it 0 degrees median over all checks; the detector
        # can still win on the frames you had to fix.
        good = _fake_model(db, "good", _em((5, 20), (0.9, 0.8)))
        assert tc.auto_promote_club(db, good)
        assert db.get(Model, good.id).status == ModelStatus.active
        from app.pipeline.run import active_club_model

        assert active_club_model(db).id == good.id
        worse_than_current = _fake_model(db, "vs-current", _em((6, 20), (0.9, 0.8), current=(5, 0.9)))
        assert not tc.auto_promote_club(db, worse_than_current)
        assert active_club_model(db).id == good.id
        db.query(Model).delete()
        db.commit()


@pytest.fixture
def four_swings(authed, tmp_path, monkeypatch):
    """Four different synthetic swings through the real pipeline (pose faked, as in test_club)."""
    from tests.test_api import _upload

    poses = []
    monkeypatch.setattr(pose_mod, "run_pose", lambda p, progress=None: poses.pop(0))
    out = []
    for seed, (fps, td) in enumerate([(120, 1), (120, -1), (240, 1), (120, 1)]):
        path, pose, truth, ev = _render(tmp_path, seed=seed, fps=fps, target_dir=td, clutter=10 + 5 * seed,
                                        motion_blur=seed % 2 == 1)
        poses.append(pose)
        sess, _ = _upload(authed, path.read_bytes(), name=f"s{seed}.mp4")
        assert worker.run_once("worker")
        sid = authed.get(f"/api/sessions/{sess['id']}").json()["videos"][0]["swing_id"]
        out.append((sid, pose, truth, ev))
    return out


def test_train_club_end_to_end(authed, four_swings):
    small = tc.ClubTrainConfig(epochs=2, width=8, max_frames_per_swing=40, batch_size=16,
                               min_swings=3, min_test_swings=1, min_test_frames=3)
    rng = random.Random(0)
    # Label three of the four swings: fixes with the clubhead clicked, and confirmations.
    for sid, pose, truth, ev in four_swings[:3]:
        club = authed.get(f"/api/swings/{sid}/club").json()
        tip, _ = _tip(pose, truth)
        tracked = [f for f, a in enumerate(club["angle_deg"]) if a is not None]
        for f in rng.sample(tracked, 6)[:3]:
            deg = math.degrees(truth[f]) % 360
            assert authed.put(f"/api/swings/{sid}/club/{f}",
                              json={"angle_deg": deg, "clubhead_xy": tip[f].tolist()}).status_code == 200
        for f in rng.sample(tracked, 6)[3:]:
            assert authed.put(f"/api/swings/{sid}/club/{f}", json={"confirm": True}).status_code == 200
        club = authed.get(f"/api/swings/{sid}/club").json()
        assert len(club["corrected"]) >= 5 and len(club["confirmed"]) >= 2
    assert authed.get("/api/training/club/status").json()["ready"] is False  # the real gate is higher

    with get_sessionmaker()() as db:
        ok, st = tc.should_train(db, small)
        assert ok and st["test_swings"] == 1 and st["labelled_swings"] == 3
        assert tc.maybe_queue_training(db, small) is not None
        assert tc.maybe_queue_training(db, small) is None  # already queued
    assert authed.post("/api/training/club").status_code == 409
    assert worker.run_once("trainer")
    with get_sessionmaker()() as db:
        job = db.query(Job).filter(Job.type == jobs.JOB_TRAIN_CLUB).one()
        assert job.status.value == "done", job.error
        m = db.query(Model).filter(Model.name == tc.MODEL_NAME).one()
        em = m.eval_metrics
        assert em["learned"]["human"]["n"] >= 3 and em["line"]["human"]["n"] >= 3
        assert em["learned"]["fixes"]["n"] >= 1 and em["line"]["human"]["n"] > em["line"]["fixes"]["n"]
        assert em["n_train_swings"] == 3 and em["n_test_swings"] == 1 and em["n_train_human"] > 0
        assert em["n_train_clubhead_points"] > 0
        ok, _ = tc.should_train(db, small)
        assert not ok  # nothing new since that run
        model_id = m.id

    # Promote it by hand: every swing is re-tracked with the detector, which also finds the clubhead.
    assert authed.post(f"/api/models/{model_id}/promote").status_code == 200
    while worker.run_once("worker"):
        pass
    sid = four_swings[3][0]
    club = authed.get(f"/api/swings/{sid}/club").json()
    assert club["model_name"] == tc.MODEL_NAME
    assert sum(a is not None for a in club["angle_deg"]) > 100
    # A 2-epoch toy model is rarely sure where the clubhead is (it says so with None); accuracy is
    # covered by test_detector_learns_to_beat_its_teacher.
    assert club["clubhead"] is not None and len(club["clubhead"]) == len(club["angle_deg"])
    # Your labels still override the detector on their frames.
    sid0 = four_swings[0][0]
    club0 = authed.get(f"/api/swings/{sid0}/club").json()
    f = club0["corrected"][0]
    assert club0["confidence"][f] == 1.0

    # A broken checkpoint never loses the swing: tracking falls back to the line tracker.
    from app.pipeline.run import club_model, latest_club_track, track_club
    from app.storage import get_storage

    with get_sessionmaker()() as db:
        db.get(Model, model_id).status = ModelStatus.deprecated
        get_storage().put_bytes("models/broken/model.pt", b"not a checkpoint")
        bad = Model(name=tc.MODEL_NAME, version="broken", task=TASK_CLUB, checkpoint_uri="models/broken/model.pt",
                    status=ModelStatus.active)
        db.add(bad)
        db.commit()
        track_club(db, uuid.UUID(sid))
        row = db.query(ClubTrack).filter(ClubTrack.swing_id == uuid.UUID(sid)).order_by(ClubTrack.created_at.desc()).first()
        assert row.model_id == club_model(db).id
        assert latest_club_track(db, uuid.UUID(sid)) is not None
        assert db.get(Swing, uuid.UUID(sid)) is not None


def test_detector_learns_to_beat_its_teacher(tmp_path):
    """Trained only on the line tracker's confident frames, the detector ends up more accurate than
    the line tracker on swings it never saw, and finds the clubhead."""
    from app.training.club_inference import track_frames

    swings = []
    for seed, (fps, td, blur) in enumerate([(120, 1, False), (120, -1, True), (240, 1, True), (120, -1, False)]):
        path, pose, truth, ev = _render(tmp_path, seed=seed, fps=fps, target_dir=td, clutter=10 + 5 * seed,
                                        motion_blur=blur)
        w = window_from_events(ev, pose.num_frames, pose.fps)
        frames, bg, _, _ = club_mod.read_frames(path, *w)
        swings.append((pose, truth, w, track_shaft(path, pose, w), frames, bg))
    data = []
    for pose, truth, w, line, frames, bg in swings[:3]:
        grip, torso, sweep = club_mod.grip_points(pose), club_mod.torso_px(pose, w[0]), tc._sweep_deg(line.angle)
        fs = [f for f in range(w[0], w[1] + 1) if line.confidence[f] >= tc.PSEUDO_CONF][::2]
        crops, heads = [], []
        for f in fs:
            i = f - w[0]
            crops.append(cm.make_crop(frames, bg, i, grip[f], torso))
            end = (shaft_end(frames[i].astype(np.float32) - bg, grip[f], line.angle[f], torso)
                   if sweep[f] <= tc.CLUBHEAD_MAX_SWEEP_DEG else None)
            heads.append(np.full(2, np.nan) if end is None else cm.apply_affine(cm.crop_matrix(grip[f], torso), end[None])[0])
        data.append(tc.SwingData("s", "train", np.stack(crops), np.array(fs), line.angle[fs], np.array(heads),
                                 np.zeros(len(fs), bool), line))
    model, _ = tc.train(data, tc.ClubTrainConfig(epochs=8, width=16), lambda s: None)

    pose, truth, w, line, frames, bg = swings[3]
    tr = track_frames(model, frames, bg, w[0], pose.num_frames, club_mod.grip_points(pose),
                      club_mod.torso_px(pose, w[0]), pose.fps)
    ok = np.isfinite(tr.angle)
    learned, teacher = tc._err_deg(tr.angle[ok], truth[ok]), tc._err_deg(line.angle[ok], truth[ok])
    assert np.median(learned) < np.median(teacher) and np.median(learned) < 3
    assert (tr.confidence[ok] >= 0.5).mean() > 0.8
    tip, length = _tip(pose, truth)
    found = np.isfinite(tr.clubhead[:, 0])
    assert found.sum() > 0.5 * ok.sum()
    assert np.median(np.linalg.norm(tr.clubhead[found] - tip[found], axis=1)) / length < 0.06
