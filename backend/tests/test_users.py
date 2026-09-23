import io

import pytest
from sqlalchemy import select

from app import auth, worker
from app.db import get_sessionmaker
from app.models import Model, RecordingSession, User
from app.users import main as users_cli
from tests.test_api import _upload, fake_pose, sample_video  # noqa: F401  (fixtures)
from tests.test_outcomes import _make_swings, fast_cv  # noqa: F401  (fixture)


def test_password_hashing():
    h = auth.hash_password("correct horse")
    assert h.startswith("scrypt$") and "correct horse" not in h
    assert auth.verify_password("correct horse", h)
    assert not auth.verify_password("wrong horse", h)
    assert auth.hash_password("correct horse") != h  # salted
    assert not auth.verify_password("x", "garbage")


def _add_user(monkeypatch, name, password):
    monkeypatch.setattr("sys.stdin", io.StringIO(password + "\n"))
    users_cli(["add", name, "--password-stdin"])


def _login(client, username, password):
    return client.post("/api/auth/login", json={"username": username, "password": password})


def test_owner_is_created_from_app_password_and_claims_old_rows(client):
    with get_sessionmaker()() as db:
        owner = db.scalar(select(User).where(User.username == "owner"))
        assert owner is not None
        db.add(RecordingSession(recorded_at=owner.created_at))  # a row from before users existed
        db.commit()
        auth.bootstrap_users(db, auth.get_settings())
        assert db.scalar(select(User.id).where(User.username != "owner")) is None  # no second owner
        assert set(db.scalars(select(RecordingSession.user_id)).all()) == {owner.id}


def test_login_logout_and_cli(client, monkeypatch):
    _add_user(monkeypatch, "Alice", "alice-password")
    assert _login(client, "alice", "nope-nope").status_code == 401
    assert _login(client, "bob", "alice-password").status_code == 401
    r = _login(client, " ALICE ", "alice-password")
    assert r.status_code == 200 and r.json()["username"] == "alice"
    assert client.get("/api/auth/me").json()["username"] == "alice"

    monkeypatch.setattr("sys.stdin", io.StringIO("new-password\n"))
    users_cli(["passwd", "alice", "--password-stdin"])
    assert _login(client, "alice", "alice-password").status_code == 401
    assert _login(client, "alice", "new-password").status_code == 200

    with pytest.raises(SystemExit):
        _add_user(monkeypatch, "alice", "whatever-pw")  # duplicate
    with pytest.raises(SystemExit):
        _add_user(monkeypatch, "carol", "short")  # too short

    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").status_code == 401


def test_old_single_user_cookie_is_rejected(client):
    token = auth._signer(auth.get_settings()).dumps({"u": "owner"})
    client.cookies.set(auth.COOKIE_NAME, token)
    assert client.get("/api/sessions").status_code == 401


def test_users_only_see_their_own_data(authed, sample_video, fake_pose, monkeypatch):  # noqa: F811
    sess, video = _upload(authed, sample_video)
    assert worker.run_once() is True
    swing_id = authed.get(f"/api/sessions/{sess['id']}").json()["videos"][0]["swing_id"]
    assert swing_id

    _add_user(monkeypatch, "bob", "bob-password")
    assert _login(authed, "bob", "bob-password").status_code == 200  # same client, now bob
    assert authed.get("/api/sessions").json() == []
    for method, path in [
        ("get", f"/api/sessions/{sess['id']}"),
        ("delete", f"/api/sessions/{sess['id']}"),
        ("post", f"/api/sessions/{sess['id']}/videos"),
        ("post", f"/api/videos/{video['id']}/reprocess"),
        ("delete", f"/api/videos/{video['id']}"),
        ("get", f"/api/swings/{swing_id}"),
        ("get", f"/api/swings/{swing_id}/pose"),
        ("put", f"/api/swings/{swing_id}/outcome"),
        ("get", f"/api/swings/{swing_id}/diagnoses"),
    ]:
        kw = {"json": {"filename": "x.mp4", "force": True}} if method in ("post", "put") else {}
        assert getattr(authed, method)(path, **kw).status_code == 404, path

    mine = authed.post("/api/sessions", json={"location": "bob's range"}).json()
    assert [s["id"] for s in authed.get("/api/sessions").json()] == [mine["id"]]

    assert _login(authed, "owner", "test-pw").status_code == 200
    assert [s["id"] for s in authed.get("/api/sessions").json()] == [sess["id"]]


def test_outcome_models_are_per_user(authed, fast_cv, monkeypatch):  # noqa: F811
    _add_user(monkeypatch, "bob", "bob-password")
    for sid, shape in _make_swings(40):  # owner's
        authed.put(f"/api/swings/{sid}/outcome", json={"shape": shape})
    bob_swings = _make_swings(12, seed=1, username="bob")
    while worker.run_once("worker"):
        pass
    assert authed.get("/api/outcomes").json()["tagged"] == 40
    owner_models = [m for m in authed.get("/api/models").json() if m["name"] == "outcome-slice"]
    assert len(owner_models) == 1 and owner_models[0]["status"] == "active"

    _login(authed, "bob", "bob-password")
    s = authed.get("/api/outcomes").json()
    assert s["tagged"] == 0 and s["with_metrics"] == 12 and s["changes_since_training"] == 0
    assert not [m for m in authed.get("/api/models").json() if m["task"].startswith("outcome_")]
    assert authed.post(f"/api/models/{owner_models[0]['id']}/promote").status_code == 404
    assert authed.get(f"/api/swings/{bob_swings[0][0]}/predictions").json() == []

    for sid, shape in bob_swings[:5]:  # bob's own tags queue bob's own retrain
        authed.put(f"/api/swings/{sid}/outcome", json={"shape": shape})
    assert authed.get("/api/outcomes").json()["training"]["status"] == "queued"
    while worker.run_once("worker"):
        pass
    with get_sessionmaker()() as db:
        owner_id = db.scalar(select(User.id).where(User.username == "owner"))
        active = db.scalars(select(Model).where(Model.task == "outcome_slice", Model.status == "active")).all()
        assert [m.user_id for m in active] == [owner_id]  # bob's 5 tags train nothing; owner's model untouched
