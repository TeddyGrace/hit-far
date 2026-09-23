import pytest

from app import worker
from app.diagnosis import claude as claude_mod
from app.diagnosis.catalog import FAULTS, upsert_catalog
from app.diagnosis.rules import MetricRow, evaluate
from app.diagnosis.service import verify
from tests.test_api import _upload, fake_pose, sample_video  # noqa: F401  (fixtures)


def _m(name, event, value, unit="deg", est=False):
    return MetricRow(name, event, value, unit, est)


# --- Rules -------------------------------------------------------------------------------------


def test_rules_fire_with_margin_and_estimate_flag():
    hits = evaluate([
        _m("hip_sway_toward_target", "top", -22.0, "% shoulder width"),
        _m("shoulder_turn", "top", 60.0, est=True),
        _m("tempo_ratio", None, 3.1, "ratio"),
    ])
    by_fault = {h.fault: h for h in hits}
    assert set(by_fault) == {"lateral_sway", "restricted_shoulder_turn"}
    assert by_fault["lateral_sway"].margin == pytest.approx(7.0)
    assert by_fault["restricted_shoulder_turn"].is_estimate


def test_abs_rule_and_missing_metrics():
    assert [h.fault for h in evaluate([_m("head_sway_toward_target", "top", -18, "%")])] == ["head_movement"]
    assert evaluate([]) == []


def test_unmeasurable_faults_have_no_rules_but_list_needs():
    f = {x.name: x for x in FAULTS}
    assert not f["open_clubface"].assessable and "club tracking" in f["open_clubface"].needs


# --- Citation verification ---------------------------------------------------------------------


def _output(**kw):
    base = dict(
        summary="s", cannot_assess=[], suggested_checks=[], narrative="See [frame 10] and [frame 999].",
        faults=[dict(fault="lateral_sway", likelihood=0.7, explanation="e", visual_observation=None,
                     frames=[10, 500],
                     evidence=[dict(metric="hip_sway_toward_target", event="top", value=-22.0, note="n"),
                               dict(metric="made_up_metric", event="top", value=1.0, note="n"),
                               dict(metric="shoulder_turn", event="top", value=95.0, note="n")])],
    )
    base.update(kw)
    return claude_mod.DiagnosisOutput.model_validate(base)


def test_verify_flags_invented_metrics_wrong_values_and_bad_frames():
    metrics = [_m("hip_sway_toward_target", "top", -22.1, "%"), _m("shoulder_turn", "top", 60.0, est=True)]
    out = verify(_output(), metrics, num_frames=360)
    problems = out["faults"][0]["unverified"]
    assert any("made_up_metric" in p for p in problems)
    assert any("shoulder_turn" in p and "measured 60.0" in p for p in problems)
    assert any("frame 500" in p for p in problems)
    assert not any("hip_sway" in p for p in problems)  # within tolerance
    assert out["narrative_unverified_frames"] == [999]


def test_catalog_upsert_is_idempotent(engine):
    from sqlalchemy import func, select

    from app.db import get_sessionmaker
    from app.models import FaultLabel

    with get_sessionmaker()() as db:
        upsert_catalog(db)
        upsert_catalog(db)
        assert db.scalar(select(func.count(FaultLabel.id))) == len(FAULTS)


# --- API ---------------------------------------------------------------------------------------


@pytest.fixture
def processed_swing(authed, sample_video, fake_pose):  # noqa: F811
    sess, video = _upload(authed, sample_video)
    worker.run_once()
    return authed.get(f"/api/sessions/{sess['id']}").json()["videos"][0]["swing_id"]


@pytest.fixture
def fake_claude(monkeypatch):
    calls = []

    def fake(system, content, schema):
        calls.append({"system": system, "content": content, "schema": schema})
        out = _output(narrative="Hips drift at [frame 150].")
        return claude_mod.ClaudeResult(output=out, model="claude-opus-5", usage={"input_tokens": 1})

    monkeypatch.setattr(claude_mod, "call_claude", fake)
    return calls


def test_diagnose_and_verdicts(authed, processed_swing, fake_claude):
    r = authed.post(f"/api/swings/{processed_swing}/diagnoses", json={"symptom_text": "I'm slicing"})
    assert r.status_code == 201, r.text
    dx = r.json()
    assert dx["error"] is None and not dx["stale"]
    assert dx["output"]["faults"][0]["fault"] == "lateral_sway"
    assert dx["served_model"] == "claude-opus-5"

    # The model saw the metrics, events, key frames (images) and the symptom.
    content = fake_claude[0]["content"]
    assert any(c["type"] == "image" for c in content)
    assert "I'm slicing" in content[-1]["text"]
    assert "Fault catalog" in fake_claude[0]["system"]

    r = authed.put(f"/api/diagnoses/{dx['id']}/faults/lateral_sway", json={"verdict": "confirmed"})
    assert r.json()["verdicts"] == {"self": {"lateral_sway": "confirmed"}}
    r = authed.put(f"/api/diagnoses/{dx['id']}/faults/quick_tempo",
                   json={"verdict": "rejected", "labeled_by": "instructor"})
    assert r.json()["verdicts"]["instructor"] == {"quick_tempo": "rejected"}
    r = authed.put(f"/api/diagnoses/{dx['id']}/faults/lateral_sway", json={"verdict": None})
    assert r.json()["verdicts"]["self"] == {}
    assert authed.put(f"/api/diagnoses/{dx['id']}/faults/nope", json={"verdict": "confirmed"}).status_code == 404

    # Verdicts are also labels (training data), including who labeled.
    from sqlalchemy import select

    from app.db import get_sessionmaker
    from app.models import Label

    with get_sessionmaker()() as db:
        labels = db.scalars(select(Label).where(Label.task == "fault").order_by(Label.created_at)).all()
    assert [(lb.corrected_value["fault"], lb.corrected_value["verdict"], lb.corrected_value["labeled_by"])
            for lb in labels] == [("lateral_sway", "confirmed", "self"), ("quick_tempo", "rejected", "instructor"),
                                  ("lateral_sway", None, "self")]
    assert labels[0].corrected_value["proposed_by_model"] is True
    assert labels[1].corrected_value["proposed_by_model"] is False

    # Moving an event changes the metrics -> the diagnosis is flagged stale.
    swing = authed.get(f"/api/swings/{processed_swing}").json()
    top = next(e for e in swing["events"] if e["event_type"] == "top")
    authed.put(f"/api/swings/{processed_swing}/events/top", json={"frame_index": top["frame_index"] - 10})
    history = authed.get(f"/api/swings/{processed_swing}/diagnoses").json()
    assert len(history) == 1 and history[0]["stale"]


def test_missing_api_key_is_503_and_nothing_stored(authed, processed_swing, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    r = authed.post(f"/api/swings/{processed_swing}/diagnoses", json={"symptom_text": "x"})
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]
    assert authed.get(f"/api/swings/{processed_swing}/diagnoses").json() == []


def test_claude_error_is_stored(authed, processed_swing, monkeypatch):
    def refuse(*a, **k):
        raise claude_mod.DiagnosisError("Claude declined this request")

    monkeypatch.setattr(claude_mod, "call_claude", refuse)
    r = authed.post(f"/api/swings/{processed_swing}/diagnoses", json={"symptom_text": "x"})
    assert r.status_code == 201 and r.json()["error"] == "Claude declined this request"


def test_fault_catalog_endpoint(authed):
    faults = {f["name"]: f for f in authed.get("/api/faults").json()}
    assert faults["quick_tempo"]["rules"] == ["tempo_ratio < 2.5"]
    assert not faults["over_the_top"]["assessable"]


# --- Real SDK request/response path against a mock HTTP transport ------------------------------


def _sdk_client(handler):
    import anthropic
    import httpx2 as httpx

    return anthropic.Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)),
                               max_retries=0)


def _message(stop_reason="end_turn", text=None, **extra):
    import json as _json

    body = {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5",
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 20},
        "content": [{"type": "text", "text": text or _json.dumps(_output().model_dump())}],
    }
    body.update(extra)
    return body


def test_sdk_request_shape_and_parsing():
    import json as _json

    import httpx2 as httpx

    from app.diagnosis.prompt import output_schema

    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = _json.loads(request.content)
        seen["beta"] = request.headers.get("anthropic-beta", "")
        return httpx.Response(200, json=_message())

    res = claude_mod.call_claude("SYS", [{"type": "text", "text": "hi"}], output_schema(), client=_sdk_client(handler))
    assert res.output.faults[0].fault == "lateral_sway" and res.model == "claude-opus-5"
    b = seen["body"]
    # Default: efficient model, no fallback parameter (only sent for models documented to accept it).
    assert b["model"] == "claude-sonnet-5" and "fallbacks" not in b
    assert "server-side-fallback" not in seen["beta"]
    assert b["thinking"] == {"type": "adaptive"}
    assert b["output_config"]["format"]["type"] == "json_schema" and b["output_config"]["effort"] == "medium"
    assert b["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_sdk_request_enables_fallback_for_opus(monkeypatch):
    import json as _json

    import httpx2 as httpx

    from app.config import get_settings
    from app.diagnosis.prompt import output_schema

    monkeypatch.setattr(get_settings(), "diagnosis_model", "claude-opus-5")
    seen = {}

    def handler(request):
        seen["body"] = _json.loads(request.content)
        seen["beta"] = request.headers.get("anthropic-beta", "")
        return httpx.Response(200, json=_message())

    claude_mod.call_claude("SYS", [], output_schema(), client=_sdk_client(handler))
    assert seen["body"]["model"] == "claude-opus-5" and seen["body"]["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in seen["beta"]


def test_sdk_refusal_and_malformed_output():
    import httpx2 as httpx

    from app.diagnosis.prompt import output_schema

    refusal = _message(stop_reason="refusal", text="",
                       stop_details={"type": "refusal", "category": None, "explanation": "nope"})
    with pytest.raises(claude_mod.DiagnosisError, match="declined.*nope"):
        claude_mod.call_claude("S", [], output_schema(), client=_sdk_client(lambda r: httpx.Response(200, json=refusal)))
    bad = _message(text='{"summary": 1}')
    with pytest.raises(claude_mod.DiagnosisError, match="malformed"):
        claude_mod.call_claude("S", [], output_schema(), client=_sdk_client(lambda r: httpx.Response(200, json=bad)))
    err = {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
    with pytest.raises(claude_mod.DiagnosisError, match="400"):
        claude_mod.call_claude("S", [], output_schema(), client=_sdk_client(lambda r: httpx.Response(400, json=err)))
