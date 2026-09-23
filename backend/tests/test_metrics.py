import numpy as np
import pytest

from app.models import EventType as E
from app.pipeline import landmarks as L
from app.pipeline.metrics import compute_metrics
from tests.synthetic import FPS, expected_frames, make_swing


def _events():
    exp = expected_frames()
    return {E.address: exp["address"], E.top: exp["top"], E.impact: exp["impact"], E.finish: exp["finish"]}


def _by_key(values):
    return {(v.metric_name, v.event_ref): v for v in values}


def test_tempo():
    m = _by_key(compute_metrics(make_swing(), _events()))
    assert m[("tempo_ratio", None)].value == pytest.approx(0.9 / 0.3, rel=1e-3)
    assert m[("backswing_time", None)].value == pytest.approx(0.9, abs=1 / FPS)
    assert not m[("tempo_ratio", None)].is_estimate


def test_turn_estimates_from_world_landmarks():
    m = _by_key(compute_metrics(make_swing(turn_at_top_deg=90, hip_turn_at_top_deg=45), _events()))
    assert m[("shoulder_turn", E.top)].value == pytest.approx(90, abs=1.0)
    assert m[("hip_turn", E.top)].value == pytest.approx(45, abs=1.0)
    assert m[("x_factor", E.top)].value == pytest.approx(45, abs=1.5)
    assert m[("shoulder_turn", E.top)].is_estimate


@pytest.mark.parametrize("target_dir", [1, -1])
def test_shoulder_tilt_sign_independent_of_mirroring(target_dir):
    pose = make_swing(target_dir=target_dir)
    a = expected_frames()["address"]
    # Lead shoulder 10 px higher than trail over a 120 px span.
    pose.kp2d[:, L.L_SHOULDER, 1] -= 10
    m = _by_key(compute_metrics(pose, _events()))
    assert m[("shoulder_tilt", E.address)].value == pytest.approx(np.degrees(np.arctan2(10, 120)), abs=0.2)
    assert pose.kp2d[a, L.L_SHOULDER, 1] < pose.kp2d[a, L.R_SHOULDER, 1]


def test_head_sway_toward_target_positive():
    pose = make_swing()
    i = expected_frames()["impact"]
    pose.kp2d[i - 5 : i + 5, L.NOSE, 0] += 12  # target is +x; shoulder width is 120 px
    m = _by_key(compute_metrics(pose, _events()))
    assert m[("head_sway_toward_target", E.impact)].value == pytest.approx(10, abs=0.5)


def test_missing_events_skip_dependent_metrics():
    out = compute_metrics(make_swing(), {})
    assert out == []
