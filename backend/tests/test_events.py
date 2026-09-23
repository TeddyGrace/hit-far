import numpy as np
import pytest

from app.models import EVENT_ORDER, EventType
from app.pipeline.events import EventDetectionError, detect_events
from tests.synthetic import FPS, expected_frames, make_swing


@pytest.mark.parametrize("target_dir", [1, -1])
def test_key_events_found_on_synthetic_swing(target_dir):
    res = detect_events(make_swing(target_dir=target_dir))
    exp = expected_frames()
    tol = int(0.05 * FPS)
    assert abs(res.events[EventType.address].frame - exp["address"]) <= tol
    assert abs(res.events[EventType.top].frame - exp["top"]) <= tol
    assert abs(res.events[EventType.impact].frame - exp["impact"]) <= tol
    assert abs(res.events[EventType.finish].frame - exp["finish"]) <= int(0.1 * FPS)


def test_events_are_ordered_and_arm_crossings_used():
    res = detect_events(make_swing())
    frames = [res.events[e].frame for e in EVENT_ORDER]
    assert frames == sorted(frames)
    assert res.events[EventType.mid_backswing].method.endswith("crossing")
    assert res.events[EventType.mid_downswing].method.endswith("crossing")
    assert all(0 < res.events[e].confidence <= 1 for e in EVENT_ORDER)


def test_missing_person_raises():
    pose = make_swing()
    pose.kp2d[:] = np.nan
    with pytest.raises(EventDetectionError):
        detect_events(pose)


def test_tolerates_dropped_frames():
    pose = make_swing()
    pose.kp2d[150:160] = np.nan  # occlusion near the top
    pose.visibility[150:160] = 0
    res = detect_events(pose)
    assert abs(res.events[EventType.impact].frame - expected_frames()["impact"]) <= int(0.05 * FPS)
