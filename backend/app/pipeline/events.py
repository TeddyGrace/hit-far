"""Rule-based swing event segmentation (v0) over the 2D pose time series.

This is a deliberately transparent baseline, registered in `models` as `rule-events` so its
outputs are attributable and it can be superseded by the trained temporal model (GolfDB-pretrained
BiLSTM/transformer) later. Designed for face-on video.

Signals:
  * hands = midpoint of both wrists, smoothed, scaled by torso length
  * lead-arm elevation = angle of shoulder->wrist from straight down (0 = hanging, 90 = horizontal)

Events:
  impact-anchored: peak hand speed p -> top = highest hands before p -> address = last still frame
  before top -> impact = hands closest to their address position between top and shortly after p
  -> finish = hands still again. The four "mid"/"toe-up" events come from arm-angle crossings (the
  club isn't tracked yet, so shaft-parallel events use arm proxies and get low confidence).
"""

from dataclasses import dataclass, field

import numpy as np

from app.models import EVENT_ORDER, EventType
from app.pipeline.landmarks import (
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    angle_between,
    interpolate_nans,
    sides,
    smooth,
)
from app.pipeline.posedata import PoseData

MODEL_NAME = "rule-events"
MODEL_VERSION = "0.1.0"
MIN_DETECTED_FRAMES = 10


class EventDetectionError(Exception):
    pass


@dataclass
class DetectedEvent:
    frame: int
    confidence: float
    method: str


@dataclass
class EventResult:
    events: dict[EventType, DetectedEvent]
    debug: dict = field(default_factory=dict)


def _first_crossing(sig: np.ndarray, lo: int, hi: int, thr: float, rising: bool) -> int | None:
    if hi <= lo:
        return None
    seg = sig[lo : hi + 1]
    above = seg >= thr
    for i in range(1, len(seg)):
        if (rising and above[i] and not above[i - 1]) or (not rising and not above[i] and above[i - 1]):
            return lo + i
    return None


def detect_events(pose: PoseData, handedness: str = "right") -> EventResult:
    T = pose.num_frames
    detected = pose.detected
    if detected.sum() < MIN_DETECTED_FRAMES:
        raise EventDetectionError(f"person detected in only {int(detected.sum())} of {T} frames")

    fps = pose.fps
    lead, trail = sides(handedness)
    kp = interpolate_nans(pose.kp2d)
    sigma = max(0.8, fps * 0.01)

    shoulders_mid = (kp[:, L_SHOULDER] + kp[:, R_SHOULDER]) / 2
    hips_mid = (kp[:, L_HIP] + kp[:, R_HIP]) / 2
    torso = float(np.nanmedian(np.linalg.norm(shoulders_mid - hips_mid, axis=-1)))
    if not np.isfinite(torso) or torso <= 1:
        raise EventDetectionError("could not establish body scale")

    hands = smooth((kp[:, L_WRIST] + kp[:, R_WRIST]) / 2, sigma) / torso  # torso lengths
    speed = np.linalg.norm(np.gradient(hands, axis=0), axis=-1) * fps  # torso lengths / s
    speed = smooth(speed, sigma)
    y = hands[:, 1]  # image y grows downward, so "higher hands" = smaller y

    down = np.array([0.0, 1.0])
    lead_elev = smooth(angle_between(kp[:, lead.wrist] - kp[:, lead.shoulder], down), sigma)
    trail_elev = smooth(angle_between(kp[:, trail.wrist] - kp[:, trail.shoulder], down), sigma)

    vis_lead = pose.visibility[:, [lead.wrist, lead.shoulder, trail.wrist]].mean(axis=1)

    p = int(np.argmax(speed))
    vmax = float(speed[p])
    still = 0.06 * vmax
    sustain = max(2, int(round(0.1 * fps)))

    # Top: highest hands in the 1.5 s before peak speed.
    lo = max(0, p - int(1.5 * fps))
    top = lo + int(np.argmin(y[lo : p + 1])) if p > lo else p

    # Address: last sustained-still frame before top, with hands in the lower half of their range.
    lo_a = max(0, top - int(2.5 * fps))
    y_low = float(y[lo_a : top + 1].max())
    mid_y = (y[top] + y_low) / 2
    address, address_method = None, "still-before-top"
    for f in range(top - 1, lo_a + sustain - 2, -1):
        window = speed[max(0, f - sustain + 1) : f + 1]
        if y[f] > mid_y and np.all(window < still):
            address = f
            break
    if address is None:
        address, address_method = lo_a, "fallback-window-start"
    else:
        # Takeaways start slowly: walk back to where the hands are genuinely at rest.
        while address > lo_a and speed[address] > 0.015 * vmax:
            address -= 1

    # Impact: hands return closest to address position, between top and shortly after peak speed.
    hi_i = min(T - 1, p + max(sustain, (p - top)))
    seg = np.linalg.norm(hands[top : hi_i + 1] - hands[address], axis=-1)
    impact = top + int(np.argmin(seg)) if hi_i > top else p

    # Finish: first sustained-still frame after impact.
    finish, finish_method = None, "still-after-impact"
    for f in range(impact + sustain, T):
        if np.all(speed[f - sustain + 1 : f + 1] < still) and y[f] < y[impact]:
            finish = f - sustain + 1
            break
    if finish is None:
        finish, finish_method = T - 1, "fallback-last-frame"

    elev0 = float(lead_elev[address])
    toe_thr = elev0 + (90.0 - elev0) * 0.5

    def crossing_or_fraction(sig, a, b, thr, rising, frac, name):
        f = _first_crossing(sig, a, b, thr, rising)
        if f is not None:
            return f, f"{name}-crossing"
        return int(round(a + frac * (b - a))), "fallback-time-fraction"

    mid_bs, mid_bs_m = crossing_or_fraction(lead_elev, address, top, 90.0, True, 0.6, "lead-arm-90")
    toe_up, toe_up_m = crossing_or_fraction(lead_elev, address, mid_bs, toe_thr, True, 0.55, "lead-arm-mid")
    mid_ds, mid_ds_m = crossing_or_fraction(lead_elev, top, impact, 90.0, False, 0.5, "lead-arm-90")
    mid_ft, mid_ft_m = crossing_or_fraction(trail_elev, impact, finish, toe_thr, True, 0.35, "trail-arm-mid")

    raw = {
        EventType.address: (address, address_method, 0.8),
        EventType.toe_up: (toe_up, toe_up_m, 0.5),
        EventType.mid_backswing: (mid_bs, mid_bs_m, 0.65),
        EventType.top: (top, "hands-highest-before-peak-speed", 0.85),
        EventType.mid_downswing: (mid_ds, mid_ds_m, 0.65),
        EventType.impact: (impact, "hands-return-to-address", 0.85),
        EventType.mid_follow_through: (mid_ft, mid_ft_m, 0.5),
        EventType.finish: (finish, finish_method, 0.7),
    }

    # A real swing has hands moving several torso lengths per second at peak.
    swing_likeness = float(np.clip(vmax / 4.0, 0.2, 1.0))

    events: dict[EventType, DetectedEvent] = {}
    prev = 0
    for et in EVENT_ORDER:
        f, method, base = raw[et]
        f = int(np.clip(max(f, prev), 0, T - 1))  # enforce temporal order
        prev = f
        if method.startswith("fallback"):
            base = min(base, 0.25)
        w = slice(max(0, f - 2), min(T, f + 3))
        vis = float(np.mean(vis_lead[w])) if detected[w].any() else 0.0
        events[et] = DetectedEvent(frame=f, confidence=round(base * vis * swing_likeness, 3), method=method)

    return EventResult(
        events=events,
        debug={"peak_speed_frame": p, "peak_speed_torso_per_s": round(vmax, 2), "torso_px": round(torso, 1)},
    )
