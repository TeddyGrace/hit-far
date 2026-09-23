"""Deterministic swing metrics over pose + events. Pure functions, no models.

Bump PIPELINE_VERSION whenever a formula changes; metrics rows are keyed by it.

Conventions (face-on camera):
  * 2D metrics are measured in the image plane of the proxy video (is_estimate=False).
  * 3D metrics come from MediaPipe world landmarks, a monocular estimate with depth ambiguity
    (is_estimate=True) until calibrated against triangulated dual-camera sessions.
  * "Toward target" is inferred from the golfer's own hips at address, so mirrored video works.
  * Lateral displacements are expressed in % of shoulder width at address (no absolute scale yet).
"""

from dataclasses import dataclass

import numpy as np

from app.models import EventType
from app.pipeline.landmarks import (
    L_HIP,
    L_SHOULDER,
    NOSE,
    R_HIP,
    R_SHOULDER,
    interpolate_nans,
    joint_angle,
    sides,
    smooth,
)
from app.pipeline.posedata import PoseData

PIPELINE_VERSION = "0.1.0"

E = EventType


@dataclass
class MetricValue:
    metric_name: str
    event_ref: EventType | None
    value: float
    unit: str
    is_estimate: bool


def _line_tilt_deg(lead_pt: np.ndarray, trail_pt: np.ndarray) -> float:
    """Tilt of the lead-trail line from horizontal in the image. Positive = lead side higher."""
    dy = trail_pt[1] - lead_pt[1]  # image y grows downward
    dx = abs(trail_pt[0] - lead_pt[0])
    return float(np.degrees(np.arctan2(dy, dx)))


def _turn_deg(v_ref: np.ndarray, v: np.ndarray) -> float:
    """Rotation (degrees) about the vertical axis between two world-space vectors (x-z plane)."""
    a = np.array([v_ref[0], v_ref[2]])
    b = np.array([v[0], v[2]])
    ang = np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b))
    return float(abs(ang))


def compute_metrics(pose: PoseData, events: dict[EventType, int], handedness: str = "right") -> list[MetricValue]:
    lead, trail = sides(handedness)
    sigma = max(0.5, pose.fps * 0.005)
    kp = smooth(interpolate_nans(pose.kp2d), sigma)
    world = smooth(interpolate_nans(pose.world), sigma)
    has_world = not np.isnan(pose.world).all()
    out: list[MetricValue] = []

    def add(name: str, ev: EventType | None, value: float, unit: str, est: bool = False) -> None:
        if value is not None and np.isfinite(value):
            out.append(MetricValue(name, ev, round(float(value), 3), unit, est))

    fps = pose.fps
    a, t, i = events.get(E.address), events.get(E.top), events.get(E.impact)

    # --- Timing
    if a is not None and t is not None and i is not None and t > a and i > t:
        add("backswing_time", None, (t - a) / fps, "s")
        add("downswing_time", None, (i - t) / fps, "s")
        add("tempo_ratio", None, (t - a) / (i - t), "ratio")

    if a is None:
        return out

    # --- 2D image-plane geometry
    target_sign = np.sign(kp[a, lead.hip, 0] - kp[a, trail.hip, 0]) or 1.0
    shoulder_w = float(np.linalg.norm(kp[a, L_SHOULDER] - kp[a, R_SHOULDER]))

    for ev in (E.address, E.top, E.impact):
        f = events.get(ev)
        if f is None:
            continue
        add("shoulder_tilt", ev, _line_tilt_deg(kp[f, lead.shoulder], kp[f, trail.shoulder]), "deg")
        add("hip_tilt", ev, _line_tilt_deg(kp[f, lead.hip], kp[f, trail.hip]), "deg")
        spine = (kp[f, L_SHOULDER] + kp[f, R_SHOULDER]) / 2 - (kp[f, L_HIP] + kp[f, R_HIP]) / 2
        lateral = spine[0] * target_sign  # + = shoulders toward target
        add("spine_tilt_away_from_target", ev, np.degrees(np.arctan2(-lateral, -spine[1])), "deg")

    if shoulder_w > 1:
        hips_mid = (kp[:, L_HIP] + kp[:, R_HIP]) / 2
        for ev in (E.top, E.impact):
            f = events.get(ev)
            if f is None:
                continue
            add("head_sway_toward_target", ev, (kp[f, NOSE, 0] - kp[a, NOSE, 0]) * target_sign / shoulder_w * 100,
                "% shoulder width")
            add("head_rise", ev, (kp[a, NOSE, 1] - kp[f, NOSE, 1]) / shoulder_w * 100, "% shoulder width")
            add("hip_sway_toward_target", ev, (hips_mid[f, 0] - hips_mid[a, 0]) * target_sign / shoulder_w * 100,
                "% shoulder width")

    # --- 3D (monocular estimate)
    if has_world:
        sh0 = world[a, lead.shoulder] - world[a, trail.shoulder]
        hp0 = world[a, lead.hip] - world[a, trail.hip]
        for ev in (E.top, E.impact):
            f = events.get(ev)
            if f is None:
                continue
            s_turn = _turn_deg(sh0, world[f, lead.shoulder] - world[f, trail.shoulder])
            h_turn = _turn_deg(hp0, world[f, lead.hip] - world[f, trail.hip])
            add("shoulder_turn", ev, s_turn, "deg", True)
            add("hip_turn", ev, h_turn, "deg", True)
            if ev == E.top:
                add("x_factor", ev, s_turn - h_turn, "deg", True)
            add("lead_elbow_angle", ev,
                joint_angle(world[f, lead.shoulder], world[f, lead.elbow], world[f, lead.wrist]), "deg", True)
        add("lead_knee_flex", E.address,
            180 - joint_angle(world[a, lead.hip], world[a, lead.knee], world[a, lead.ankle]), "deg", True)
        add("trail_knee_flex", E.address,
            180 - joint_angle(world[a, trail.hip], world[a, trail.knee], world[a, trail.ankle]), "deg", True)

    return out
