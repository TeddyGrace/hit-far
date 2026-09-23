"""Synthetic face-on swing: a stick figure whose hands sweep through a known arc on a known timeline."""

import numpy as np

from app.pipeline import landmarks as L
from app.pipeline.posedata import PoseData

FPS = 120.0
ADDRESS_S, TOP_S, IMPACT_S, FINISH_S, END_S = 0.5, 1.4, 1.7, 2.2, 3.0


def _phi(t: float) -> float:
    """Signed hand angle (deg) around the shoulder centre. 0 = hanging, - = trail side, + = target side."""
    if t <= ADDRESS_S:
        return 0.0
    if t <= TOP_S:  # smooth backswing
        u = (t - ADDRESS_S) / (TOP_S - ADDRESS_S)
        return -170.0 * (1 - np.cos(np.pi * u)) / 2
    if t <= IMPACT_S:  # accelerating downswing: peak speed at impact
        u = (t - TOP_S) / (IMPACT_S - TOP_S)
        return -170.0 * (1 - u**2)
    if t <= FINISH_S:  # decelerating follow-through
        u = (t - IMPACT_S) / (FINISH_S - IMPACT_S)
        return 160.0 * (1 - (1 - u) ** 2)
    return 160.0


def make_swing(target_dir: int = 1, turn_at_top_deg: float = 90.0, hip_turn_at_top_deg: float = 45.0) -> PoseData:
    """target_dir=+1 means the target is toward +x in the image (lead side on the right of the image)."""
    T = int(END_S * FPS)
    kp = np.zeros((T, 33, 2))
    world = np.zeros((T, 33, 3))
    cx, sh_y, hip_y, arm = 500.0, 400.0, 600.0, 180.0
    half_sh, half_hip = 60.0, 40.0
    lead_sh, trail_sh = L.L_SHOULDER, L.R_SHOULDER  # right-handed golfer: lead = left
    for f in range(T):
        t = f / FPS
        phi = np.radians(_phi(t))
        s = target_dir
        kp[f, lead_sh] = [cx + s * half_sh, sh_y]
        kp[f, trail_sh] = [cx - s * half_sh, sh_y]
        kp[f, L.L_HIP] = [cx + s * half_hip, hip_y]
        kp[f, L.R_HIP] = [cx - s * half_hip, hip_y]
        kp[f, L.L_KNEE] = [cx + s * half_hip, hip_y + 150]
        kp[f, L.R_KNEE] = [cx - s * half_hip, hip_y + 150]
        kp[f, L.L_ANKLE] = [cx + s * half_hip, hip_y + 300]
        kp[f, L.R_ANKLE] = [cx - s * half_hip, hip_y + 300]
        kp[f, L.NOSE] = [cx, sh_y - 80]
        hands = np.array([cx + s * arm * np.sin(phi), sh_y + arm * np.cos(phi)])
        kp[f, L.L_WRIST] = kp[f, L.R_WRIST] = hands
        kp[f, L.L_ELBOW] = (kp[f, lead_sh] + hands) / 2
        kp[f, L.R_ELBOW] = (kp[f, trail_sh] + hands) / 2

        # World: rotate shoulders/hips about vertical proportionally to backswing progress.
        prog = max(0.0, -_phi(t)) / 170.0 if t <= IMPACT_S else 0.0
        for (li, ri, half, turn) in ((lead_sh, trail_sh, 0.2, turn_at_top_deg),
                                     (L.L_HIP, L.R_HIP, 0.15, hip_turn_at_top_deg)):
            a = np.radians(turn * prog)
            v = np.array([np.cos(a), 0.0, np.sin(a)]) * half
            y = -0.5 if half == 0.2 else 0.0
            world[f, li] = [v[0], y, v[2]]
            world[f, ri] = [-v[0], y, -v[2]]
        world[f, L.L_ELBOW] = world[f, lead_sh] + [0, 0.3, 0]
        world[f, L.L_WRIST] = world[f, lead_sh] + [0, 0.6, 0]
        for knee, hip, ankle in ((L.L_KNEE, L.L_HIP, L.L_ANKLE), (L.R_KNEE, L.R_HIP, L.R_ANKLE)):
            world[f, knee] = world[f, hip] + [0, 0.45, 0.1]
            world[f, ankle] = world[f, knee] + [0, 0.45, -0.1]
    vis = np.ones((T, 33)) * 0.95
    return PoseData(kp2d=kp, visibility=vis, world=world, fps=FPS, width=1000, height=1200)


def expected_frames() -> dict[str, int]:
    return {k: int(round(v * FPS)) for k, v in
            {"address": ADDRESS_S, "top": TOP_S, "impact": IMPACT_S, "finish": FINISH_S}.items()}
