"""Synthetic face-on swing: a stick figure whose hands sweep through a known arc on a known timeline."""

import numpy as np

from app.pipeline import landmarks as L
from app.pipeline.posedata import PoseData

FPS = 120.0
ADDRESS_S, TOP_S, IMPACT_S, FINISH_S, END_S = 0.5, 1.4, 1.7, 2.2, 3.0


def _phi(t: float, a=ADDRESS_S, top=TOP_S, imp=IMPACT_S, fin=FINISH_S) -> float:
    """Signed hand angle (deg) around the shoulder centre. 0 = hanging, - = trail side, + = target side."""
    if t <= a:
        return 0.0
    if t <= top:  # smooth backswing
        u = (t - a) / (top - a)
        return -170.0 * (1 - np.cos(np.pi * u)) / 2
    if t <= imp:  # accelerating downswing: peak speed at impact
        u = (t - top) / (imp - top)
        return -170.0 * (1 - u**2)
    if t <= fin:  # decelerating follow-through
        u = (t - imp) / (fin - imp)
        return 160.0 * (1 - (1 - u) ** 2)
    return 160.0


def make_swing(target_dir: int = 1, turn_at_top_deg: float = 90.0, hip_turn_at_top_deg: float = 45.0,
               times: tuple[float, float, float, float, float] | None = None, fps: float = FPS,
               shoulders_open_deg: float = 0.0, hips_open_deg: float = 0.0,
               wrist_bow_deg: float = 0.0, hand_roll_top_deg: float = -60.0,
               hand_roll_impact_deg: float = 0.0) -> PoseData:
    """target_dir=+1 means the target is toward +x in the image (lead side on the right of the image).
    times = (address, top, impact, finish, end) in seconds. *_open_deg = rotation past address
    toward the target reached at impact (ramped in over the downswing). The lead hand rolls about
    the forearm (relative to the shoulder line) to hand_roll_top_deg at the top and to
    hand_roll_impact_deg at impact; wrist_bow_deg flexes it toward the palm throughout."""
    a_s, top_s, imp_s, fin_s, end_s = times or (ADDRESS_S, TOP_S, IMPACT_S, FINISH_S, END_S)
    phi_at = lambda t: _phi(t, a_s, top_s, imp_s, fin_s)  # noqa: E731
    T = int(end_s * fps)
    kp = np.zeros((T, 33, 2))
    world = np.zeros((T, 33, 3))
    cx, sh_y, hip_y, arm = 500.0, 400.0, 600.0, 180.0
    half_sh, half_hip = 60.0, 40.0
    lead_sh, trail_sh = L.L_SHOULDER, L.R_SHOULDER  # right-handed golfer: lead = left
    for f in range(T):
        t = f / fps
        phi = np.radians(phi_at(t))
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
        for j in (L.L_INDEX, L.R_INDEX, L.L_PINKY, L.R_PINKY):
            kp[f, j] = hands
        kp[f, L.L_ELBOW] = (kp[f, lead_sh] + hands) / 2
        kp[f, L.R_ELBOW] = (kp[f, trail_sh] + hands) / 2

        # World: rotate shoulders/hips about vertical proportionally to backswing progress.
        prog = max(0.0, -phi_at(t)) / 170.0 if t <= imp_s else 0.0
        down = 0.0 if t <= top_s else (1.0 if t >= imp_s else ((t - top_s) / (imp_s - top_s)) ** 2)
        for (li, ri, half, turn, opened) in ((lead_sh, trail_sh, 0.2, turn_at_top_deg, shoulders_open_deg),
                                             (L.L_HIP, L.R_HIP, 0.15, hip_turn_at_top_deg, hips_open_deg)):
            a = np.radians(turn * prog - opened * down)
            v = np.array([np.cos(a), 0.0, np.sin(a)]) * half
            y = -0.5 if half == 0.2 else 0.0
            world[f, li] = [v[0], y, v[2]]
            world[f, ri] = [-v[0], y, -v[2]]
        world[f, L.L_ELBOW] = world[f, lead_sh] + [0, 0.3, 0]
        world[f, L.L_WRIST] = world[f, lead_sh] + [0, 0.6, 0]
        # Lead (left) hand: fingers continue down the forearm, back of the hand toward the camera
        # at address; index/pinky knuckles either side. Rotated about the forearm (vertical) with
        # the torso plus the hand's own roll.
        torso = np.radians(turn_at_top_deg * prog - shoulders_open_deg * down)
        if t <= top_s:
            roll = hand_roll_top_deg * prog
        elif t <= imp_s:
            roll = hand_roll_top_deg + (hand_roll_impact_deg - hand_roll_top_deg) * down
        else:
            roll = hand_roll_impact_deg
        th = torso + np.radians(roll)
        b = np.radians(wrist_bow_deg)

        def rot(v, th=th):
            return np.array([v[0] * np.cos(th) - v[2] * np.sin(th), v[1], v[0] * np.sin(th) + v[2] * np.cos(th)])

        hand_dir = rot(np.array([0.0, np.cos(b), np.sin(b)]))
        lateral = rot(np.array([-1.0, 0.0, 0.0]))
        world[f, L.L_INDEX] = world[f, L.L_WRIST] + 0.08 * hand_dir + 0.04 * lateral
        world[f, L.L_PINKY] = world[f, L.L_WRIST] + 0.08 * hand_dir - 0.04 * lateral
        for knee, hip, ankle in ((L.L_KNEE, L.L_HIP, L.L_ANKLE), (L.R_KNEE, L.R_HIP, L.R_ANKLE)):
            world[f, knee] = world[f, hip] + [0, 0.45, 0.1]
            world[f, ankle] = world[f, knee] + [0, 0.45, -0.1]
    vis = np.ones((T, 33)) * 0.95
    return PoseData(kp2d=kp, visibility=vis, world=world, fps=fps, width=1000, height=1200)


def random_swing(rng: np.random.Generator, fps: float = 30.0) -> tuple[PoseData, list[int]]:
    """Synthetic swing with random tempo/timing, plus its 8 ground-truth event frames (derived from
    the arm-angle trajectory the same way GolfDB defines them, approximately)."""
    a = rng.uniform(0.3, 0.8)
    top = a + rng.uniform(0.7, 1.1)
    imp = top + rng.uniform(0.22, 0.35)
    fin = imp + rng.uniform(0.4, 0.7)
    end = fin + rng.uniform(0.2, 0.6)
    pose = make_swing(target_dir=int(rng.choice([-1, 1])), times=(a, top, imp, fin, end), fps=fps)
    phi = np.array([_phi(f / fps, a, top, imp, fin) for f in range(pose.num_frames)])
    fr = lambda s: int(round(s * fps))  # noqa: E731
    back = np.arange(fr(a), fr(top) + 1)
    down = np.arange(fr(top), fr(imp) + 1)
    thru = np.arange(fr(imp), fr(fin) + 1)
    first = lambda idx, cond: int(idx[np.argmax(cond)]) if cond.any() else int(idx[len(idx) // 2])  # noqa: E731
    events = [
        fr(a),
        first(back, phi[back] <= -45),  # toe-up proxy
        first(back, phi[back] <= -90),  # mid-backswing
        fr(top),
        first(down, phi[down] >= -90),  # mid-downswing
        fr(imp),
        first(thru, phi[thru] >= 90),  # mid-follow-through
        fr(fin),
    ]
    for i in range(1, 8):
        events[i] = max(events[i], events[i - 1] + 1)
    pose.visibility[:] = rng.uniform(0.8, 1.0)
    pose.kp2d = pose.kp2d + rng.normal(0, 2.0, pose.kp2d.shape)
    return pose, events


def expected_frames() -> dict[str, int]:
    return {k: int(round(v * FPS)) for k, v in
            {"address": ADDRESS_S, "top": TOP_S, "impact": IMPACT_S, "finish": FINISH_S}.items()}


def shaft_path(pose: PoseData, hinge_top_deg: float = 90.0, lean_impact_deg: float = 0.0) -> np.ndarray:
    """Known image-plane shaft direction (radians, grip -> clubhead) for a make_swing() pose: along
    the arm (shoulder centre -> hands), cocked by up to hinge_top_deg at the top and released by
    impact; lean_impact_deg tilts it at impact (+ = hands ahead, i.e. clubhead trailing)."""
    sh = (pose.kp2d[:, L.L_SHOULDER] + pose.kp2d[:, L.R_SHOULDER]) / 2
    arm = pose.kp2d[:, L.L_WRIST] - sh
    base = np.arctan2(arm[:, 1], arm[:, 0])
    reach = np.linalg.norm(arm, axis=1)
    # Backswing progress from how far the hands have swung up (0 at address, 1 at the top).
    height = np.clip((pose.kp2d[:, L.L_WRIST, 1] - sh[:, 1]) / reach, -1, 1)
    prog = np.clip((1 - height) / 2 * 1.2, 0, 1)
    target_sign = np.sign(pose.kp2d[0, L.L_HIP, 0] - pose.kp2d[0, L.R_HIP, 0]) or 1.0
    # Cock toward "clubhead up": rotate against the direction the hands travel in the backswing.
    top = int(np.argmin(pose.kp2d[:, L.L_WRIST, 1]))
    impact_like = top + int(np.argmax(np.linalg.norm(np.diff(pose.kp2d[top:, L.L_WRIST], axis=0), axis=1)))
    ang = base.copy()
    t = np.arange(len(base))
    cock = np.where(t <= top, prog, np.clip(1 - (t - top) / max(1, impact_like - top), 0, 1))
    ang += np.radians(hinge_top_deg) * cock * target_sign
    near = np.exp(-0.5 * ((t - impact_like) / max(1.0, 0.03 * pose.fps)) ** 2)
    ang += np.radians(lean_impact_deg) * near * target_sign
    return ang


def render_video(path, pose: PoseData, shaft: np.ndarray, clutter: int = 0, seed: int = 0,
                 scale: float = 0.5, shaft_gray: int = 30, motion_blur: bool = False) -> PoseData:
    """Render a stick figure with a thin shaft to an H.264 file. Returns the pose rescaled to the
    video's pixels (what the pipeline would see)."""
    import subprocess

    import cv2

    from app.pipeline.ingest import ffmpeg_bin

    rng = np.random.default_rng(seed)
    W, H = int(pose.width * scale), int(pose.height * scale)
    W -= W % 2
    H -= H % 2
    kp = pose.kp2d * scale
    bg = np.full((H, W, 3), 110, np.uint8)
    bg = np.clip(bg.astype(int) + rng.normal(0, 6, bg.shape), 0, 255).astype(np.uint8)
    for _ in range(clutter):
        p1 = rng.uniform([0, 0], [W, H]).astype(int)
        p2 = rng.uniform([0, 0], [W, H]).astype(int)
        cv2.line(bg, tuple(int(v) for v in p1), tuple(int(v) for v in p2), (int(rng.uniform(60, 200)),) * 3, 2)
    torso = float(np.linalg.norm((kp[0, L.L_SHOULDER] + kp[0, L.R_SHOULDER]) / 2 - (kp[0, L.L_HIP] + kp[0, L.R_HIP]) / 2))
    proc = subprocess.Popen(
        [ffmpeg_bin(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(pose.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)],
        stdin=subprocess.PIPE)
    limbs = [(L.L_SHOULDER, L.R_SHOULDER), (L.L_HIP, L.R_HIP), (L.L_SHOULDER, L.L_HIP), (L.R_SHOULDER, L.R_HIP),
             (L.L_HIP, L.L_KNEE), (L.L_KNEE, L.L_ANKLE), (L.R_HIP, L.R_KNEE), (L.R_KNEE, L.R_ANKLE),
             (L.L_SHOULDER, L.L_ELBOW), (L.L_ELBOW, L.L_WRIST), (L.R_SHOULDER, L.R_ELBOW), (L.R_ELBOW, L.R_WRIST)]
    for f in range(pose.num_frames):
        img = bg.copy()
        for a, b in limbs:
            cv2.line(img, tuple(int(v) for v in kp[f, a]), tuple(int(v) for v in kp[f, b]), (150, 170, 200),
                     max(4, int(torso * 0.1)), cv2.LINE_AA)
        cv2.circle(img, tuple(int(v) for v in kp[f, L.NOSE]), int(torso * 0.18), (150, 170, 200), -1)
        g = kp[f, L.L_WRIST]
        # Motion blur: the shaft smeared over the angle it sweeps during the exposure (a full
        # frame interval), like a real camera near impact.
        prev = shaft[f - 1] if f > 0 else shaft[f]
        sweep = np.linspace(prev, shaft[f], 12) if motion_blur else [shaft[f]]
        acc = np.zeros(img.shape, np.float32)
        for a in sweep:
            layer = img.copy()
            tip = g + 1.9 * torso * np.array([np.cos(a), np.sin(a)])
            cv2.line(layer, tuple(int(v) for v in g), tuple(int(v) for v in tip), (shaft_gray,) * 3, 2, cv2.LINE_AA)
            acc += layer
        img = (acc / len(sweep)).astype(np.uint8)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()
    return PoseData(kp2d=kp, visibility=pose.visibility, world=pose.world, fps=pose.fps, width=W, height=H)
