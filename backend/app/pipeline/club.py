"""Stage-1 club tracking: a label-free shaft-line tracker.

The pose model already gives the hands, and the shaft is a thin straight line leaving them. For
every frame of the swing we score each direction around the grip by how strongly the image looks
like a thin line along it (a ridge: the centre differs from both sides, with a consistent sign),
then pick one direction per frame with dynamic programming so the shaft can't teleport between
frames. Confidence is how far the chosen direction's score stands out from the frame's other
directions.

This is a baseline, like the rule-based event detector. It reports direction only (not the
clubhead position or face orientation), it is weakest near impact where the shaft blurs, and it
says so through low confidence rather than guessing. Stage 2 replaces it with a learned detector
trained on its confident frames plus your corrections.
"""

import io
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.pipeline.landmarks import (
    L_ELBOW,
    L_HIP,
    L_INDEX,
    L_SHOULDER,
    L_WRIST,
    R_ELBOW,
    R_HIP,
    R_INDEX,
    R_SHOULDER,
    R_WRIST,
    interpolate_nans,
    smooth,
)
from app.pipeline.posedata import PoseData

MODEL_NAME = "shaft-line-tracker"
MODEL_VERSION = "0.1.0"

N_BINS = 180  # 2-degree directions
MAX_SPEED_DEG_S = 3500.0  # faster than any shaft rotation near impact
CONF_Z = 3.0  # ridge z-score at which confidence reaches 0.5
BLUR_DEG = 20.0  # per-frame shaft sweep at which confidence halves
MARGIN_S = 0.15  # tracked window: address - margin .. finish + margin


@dataclass
class ShaftTrack:
    angle: np.ndarray  # (T,) radians in the image plane, grip -> clubhead; NaN = not tracked
    confidence: np.ndarray  # (T,) 0..1
    grip: np.ndarray  # (T, 2) pixels
    length_px: float  # drawing length (typical shaft length relative to the torso)
    fps: float
    clubhead: np.ndarray | None = None  # (T, 2) pixels, NaN = unknown; learned detector only

    def clubhead_at(self, frame: int) -> np.ndarray | None:
        if self.clubhead is None or not 0 <= frame < len(self.clubhead):
            return None
        p = self.clubhead[frame]
        return p if np.all(np.isfinite(p)) else None

    @property
    def num_frames(self) -> int:
        return len(self.angle)

    def to_npz(self) -> bytes:
        buf = io.BytesIO()
        extra = {} if self.clubhead is None else {"clubhead": self.clubhead}
        np.savez_compressed(buf, angle=self.angle, confidence=self.confidence, grip=self.grip,
                            length_px=self.length_px, fps=self.fps, **extra)
        return buf.getvalue()

    @classmethod
    def from_npz(cls, data: bytes) -> "ShaftTrack":
        z = np.load(io.BytesIO(data))
        return cls(angle=z["angle"], confidence=z["confidence"], grip=z["grip"],
                   length_px=float(z["length_px"]), fps=float(z["fps"]),
                   clubhead=z["clubhead"] if "clubhead" in z.files else None)


def grip_points(pose: PoseData) -> np.ndarray:
    """Grip = mean of both wrists and index knuckles (the hands overlap on the club)."""
    pts = pose.kp2d[:, [L_WRIST, R_WRIST, L_INDEX, R_INDEX]]
    vis = pose.visibility[:, [L_WRIST, R_WRIST, L_INDEX, R_INDEX]]
    w = np.where(np.isnan(pts[..., 0]), 0, np.maximum(vis, 0.05))
    tot = w.sum(axis=1)
    g = np.nansum(pts * w[..., None], axis=1) / np.where(tot == 0, np.nan, tot)[:, None]
    return smooth(interpolate_nans(g), max(0.5, pose.fps * 0.004))


def torso_px(pose: PoseData, frame: int) -> float:
    sh = (pose.kp2d[frame, L_SHOULDER] + pose.kp2d[frame, R_SHOULDER]) / 2
    hp = (pose.kp2d[frame, L_HIP] + pose.kp2d[frame, R_HIP]) / 2
    v = float(np.linalg.norm(sh - hp))
    if not np.isfinite(v) or v < 10:
        sh = (pose.kp2d[:, L_SHOULDER] + pose.kp2d[:, R_SHOULDER]) / 2
        hp = (pose.kp2d[:, L_HIP] + pose.kp2d[:, R_HIP]) / 2
        v = float(np.nanmedian(np.linalg.norm(sh - hp, axis=1)))
    return v if np.isfinite(v) and v > 10 else 0.25 * pose.height


class _Scorer:
    """Ridge score for every direction around a point in one grayscale frame."""

    def __init__(self, torso: float, width: int, height: int):
        th = np.arange(N_BINS) * (2 * np.pi / N_BINS)
        self.dirs = np.stack([np.cos(th), np.sin(th)], axis=1)  # (B, 2)
        self.normals = np.stack([-np.sin(th), np.cos(th)], axis=1)
        self.radii = np.linspace(0.15 * torso, 1.9 * torso, 48)
        d = float(np.clip(0.02 * torso, 1.5, 8.0))
        self.offsets = (d, 2 * d)
        self.w, self.h = width, height

    def __call__(self, gray: np.ndarray, g: np.ndarray) -> np.ndarray:
        """gray: the frame, or the frame minus the static background."""
        pts = g[None, None, :] + self.radii[None, :, None] * self.dirs[:, None, :]  # (B, K, 2)
        best = np.zeros(N_BINS)
        for d in self.offsets:
            ridge, valid = _ridge(gray, pts, self.normals[:, None, :] * d, self.w, self.h)
            cnt = valid.sum(axis=1)
            mean = np.where(cnt >= 0.4 * pts.shape[1], (ridge * valid).sum(axis=1) / np.maximum(cnt, 1), 0.0)
            best = np.maximum(best, np.abs(mean))
        return best


def _ridge(gray: np.ndarray, pts: np.ndarray, n: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Signed ridge strength at each point (normal offset n) and whether all samples were inside."""
    samples = []
    valid = np.ones(pts.shape[:-1], dtype=bool)
    for p in (pts, pts + n, pts - n):
        xi, yi = np.rint(p[..., 0]).astype(int), np.rint(p[..., 1]).astype(int)
        valid &= (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        samples.append(gray[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)])
    # Symmetric ridge: the centre must differ from BOTH sides in the same direction, so the edge
    # of a thick limb or a step in the background doesn't count as a line.
    a, b = samples[0] - samples[1], samples[0] - samples[2]
    return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0), valid


def shaft_end(gray: np.ndarray, g: np.ndarray, angle: float, torso: float) -> np.ndarray | None:
    """Where the shaft ends along a roughly known direction (the clubhead), from the
    background-subtracted frame. None when the line is too weak to tell.

    The tracked direction is only good to a few degrees, which is enough to miss a thin shaft far
    from the hands, so first refit the line locally (direction and a small sideways offset from
    the grip), then walk out along it until it stops. Used for the learned detector's clubhead
    pseudo-labels on confidently tracked frames."""
    h, w = gray.shape
    step = max(1.0, torso / 60)
    d = float(np.clip(0.02 * torso, 1.5, 8.0))

    def profile(a: float, off: float, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.array([np.cos(a), np.sin(a)])
        nrm = np.array([-u[1], u[0]])
        pts = g[None, :] + off * nrm[None, :] + r[:, None] * u[None, :]
        strength = np.zeros(len(r))
        inside = np.ones(len(r), dtype=bool)
        for dd in (d, 2 * d):
            ridge, valid = _ridge(gray, pts, nrm * dd, w, h)
            strength = np.maximum(strength, np.abs(ridge) * valid)
            inside &= valid
        return strength, pts, inside

    near = np.arange(0.2 * torso, 1.0 * torso, step)
    best = (-1.0, angle, 0.0)
    for da in np.radians(np.arange(-8, 8.01, 0.5)):
        for off in np.arange(-0.06 * torso, 0.0601 * torso, max(1.0, 0.01 * torso)):
            s, _, ins = profile(angle + da, off, near)
            score = float(s[ins].mean()) if ins.sum() >= 0.5 * len(near) else 0.0
            if score > best[0]:
                best = (score, angle + da, off)
    _, a, off = best
    r = np.arange(0.15 * torso, 2.6 * torso, step)
    strength, pts, inside = profile(a, off, r)
    k = max(1, int(round(0.06 * torso / step)))
    sm = np.convolve(strength, np.ones(k) / k, mode="same")
    head = sm[(r <= 1.0 * torso) & inside]
    ref = float(np.percentile(head, 75)) if len(head) else 0.0
    if ref < 4.0:  # gray levels: no clear line near the hands
        return None
    gap = max(2, int(round(0.12 * torso / step)))  # a real end: the line stays gone this long
    weak = sm < 0.3 * ref
    for i in range(len(r) - gap):
        if not inside[i]:
            return None  # the shaft runs out of the frame: its end isn't visible
        if weak[i:i + gap].all():
            return None if r[i] < 0.5 * torso else pts[max(0, i - 1)]
    return None


def _exclude_forearms(z: np.ndarray, g: np.ndarray, elbows: np.ndarray) -> None:
    """The forearms are ridge-like too, but the shaft never points back up an arm."""
    for e in elbows:
        v = e - g
        if not np.all(np.isfinite(v)) or np.linalg.norm(v) < 1:
            continue
        b = int(round(np.arctan2(v[1], v[0]) % (2 * np.pi) / (2 * np.pi) * N_BINS)) % N_BINS
        for k in range(-12, 13):  # +-24 degrees
            z[(b + k) % N_BINS] = min(z[(b + k) % N_BINS], 0.0)


def _decode(z: np.ndarray, fps: float) -> np.ndarray:
    """Best direction path: maximize sum of z minus a smoothness penalty, with a hard cap on
    how far the shaft can turn between frames."""
    T = len(z)
    w = int(np.ceil(MAX_SPEED_DEG_S / fps / (360 / N_BINS)))
    w = min(w, N_BINS // 2)
    shifts = np.arange(-w, w + 1)
    pen = 0.5 * (shifts / max(w, 1)) ** 2 * 4.0
    score = z[0].copy()
    back = np.zeros((T, N_BINS), dtype=np.int16)
    for t in range(1, T):
        cand = np.stack([np.roll(score, s) - p for s, p in zip(shifts, pen)])  # value arriving at b from b - s
        k = cand.argmax(axis=0)
        back[t] = (np.arange(N_BINS) - shifts[k]) % N_BINS
        score = cand[k, np.arange(N_BINS)] + z[t]
    path = np.zeros(T, dtype=int)
    path[-1] = int(score.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def read_frames(video_path: Path, f0: int, f1: int) -> tuple[list[np.ndarray], np.ndarray | None, int, int]:
    """Lightly blurred grayscale frames f0..f1, the static background, and the frame size.

    Background (tripod camera): per-pixel median over the swing. The club never stays put long
    enough to be part of it, so subtracting it removes range clutter (mats, nets, alignment sticks)
    while keeping the shaft."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for _ in range(f0):
        cap.grab()
    frames = []
    for _ in range(f0, f1 + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0))
    cap.release()
    step = max(1, len(frames) // 40)
    background = np.median(np.stack(frames[::step]), axis=0).astype(np.float32) if frames else None
    return frames, background, width, height


def track_from_scores(z: np.ndarray, seen: np.ndarray, f0: int, T: int, grip: np.ndarray, torso: float,
                      fps: float) -> ShaftTrack:
    """Direction scores per frame (window rows, N_BINS z-scores) -> smoothed track + confidence.
    Shared by the line tracker and the learned detector (which only replaces the scores)."""
    angle = np.full(T, np.nan)
    conf = np.zeros(T)
    path = _decode(z, fps)
    zc = z[np.arange(len(path)), path]
    c = 1 / (1 + np.exp(-(zc - CONF_Z) / 0.75))
    # Motion blur: when the shaft sweeps more than ~20 degrees per frame (fast swing, low frame
    # rate) each frame shows a smeared wedge and the direction is uncertain by half the sweep.
    sweep = np.degrees(np.abs(np.gradient(np.unwrap(path * (2 * np.pi / N_BINS))))) if len(path) > 1 else np.zeros(1)
    c = c / (1 + np.exp(-(BLUR_DEG - sweep) / 3.0))
    n = len(path)
    angle[f0:f0 + n] = path * (2 * np.pi / N_BINS)
    conf[f0:f0 + n] = np.where(seen[:n], c, 0.0)
    return ShaftTrack(angle=angle, confidence=conf, grip=grip, length_px=1.9 * torso, fps=fps)


def normalize_scores(s: np.ndarray, g: np.ndarray, elbows: np.ndarray) -> np.ndarray:
    """Raw per-direction scores -> robust z-scores, with the forearm directions suppressed."""
    med = np.median(s)
    mad = np.median(np.abs(s - med)) * 1.4826 + 1e-6
    zi = (s - med) / mad
    _exclude_forearms(zi, g, elbows)
    return np.clip(zi, -5, 25)


def track_shaft(video_path: Path, pose: PoseData, window: tuple[int, int] | None = None,
                progress: Callable[[int, int], None] | None = None) -> ShaftTrack:
    T = pose.num_frames
    f0, f1 = window if window else (0, T - 1)
    f0, f1 = max(0, f0), min(T - 1, f1)
    grip = grip_points(pose)
    torso = torso_px(pose, f0)
    if f1 - f0 < 2 or np.isnan(grip[f0:f1 + 1]).all():
        return ShaftTrack(np.full(T, np.nan), np.zeros(T), grip, 1.9 * torso, pose.fps)

    frames, background, width, height = read_frames(video_path, f0, f1)
    scorer = _Scorer(torso, width or pose.width, height or pose.height)
    z = np.zeros((f1 - f0 + 1, N_BINS))
    seen = np.zeros(f1 - f0 + 1, dtype=bool)
    elbows = pose.kp2d[:, [L_ELBOW, R_ELBOW]]
    for i, gray in enumerate(frames):
        f = f0 + i
        g = grip[f]
        if not np.all(np.isfinite(g)):
            continue
        z[i] = normalize_scores(scorer(gray.astype(np.float32) - background, g), g, elbows[f])
        seen[i] = True
        if progress and i % 60 == 0:
            progress(i, f1 - f0 + 1)
    return track_from_scores(z, seen, f0, T, grip, torso, pose.fps)


def window_from_events(events: dict, num_frames: int, fps: float) -> tuple[int, int]:
    from app.models import EventType

    m = int(round(MARGIN_S * fps))
    a = events.get(EventType.address, 0)
    fin = events.get(EventType.finish, num_frames - 1)
    return max(0, a - m), min(num_frames - 1, fin + m)


def apply_corrections(track: ShaftTrack, corrections: dict[int, float]) -> ShaftTrack:
    """Your corrected frames (degrees) override the tracker at full confidence. A detected
    clubhead on a corrected frame is swung onto the corrected direction (same distance)."""
    angle, conf = track.angle.copy(), track.confidence.copy()
    head = None if track.clubhead is None else track.clubhead.copy()
    for f, deg in corrections.items():
        if 0 <= f < len(angle):
            angle[f] = np.radians(deg) % (2 * np.pi)
            conf[f] = 1.0
            if head is not None:
                r = float(np.linalg.norm(head[f] - track.grip[f]))
                if np.isfinite(r):
                    head[f] = track.grip[f] + r * np.array([np.cos(angle[f]), np.sin(angle[f])])
    return ShaftTrack(angle, conf, track.grip, track.length_px, track.fps, clubhead=head)
