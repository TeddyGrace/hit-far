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

    @property
    def num_frames(self) -> int:
        return len(self.angle)

    def to_npz(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(buf, angle=self.angle, confidence=self.confidence, grip=self.grip,
                            length_px=self.length_px, fps=self.fps)
        return buf.getvalue()

    @classmethod
    def from_npz(cls, data: bytes) -> "ShaftTrack":
        z = np.load(io.BytesIO(data))
        return cls(angle=z["angle"], confidence=z["confidence"], grip=z["grip"],
                   length_px=float(z["length_px"]), fps=float(z["fps"]))


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
            n = self.normals[:, None, :] * d
            samples = []
            valid = np.ones(pts.shape[:2], dtype=bool)
            for p in (pts, pts + n, pts - n):
                xi, yi = np.rint(p[..., 0]).astype(int), np.rint(p[..., 1]).astype(int)
                ok = (xi >= 0) & (xi < self.w) & (yi >= 0) & (yi < self.h)
                valid &= ok
                samples.append(gray[np.clip(yi, 0, self.h - 1), np.clip(xi, 0, self.w - 1)])
            # Symmetric ridge: the centre must differ from BOTH sides in the same direction, so
            # the edge of a thick limb or a step in the background doesn't count as a line.
            a, b = samples[0] - samples[1], samples[0] - samples[2]
            ridge = np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)
            cnt = valid.sum(axis=1)
            mean = np.where(cnt >= 0.4 * pts.shape[1], (ridge * valid).sum(axis=1) / np.maximum(cnt, 1), 0.0)
            best = np.maximum(best, np.abs(mean))
        return best


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


def track_shaft(video_path: Path, pose: PoseData, window: tuple[int, int] | None = None,
                progress: Callable[[int, int], None] | None = None) -> ShaftTrack:
    import cv2

    T = pose.num_frames
    f0, f1 = window if window else (0, T - 1)
    f0, f1 = max(0, f0), min(T - 1, f1)
    grip = grip_points(pose)
    torso = torso_px(pose, f0)
    angle = np.full(T, np.nan)
    conf = np.zeros(T)
    if f1 - f0 < 2 or np.isnan(grip[f0:f1 + 1]).all():
        return ShaftTrack(angle, conf, grip, 1.9 * torso, pose.fps)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or pose.width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or pose.height
    scorer = _Scorer(torso, width, height)
    for _ in range(f0):
        cap.grab()
    frames = []
    for _ in range(f0, f1 + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0))
    cap.release()
    # Static background (tripod camera): per-pixel median over the swing. The club never stays
    # put long enough to be part of it, so subtracting it removes range clutter (mats, nets,
    # alignment sticks) while keeping the shaft.
    step = max(1, len(frames) // 40)
    background = np.median(np.stack(frames[::step]), axis=0).astype(np.float32) if frames else None
    z = np.zeros((f1 - f0 + 1, N_BINS))
    seen = np.zeros(f1 - f0 + 1, dtype=bool)
    elbows = pose.kp2d[:, [L_ELBOW, R_ELBOW]]
    for i, gray in enumerate(frames):
        f = f0 + i
        g = grip[f]
        if not np.all(np.isfinite(g)):
            continue
        s = scorer(gray.astype(np.float32) - background, g)
        med = np.median(s)
        mad = np.median(np.abs(s - med)) * 1.4826 + 1e-6
        zi = (s - med) / mad
        _exclude_forearms(zi, g, elbows[f])
        z[i] = np.clip(zi, -5, 25)
        seen[i] = True
        if progress and i % 60 == 0:
            progress(i, f1 - f0 + 1)

    path = _decode(z, pose.fps)
    zc = z[np.arange(len(path)), path]
    c = 1 / (1 + np.exp(-(zc - CONF_Z) / 0.75))
    # Motion blur: when the shaft sweeps more than ~20 degrees per frame (fast swing, low frame
    # rate) each frame shows a smeared wedge and the direction is uncertain by half the sweep.
    sweep = np.degrees(np.abs(np.gradient(np.unwrap(path * (2 * np.pi / N_BINS)))))
    c = c / (1 + np.exp(-(BLUR_DEG - sweep) / 3.0))
    angle[f0:f1 + 1] = path * (2 * np.pi / N_BINS)
    conf[f0:f1 + 1] = np.where(seen, c, 0.0)
    return ShaftTrack(angle=angle, confidence=conf, grip=grip, length_px=1.9 * torso, fps=pose.fps)


def window_from_events(events: dict, num_frames: int, fps: float) -> tuple[int, int]:
    from app.models import EventType

    m = int(round(MARGIN_S * fps))
    a = events.get(EventType.address, 0)
    fin = events.get(EventType.finish, num_frames - 1)
    return max(0, a - m), min(num_frames - 1, fin + m)


def apply_corrections(track: ShaftTrack, corrections: dict[int, float]) -> ShaftTrack:
    """Your corrected frames (degrees) override the tracker at full confidence."""
    angle, conf = track.angle.copy(), track.confidence.copy()
    for f, deg in corrections.items():
        if 0 <= f < len(angle):
            angle[f] = np.radians(deg) % (2 * np.pi)
            conf[f] = 1.0
    return ShaftTrack(angle, conf, track.grip, track.length_px, track.fps)
