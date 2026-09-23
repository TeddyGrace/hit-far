"""MediaPipe Pose 33-landmark schema ("mediapipe33") and small geometry helpers."""

from dataclasses import dataclass

import numpy as np

SCHEMA_VERSION = "mediapipe33"

NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# Skeleton edges drawn by the viewer (subset of MediaPipe's POSE_CONNECTIONS, no face/hands).
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),
    (0, 11), (0, 12),
]


@dataclass(frozen=True)
class Side:
    """Lead/trail landmark indices. Lead = the side facing the target (left for a right-hander)."""

    shoulder: int
    elbow: int
    wrist: int
    hip: int
    knee: int
    ankle: int


LEFT = Side(L_SHOULDER, L_ELBOW, L_WRIST, L_HIP, L_KNEE, L_ANKLE)
RIGHT = Side(R_SHOULDER, R_ELBOW, R_WRIST, R_HIP, R_KNEE, R_ANKLE)


def sides(handedness: str) -> tuple[Side, Side]:
    """Returns (lead, trail)."""
    return (LEFT, RIGHT) if handedness == "right" else (RIGHT, LEFT)


def interpolate_nans(x: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaNs along axis 0 (edges held constant). Works on any trailing shape."""
    x = np.array(x, dtype=float, copy=True)
    flat = x.reshape(len(x), -1)
    idx = np.arange(len(x))
    for c in range(flat.shape[1]):
        col = flat[:, c]
        ok = ~np.isnan(col)
        if ok.sum() == 0:
            continue
        if not ok.all():
            col[~ok] = np.interp(idx[~ok], idx[ok], col[ok])
    return flat.reshape(x.shape)


def smooth(x: np.ndarray, sigma_frames: float) -> np.ndarray:
    """Gaussian smoothing along axis 0 with reflected edges."""
    if sigma_frames <= 0:
        return np.asarray(x, dtype=float)
    radius = max(1, int(round(3 * sigma_frames)))
    t = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (t / sigma_frames) ** 2)
    k /= k.sum()
    x = np.asarray(x, dtype=float)
    flat = x.reshape(len(x), -1)
    padded = np.pad(flat, ((radius, radius), (0, 0)), mode="reflect" if len(x) > radius else "edge")
    out = np.stack([np.convolve(padded[:, c], k, mode="valid") for c in range(flat.shape[1])], axis=1)
    return out.reshape(x.shape)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Unsigned angle in degrees between vectors along the last axis."""
    n = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    cos = np.sum(v1 * v2, axis=-1) / np.where(n == 0, np.nan, n)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at b (degrees) formed by a-b-c."""
    return angle_between(a - b, c - b)
