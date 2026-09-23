"""Pose sequence -> per-frame feature matrix for the learned event model.

Shared by training (GolfDB + your swings) and inference, so both see identical inputs.
Features are resolution- and position-invariant: joints are centred on the sequence's median hip
centre and scaled by median torso length. Left-handed swings are mirrored so the model always
sees a right-hander.
"""

import numpy as np

from app.pipeline import landmarks as L
from app.pipeline.landmarks import interpolate_nans, smooth
from app.pipeline.posedata import PoseData

FEATURE_VERSION = "1"
# nose, shoulders, elbows, wrists, hips, knees, ankles
JOINTS = [L.NOSE, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
# left/right swap for mirroring, expressed as positions within JOINTS
_MIRROR = {11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15, 23: 24, 24: 23, 25: 26, 26: 25, 27: 28, 28: 27}
MIRROR_PERM = [JOINTS.index(_MIRROR.get(j, j)) for j in JOINTS]
N_JOINTS = len(JOINTS)
N_FEATURES = N_JOINTS * 2 * 2 + N_JOINTS  # xy + velocity xy + visibility


def raw_sequence(pose: PoseData) -> tuple[np.ndarray, np.ndarray]:
    """(T, J, 2) normalized joint positions and (T, J) visibility. Missing frames interpolated."""
    kp = pose.kp2d[:, JOINTS, :]
    vis = pose.visibility[:, JOINTS].astype(np.float32)
    if np.isnan(kp[:, 0, 0]).all():
        return np.zeros((pose.num_frames, N_JOINTS, 2), np.float32), np.zeros_like(vis)
    kp = interpolate_nans(kp)
    hips = (kp[:, JOINTS.index(23)] + kp[:, JOINTS.index(24)]) / 2
    shoulders = (kp[:, JOINTS.index(11)] + kp[:, JOINTS.index(12)]) / 2
    torso = float(np.median(np.linalg.norm(shoulders - hips, axis=-1)))
    centre = np.median(hips, axis=0)
    xy = (kp - centre) / max(torso, 1e-6)
    return xy.astype(np.float32), vis


def mirror(xy: np.ndarray, vis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = xy[:, MIRROR_PERM].copy()
    xy[..., 0] *= -1
    return xy, vis[:, MIRROR_PERM]


def to_features(xy: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """(T, J, 2), (T, J) -> (T, N_FEATURES). Velocity is per-frame, so it scales with the
    sequence's effective frame rate - the model is trained across rates via resampling."""
    xy_s = smooth(xy, 0.8) if len(xy) > 3 else xy
    vel = np.gradient(xy_s, axis=0) if len(xy) > 1 else np.zeros_like(xy)
    return np.concatenate(
        [xy_s.reshape(len(xy), -1), 10.0 * vel.reshape(len(xy), -1), vis], axis=1
    ).astype(np.float32)


def resample(arr: np.ndarray, factor: float) -> np.ndarray:
    """Linear time resampling along axis 0. factor > 1 = more frames (slower)."""
    T = len(arr)
    n = max(2, int(round(T * factor)))
    src = np.linspace(0, T - 1, n)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (src - i0).reshape(-1, *([1] * (arr.ndim - 1)))
    return (arr[i0] * (1 - w) + arr[i1] * w).astype(arr.dtype)


def pose_features(pose: PoseData, handedness: str = "right") -> tuple[np.ndarray, np.ndarray]:
    xy, vis = raw_sequence(pose)
    if handedness == "left":
        xy, vis = mirror(xy, vis)
    return xy, vis
