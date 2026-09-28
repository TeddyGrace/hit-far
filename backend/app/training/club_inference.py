"""Run a trained club detector (checkpoint in the bucket) on a swing video."""

import io
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.pipeline import club as club_mod
from app.pipeline.posedata import PoseData
from app.storage import get_storage
from app.training.club_model import (
    CROP_VERSION,
    apply_affine,
    build_model,
    clubhead_along,
    crop_matrix,
    direction_scores,
    heatmap_probs,
    make_crop,
    to_input,
)

MIN_HEAD_MASS = 0.3  # report a clubhead only where the model is fairly sure of it


@lru_cache(maxsize=4)
def load_checkpoint(checkpoint_uri: str):
    import torch

    from app.resources import available_cpus

    torch.set_num_threads(min(4, available_cpus()))
    ckpt = torch.load(io.BytesIO(get_storage().get_bytes(checkpoint_uri)), map_location="cpu", weights_only=True)
    if ckpt.get("crop_version") != CROP_VERSION:
        raise RuntimeError(f"checkpoint crops v{ckpt.get('crop_version')} != code v{CROP_VERSION}")
    model = build_model(ckpt["width"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def track_frames(model, frames: list[np.ndarray], background: np.ndarray, f0: int, T: int, grip: np.ndarray,
                 torso: float, fps: float) -> club_mod.ShaftTrack:
    """Learned per-frame evidence + the stage-1 temporal decoder."""
    idx = [i for i in range(len(frames)) if np.all(np.isfinite(grip[f0 + i]))]
    n = len(frames)
    z = np.zeros((n, club_mod.N_BINS))
    seen = np.zeros(n, dtype=bool)
    probs = None
    if idx:
        crops = np.stack([make_crop(frames, background, i, grip[f0 + i], torso) for i in idx])
        probs = heatmap_probs(model, to_input(crops))
        z[idx] = direction_scores(probs[:, 1])
        seen[idx] = True
    track = club_mod.track_from_scores(z, seen, f0, T, grip, torso, fps)
    head = np.full((T, 2), np.nan)
    if probs is not None:
        for k, i in enumerate(idx):
            f = f0 + i
            if not np.isfinite(track.angle[f]):
                continue
            pt, mass = clubhead_along(probs[k, 1], track.angle[f])
            if pt is not None and mass >= MIN_HEAD_MASS:
                M = crop_matrix(grip[f], torso)
                Minv = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]
                head[f] = apply_affine(Minv, pt[None])[0]
    track.clubhead = head
    return track


def track_shaft_learned(checkpoint_uri: str, video_path: Path, pose: PoseData, window: tuple[int, int] | None = None,
                        progress: Callable[[int, int], None] | None = None) -> club_mod.ShaftTrack:  # noqa: ARG001
    model = load_checkpoint(checkpoint_uri)
    T = pose.num_frames
    f0, f1 = window if window else (0, T - 1)
    f0, f1 = max(0, f0), min(T - 1, f1)
    grip = club_mod.grip_points(pose)
    torso = club_mod.torso_px(pose, f0)
    if f1 - f0 < 2 or np.isnan(grip[f0:f1 + 1]).all():
        return club_mod.ShaftTrack(np.full(T, np.nan), np.zeros(T), grip, 1.9 * torso, pose.fps)
    frames, background, _, _ = club_mod.read_frames(video_path, f0, f1)
    return track_frames(model, frames, background, f0, T, grip, torso, pose.fps)
