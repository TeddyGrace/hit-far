"""Baseline 2D pose + 3D world landmarks via MediaPipe Pose Landmarker (heavy).

This is the free baseline the brief calls for. It's registered in `models` like any trained model
so its outputs can be compared against fine-tuned successors.
"""

import urllib.request
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from app.config import get_settings
from app.pipeline.posedata import PoseData

MODEL_NAME = "mediapipe-pose-landmarker-heavy"
MODEL_VERSION = "float16-latest"
NUM_LANDMARKS = 33


def ensure_pose_model() -> Path:
    s = get_settings()
    path = s.pose_model_path.resolve()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(s.pose_model_url, tmp)
        tmp.rename(path)
    return path


def run_pose(video_path: Path, progress: Callable[[int, int], None] | None = None) -> PoseData:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_pose_model())),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    kp2d, vis, world = [], [], []
    last_ts = -1
    with PoseLandmarker.create_from_options(options) as landmarker:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = max(last_ts + 1, int(round(i * 1000.0 / fps)))
            last_ts = ts
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            if res.pose_landmarks:
                lms = res.pose_landmarks[0]
                kp2d.append([[lm.x * w, lm.y * h] for lm in lms])
                vis.append([lm.visibility if lm.visibility is not None else 0.0 for lm in lms])
                wl = res.pose_world_landmarks[0] if res.pose_world_landmarks else None
                world.append([[p.x, p.y, p.z] for p in wl] if wl else np.full((NUM_LANDMARKS, 3), np.nan))
            else:
                kp2d.append(np.full((NUM_LANDMARKS, 2), np.nan))
                vis.append(np.zeros(NUM_LANDMARKS))
                world.append(np.full((NUM_LANDMARKS, 3), np.nan))
            i += 1
            if progress and i % 30 == 0:
                progress(i, total)
    cap.release()

    return PoseData(
        kp2d=np.asarray(kp2d, dtype=float).reshape(-1, NUM_LANDMARKS, 2),
        visibility=np.asarray(vis, dtype=float).reshape(-1, NUM_LANDMARKS),
        world=np.asarray(world, dtype=float).reshape(-1, NUM_LANDMARKS, 3),
        fps=float(fps),
        width=w,
        height=h,
    )
