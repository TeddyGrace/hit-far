"""Run a trained event model (checkpoint in the bucket) on a pose sequence."""

import io
from functools import lru_cache

import numpy as np

from app.models import EVENT_ORDER, EventType
from app.pipeline.events import DetectedEvent
from app.pipeline.posedata import PoseData
from app.storage import get_storage
from app.training.features import FEATURE_VERSION, pose_features, resample, to_features
from app.training.model import N_EVENTS, build_model, predict


@lru_cache(maxsize=4)
def load_checkpoint(checkpoint_uri: str):
    import torch

    ckpt = torch.load(io.BytesIO(get_storage().get_bytes(checkpoint_uri)), map_location="cpu", weights_only=True)
    if ckpt.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError(f"checkpoint features v{ckpt.get('feature_version')} != code v{FEATURE_VERSION}")
    model = build_model(ckpt["hidden"], ckpt["layers"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def detect_events_learned(checkpoint_uri: str, pose: PoseData, handedness: str) -> dict[EventType, DetectedEvent]:
    import torch

    model = load_checkpoint(checkpoint_uri)
    xy, vis = pose_features(pose, handedness)
    pred = predict(model, xy, vis)
    # Confidence = probability mass within +-1 frame of the chosen frame (at the model's scale);
    # the soft training targets spread each event over 3 frames.
    n = max(N_EVENTS + 1, int(round(len(xy) * pred.scale)))
    with torch.no_grad():
        feats = to_features(resample(xy, n / len(xy)), resample(vis, n / len(vis)))
        probs = torch.softmax(model(torch.from_numpy(feats)[None])[0], -1).numpy()
    out = {}
    for i, et in enumerate(EVENT_ORDER):
        f_scaled = min(n - 1, int(round(pred.frames[i] * pred.scale)))
        conf = float(probs[max(0, f_scaled - 1): f_scaled + 2, i].sum())
        out[et] = DetectedEvent(frame=pred.frames[i], confidence=round(float(np.clip(conf, 0, 1)), 3),
                                method="event-bilstm")
    return out
