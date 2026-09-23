"""Train the event model: GolfDB (+ your reviewed swings) -> BiLSTM -> evaluate -> register.

The new model is registered `experimental`; promoting it to `active` is a separate, explicit step
(Models page) after comparing its held-out numbers against the rule-based baseline.
"""

import io
import json
import logging
import os
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import EVENT_ORDER, Dataset, DatasetSource, Label, Model, ModelStatus, Swing
from app.pipeline import events as rule_events
from app.pipeline.registry import TASK_EVENTS
from app.pipeline.run import effective_events, latest_pose_sequence, load_pose
from app.storage import get_storage
from app.training import golfdb
from app.training.features import FEATURE_VERSION, mirror, pose_features, resample, to_features
from app.training.model import N_CLASSES, N_EVENTS, build_model, pce, predict

log = logging.getLogger(__name__)

MODEL_NAME = "event-bilstm"
REVIEW_LABEL_TASK = "event_review"
TEST_SPLIT = 1  # GolfDB split held out for evaluation (never trained on)


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 16
    lr: float = 1e-3
    hidden: int = 128
    layers: int = 2
    window: int = 384  # max frames per training crop (after time-scaling)
    user_weight: int = 20  # oversampling factor for your reviewed swings
    patience: int = 8  # early stopping on validation PCE
    seed: int = 0
    golfdb: bool = True
    max_golfdb_clips: int | None = None  # for smoke runs


@dataclass
class Sample:
    xy: np.ndarray  # (T, J, 2)
    vis: np.ndarray  # (T, J)
    events: list[int]
    source: str  # golfdb | self
    ref: str  # clip id / swing id
    view: str = "face-on"
    split: str = "train"  # train | val | test
    pose: object = field(default=None, repr=False)  # PoseData, kept for the rule baseline


# --- Data --------------------------------------------------------------------------------------


def reviewed_swing_ids(db: Session) -> list:
    rows = db.scalars(
        select(Label).where(Label.task == REVIEW_LABEL_TASK, Label.target_type == "swing").order_by(Label.created_at)
    ).all()
    state: dict = {}
    for r in rows:
        state[r.target_id] = bool(r.corrected_value.get("reviewed"))
    return [sid for sid, ok in state.items() if ok]


def user_samples(db: Session, handedness: str) -> list[Sample]:
    out = []
    for i, sid in enumerate(sorted(reviewed_swing_ids(db), key=str)):
        swing = db.get(Swing, sid)
        seq = latest_pose_sequence(db, sid) if swing else None
        ev = effective_events(db, sid) if seq else {}
        if seq is None or len(ev) < N_EVENTS:
            continue
        pose = load_pose(seq)
        xy, vis = pose_features(pose, handedness)
        frames = [ev[e] for e in EVENT_ORDER]
        if frames != sorted(frames):
            continue
        # Every 5th reviewed swing is held out to measure how well the model fits *your* swings.
        out.append(Sample(xy, vis, frames, "self", str(sid), split="test" if i % 5 == 4 else "train", pose=pose))
    return out


def golfdb_samples(cfg: TrainConfig, progress: Callable[[str], None]) -> tuple[list[Sample], dict]:
    clips = golfdb.load_labels()
    if cfg.max_golfdb_clips:
        clips = clips[: cfg.max_golfdb_clips]
    errors = golfdb.ensure_pose(clips, progress)
    rng = random.Random(cfg.seed)
    out = []
    for i, c in enumerate(clips):
        if i % 50 == 0:
            progress(f"loading pose {i}/{len(clips)}")
        pose = golfdb.load_pose(c.id)
        if pose is None or pose.detected.sum() < 10:
            continue
        xy, vis = pose_features(pose)
        if c.events[-1] >= len(xy):
            continue
        split = "test" if c.split == TEST_SPLIT else ("val" if rng.random() < 0.1 else "train")
        out.append(Sample(xy, vis, c.events, "golfdb", str(c.id), c.view, split, pose=pose))
    return out, {"pose_errors": len(errors), "clips": len(clips)}


# --- Training ----------------------------------------------------------------------------------


def _soft_targets(T: int, events: list[int]) -> np.ndarray:
    y = np.zeros((T, N_CLASSES), np.float32)
    y[:, -1] = 1.0
    for e, f in enumerate(events):
        for d, w in ((0, 0.5), (-1, 0.25), (1, 0.25)):
            t = f + d
            if 0 <= t < T:
                y[t, e] += w
                y[t, -1] -= w
    return np.clip(y, 0, 1)


def augment(s: Sample, rng: random.Random, window: int) -> tuple[np.ndarray, np.ndarray]:
    xy, vis, ev = s.xy, s.vis, list(s.events)
    if rng.random() < 0.5:
        xy, vis = mirror(xy, vis)
    # Time scale: covers slow-motion vs real-time capture and tempo differences.
    scale = float(np.exp(rng.uniform(np.log(0.4), np.log(2.5))))
    T0 = len(xy)
    n = max(N_EVENTS + 2, int(round(T0 * scale)))
    xy, vis = resample(xy, n / T0), resample(vis, n / T0)
    ev = [min(n - 1, int(round(f * n / T0))) for f in ev]
    # Idle padding (your clips have waggles/stillness GolfDB clips don't).
    pre, post = rng.randint(0, 60), rng.randint(0, 60)
    xy = np.concatenate([np.repeat(xy[:1], pre, 0), xy, np.repeat(xy[-1:], post, 0)])
    vis = np.concatenate([np.repeat(vis[:1], pre, 0), vis, np.repeat(vis[-1:], post, 0)])
    ev = [f + pre for f in ev]
    xy = xy + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 0.01, xy.shape)
    feats = to_features(xy.astype(np.float32), vis)
    y = _soft_targets(len(feats), ev)
    if len(feats) > window:  # random crop
        start = rng.randint(0, len(feats) - window)
        feats, y = feats[start:start + window], y[start:start + window]
    return feats, y


def _batches(samples: list[Sample], cfg: TrainConfig, rng: random.Random):
    import torch

    order = samples[:]
    rng.shuffle(order)
    for i in range(0, len(order), cfg.batch_size):
        items = [augment(s, rng, cfg.window) for s in order[i:i + cfg.batch_size]]
        T = max(len(f) for f, _ in items)
        X = np.zeros((len(items), T, items[0][0].shape[1]), np.float32)
        Y = np.zeros((len(items), T, N_CLASSES), np.float32)
        M = np.zeros((len(items), T), np.float32)
        for b, (f, y) in enumerate(items):
            X[b, : len(f)], Y[b, : len(f)], M[b, : len(f)] = f, y, 1
        yield torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(M)


def evaluate(model, samples: list[Sample]) -> dict:
    if not samples:
        return {"n": 0}
    correct = np.array([pce(predict(model, s.xy, s.vis).frames, s.events) for s in samples])
    return {"n": len(samples), "pce": round(float(correct.mean()), 4),
            "per_event": {e.value: round(float(v), 4) for e, v in zip(EVENT_ORDER, correct.mean(axis=0))}}


def evaluate_rules(samples: list[Sample]) -> dict:
    correct = []
    for s in samples:
        try:
            res = rule_events.detect_events(s.pose)
            pred = [res.events[e].frame for e in EVENT_ORDER]
        except rule_events.EventDetectionError:
            pred = [0] * N_EVENTS
        correct.append(pce(pred, s.events))
    if not correct:
        return {"n": 0}
    c = np.array(correct)
    return {"n": len(c), "pce": round(float(c.mean()), 4),
            "per_event": {e.value: round(float(v), 4) for e, v in zip(EVENT_ORDER, c.mean(axis=0))}}


def train(samples: list[Sample], cfg: TrainConfig, progress: Callable[[str], None]):
    import torch

    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    train_set = [s for s in samples if s.split == "train"]
    train_set = train_set + [s for s in train_set if s.source == "self"] * (cfg.user_weight - 1)
    val_set = [s for s in samples if s.split == "val"] or [s for s in samples if s.split == "train"][:20]
    model = build_model(cfg.hidden, cfg.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    class_w = torch.ones(N_CLASSES)
    class_w[-1] = 0.1  # background dominates frame counts
    best, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, nb = 0.0, 0
        for X, Y, M in _batches(train_set, cfg, rng):
            logp = torch.log_softmax(model(X), dim=-1)
            loss_t = -(Y * logp * class_w).sum(-1)
            loss = (loss_t * M).sum() / M.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total, nb = total + loss.item(), nb + 1
        sched.step()
        val = evaluate(model, val_set)["pce"]
        progress(f"train epoch {epoch}/{cfg.epochs} loss {total / max(nb, 1):.3f} val {val:.3f}")
        if val > best:
            best, best_epoch, stale = val, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_val_pce": round(best, 4), "best_epoch": best_epoch, "epochs_run": epoch}


# --- Entry point (trainer job) -----------------------------------------------------------------


def run_training(db: Session, payload: dict, on_stage: Callable[[str], None] = lambda s: None) -> Model:
    import torch

    cfg = TrainConfig(**payload.get("config", {}))
    s = get_settings()
    torch.set_num_threads(s.train_workers or os.cpu_count() or 1)
    on_stage("loading data")
    samples: list[Sample] = []
    info: dict = {}
    if cfg.golfdb:
        g, info = golfdb_samples(cfg, on_stage)
        samples += g
    mine = user_samples(db, s.golfer_handedness)
    samples += mine
    if not any(x.split == "train" for x in samples):
        raise ValueError("no training data (GolfDB disabled and no reviewed swings)")

    version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    st = get_storage()
    manifest = {
        "version": version, "feature_version": FEATURE_VERSION, "config": asdict(cfg),
        "samples": [{"source": x.source, "ref": x.ref, "split": x.split, "view": x.view} for x in samples],
        **info,
    }
    manifest_key = f"datasets/event_training/{version}.json"
    st.put_bytes(manifest_key, json.dumps(manifest).encode(), "application/json")
    ds = Dataset(name=f"events-{version}", task=TASK_EVENTS,
                 source=DatasetSource.mixed if mine and cfg.golfdb else
                 (DatasetSource.golfdb if cfg.golfdb else DatasetSource.self_labeled),
                 manifest_uri=manifest_key, num_samples=len(samples))
    db.add(ds)
    db.commit()

    model, train_info = train(samples, cfg, on_stage)

    on_stage("evaluating")
    test = [x for x in samples if x.split == "test" and x.source == "golfdb"]
    test_face_on = [x for x in test if x.view == "face-on"]
    user_test = [x for x in samples if x.split == "test" and x.source == "self"]
    metrics = {
        "golfdb_test": evaluate(model, test),
        "golfdb_test_face_on": evaluate(model, test_face_on),
        "rules_golfdb_test_face_on": evaluate_rules(test_face_on),
        "self_holdout": evaluate(model, user_test),
        "rules_self_holdout": evaluate_rules(user_test),
        "n_train_golfdb": sum(1 for x in samples if x.split == "train" and x.source == "golfdb"),
        "n_train_self": sum(1 for x in samples if x.split == "train" and x.source == "self"),
        "pce_tolerance": "GolfDB: max(1, round((impact - address) / 30)) frames",
        **train_info,
    }

    buf = io.BytesIO()
    torch.save({"state_dict": model.state_dict(), "hidden": cfg.hidden, "layers": cfg.layers,
                "feature_version": FEATURE_VERSION}, buf)
    ckpt_key = f"models/{MODEL_NAME}/{version}/model.pt"
    st.put_bytes(ckpt_key, buf.getvalue())
    m = Model(name=MODEL_NAME, version=version, task=TASK_EVENTS, checkpoint_uri=ckpt_key,
              trained_on_dataset_id=ds.id, eval_metrics=metrics, status=ModelStatus.experimental,
              notes="BiLSTM over pose features; trained on GolfDB + reviewed swings")
    db.add(m)
    db.commit()
    on_stage("done")
    log.info("trained %s %s: %s", MODEL_NAME, version, json.dumps(metrics)[:500])
    return m

