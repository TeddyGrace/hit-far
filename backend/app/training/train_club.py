"""Stage-2 club detector training: line-tracker pseudo-labels + your shaft checks -> heatmap CNN.

Training data per swing:
  * pseudo-labels: frames the stage-1 line tracker is confident about (direction, plus the clubhead
    where the shaft's end is visible and not smeared by motion blur);
  * your labels (`labels` rows with task=club): shaft fixes (direction + the clubhead you clicked)
    and "looks right" confirmations. They override the tracker on their frames.

Swings are split, not frames: every TEST_EVERY-th swing you have labelled is held out entirely. The
new model is compared with the line tracker (and the currently active detector) on those held-out
swings, exactly as it would run in the pipeline (full decoding), and promoted automatically only
if it is better on your labels without disagreeing with the tracker where the tracker was sure.

Nothing trains until there is enough data (the gate): see `data_status`.
"""

import hashlib
import io
import json
import logging
import random
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import jobs
from app.config import get_settings
from app.models import ClubTrack, Dataset, DatasetSource, Job, Label, Model, ModelStatus, PoseSequence, Swing, Video
from app.pipeline import club as club_mod
from app.pipeline.registry import TASK_CLUB
from app.pipeline.run import (
    CLUB_LABEL_TASK,
    active_club_model,
    club_labels,
    club_model,
    effective_events,
    latest_pose_sequence,
    load_pose,
)
from app.resources import available_cpus
from app.storage import get_storage
from app.training.club_model import (
    CROP,
    CROP_VERSION,
    HM,
    TORSO_CROP_PX,
    apply_affine,
    build_model,
    crop_matrix,
    gaussian_target,
    make_crop,
    to_input,
    transform_angle,
    wedge_weights,
)

log = logging.getLogger(__name__)

MODEL_NAME = "club-heatmap-cnn"
PSEUDO_CONF = 0.8  # line-tracker confidence for a frame to become a pseudo-label
# The shaft's end is only reliable when motion blur doesn't smear it: on synthetic video the
# ridge-end search is within ~1-2% of the shaft length up to this sweep, and falls apart above it.
CLUBHEAD_MAX_SWEEP_DEG = 1.2
TEST_EVERY = 3  # every 3rd labelled swing is held out
MAX_AGREEMENT_DEG = 6.0  # promotion: median disagreement with the tracker where it was confident


@dataclass
class ClubTrainConfig:
    epochs: int = 15
    batch_size: int = 32
    lr: float = 2e-3
    width: int = 24
    patience: int = 4
    seed: int = 0
    human_weight: int = 5  # oversampling factor for your labelled frames
    max_frames_per_swing: int = 150
    max_samples: int = 12000
    # The data gate.
    min_swings: int = 8
    min_test_swings: int = 2
    min_test_frames: int = 20


# --- Data gate ---------------------------------------------------------------------------------


def _hash(sid) -> str:
    return hashlib.sha256(str(sid).encode()).hexdigest()


def trackable_swings(db: Session) -> list[uuid.UUID]:
    """Swings with pose and a playback proxy (what the detector can learn from)."""
    has_pose = select(PoseSequence.swing_id).distinct()
    return sorted(db.scalars(select(Swing.id).where(Swing.id.in_(has_pose))).all(), key=_hash)


def labelled_counts(db: Session, swing_ids: list[uuid.UUID]) -> dict[uuid.UUID, int]:
    ids = set(db.scalars(select(Label.target_id).where(Label.task == CLUB_LABEL_TASK, Label.target_type == "swing")
                         .distinct()).all()) & set(swing_ids)
    counts = {sid: len(club_labels(db, sid)) for sid in ids}
    return {sid: n for sid, n in counts.items() if n}


def split_swings(labelled: dict[uuid.UUID, int]) -> set[uuid.UUID]:
    """Held-out swings: every TEST_EVERY-th labelled swing in a fixed (hash) order."""
    order = sorted(labelled, key=_hash)
    return {sid for i, sid in enumerate(order) if i % TEST_EVERY == TEST_EVERY - 1}


def data_status(db: Session, cfg: ClubTrainConfig | None = None) -> dict:
    cfg = cfg or ClubTrainConfig()
    swings = trackable_swings(db)
    labelled = labelled_counts(db, swings)
    test = split_swings(labelled)
    test_frames = sum(labelled[s] for s in test)
    need = []
    if len(swings) < cfg.min_swings:
        need.append(f"{cfg.min_swings - len(swings)} more swings")
    if len(test) < cfg.min_test_swings:
        need.append(f"shaft checks on {cfg.min_test_swings * TEST_EVERY - len(labelled)} more swings")
    if test_frames < cfg.min_test_frames:
        # Only a third of labelled swings are held out, so ask for ~3x the missing frames overall.
        need.append(f"~{(cfg.min_test_frames - test_frames) * TEST_EVERY} more checked frames")
    return {
        "swings": len(swings), "labelled_swings": len(labelled), "labelled_frames": sum(labelled.values()),
        "test_swings": len(test), "test_frames": test_frames, "ready": not need,
        "min_swings": cfg.min_swings, "min_test_swings": cfg.min_test_swings, "min_test_frames": cfg.min_test_frames,
        "needs": need,
    }


def _last_training_counts(db: Session) -> dict | None:
    job = db.scalar(select(Job).where(Job.type == jobs.JOB_TRAIN_CLUB).order_by(Job.created_at.desc()).limit(1))
    return (job.payload or {}).get("data") if job else None


def should_train(db: Session, cfg: ClubTrainConfig | None = None) -> tuple[bool, dict]:
    """Hands-off retraining: when the gate is met and there's meaningfully more data than the last
    run saw (new shaft checks or new swings). A failed run isn't retried until the data grows."""
    st = data_status(db, cfg)
    if not st["ready"] or jobs.pending(db, jobs.JOB_TRAIN_CLUB):
        return False, st
    last = _last_training_counts(db)
    if last is None:
        return True, st
    more_labels = st["labelled_frames"] - last.get("labelled_frames", 0) >= get_settings().club_retrain_every
    more_swings = st["swings"] >= last.get("swings", 0) + max(5, last.get("swings", 0) // 2)
    return more_labels or more_swings, st


def queue_training(db: Session, st: dict, config: dict | None = None) -> Job:
    job = jobs.enqueue(db, jobs.JOB_TRAIN_CLUB, {"config": config or {}, "data": {
        "labelled_frames": st["labelled_frames"], "swings": st["swings"]}})
    job.max_attempts = 2
    return job


def maybe_queue_training(db: Session, cfg: ClubTrainConfig | None = None) -> Job | None:
    ok, st = should_train(db, cfg)
    if not ok:
        return None
    job = queue_training(db, st, asdict(cfg) if cfg else None)
    db.commit()
    log.info("queued club detector training: %s", st)
    return job


# --- Per-swing data ----------------------------------------------------------------------------


@dataclass
class SwingData:
    swing_id: str
    split: str  # train | test
    crops: np.ndarray  # (N, CROP, CROP, 3) uint8
    frames: np.ndarray  # (N,) frame indices
    angle: np.ndarray  # (N,) label direction (radians, image), NaN = none
    head: np.ndarray  # (N, 2) clubhead label in crop pixels, NaN = none
    human: np.ndarray  # (N,) bool: your label (vs a pseudo-label)
    line: club_mod.ShaftTrack  # raw line-tracker output (no corrections)


def _line_track(db: Session, swing_id, proxy: Path, pose, window) -> club_mod.ShaftTrack:
    """The raw stage-1 track: stored if the swing has one, otherwise computed now."""
    line = club_model(db)
    row = db.scalar(select(ClubTrack).where(ClubTrack.swing_id == swing_id, ClubTrack.model_id == line.id)
                    .order_by(ClubTrack.created_at.desc()).limit(1))
    if row is not None:
        return club_mod.ShaftTrack.from_npz(get_storage().get_bytes(row.track_uri))
    return club_mod.track_shaft(proxy, pose, window)


def _swing_inputs(db: Session, swing_id, tmp: Path):
    swing = db.get(Swing, swing_id)
    seq = latest_pose_sequence(db, swing_id)
    video = db.get(Video, swing.video_ids[0]) if swing else None
    if seq is None or video is None or not video.proxy_uri:
        return None
    pose = load_pose(seq)
    window = club_mod.window_from_events(effective_events(db, swing_id), pose.num_frames, pose.fps)
    proxy = tmp / f"{swing_id}.mp4"
    if not proxy.exists():
        get_storage().download_file(video.proxy_uri, proxy)
    return seq, pose, window, proxy


def _sweep_deg(angle: np.ndarray) -> np.ndarray:
    a = np.where(np.isfinite(angle), angle, 0.0)
    return np.degrees(np.abs(np.gradient(np.unwrap(a)))) if len(a) > 1 else np.zeros(len(a))


def _build_cache(db: Session, swing_id, labels: dict[int, dict], cfg: ClubTrainConfig, tmp: Path) -> dict | None:
    got = _swing_inputs(db, swing_id, tmp)
    if got is None:
        return None
    seq, pose, window, proxy = got
    line = _line_track(db, swing_id, proxy, pose, window)
    T = pose.num_frames
    f0, f1 = window
    human = sorted(f for f in labels if 0 <= f < T)
    lo, hi = min([f0] + human), max([f1] + human)
    pseudo = [f for f in range(f0, f1 + 1) if f not in labels and line.confidence[f] >= PSEUDO_CONF]
    room = max(0, cfg.max_frames_per_swing - len(human))
    if len(pseudo) > room:
        pseudo = [pseudo[int(i)] for i in np.linspace(0, len(pseudo) - 1, room)] if room else []
    chosen = sorted(set(human) | set(pseudo))
    frames, background, _, _ = club_mod.read_frames(proxy, lo, hi)
    grip = club_mod.grip_points(pose)
    torso = club_mod.torso_px(pose, f0)
    sweep = _sweep_deg(line.angle)
    crops, keep, heads = [], [], []
    for f in chosen:
        i = f - lo
        if i >= len(frames) or not np.all(np.isfinite(grip[f])):
            continue
        crops.append(make_crop(frames, background, i, grip[f], torso))
        keep.append(f)
        head = np.full(2, np.nan)
        if f in pseudo and sweep[f] <= CLUBHEAD_MAX_SWEEP_DEG:
            end = club_mod.shaft_end(frames[i].astype(np.float32) - background, grip[f], line.angle[f], torso)
            if end is not None:
                head = apply_affine(crop_matrix(grip[f], torso), end[None])[0]
        heads.append(head)
    to_crop = {f: crop_matrix(grip[f], torso) for f in keep}
    return {"crops": np.stack(crops) if crops else np.zeros((0, CROP, CROP, 3), np.uint8),
            "frames": np.array(keep, dtype=int), "pseudo_head": np.array(heads).reshape(-1, 2),
            "affine": np.array([to_crop[f] for f in keep]).reshape(-1, 2, 3), "line": line.to_npz(),
            "pose_sequence": str(seq.id)}


def swing_data(db: Session, swing_id, split: str, cfg: ClubTrainConfig, tmp: Path) -> SwingData | None:
    """Crops are cached in the bucket per pose sequence; labels are applied fresh each run."""
    st = get_storage()
    labels = club_labels(db, swing_id)
    seq = latest_pose_sequence(db, swing_id)
    if seq is None:
        return None
    key = f"datasets/club/crops/{swing_id}/{seq.id}-v{CROP_VERSION}.npz"
    cache = None
    if st.exists(key):
        z = np.load(io.BytesIO(st.get_bytes(key)))
        cache = {k: z[k] for k in z.files}
        if not set(labels) <= set(cache["frames"].tolist()):
            cache = None  # new labels on frames the cache doesn't have
    if cache is None:
        cache = _build_cache(db, swing_id, labels, cfg, tmp)
        if cache is None:
            return None
        buf = io.BytesIO()
        np.savez_compressed(buf, **{k: (np.frombuffer(v, np.uint8) if isinstance(v, bytes) else
                                        np.array(v) if isinstance(v, str) else v) for k, v in cache.items()})
        st.put_bytes(key, buf.getvalue())
        cache["line"] = np.frombuffer(cache["line"], np.uint8)
    line = club_mod.ShaftTrack.from_npz(bytes(cache["line"]))
    frames = cache["frames"].astype(int)
    angle = np.full(len(frames), np.nan)
    head = cache["pseudo_head"].astype(float).copy()
    human = np.zeros(len(frames), dtype=bool)
    for k, f in enumerate(frames):
        lab = labels.get(int(f))
        if lab is None:
            angle[k] = line.angle[f] if line.confidence[f] >= PSEUDO_CONF else np.nan
            continue
        human[k] = True
        angle[k] = np.radians(float(lab["angle_deg"])) % (2 * np.pi)
        xy = lab.get("clubhead_xy")
        if xy is not None:
            head[k] = apply_affine(cache["affine"][k], np.array(xy, float)[None])[0]
        elif not lab.get("confirmed"):
            head[k] = np.nan  # the pseudo clubhead was on the tracker's (wrong) direction
    ok = np.isfinite(angle)
    return SwingData(str(swing_id), split, cache["crops"][ok], frames[ok], angle[ok], head[ok], human[ok], line)


# --- Training ----------------------------------------------------------------------------------


def augment(crop: np.ndarray, angle: float, head: np.ndarray, rng: random.Random):
    import cv2

    rot = rng.uniform(-np.pi, np.pi) if rng.random() < 0.8 else 0.0
    mirror = rng.random() < 0.5
    scale = rng.uniform(0.85, 1.15)
    # Crop pixels -> crop pixels: about the centre (the grip), where one torso is TORSO_CROP_PX.
    M = crop_matrix(np.array([CROP / 2, CROP / 2]), TORSO_CROP_PX, rot, scale, mirror)
    img = cv2.warpAffine(crop, M, (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0, 128, 0))
    img = img.astype(np.float32)
    img[..., 0] = np.clip(img[..., 0] * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25), 0, 255)
    if rng.random() < 0.2:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    h = apply_affine(M, head[None])[0] if np.all(np.isfinite(head)) else head
    return img.astype(np.uint8), transform_angle(angle, rot, mirror), h


def _batch(items, rng: random.Random):
    import torch

    crops, angles, heads = [], [], []
    for crop, a, h in items:
        c, a2, h2 = augment(crop, a, h, rng)
        crops.append(c)
        angles.append(a2)
        heads.append(h2)
    x = torch.from_numpy(to_input(np.stack(crops)))
    wedge = torch.from_numpy(wedge_weights(np.array(angles)))
    head_t = np.zeros((len(items), HM, HM), np.float32)
    has_head = np.zeros(len(items), np.float32)
    for b, h in enumerate(heads):
        if np.all(np.isfinite(h)) and 0 <= h[0] < CROP and 0 <= h[1] < CROP:
            head_t[b] = gaussian_target(h)
            has_head[b] = 1
    grip_t = torch.from_numpy(np.repeat(gaussian_target(np.array([CROP / 2, CROP / 2]))[None], len(items), 0))
    return x, wedge, torch.from_numpy(head_t), torch.from_numpy(has_head), grip_t


def loss_fn(logits, wedge, head_t, has_head, grip_t):
    import torch

    B = logits.shape[0]
    logp = torch.log_softmax(logits.reshape(B, 2, -1), -1).reshape(B, 2, HM, HM)
    p_head = logp[:, 1].exp()
    direction = -torch.log((p_head * wedge).sum((1, 2)).clamp_min(1e-6)).mean()
    point = (-(head_t * logp[:, 1]).sum((1, 2)) * has_head).sum() / has_head.sum().clamp_min(1.0)
    grip = -(grip_t * logp[:, 0]).sum((1, 2)).mean()
    return direction + point + 0.1 * grip


def _items(data: list[SwingData], cfg: ClubTrainConfig, rng: random.Random):
    items = []
    for d in data:
        for k in range(len(d.frames)):
            items += [(d.crops[k], float(d.angle[k]), d.head[k])] * (cfg.human_weight if d.human[k] else 1)
    if len(items) > cfg.max_samples:
        items = rng.sample(items, cfg.max_samples)
    return items


def train(train_data: list[SwingData], cfg: ClubTrainConfig, progress: Callable[[str], None]):
    import torch

    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    items = _items(train_data, cfg, rng)
    rng.shuffle(items)
    n_val = max(1, len(items) // 10)
    val, tr = items[:n_val], items[n_val:] or items
    model = build_model(cfg.width)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    val_rng = random.Random(cfg.seed + 1)
    val_batches = [_batch(val[i:i + cfg.batch_size], val_rng) for i in range(0, len(val), cfg.batch_size)]
    best, best_state, best_epoch, stale, epoch = float("inf"), None, 0, 0, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        rng.shuffle(tr)
        total, nb = 0.0, 0
        for i in range(0, len(tr), cfg.batch_size):
            chunk = tr[i:i + cfg.batch_size]
            if len(chunk) < 2:
                continue  # BatchNorm needs more than one sample
            b = _batch(chunk, rng)
            loss = loss_fn(model(b[0]), *b[1:])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total, nb = total + loss.item(), nb + 1
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([loss_fn(model(b[0]), *b[1:]).item() for b in val_batches]))
        progress(f"club epoch {epoch}/{cfg.epochs} loss {total / max(nb, 1):.2f} val {vl:.2f}")
        if vl < best:
            best, best_epoch, stale = vl, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_val_loss": round(best, 4), "best_epoch": best_epoch, "epochs_run": epoch,
                   "n_train_items": len(tr)}


# --- Evaluation --------------------------------------------------------------------------------


def _err_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.abs(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b))))))


def _summary(errs: list[float]) -> dict:
    if not errs:
        return {"n": 0}
    e = np.array(errs)
    return {"n": len(e), "median_err_deg": round(float(np.median(e)), 2),
            "within_10": round(float((e <= 10).mean()), 4)}


def evaluate_tracks(tracks: dict[str, list[club_mod.ShaftTrack]], test: list[tuple[SwingData, dict]]) -> dict:
    """tracks[name][i] is that model's full track on test swing i (as deployed, no corrections).

    human: error on all your labelled frames; fixes: only the frames you corrected (where the
    tracker of the day was wrong). Confirmations ("looks right") store the shown angle, so on them
    whichever model you confirmed scores exactly 0 - that's why promotion compares medians on fixes
    only and uses the within-10-degrees rate over everything. agreement: disagreement with the line
    tracker on frames it was confident about (and you didn't label). coverage: share of swing frames
    confident enough for metrics (>= 0.5)."""
    out = {}
    for name, per_swing in tracks.items():
        human, fixes, conf_human, agree, cover = [], [], [], [], []
        for tr, (d, labels) in zip(per_swing, test):
            line = d.line
            for f, lab in labels.items():
                if not 0 <= f < len(tr.angle):
                    continue
                if np.isfinite(tr.angle[f]):
                    e = float(_err_deg(tr.angle[f], np.radians(float(lab["angle_deg"]))))
                    if tr.confidence[f] >= 0.5:
                        conf_human.append(e)
                else:
                    e = 180.0  # not tracked on a frame you labelled: count as a miss
                human.append(e)
                if not lab.get("confirmed"):
                    fixes.append(e)
            sure = [f for f in range(len(line.angle)) if line.confidence[f] >= PSEUDO_CONF and f not in labels
                    and np.isfinite(tr.angle[f])]
            agree += _err_deg(tr.angle[sure], line.angle[sure]).tolist() if sure else []
            tracked = np.isfinite(line.angle)
            if tracked.any():
                cover.append(float((tr.confidence[tracked] >= 0.5).mean()))
        out[name] = {"human": _summary(human), "fixes": _summary(fixes), "human_confident": _summary(conf_human),
                     "agreement": {"n": len(agree), "median_diff_deg":
                                   round(float(np.median(agree)), 2) if agree else None},
                     "coverage": round(float(np.mean(cover)), 4) if cover else None}
    return out


def _deployed_tracks(models: dict, db: Session, test: list[SwingData], tmp: Path,
                     progress: Callable[[str], None]) -> dict[str, list[club_mod.ShaftTrack]]:
    """Run each model over each held-out swing the way the pipeline would."""
    from app.training.club_inference import track_frames

    out = {name: [] for name in ["line", *models]}
    for n, d in enumerate(test, 1):
        progress(f"evaluating swing {n}/{len(test)}")
        seq, pose, window, proxy = _swing_inputs(db, uuid.UUID(d.swing_id), tmp)
        out["line"].append(d.line)
        f0, f1 = window
        frames, background, _, _ = club_mod.read_frames(proxy, f0, f1)
        grip = club_mod.grip_points(pose)
        torso = club_mod.torso_px(pose, f0)
        for name, m in models.items():
            out[name].append(track_frames(m, frames, background, f0, pose.num_frames, grip, torso, pose.fps))
    return out


def auto_promote_club(db: Session, m: Model, cfg: ClubTrainConfig | None = None) -> bool:
    """Hands-off promotion, judged on your held-out shaft checks: more accurate than the line
    tracker where you had to fix the shaft (lower median error), at least as many of all your
    checked frames within 10 degrees (so it doesn't lose the ones you confirmed), agreeing with the
    tracker where the tracker was confident, and no worse than the detector active now. Then every
    swing is re-tracked with it. With confirmations only there's no evidence it's better, so the
    tracker stays (you can still promote it by hand)."""
    cfg = cfg or ClubTrainConfig()
    em = m.eval_metrics or {}

    def part(model: str, key: str) -> dict:
        return (em.get(model) or {}).get(key) or {}

    mine, line = part("learned", "human"), part("line", "human")
    mine_fix, line_fix = part("learned", "fixes"), part("line", "fixes")
    if mine.get("n", 0) < cfg.min_test_frames or not line.get("n") or not mine_fix.get("n") or not line_fix.get("n"):
        return False
    if mine_fix["median_err_deg"] >= line_fix["median_err_deg"] or mine["within_10"] < line["within_10"]:
        return False
    agree = part("learned", "agreement")
    if agree.get("n") and agree["median_diff_deg"] > MAX_AGREEMENT_DEG:
        return False
    cur, cur_fix = part("current", "human"), part("current", "fixes")
    if cur.get("n") and mine["within_10"] < cur["within_10"]:
        return False
    if cur_fix.get("n") and mine_fix["median_err_deg"] > cur_fix["median_err_deg"]:
        return False
    for other in db.scalars(select(Model).where(Model.task == TASK_CLUB, Model.status == ModelStatus.active)):
        other.status = ModelStatus.deprecated
    m.status = ModelStatus.active
    db.flush()
    from app.automation import queue_club_retrack

    queue_club_retrack(db)
    db.commit()
    log.info("auto-promoted %s %s (fixes: median %.1f deg vs line %.1f)", m.name, m.version,
             mine_fix["median_err_deg"], line_fix["median_err_deg"])
    return True


# --- Entry point (trainer job) -----------------------------------------------------------------


def run_club_training(db: Session, payload: dict, on_stage: Callable[[str], None] = lambda s: None) -> Model | None:
    import torch

    cfg = ClubTrainConfig(**payload.get("config", {}))
    torch.set_num_threads(get_settings().train_workers or available_cpus())
    st = data_status(db, cfg)
    if not st["ready"]:
        # Not a failure: the gate re-opens by itself as swings and shaft checks come in.
        on_stage("not enough data yet")
        log.info("club training skipped, gate not met: %s", st)
        return None

    swings = trackable_swings(db)
    test_ids = split_swings(labelled_counts(db, swings))
    data: list[SwingData] = []
    with tempfile.TemporaryDirectory(prefix="hitfar-club-") as tmp_s:
        tmp = Path(tmp_s)
        for n, sid in enumerate(swings, 1):
            on_stage(f"club data {n}/{len(swings)}")
            try:
                d = swing_data(db, sid, "test" if sid in test_ids else "train", cfg, tmp)
            except Exception:
                log.exception("club data for swing %s failed; skipping it", sid)
                continue
            if d is not None and len(d.frames):
                data.append(d)
            for p in tmp.glob("*.mp4"):
                if p.stem != str(sid):
                    p.unlink(missing_ok=True)  # keep disk use to one video at a time
        train_data = [d for d in data if d.split == "train"]
        test_data = [d for d in data if d.split == "test"]
        if not train_data or not test_data:
            on_stage("not enough usable data yet")
            return None

        version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        storage = get_storage()
        manifest = {
            "version": version, "crop_version": CROP_VERSION, "config": asdict(cfg), "data": st,
            "swings": [{"swing_id": d.swing_id, "split": d.split, "frames": int(len(d.frames)),
                        "human": int(d.human.sum()), "clubhead_points": int(np.isfinite(d.head[:, 0]).sum())}
                       for d in data],
        }
        manifest_key = f"datasets/club_training/{version}.json"
        storage.put_bytes(manifest_key, json.dumps(manifest).encode(), "application/json")
        ds = Dataset(name=f"club-{version}", task=TASK_CLUB, source=DatasetSource.self_labeled,
                     manifest_uri=manifest_key, num_samples=int(sum(len(d.frames) for d in data)))
        db.add(ds)
        db.commit()

        model, train_info = train(train_data, cfg, on_stage)

        on_stage("evaluating")
        candidates = {"learned": model}
        current = active_club_model(db)
        if current.checkpoint_uri:
            from app.training.club_inference import load_checkpoint

            try:
                candidates["current"] = load_checkpoint(current.checkpoint_uri)
            except Exception:
                log.exception("could not load the active club model for comparison")
        test_labels = [(d, club_labels(db, uuid.UUID(d.swing_id))) for d in test_data]
        tracks = _deployed_tracks(candidates, db, test_data, tmp, on_stage)
    metrics = {
        **evaluate_tracks(tracks, test_labels),
        "data": {"labelled_frames": st["labelled_frames"], "swings": st["swings"]},
        "n_train_swings": len(train_data), "n_test_swings": len(test_data),
        "n_train_human": int(sum(d.human.sum() for d in train_data)),
        "n_train_pseudo": int(sum((~d.human).sum() for d in train_data)),
        "n_train_clubhead_points": int(sum(np.isfinite(d.head[:, 0]).sum() for d in train_data)),
        **({"current_version": current.version} if "current" in candidates else {}),
        **train_info,
    }

    buf = io.BytesIO()
    torch.save({"state_dict": model.state_dict(), "width": cfg.width, "crop_version": CROP_VERSION}, buf)
    ckpt_key = f"models/{MODEL_NAME}/{version}/model.pt"
    storage.put_bytes(ckpt_key, buf.getvalue())
    m = db.scalar(select(Model).where(Model.name == MODEL_NAME, Model.version == version))
    if m is not None:  # two runs in the same minute: keep names unique
        version = f"{version}-{uuid.uuid4().hex[:4]}"
    m = Model(name=MODEL_NAME, version=version, task=TASK_CLUB, checkpoint_uri=ckpt_key, trained_on_dataset_id=ds.id,
              eval_metrics=metrics, status=ModelStatus.experimental,
              notes="Heatmap CNN (shaft direction + clubhead) on crops around the hands; trained on line-tracker "
                    "pseudo-labels + your shaft checks (stage 2)")
    db.add(m)
    db.commit()
    if auto_promote_club(db, m, cfg):
        metrics["auto_promoted"] = True
        m.eval_metrics = dict(metrics)
        db.commit()
    on_stage("done")
    log.info("trained %s %s: %s", MODEL_NAME, version, json.dumps(metrics)[:800])
    return m

