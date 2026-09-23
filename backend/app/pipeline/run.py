"""Pipeline orchestration: preprocess -> swing -> pose -> events -> metrics.

Each stage is idempotent so a retried job resumes cleanly.
"""

import logging
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import (
    EventType,
    Label,
    Metric,
    Model,
    ModelStatus,
    Pose3DMethod,
    Pose3DSequence,
    PoseSequence,
    Swing,
    SwingEvent,
    Video,
    VideoStatus,
)
from app.pipeline import events as events_mod
from app.pipeline import ingest, pose as pose_mod
from app.pipeline.metrics import PIPELINE_VERSION, compute_metrics
from app.pipeline.posedata import PoseData
from app.pipeline.registry import TASK_EVENTS, TASK_POSE_2D, get_or_create_model
from app.storage import get_storage

log = logging.getLogger(__name__)

EVENT_LABEL_TASK = "event"


class PermanentError(Exception):
    """Failure that retrying won't fix (bad input)."""


# --- Stages ------------------------------------------------------------------------------------


def preprocess_video(db: Session, video: Video, workdir: Path) -> None:
    if video.status == VideoStatus.preprocessed and video.proxy_uri:
        return
    st = get_storage()
    ext = Path(video.original_filename or "video.mp4").suffix.lower() or ".mp4"
    raw = workdir / f"raw{ext}"
    st.download_file(video.file_uri, raw)

    checksum = ingest.sha256_file(raw)
    dup = db.scalar(select(Video).where(Video.checksum == checksum, Video.id != video.id))
    if dup is not None:
        st.delete(video.file_uri)
        raise PermanentError(f"duplicate of video {dup.id} (same SHA-256)")

    canonical = f"videos/{checksum}{ext}"
    if video.file_uri != canonical:
        st.copy(video.file_uri, canonical)
        st.delete(video.file_uri)
        video.file_uri = canonical
    video.checksum = checksum

    proxy = workdir / "proxy.mp4"
    try:
        ingest.make_proxy(raw, proxy)
        info = ingest.probe(proxy)
    except ingest.IngestError as e:
        raise PermanentError(str(e)) from e
    proxy_key = f"proxies/{video.id}.mp4"
    st.upload_file(proxy_key, proxy, "video/mp4")
    video.proxy_uri = proxy_key
    video.fps, video.width, video.height = info.fps, info.width, info.height
    video.num_frames, video.duration_s = info.num_frames, info.duration_s
    video.status = VideoStatus.preprocessed
    video.error = None
    db.commit()


def ensure_swing(db: Session, video: Video) -> Swing:
    """v0: one swing per uploaded video, spanning the whole clip."""
    swing = db.scalar(select(Swing).where(Swing.video_ids.any(video.id)).order_by(Swing.created_at).limit(1))
    if swing is None:
        swing = Swing(
            session_id=video.session_id, video_ids=[video.id], clip_start=0, clip_end=video.num_frames,
            club_used=video.session.club_used,
        )
        db.add(swing)
        db.commit()
    return swing


def run_pose_stage(
    db: Session, swing: Swing, video: Video, workdir: Path, force: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> PoseSequence:
    model = get_or_create_model(db, pose_mod.MODEL_NAME, pose_mod.MODEL_VERSION, TASK_POSE_2D,
                                notes="MediaPipe baseline; world landmarks used as monocular 3D estimate")
    existing = db.scalar(
        select(PoseSequence)
        .where(PoseSequence.swing_id == swing.id, PoseSequence.model_id == model.id)
        .order_by(PoseSequence.created_at.desc())
        .limit(1)
    )
    if existing is not None and not force:
        return existing

    st = get_storage()
    proxy = workdir / "proxy.mp4"
    if not proxy.exists():
        st.download_file(video.proxy_uri, proxy)
    pose = pose_mod.run_pose(proxy, progress=progress)

    seq_id = uuid.uuid4()
    key = f"artifacts/{swing.id}/{model.id}/pose-{seq_id}.npz"  # never overwritten
    st.put_bytes(key, pose.to_npz())
    seq = PoseSequence(
        id=seq_id, swing_id=swing.id, video_id=video.id, model_id=model.id, keypoint_uri=key,
        landmark_schema_version=pose.schema, num_frames=pose.num_frames, mean_confidence=pose.mean_confidence(),
    )
    db.add(seq)
    # World landmarks live in the same artifact; recorded as a monocular 3D estimate with no
    # error bound until triangulated calibration sessions exist.
    db.add(Pose3DSequence(swing_id=swing.id, model_id=model.id, joint_uri=key, method=Pose3DMethod.monocular_lift))
    db.commit()
    return seq


def active_event_model(db: Session) -> Model | None:
    """The active event model; a trained (checkpointed) model wins over the rule baseline."""
    rows = db.scalars(
        select(Model).where(Model.task == TASK_EVENTS, Model.status == ModelStatus.active)
        .order_by(Model.created_at.desc())
    ).all()
    trained = [m for m in rows if m.checkpoint_uri]
    return trained[0] if trained else (rows[0] if rows else None)


def run_event_stage(db: Session, swing: Swing, pose: PoseData) -> None:
    rules = get_or_create_model(db, events_mod.MODEL_NAME, events_mod.MODEL_VERSION, TASK_EVENTS,
                                notes="Rule-based baseline over the 2D pose time series (face-on)")
    model = active_event_model(db) or rules
    handedness = get_settings().golfer_handedness
    detected = None
    if model.checkpoint_uri:
        try:
            from app.training.inference import detect_events_learned

            detected = detect_events_learned(model.checkpoint_uri, pose, handedness)
        except Exception:
            # Never lose a swing to a model problem: fall back to the rule baseline.
            log.exception("learned event model %s failed; using rules", model.version)
            model = rules
    if detected is None:
        try:
            detected = events_mod.detect_events(pose, handedness).events
        except events_mod.EventDetectionError as e:
            raise PermanentError(str(e)) from e
    # Same model + version is deterministic, so replace its rows; other models' rows are kept.
    db.execute(delete(SwingEvent).where(SwingEvent.swing_id == swing.id, SwingEvent.model_id == model.id))
    for et, ev in detected.items():
        db.add(SwingEvent(swing_id=swing.id, model_id=model.id, event_type=et, frame_index=ev.frame,
                          confidence=ev.confidence))
    db.commit()


def redetect_events(db: Session, swing_id: uuid.UUID) -> None:
    """Re-run event detection with the currently active model (reuses cached pose)."""
    swing = db.get(Swing, swing_id)
    seq = latest_pose_sequence(db, swing_id) if swing else None
    if seq is None:
        raise PermanentError(f"swing {swing_id} has no pose yet")
    pose = load_pose(seq)
    run_event_stage(db, swing, pose)
    recompute_metrics(db, swing, pose)


# --- Queries shared with the API ---------------------------------------------------------------


def latest_pose_sequence(db: Session, swing_id: uuid.UUID) -> PoseSequence | None:
    return db.scalar(
        select(PoseSequence).where(PoseSequence.swing_id == swing_id).order_by(PoseSequence.created_at.desc()).limit(1)
    )


def load_pose(seq: PoseSequence) -> PoseData:
    return PoseData.from_npz(get_storage().get_bytes(seq.keypoint_uri))


def predicted_events(db: Session, swing_id: uuid.UUID) -> dict[EventType, SwingEvent]:
    """Newest prediction per event type (the most recent model run wins)."""
    rows = db.scalars(
        select(SwingEvent).where(SwingEvent.swing_id == swing_id).order_by(SwingEvent.created_at)
    ).all()
    out: dict[EventType, SwingEvent] = {}
    for r in rows:
        out[r.event_type] = r
    return out


def event_corrections(db: Session, swing_id: uuid.UUID) -> dict[EventType, Label]:
    """Latest human correction per event type; a null frame_index means 'reverted'."""
    rows = db.scalars(
        select(Label)
        .where(Label.task == EVENT_LABEL_TASK, Label.target_type == "swing", Label.target_id == swing_id)
        .order_by(Label.created_at)
    ).all()
    out: dict[EventType, Label] = {}
    for r in rows:
        et = EventType(r.corrected_value["event_type"])
        if r.corrected_value.get("frame_index") is None:
            out.pop(et, None)
        else:
            out[et] = r
    return out


def effective_events(db: Session, swing_id: uuid.UUID) -> dict[EventType, int]:
    frames = {et: e.frame_index for et, e in predicted_events(db, swing_id).items()}
    for et, lbl in event_corrections(db, swing_id).items():
        frames[et] = int(lbl.corrected_value["frame_index"])
    return frames


def recompute_metrics(db: Session, swing: Swing, pose: PoseData | None = None) -> list[Metric]:
    if pose is None:
        seq = latest_pose_sequence(db, swing.id)
        if seq is None:
            return []
        pose = load_pose(seq)
    values = compute_metrics(pose, effective_events(db, swing.id), get_settings().golfer_handedness)
    db.execute(delete(Metric).where(Metric.swing_id == swing.id, Metric.pipeline_version == PIPELINE_VERSION))
    rows = [
        Metric(swing_id=swing.id, pipeline_version=PIPELINE_VERSION, metric_name=v.metric_name,
               event_ref=v.event_ref, value=v.value, unit=v.unit, is_estimate=v.is_estimate)
        for v in values
    ]
    db.add_all(rows)
    db.commit()
    return rows


def swings_needing_metrics(db: Session) -> list[uuid.UUID]:
    """Swings with pose but no metrics at the current PIPELINE_VERSION (e.g. after a formula bump)."""
    has_pose = select(PoseSequence.swing_id).distinct()
    current = select(Metric.swing_id).where(Metric.pipeline_version == PIPELINE_VERSION).distinct()
    return list(db.scalars(select(Swing.id).where(Swing.id.in_(has_pose), Swing.id.not_in(current))).all())


def recompute_all_metrics(db: Session, on_stage: Callable[[str], None] = lambda s: None) -> int:
    """Recompute metrics for every swing with pose (pose is cached, so this is seconds per swing)."""
    ids = list(db.scalars(select(PoseSequence.swing_id).distinct()).all())
    for n, sid in enumerate(ids, 1):
        swing = db.get(Swing, sid)
        if swing is not None:
            recompute_metrics(db, swing)
        on_stage(f"metrics {n}/{len(ids)}")
    return len(ids)


# --- Entry point used by the worker ------------------------------------------------------------


def process_video(db: Session, video_id: uuid.UUID, force: bool = False,
                  on_stage: Callable[[str], None] = lambda s: None) -> None:
    video = db.get(Video, video_id)
    if video is None:
        raise PermanentError(f"video {video_id} not found")
    with tempfile.TemporaryDirectory(prefix="hitfar-") as tmp:
        workdir = Path(tmp)
        try:
            on_stage("preprocess")
            preprocess_video(db, video, workdir)
        except PermanentError as e:
            video.status = VideoStatus.failed
            video.error = str(e)
            db.commit()
            raise
        on_stage("swing")
        swing = ensure_swing(db, video)
        on_stage("pose")
        seq = run_pose_stage(db, swing, video, workdir, force=force)
        pose = load_pose(seq)
        on_stage("events")
        run_event_stage(db, swing, pose)
        on_stage("metrics")
        recompute_metrics(db, swing, pose)
