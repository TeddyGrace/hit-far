import math
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import jobs
from app.auth import current_user, require_auth
from app.db import get_db
from app.models import EVENT_ORDER, EventType, Label, Metric, Model, RecordingSession, Swing, User, Video
from app.outcomes.service import get_outcome
from app.pipeline.landmarks import CONNECTIONS
from app.pipeline.metrics import PIPELINE_VERSION
from app.pipeline.run import (
    CLUB_LABEL_TASK,
    EVENT_LABEL_TASK,
    club_labels,
    effective_club,
    event_corrections,
    latest_club_track,
    latest_pose_sequence,
    load_pose,
    predicted_events,
    recompute_metrics,
)
from app.schemas import (
    ClubCorrectionIn,
    ClubFrames,
    EventCorrectionIn,
    EventOut,
    JobOut,
    MetricOut,
    ModelRef,
    PoseFrames,
    PoseInfo,
    SwingDetail,
    SwingPatch,
    SwingVideo,
    VideoOut,
)
from app.storage import get_storage

router = APIRouter(prefix="/api/swings", tags=["swings"], dependencies=[Depends(require_auth)])


def get_swing_or_404(db: Session, swing_id: uuid.UUID, user: User) -> Swing:
    s = db.get(Swing, swing_id)
    if s is None or db.get(RecordingSession, s.session_id).user_id != user.id:
        raise HTTPException(404, "swing not found")
    return s


def _primary_video(db: Session, swing: Swing) -> Video:
    return db.get(Video, swing.video_ids[0])


def _is_reviewed(db: Session, swing_id: uuid.UUID) -> bool:
    last = db.scalar(
        select(Label).where(Label.task == "event_review", Label.target_type == "swing", Label.target_id == swing_id)
        .order_by(Label.created_at.desc()).limit(1)
    )
    return bool(last and last.corrected_value.get("reviewed"))


def swing_detail(db: Session, swing: Swing) -> SwingDetail:
    video = _primary_video(db, swing)
    playback = get_storage().presign_get(video.proxy_uri, 6 * 3600) if video.proxy_uri else None
    job = jobs.latest_for(db, video.id)

    models: dict[uuid.UUID, Model] = {}

    def model_ref(mid: uuid.UUID | None) -> ModelRef | None:
        if mid is None:
            return None
        if mid not in models:
            models[mid] = db.get(Model, mid)
        return ModelRef.model_validate(models[mid])

    seq = latest_pose_sequence(db, swing.id)
    pose = None
    if seq is not None:
        pose = PoseInfo(model=model_ref(seq.model_id), num_frames=seq.num_frames,
                        mean_confidence=seq.mean_confidence, landmark_schema_version=seq.landmark_schema_version)

    preds = predicted_events(db, swing.id)
    corrs = event_corrections(db, swing.id)
    events = []
    for et in EVENT_ORDER:
        p, c = preds.get(et), corrs.get(et)
        if p is None and c is None:
            continue
        events.append(EventOut(
            event_type=et,
            frame_index=int(c.corrected_value["frame_index"]) if c else p.frame_index,
            confidence=p.confidence if p else None,
            predicted_frame_index=p.frame_index if p else None,
            corrected=c is not None,
            model=model_ref(p.model_id) if p else None,
        ))

    metrics = db.scalars(
        select(Metric).where(Metric.swing_id == swing.id, Metric.pipeline_version == PIPELINE_VERSION)
        .order_by(Metric.metric_name)
    ).all()

    return SwingDetail(
        id=swing.id, session_id=swing.session_id, is_reference=swing.is_reference, club_used=swing.club_used,
        created_at=swing.created_at,
        video=SwingVideo(**VideoOut.model_validate(video).model_dump(), playback_url=playback),
        job=JobOut.model_validate(job) if job else None,
        pose=pose, events=events, events_reviewed=_is_reviewed(db, swing.id),
        outcome=get_outcome(db, swing.id),
        metrics=[MetricOut.model_validate(m) for m in metrics],
        pipeline_version=PIPELINE_VERSION,
    )


@router.get("/{swing_id}", response_model=SwingDetail)
def get_swing(swing_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(current_user)):
    return swing_detail(db, get_swing_or_404(db, swing_id, user))


@router.patch("/{swing_id}", response_model=SwingDetail)
def update_swing(swing_id: uuid.UUID, body: SwingPatch, db: Session = Depends(get_db),
                 user: User = Depends(current_user)):
    swing = get_swing_or_404(db, swing_id, user)
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(swing, k, v)
    db.commit()
    return swing_detail(db, swing)


@router.get("/{swing_id}/pose", response_model=PoseFrames)
def get_pose(swing_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(current_user)):
    swing = get_swing_or_404(db, swing_id, user)
    seq = latest_pose_sequence(db, swing.id)
    if seq is None:
        raise HTTPException(404, "no pose for this swing yet")
    pose = load_pose(seq)
    frames: list[list[float] | None] = []
    for t in range(pose.num_frames):
        if math.isnan(pose.kp2d[t, 0, 0]):
            frames.append(None)
            continue
        flat: list[float] = []
        for j in range(pose.kp2d.shape[1]):
            x, y = pose.kp2d[t, j]
            flat += [round(float(x), 1), round(float(y), 1), round(float(pose.visibility[t, j]), 2)]
        frames.append(flat)
    return PoseFrames(fps=pose.fps, width=pose.width, height=pose.height, connections=CONNECTIONS, frames=frames)


@router.put("/{swing_id}/events/{event_type}", response_model=SwingDetail)
def correct_event(swing_id: uuid.UUID, event_type: EventType, body: EventCorrectionIn, db: Session = Depends(get_db),
                  user: User = Depends(current_user)):
    swing = get_swing_or_404(db, swing_id, user)
    video = _primary_video(db, swing)
    if body.frame_index is not None and video.num_frames and not 0 <= body.frame_index < video.num_frames:
        raise HTTPException(400, f"frame_index must be in [0, {video.num_frames})")
    pred = predicted_events(db, swing.id).get(event_type)
    db.add(Label(
        task=EVENT_LABEL_TASK, target_type="swing", target_id=swing.id,
        corrected_value={
            "event_type": event_type.value,
            "frame_index": body.frame_index,
            "predicted_frame_index": pred.frame_index if pred else None,
            "predicted_model_id": str(pred.model_id) if pred else None,
            "video_id": str(video.id),
        },
    ))
    db.commit()
    recompute_metrics(db, swing)
    return swing_detail(db, swing)


def _xy(p) -> list[float] | None:
    return None if not all(map(math.isfinite, p)) else [round(float(p[0]), 1), round(float(p[1]), 1)]


@router.get("/{swing_id}/club", response_model=ClubFrames)
def get_club(swing_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(current_user)):
    swing = get_swing_or_404(db, swing_id, user)
    track = effective_club(db, swing.id)
    if track is None:
        raise HTTPException(404, "no club track for this swing yet")
    labels = club_labels(db, swing.id)
    model = db.get(Model, latest_club_track(db, swing.id).model_id)
    return ClubFrames(
        fps=track.fps, length_px=round(track.length_px, 1),
        angle_deg=[None if not math.isfinite(a) else round(math.degrees(a) % 360, 1) for a in track.angle],
        confidence=[round(float(c), 2) for c in track.confidence],
        grip=[_xy(g) for g in track.grip],
        corrected=sorted(labels),
        confirmed=sorted(f for f, v in labels.items() if v.get("confirmed")),
        clubhead=None if track.clubhead is None else [_xy(p) for p in track.clubhead],
        model_name=model.name if model else None, model_version=model.version if model else None,
    )


@router.put("/{swing_id}/club/{frame_index}", response_model=SwingDetail)
def correct_club(swing_id: uuid.UUID, frame_index: int, body: ClubCorrectionIn, db: Session = Depends(get_db),
                 user: User = Depends(current_user)):
    """Label the shaft on one frame: a fix (direction in degrees, image plane, grip -> clubhead,
    0 = right, 90 = down, plus the clubhead point you clicked), or `confirm` that the shaft shown is
    right. angle_deg null (without confirm) reverts that frame to the tracker. Labels train the club
    detector and decide whether it replaces the line tracker."""
    swing = get_swing_or_404(db, swing_id, user)
    video = _primary_video(db, swing)
    if video.num_frames and not 0 <= frame_index < video.num_frames:
        raise HTTPException(400, f"frame_index must be in [0, {video.num_frames})")
    track = effective_club(db, swing.id)
    if track is None:
        raise HTTPException(409, "no club track for this swing yet")
    predicted = track.angle[frame_index] if frame_index < len(track.angle) else float("nan")
    angle = body.angle_deg
    if body.confirm:
        if not math.isfinite(predicted):
            raise HTTPException(409, "the shaft isn't tracked on this frame; fix it instead")
        angle = math.degrees(predicted)
    head = body.clubhead_xy
    if body.confirm and head is None:
        p = track.clubhead_at(frame_index)
        head = None if p is None else (float(p[0]), float(p[1]))
    db.add(Label(task=CLUB_LABEL_TASK, target_type="swing", target_id=swing.id, corrected_value={
        "frame_index": frame_index,
        "angle_deg": None if angle is None else angle % 360,
        "predicted_angle_deg": round(math.degrees(predicted) % 360, 2) if math.isfinite(predicted) else None,
        "clubhead_xy": None if head is None or angle is None else [round(head[0], 1), round(head[1], 1)],
        "confirmed": bool(body.confirm),
        "video_id": str(video.id),
    }))
    db.commit()
    recompute_metrics(db, swing)
    from app.training.train_club import maybe_queue_training

    maybe_queue_training(db)
    return swing_detail(db, swing)
