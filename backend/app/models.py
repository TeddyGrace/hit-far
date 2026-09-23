"""Database schema.

Mirrors the project brief's data model. Large artifacts (videos, keypoint arrays) live in object
storage; rows hold their URIs. Every model output row carries the `model_id` that produced it.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _created() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


def _pg_enum(cls: type[enum.Enum], name: str) -> Enum:
    return Enum(cls, name=name, values_callable=lambda e: [m.value for m in e])


class CameraRole(str, enum.Enum):
    face_on = "face_on"
    down_the_line = "down_the_line"
    other = "other"


class VideoStatus(str, enum.Enum):
    pending_upload = "pending_upload"
    uploaded = "uploaded"
    preprocessed = "preprocessed"
    failed = "failed"


class Pose3DMethod(str, enum.Enum):
    triangulation = "triangulation"
    monocular_lift = "monocular_lift"


class EventType(str, enum.Enum):
    address = "address"
    toe_up = "toe_up"
    mid_backswing = "mid_backswing"
    top = "top"
    mid_downswing = "mid_downswing"
    impact = "impact"
    mid_follow_through = "mid_follow_through"
    finish = "finish"


EVENT_ORDER = list(EventType)
EVENT_TYPE_ENUM = _pg_enum(EventType, "event_type")


class ProfileType(str, enum.Enum):
    archetype = "archetype"
    reference_swing = "reference_swing"


class ModelStatus(str, enum.Enum):
    active = "active"
    experimental = "experimental"
    deprecated = "deprecated"


class DatasetSource(str, enum.Enum):
    golfdb = "golfdb"
    self_labeled = "self_labeled"
    mixed = "mixed"


class JobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


# --- Raw capture -------------------------------------------------------------------------------


class RecordingSession(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = _pk()
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    location: Mapped[str | None] = mapped_column(String(200))
    club_used: Mapped[str | None] = mapped_column(String(50))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()

    videos: Mapped[list["Video"]] = relationship(
        back_populates="session", order_by="Video.uploaded_at", cascade="all, delete-orphan", passive_deletes=True
    )
    swings: Mapped[list["Swing"]] = relationship(
        back_populates="session", order_by="Swing.created_at", cascade="all, delete-orphan", passive_deletes=True
    )


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[uuid.UUID] = _pk()
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    uploaded_at: Mapped[datetime] = _created()
    original_filename: Mapped[str | None] = mapped_column(String(300))
    content_type: Mapped[str | None] = mapped_column(String(100))
    file_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    # Browser-playable, constant-frame-rate, rotation-applied H.264 copy. All frame indices in the
    # system (pose, events) refer to frames of this proxy so the viewer and the models agree.
    proxy_uri: Mapped[str | None] = mapped_column(String(500))
    checksum: Mapped[str | None] = mapped_column(String(64), unique=True)
    camera_role: Mapped[CameraRole] = mapped_column(
        _pg_enum(CameraRole, "camera_role"), default=CameraRole.face_on, nullable=False
    )
    fps: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    num_frames: Mapped[int | None] = mapped_column(Integer)
    duration_s: Mapped[float | None] = mapped_column(Float)
    status: Mapped[VideoStatus] = mapped_column(
        _pg_enum(VideoStatus, "video_status"), default=VideoStatus.pending_upload, nullable=False
    )
    error: Mapped[str | None] = mapped_column(Text)

    session: Mapped[RecordingSession] = relationship(back_populates="videos")


class Swing(Base):
    """One physical swing, clipped from one (default) or two synced camera angles."""

    __tablename__ = "swings"
    __table_args__ = (Index("ix_swings_video_ids", "video_ids", postgresql_using="gin"),)

    id: Mapped[uuid.UUID] = _pk()
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    video_ids: Mapped[list[uuid.UUID]] = mapped_column(ARRAY(UUID(as_uuid=True)), nullable=False)
    sync_offset_frames: Mapped[dict | None] = mapped_column(JSONB)
    clip_start: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # proxy frame index
    clip_end: Mapped[int | None] = mapped_column(Integer)  # exclusive
    is_reference: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    club_used: Mapped[str | None] = mapped_column(String(50))
    created_at: Mapped[datetime] = _created()

    session: Mapped[RecordingSession] = relationship(back_populates="swings")


# --- MLOps backbone ----------------------------------------------------------------------------


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = _pk()
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    task: Mapped[str] = mapped_column(String(50), nullable=False)
    source: Mapped[DatasetSource] = mapped_column(_pg_enum(DatasetSource, "dataset_source"), nullable=False)
    manifest_uri: Mapped[str | None] = mapped_column(String(500))
    num_samples: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = _created()


class Model(Base):
    __tablename__ = "models"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[uuid.UUID] = _pk()
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    task: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    checkpoint_uri: Mapped[str | None] = mapped_column(String(500))
    trained_on_dataset_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("datasets.id"))
    eval_metrics: Mapped[dict | None] = mapped_column(JSONB)
    status: Mapped[ModelStatus] = mapped_column(
        _pg_enum(ModelStatus, "model_status"), default=ModelStatus.experimental, nullable=False
    )
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()


# --- Model outputs -----------------------------------------------------------------------------


class PoseSequence(Base):
    __tablename__ = "pose_sequences"

    id: Mapped[uuid.UUID] = _pk()
    swing_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("swings.id", ondelete="CASCADE"), index=True)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("models.id"))
    keypoint_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    landmark_schema_version: Mapped[str] = mapped_column(String(50), nullable=False)
    num_frames: Mapped[int] = mapped_column(Integer, nullable=False)
    mean_confidence: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = _created()


class Pose3DSequence(Base):
    __tablename__ = "pose_3d_sequences"

    id: Mapped[uuid.UUID] = _pk()
    swing_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("swings.id", ondelete="CASCADE"), index=True)
    model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("models.id"))
    joint_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    method: Mapped[Pose3DMethod] = mapped_column(_pg_enum(Pose3DMethod, "pose_3d_method"), nullable=False)
    # Populated for monocular_lift once triangulated calibration data exists.
    error_estimate: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _created()


class ClubTrack(Base):
    __tablename__ = "club_tracks"

    id: Mapped[uuid.UUID] = _pk()
    swing_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("swings.id", ondelete="CASCADE"), index=True)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("models.id"))
    track_uri: Mapped[str] = mapped_column(String(500), nullable=False)
    num_frames: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = _created()


class SwingEvent(Base):
    __tablename__ = "swing_events"
    __table_args__ = (UniqueConstraint("swing_id", "model_id", "event_type"),)

    id: Mapped[uuid.UUID] = _pk()
    swing_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("swings.id", ondelete="CASCADE"), index=True)
    model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("models.id"))
    event_type: Mapped[EventType] = mapped_column(EVENT_TYPE_ENUM, nullable=False)
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = _created()


class Metric(Base):
    """Deterministic geometry over pose + events. Never a model output."""

    __tablename__ = "metrics"
    __table_args__ = (Index("ix_metrics_swing_version", "swing_id", "pipeline_version"),)

    id: Mapped[uuid.UUID] = _pk()
    swing_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("swings.id", ondelete="CASCADE"))
    pipeline_version: Mapped[str] = mapped_column(String(20), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    event_ref: Mapped[EventType | None] = mapped_column(EVENT_TYPE_ENUM)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(30), nullable=False)
    # True when derived from monocular 3D (an estimate with unknown error), False for 2D image-plane
    # geometry or timing.
    is_estimate: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = _created()


# --- Comparison + diagnosis (schema only in v0) ------------------------------------------------


class ReferenceProfile(Base):
    __tablename__ = "reference_profiles"

    id: Mapped[uuid.UUID] = _pk()
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    type: Mapped[ProfileType] = mapped_column(_pg_enum(ProfileType, "profile_type"), nullable=False)
    source_swing_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("swings.id", ondelete="SET NULL"))
    metric_targets: Mapped[dict | None] = mapped_column(JSONB)  # archetype: {metric: {mean, range}}
    created_at: Mapped[datetime] = _created()


class Comparison(Base):
    __tablename__ = "comparisons"

    id: Mapped[uuid.UUID] = _pk()
    swing_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("swings.id", ondelete="CASCADE"), index=True)
    reference_profile_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("reference_profiles.id", ondelete="CASCADE"))
    alignment: Mapped[dict | None] = mapped_column(JSONB)  # DTW map: user event frame -> reference frame
    metric_deltas: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _created()


class FaultLabel(Base):
    __tablename__ = "fault_labels"

    id: Mapped[uuid.UUID] = _pk()
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    metric_signature: Mapped[dict | None] = mapped_column(JSONB)


class Diagnosis(Base):
    __tablename__ = "diagnoses"

    id: Mapped[uuid.UUID] = _pk()
    swing_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("swings.id", ondelete="CASCADE"), index=True)
    user_symptom_text: Mapped[str | None] = mapped_column(Text)
    predicted_faults: Mapped[dict | None] = mapped_column(JSONB)  # {fault_id: probability}
    model_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("models.id"))
    human_confirmed_faults: Mapped[dict | None] = mapped_column(JSONB)
    narrative: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()


# --- Human-in-the-loop -------------------------------------------------------------------------


class Label(Base):
    """A human correction. Shared by pose / club / event / fault workflows.

    For event corrections (v0): task='event', target_type='swing', target_id=swing id,
    corrected_value={"event_type": ..., "frame_index": int | null, "predicted_frame_index": ...,
    "predicted_model_id": ...}. frame_index=null means "revert to the model prediction".
    """

    __tablename__ = "labels"
    __table_args__ = (Index("ix_labels_target", "target_type", "target_id"),)

    id: Mapped[uuid.UUID] = _pk()
    task: Mapped[str] = mapped_column(String(30), nullable=False)
    target_type: Mapped[str] = mapped_column(String(30), nullable=False)
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    corrected_value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = _created()


# --- Job queue ---------------------------------------------------------------------------------


class Job(Base):
    """Postgres-backed work queue, consumed with SELECT ... FOR UPDATE SKIP LOCKED."""

    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_claim", "status", "run_after"),)

    id: Mapped[uuid.UUID] = _pk()
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    # The row the job operates on (video or swing), for status lookups from the UI.
    subject_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), index=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        _pg_enum(JobStatus, "job_status"), default=JobStatus.queued, nullable=False
    )
    stage: Mapped[str | None] = mapped_column(String(50))
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
