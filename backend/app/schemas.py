import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.models import CameraRole, EventType, JobStatus, ModelStatus, VideoStatus


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class SessionIn(BaseModel):
    recorded_at: datetime | None = None
    name: str | None = None
    location: str | None = None
    club_used: str | None = None
    notes: str | None = None


class SessionPatch(BaseModel):
    recorded_at: datetime | None = None
    name: str | None = None
    location: str | None = None
    club_used: str | None = None
    notes: str | None = None


class SessionOut(ORM):
    id: uuid.UUID
    recorded_at: datetime
    name: str | None
    location: str | None
    club_used: str | None
    notes: str | None
    created_at: datetime


class SessionSummary(SessionOut):
    num_videos: int
    num_swings: int


class JobOut(ORM):
    id: uuid.UUID
    type: str
    status: JobStatus
    stage: str | None
    attempts: int
    max_attempts: int
    error: str | None
    updated_at: datetime


class VideoOut(ORM):
    id: uuid.UUID
    session_id: uuid.UUID
    uploaded_at: datetime
    original_filename: str | None
    camera_role: CameraRole
    status: VideoStatus
    error: str | None
    fps: float | None
    width: int | None
    height: int | None
    num_frames: int | None
    duration_s: float | None


Shape = Literal["slice", "fade", "straight", "draw", "hook"]
StartLine = Literal["left", "straight", "right"]
Contact = Literal["fat", "solid", "thin"]


class ShotOutcomeIn(BaseModel):
    """Fields left out are unchanged; null clears one. Clearing everything removes the tag."""

    shape: Shape | None = None
    start_line: StartLine | None = None
    contact: Contact | None = None
    source: Literal["self", "launch_monitor"] | None = None
    club_path: float | None = None
    face_to_path: float | None = None
    face_angle: float | None = None
    carry: float | None = None
    offline: float | None = None


class ShotOutcomeOut(ORM):
    shape: str | None
    start_line: str | None
    contact: str | None
    source: str
    club_path: float | None
    face_to_path: float | None
    face_angle: float | None
    carry: float | None
    offline: float | None
    updated_at: datetime


class VideoWithStatus(VideoOut):
    job: JobOut | None = None
    swing_id: uuid.UUID | None = None
    outcome: ShotOutcomeOut | None = None


class SessionDetail(SessionOut):
    videos: list[VideoWithStatus]


class UploadIn(BaseModel):
    filename: str
    content_type: str = "video/mp4"
    camera_role: CameraRole = CameraRole.face_on


class UploadOut(BaseModel):
    video: VideoOut
    upload_url: str
    upload_headers: dict[str, str]


class ReprocessIn(BaseModel):
    force: bool = True


class ModelRef(ORM):
    id: uuid.UUID
    name: str
    version: str


class ModelOut(ORM):
    id: uuid.UUID
    name: str
    task: str
    version: str
    status: ModelStatus
    checkpoint_uri: str | None
    eval_metrics: dict | None
    notes: str | None
    created_at: datetime


class EventOut(BaseModel):
    event_type: EventType
    frame_index: int
    confidence: float | None
    predicted_frame_index: int | None
    corrected: bool
    model: ModelRef | None


class EventCorrectionIn(BaseModel):
    frame_index: int | None  # null reverts to the model prediction


class MetricOut(ORM):
    metric_name: str
    event_ref: EventType | None
    value: float
    unit: str
    is_estimate: bool


class PoseInfo(BaseModel):
    model: ModelRef
    num_frames: int
    mean_confidence: float | None
    landmark_schema_version: str


class SwingVideo(VideoOut):
    playback_url: str | None


class SwingDetail(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    is_reference: bool
    club_used: str | None
    created_at: datetime
    video: SwingVideo
    job: JobOut | None
    pose: PoseInfo | None
    events: list[EventOut]
    events_reviewed: bool
    outcome: ShotOutcomeOut | None
    metrics: list[MetricOut]
    pipeline_version: str


class SwingPatch(BaseModel):
    is_reference: bool | None = None
    club_used: str | None = None


class PoseFrames(BaseModel):
    fps: float
    width: int
    height: int
    connections: list[tuple[int, int]]
    # frames[t] = flat [x0, y0, v0, x1, y1, v1, ...] in pixels, or null when no person detected
    frames: list[list[float] | None]


class ClubFrames(BaseModel):
    fps: float
    length_px: float
    angle_deg: list[float | None]  # image plane, grip -> clubhead, 0 = +x (right), 90 = +y (down)
    confidence: list[float]
    grip: list[list[float] | None]
    corrected: list[int]  # frames you labelled (fixed or confirmed)
    confirmed: list[int] = []  # of those, the ones you confirmed as right
    # Clubhead per frame (pixels) where the tracker finds it (learned detector only).
    clubhead: list[list[float] | None] | None = None
    model_name: str | None = None
    model_version: str | None = None


class ClubCorrectionIn(BaseModel):
    angle_deg: float | None = None  # null (without confirm) reverts that frame to the tracker
    clubhead_xy: tuple[float, float] | None = None  # where you clicked the clubhead (pixels)
    confirm: bool = False  # "the shaft shown on this frame is right"


# --- Diagnosis ---------------------------------------------------------------------------------

Verdict = Literal["confirmed", "rejected", "unsure"]
Labeler = Literal["self", "instructor"]


class FaultOut(BaseModel):
    name: str
    title: str
    description: str
    assessable: bool
    partial: bool
    views: list[str]
    needs: list[str]
    rules: list[str]


class DiagnoseIn(BaseModel):
    symptom_text: str = ""


class VerdictIn(BaseModel):
    verdict: Verdict | None  # null clears the verdict
    labeled_by: Labeler = "self"


class DiagnosisOut(BaseModel):
    id: uuid.UUID
    swing_id: uuid.UUID
    created_at: datetime
    symptom_text: str | None
    error: str | None
    output: dict | None
    rule_hits: list | None
    # {"self": {fault: verdict}, "instructor": {fault: verdict}}
    verdicts: dict[str, dict[str, str]]
    model: ModelRef | None
    served_model: str | None
    stale: bool
