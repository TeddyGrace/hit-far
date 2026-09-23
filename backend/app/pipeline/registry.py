"""Model registry helpers: every model output row points at a `models` row."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Model, ModelStatus

TASK_POSE_2D = "pose_2d"
TASK_EVENTS = "event_segmentation"
TASK_CLUB = "club_tracking"


def get_or_create_model(
    db: Session, name: str, version: str, task: str, notes: str | None = None, checkpoint_uri: str | None = None
) -> Model:
    m = db.scalar(select(Model).where(Model.name == name, Model.version == version))
    if m is None:
        # Baselines are active on first registration; trained candidates are registered as
        # experimental by the training workflow and promoted explicitly.
        m = Model(name=name, version=version, task=task, notes=notes, checkpoint_uri=checkpoint_uri,
                  status=ModelStatus.active)
        db.add(m)
        db.flush()
    return m
