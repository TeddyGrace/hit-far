"""One feature row per swing: every deterministic metric at every event it's measured at."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Metric, ShotOutcome, Swing
from app.pipeline.metrics import PIPELINE_VERSION

# A feature must be measured on at least this share of the swings to be used; the models handle
# the remaining gaps (imputation / native missing-value support).
MIN_COVERAGE = 0.6


def feature_name(metric_name: str, event_ref: str | None) -> str:
    return f"{metric_name}@{event_ref}" if event_ref else metric_name


@dataclass
class FeatureInfo:
    unit: str
    is_estimate: bool


@dataclass
class Table:
    """Metrics (current pipeline version) and outcomes for every swing that has metrics."""

    swing_ids: list[uuid.UUID]
    created_at: dict[uuid.UUID, datetime]
    session_ids: dict[uuid.UUID, uuid.UUID]
    values: dict[uuid.UUID, dict[str, float]]
    outcomes: dict[uuid.UUID, dict]
    info: dict[str, FeatureInfo] = field(default_factory=dict)

    def matrix(self, swing_ids: list[uuid.UUID], names: list[str]) -> np.ndarray:
        return np.array([[self.values[s].get(n, np.nan) for n in names] for s in swing_ids], dtype=float).reshape(
            len(swing_ids), len(names))


def outcome_dict(o: ShotOutcome) -> dict:
    return {"shape": o.shape, "start_line": o.start_line, "contact": o.contact, "source": o.source,
            "club_path": o.club_path, "face_to_path": o.face_to_path, "face_angle": o.face_angle,
            "carry": o.carry, "offline": o.offline}


def load_table(db: Session) -> Table:
    rows = db.execute(
        select(Metric.swing_id, Metric.metric_name, Metric.event_ref, Metric.value, Metric.unit, Metric.is_estimate)
        .where(Metric.pipeline_version == PIPELINE_VERSION)
    ).all()
    values: dict[uuid.UUID, dict[str, float]] = {}
    info: dict[str, FeatureInfo] = {}
    for sid, name, ev, value, unit, est in rows:
        fname = feature_name(name, ev.value if ev is not None else None)
        values.setdefault(sid, {})[fname] = value
        info.setdefault(fname, FeatureInfo(unit, est))
    swings = db.execute(select(Swing.id, Swing.created_at, Swing.session_id).where(Swing.id.in_(list(values)))).all()
    swings = sorted(swings, key=lambda r: r.created_at)
    outcomes = {o.swing_id: outcome_dict(o) for o in db.scalars(select(ShotOutcome)).all()}
    return Table(
        swing_ids=[r.id for r in swings],
        created_at={r.id: r.created_at for r in swings},
        session_ids={r.id: r.session_id for r in swings},
        values=values,
        outcomes={k: v for k, v in outcomes.items() if k in values},
        info=info,
    )


def select_features(X: np.ndarray, names: list[str]) -> list[int]:
    """Columns measured on enough swings and not constant."""
    keep = []
    for j in range(X.shape[1]):
        col = X[:, j]
        ok = ~np.isnan(col)
        if ok.mean() >= MIN_COVERAGE and ok.sum() >= 2 and np.nanstd(col) > 1e-9:
            keep.append(j)
    return keep
