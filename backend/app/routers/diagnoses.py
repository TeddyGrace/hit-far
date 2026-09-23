import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import current_user, require_auth
from app.db import get_db
from app.diagnosis import claude as claude_mod
from app.diagnosis.catalog import BY_NAME, FAULTS
from app.diagnosis.service import current_events, current_metrics, inputs_hash, run_diagnosis
from app.models import Diagnosis, Label, Model, User
from app.routers.swings import get_swing_or_404
from app.schemas import DiagnoseIn, DiagnosisOut, FaultOut, ModelRef, VerdictIn

router = APIRouter(prefix="/api", tags=["diagnosis"], dependencies=[Depends(require_auth)])

FAULT_LABEL_TASK = "fault"


def _out(db: Session, dx: Diagnosis, current_hash: str) -> DiagnosisOut:
    model = db.get(Model, dx.model_id) if dx.model_id else None
    snap = dx.inputs_snapshot or {}
    return DiagnosisOut(
        id=dx.id, swing_id=dx.swing_id, created_at=dx.created_at, symptom_text=dx.user_symptom_text,
        error=dx.error, output=dx.output, rule_hits=dx.rule_hits,
        verdicts=dx.human_confirmed_faults or {},
        model=ModelRef.model_validate(model) if model else None,
        served_model=(dx.meta or {}).get("served_model"),
        stale=snap.get("inputs_hash") != current_hash,
    )


def _current_hash(db: Session, swing_id: uuid.UUID) -> str:
    return inputs_hash(current_metrics(db, swing_id), current_events(db, swing_id))


@router.get("/faults", response_model=list[FaultOut])
def list_faults():
    return [
        FaultOut(
            name=f.name, title=f.title, description=f.description, assessable=f.assessable, partial=f.partial,
            views=list(f.views), needs=list(f.needs),
            rules=[f"{r.metric}{'@' + r.event if r.event else ''} {r.op} {r.threshold:g}" for r in f.rules],
        )
        for f in FAULTS
    ]


@router.get("/swings/{swing_id}/diagnoses", response_model=list[DiagnosisOut])
def list_diagnoses(swing_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(current_user)):
    swing = get_swing_or_404(db, swing_id, user)
    h = _current_hash(db, swing.id)
    rows = db.scalars(
        select(Diagnosis).where(Diagnosis.swing_id == swing.id).order_by(Diagnosis.created_at.desc())
    ).all()
    return [_out(db, dx, h) for dx in rows]


@router.post("/swings/{swing_id}/diagnoses", response_model=DiagnosisOut, status_code=201)
def create_diagnosis(swing_id: uuid.UUID, body: DiagnoseIn, db: Session = Depends(get_db),
                     user: User = Depends(current_user)):
    swing = get_swing_or_404(db, swing_id, user)
    if len(body.symptom_text) > 2000:
        raise HTTPException(400, "symptom text is too long (max 2000 characters)")
    try:
        dx = run_diagnosis(db, swing, body.symptom_text)
    except claude_mod.NotConfigured as e:
        raise HTTPException(503, f"Diagnosis is not configured: {e}") from None
    return _out(db, dx, _current_hash(db, swing.id))


@router.put("/diagnoses/{diagnosis_id}/faults/{fault}", response_model=DiagnosisOut)
def set_verdict(diagnosis_id: uuid.UUID, fault: str, body: VerdictIn, db: Session = Depends(get_db),
                user: User = Depends(current_user)):
    dx = db.get(Diagnosis, diagnosis_id)
    if dx is None:
        raise HTTPException(404, "diagnosis not found")
    get_swing_or_404(db, dx.swing_id, user)
    if fault not in BY_NAME:
        raise HTTPException(404, f"unknown fault {fault!r}")
    predicted = (dx.predicted_faults or {}).get(fault)

    verdicts = {k: dict(v) for k, v in (dx.human_confirmed_faults or {}).items()}
    mine = verdicts.setdefault(body.labeled_by, {})
    if body.verdict is None:
        mine.pop(fault, None)
    else:
        mine[fault] = body.verdict
    dx.human_confirmed_faults = verdicts  # reassign so SQLAlchemy sees the JSONB change
    dx.confirmed_at = datetime.now(timezone.utc)

    # Every verdict is also an append-only label: the training set for the fault classifier.
    db.add(Label(
        task=FAULT_LABEL_TASK, target_type="diagnosis", target_id=dx.id,
        corrected_value={
            "fault": fault,
            "verdict": body.verdict,
            "labeled_by": body.labeled_by,
            "swing_id": str(dx.swing_id),
            "proposed_by_model": predicted is not None,
            "predicted_likelihood": predicted.get("likelihood") if predicted else None,
            "inputs_hash": (dx.inputs_snapshot or {}).get("inputs_hash"),
        },
    ))
    db.commit()
    return _out(db, dx, _current_hash(db, dx.swing_id))
