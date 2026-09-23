import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import current_user, require_auth
from app.config import get_settings
from app.db import get_db
from app.models import User
from app.outcomes import service
from app.outcomes.problems import PROBLEMS
from app.outcomes.train import jsonsafe
from app.routers.swings import get_swing_or_404
from app.schemas import ShotOutcomeIn, ShotOutcomeOut

router = APIRouter(prefix="/api", tags=["outcomes"], dependencies=[Depends(require_auth)])


def _problem(key: str) -> str:
    if key not in PROBLEMS:
        raise HTTPException(404, f"unknown problem {key!r}; expected one of {sorted(PROBLEMS)}")
    return key


@router.put("/swings/{swing_id}/outcome", response_model=ShotOutcomeOut | None)
def set_outcome(swing_id: uuid.UUID, body: ShotOutcomeIn, db: Session = Depends(get_db),
                user: User = Depends(current_user)):
    swing = get_swing_or_404(db, swing_id, user)
    return service.set_outcome(db, swing, body.model_dump(exclude_unset=True))


@router.get("/swings/{swing_id}/predictions")
def swing_predictions(swing_id: uuid.UUID, db: Session = Depends(get_db),
                      user: User = Depends(current_user)) -> list[dict]:
    get_swing_or_404(db, swing_id, user)
    return jsonsafe(service.predictions(db, swing_id, user.id))


@router.get("/outcomes")
def outcomes_summary(db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict:
    return jsonsafe(service.summary(db, user.id))


@router.post("/outcomes/train", status_code=202)
def train_now(db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict:
    """Retrain now instead of waiting for the next few tags."""
    return {"queued": service.maybe_queue_training(db, user.id, force=True)}


@router.get("/outcomes/{problem}")
def outcome_analysis(problem: str, db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict:
    return jsonsafe(service.analysis(db, _problem(problem), user.id))


@router.post("/outcomes/{problem}/explain")
def explain(problem: str, db: Session = Depends(get_db), user: User = Depends(current_user)) -> dict:
    if not get_settings().outcome_llm_explain:
        raise HTTPException(403, "LLM explanations are off (set OUTCOME_LLM_EXPLAIN=true on the API service)")
    from app.diagnosis.claude import DiagnosisError
    from app.outcomes.explain import explain as run

    payload = service.explain_payload(db, _problem(problem), user.id)
    if payload["model"] is None:
        raise HTTPException(409, "no trained model for this problem yet")
    try:
        text, served = run(payload)
    except DiagnosisError as e:
        raise HTTPException(502, str(e)) from e
    return jsonsafe({"explanation": text, "served_model": served, "inputs": payload})
