"""Outcome tagging, automatic (re)training and promotion, and the queries behind the Analysis page."""

import io
import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import fields
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import jobs
from app.config import get_settings
from app.models import (
    Dataset,
    DatasetSource,
    EventType,
    Job,
    Label,
    Model,
    ModelStatus,
    ShotOutcome,
    Swing,
    Video,
)
from app.outcomes.features import Table, load_table, outcome_dict, select_features
from app.outcomes.problems import PROBLEMS, Problem
from app.outcomes.train import CVSettings, eligibility, jsonsafe, recipe_auc, train_problem, what_if
from app.pipeline.metrics import PIPELINE_VERSION
from app.pipeline.run import effective_events
from app.storage import get_storage

log = logging.getLogger(__name__)

OUTCOME_LABEL_TASK = "outcome"
DATASET_TASK = "outcome"


def model_name(problem: str) -> str:
    return f"outcome-{problem}"


def model_task(problem: str) -> str:
    return f"outcome_{problem}"


# --- Tagging -----------------------------------------------------------------------------------


def set_outcome(db: Session, swing: Swing, values: dict) -> ShotOutcome | None:
    """Upsert the swing's outcome (fields not given are left as they were; all-empty deletes it),
    append an audit label and queue retraining when enough has changed."""
    row = db.scalar(select(ShotOutcome).where(ShotOutcome.swing_id == swing.id))
    if row is None:
        row = ShotOutcome(swing_id=swing.id, source="self")
        db.add(row)
    for k, v in values.items():
        setattr(row, k, v)
    snapshot = outcome_dict(row)
    db.add(Label(task=OUTCOME_LABEL_TASK, target_type="swing", target_id=swing.id, corrected_value=snapshot))
    if all(snapshot[k] is None for k in snapshot if k != "source"):
        if row in db.new:
            db.expunge(row)
        else:
            db.delete(row)
        row = None
    db.commit()
    maybe_queue_training(db)
    return row


def get_outcome(db: Session, swing_id: uuid.UUID) -> ShotOutcome | None:
    return db.scalar(select(ShotOutcome).where(ShotOutcome.swing_id == swing_id))


def _last_training_dataset(db: Session) -> Dataset | None:
    return db.scalar(select(Dataset).where(Dataset.task == DATASET_TASK).order_by(Dataset.created_at.desc()).limit(1))


def changes_since_training(db: Session) -> int:
    last = _last_training_dataset(db)
    q = select(func.count()).select_from(Label).where(Label.task == OUTCOME_LABEL_TASK)
    if last is not None:
        q = q.where(Label.created_at > last.created_at)
    return int(db.scalar(q) or 0)


def maybe_queue_training(db: Session, force: bool = False) -> bool:
    """Every N outcome changes (tags or edits), retrain. Cheap: seconds on the worker."""
    n = changes_since_training(db)
    if not force and (n == 0 or n < get_settings().outcome_retrain_every):
        return False
    job = jobs.enqueue_once(db, jobs.JOB_TRAIN_OUTCOMES, {})
    db.commit()
    return job is not None


# --- Training ----------------------------------------------------------------------------------


def _labels(table: Table, problem: Problem) -> tuple[list[uuid.UUID], np.ndarray]:
    ids, y = [], []
    for sid in table.swing_ids:
        o = table.outcomes.get(sid)
        lab = problem.label(o) if o else None
        if lab is not None:
            ids.append(sid)
            y.append(lab)
    return ids, np.array(y, dtype=int)


def _feature_info(table: Table) -> dict[str, dict]:
    out = {}
    for name, fi in table.info.items():
        metric, _, event = name.partition("@")
        out[name] = {"metric": metric, "event": event or None, "unit": fi.unit, "is_estimate": fi.is_estimate}
    return out


def _cv_settings(overrides: dict | None) -> CVSettings:
    names = {f.name for f in fields(CVSettings)}
    return CVSettings(**{k: v for k, v in (overrides or {}).items() if k in names})


def active_outcome_model(db: Session, problem: str) -> Model | None:
    return db.scalar(select(Model).where(Model.task == model_task(problem), Model.status == ModelStatus.active)
                     .order_by(Model.created_at.desc()).limit(1))


def run_outcome_training(db: Session, payload: dict | None = None,
                         on_stage: Callable[[str], None] = lambda s: None) -> dict:
    """Train every problem that has enough tagged swings; auto-promote a new version when it scores
    at least as well as the active one on the same swings with the same protocol."""
    import joblib

    cfg = _cv_settings((payload or {}).get("cv"))
    on_stage("loading")
    table = load_table(db)
    all_names = sorted({n for v in table.values.values() for n in v})
    finfo = _feature_info(table)
    now = datetime.now(timezone.utc)
    version = now.strftime("%Y%m%d-%H%M%S-") + f"{now.microsecond // 1000:03d}"
    st = get_storage()

    tagged = [s for s in table.swing_ids if s in table.outcomes]
    manifest = {
        "version": version, "pipeline_version": PIPELINE_VERSION, "features": all_names,
        "swings": [{"swing_id": str(s), "outcome": table.outcomes[s],
                    "values": [table.values[s].get(n) for n in all_names]} for s in tagged],
    }
    manifest_key = f"datasets/outcomes/{version}.json"
    st.put_bytes(manifest_key, json.dumps(manifest).encode(), "application/json")
    ds = Dataset(name=f"outcomes-{version}", task=DATASET_TASK, source=DatasetSource.self_labeled,
                 manifest_uri=manifest_key, num_samples=len(tagged))
    db.add(ds)
    db.commit()

    summary: dict[str, dict] = {}
    for problem in PROBLEMS.values():
        ids, y = _labels(table, problem)
        elig = eligibility(y, cfg)
        if not elig["eligible"]:
            summary[problem.key] = {"trained": False, **elig}
            continue
        on_stage(f"training {problem.key}")
        X_all = table.matrix(ids, all_names)
        keep = select_features(X_all, all_names)
        names = [all_names[j] for j in keep]
        X = X_all[:, keep]
        res = train_problem(X, y, names, [str(s) for s in ids], cfg, finfo)

        # Same-data comparison with the active model's recipe (model kind + feature set).
        current = active_outcome_model(db, problem.key)
        promote, compared = True, None
        if current is not None and current.eval_metrics:
            old_kind = current.eval_metrics.get("kind")
            old_feats = [f for f in current.eval_metrics.get("features", []) if f in all_names]
            if old_kind == res.kind and old_feats == names:
                old_auc = res.eval["cv_auc"]  # identical recipe: more data can only help
            elif old_feats:
                cols = [all_names.index(f) for f in old_feats]
                old_auc = recipe_auc(old_kind, X_all[:, cols], y, cfg)
            else:
                old_auc = float("-inf")
            promote = res.eval["cv_auc"] >= old_auc - 1e-9
            compared = {"version": current.version, "cv_auc_same_data": old_auc if np.isfinite(old_auc) else None}

        buf = io.BytesIO()
        joblib.dump({"estimator": res.estimator, "kind": res.kind, "features": names, "problem": problem.key,
                     "oof": res.oof, "factors": res.eval["factors"], "version": version,
                     "pipeline_version": PIPELINE_VERSION}, buf)
        ckpt = f"models/{model_name(problem.key)}/{version}/model.joblib"
        st.put_bytes(ckpt, buf.getvalue())
        m = Model(name=model_name(problem.key), version=version, task=model_task(problem.key),
                  checkpoint_uri=ckpt, trained_on_dataset_id=ds.id,
                  eval_metrics=jsonsafe({**res.eval, "features": names, "pipeline_version": PIPELINE_VERSION,
                                         "compared_to": compared, "auto_promoted": promote}),
                  status=ModelStatus.experimental,
                  notes=f"{problem.title}: {problem.description}")
        db.add(m)
        db.flush()
        if promote:
            for other in db.scalars(select(Model).where(Model.task == m.task, Model.status == ModelStatus.active)):
                other.status = ModelStatus.deprecated
            m.status = ModelStatus.active
        db.commit()
        summary[problem.key] = {"trained": True, "version": version, "promoted": promote, **elig,
                                "cv_auc": res.eval["cv_auc"], "reliable": res.eval["reliable"]}
        log.info("outcome model %s %s: auc %.3f reliable=%s promoted=%s", problem.key, version,
                 res.eval["cv_auc"], res.eval["reliable"], promote)
    on_stage("done")
    return summary


@lru_cache(maxsize=16)
def load_checkpoint(uri: str) -> dict:
    import joblib

    return joblib.load(io.BytesIO(get_storage().get_bytes(uri)))


# --- Queries for the UI ------------------------------------------------------------------------


def _model_summary(m: Model | None) -> dict | None:
    if m is None:
        return None
    em = m.eval_metrics or {}
    return {
        "id": str(m.id), "version": m.version, "created_at": m.created_at.isoformat(),
        "kind": em.get("kind"), "kind_title": em.get("kind_title"), "n": em.get("n"),
        "cv_auc": em.get("cv_auc"), "cv_auc_ci": em.get("cv_auc_ci"), "null_auc_95": em.get("null_auc_95"),
        "reliable": em.get("reliable", False), "reliable_rule": em.get("reliable_rule"),
        "candidates": em.get("candidates"), "protocol": em.get("protocol"),
        "pipeline_version": em.get("pipeline_version"),
    }


def summary(db: Session) -> dict:
    cfg = CVSettings()
    table = load_table(db)
    problems = []
    for p in PROBLEMS.values():
        _, y = _labels(table, p)
        problems.append({"key": p.key, "title": p.title, "description": p.description,
                         **eligibility(y, cfg), "model": _model_summary(active_outcome_model(db, p.key))})
    job = db.scalar(select(Job).where(Job.type == jobs.JOB_TRAIN_OUTCOMES).order_by(Job.created_at.desc()).limit(1))
    s = get_settings()
    return {
        "problems": problems,
        "tagged": len(table.outcomes),
        "with_metrics": len(table.swing_ids),
        "changes_since_training": changes_since_training(db),
        "retrain_every": s.outcome_retrain_every,
        "training": None if job is None else {"status": job.status.value, "stage": job.stage,
                                              "error": job.error, "updated_at": job.updated_at.isoformat()},
        "llm_enabled": s.outcome_llm_explain,
        "pipeline_version": PIPELINE_VERSION,
    }


def _impact_ref(db: Session, swing_id: uuid.UUID, prob: float | None, outcome: dict | None) -> dict | None:
    swing = db.get(Swing, swing_id)
    if swing is None:
        return None
    video = db.get(Video, swing.video_ids[0])
    ev = effective_events(db, swing_id)
    return {
        "swing_id": str(swing_id), "filename": video.original_filename if video else None,
        "playback_url": get_storage().presign_get(video.proxy_uri, 6 * 3600) if video and video.proxy_uri else None,
        "fps": video.fps if video else None, "impact_frame": ev.get(EventType.impact),
        "probability": prob, "outcome": outcome,
    }


def analysis(db: Session, problem_key: str) -> dict:
    problem = PROBLEMS[problem_key]
    cfg = CVSettings()
    table = load_table(db)
    ids, y = _labels(table, problem)
    m = active_outcome_model(db, problem_key)
    out = {"problem": {"key": problem.key, "title": problem.title, "description": problem.description},
           **eligibility(y, cfg), "model": _model_summary(m), "factors": [], "references": None, "latest": None}
    if m is None or not m.checkpoint_uri:
        return out
    ck = load_checkpoint(m.checkpoint_uri)
    feats: list[str] = ck["features"]
    label_of = dict(zip(ids, y.tolist()))

    # Latest session with metrics: where your recent swings sit on each factor.
    latest_session = table.session_ids[table.swing_ids[-1]] if table.swing_ids else None
    latest_ids = [s for s in table.swing_ids if table.session_ids[s] == latest_session]

    factors = []
    for f in ck["factors"]:
        name = f["feature"]
        pts = [{"swing_id": str(s), "value": table.values[s][name], "bad": bool(label_of[s])}
               for s in ids if name in table.values[s]]
        latest_vals = [table.values[s][name] for s in latest_ids if name in table.values[s]]
        factors.append({**f, "points": pts,
                        "latest_median": float(np.median(latest_vals)) if latest_vals else None})
    out["factors"] = factors
    out["latest"] = {"session_id": str(latest_session) if latest_session else None, "n": len(latest_ids)}

    # Reference pair (out-of-fold probabilities, so not flattered by training fit): your most
    # clearly-good tagged swing, and a typical bad one.
    oof = {uuid.UUID(k): v for k, v in ck["oof"].items()}
    good = [(p, s) for s, p in oof.items() if label_of.get(s) == 0]
    bad = [(p, s) for s, p in oof.items() if label_of.get(s) == 1]
    if good and bad:
        g = min(good, key=lambda t: t[0])
        med = float(np.median([p for p, _ in bad]))
        b = min(bad, key=lambda t: abs(t[0] - med))
        out["references"] = {
            "good": _impact_ref(db, g[1], g[0], table.outcomes.get(g[1])),
            "bad": _impact_ref(db, b[1], b[0], table.outcomes.get(b[1])),
        }

    # What-if on your most recent swing.
    if table.swing_ids:
        last = table.swing_ids[-1]
        x = table.matrix([last], feats)[0]
        out["what_if"] = {"swing_id": str(last), "items": what_if(ck["estimator"], x, feats, ck["factors"])}
    return out


def predictions(db: Session, swing_id: uuid.UUID) -> list[dict]:
    table = load_table(db)
    if swing_id not in table.values:
        return []
    out = []
    for p in PROBLEMS.values():
        m = active_outcome_model(db, p.key)
        if m is None or not m.checkpoint_uri:
            continue
        ck = load_checkpoint(m.checkpoint_uri)
        x = table.matrix([swing_id], ck["features"])[0]
        prob = float(ck["estimator"].predict_proba(x[None, :])[0, 1])
        out.append({"problem": p.key, "title": p.title, "probability": prob,
                    "reliable": bool((m.eval_metrics or {}).get("reliable")),
                    "model_version": m.version,
                    "what_if": what_if(ck["estimator"], x, ck["features"], ck["factors"])[:3]})
    return out


def explain_payload(db: Session, problem_key: str) -> dict:
    """Exactly what the optional LLM explanation sees: model output, no video or raw data."""
    a = analysis(db, problem_key)
    return {
        "problem": a["problem"], "n": a["n"], "n_pos": a["n_pos"], "n_neg": a["n_neg"],
        "model": a["model"],
        "factors": [{k: v for k, v in f.items() if k != "points"} for f in a["factors"] if f.get("supported")],
        "what_if_latest_swing": (a.get("what_if") or {}).get("items", []),
    }
