"""Diagnosis orchestration: context -> rules -> Claude -> citation check -> persist.

Claude's output is a *proposal*. Nothing here writes labels; only the golfer's verdicts do
(see routers/diagnoses.py).
"""

import base64
import hashlib
import json
import re
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.diagnosis import claude as claude_mod
from app.diagnosis.catalog import BY_NAME, CATALOG_VERSION
from app.diagnosis.prompt import PROMPT_VERSION, output_schema, system_prompt
from app.diagnosis.rules import MetricRow, evaluate
from app.models import Diagnosis, EventType, Metric, Swing, Video
from app.pipeline.landmarks import CONNECTIONS
from app.pipeline.metrics import PIPELINE_VERSION
from app.pipeline.posedata import PoseData
from app.pipeline.registry import get_or_create_model
from app.pipeline.run import event_corrections, latest_pose_sequence, load_pose, predicted_events
from app.storage import get_storage

TASK_FAULTS = "fault_diagnosis"
KEYFRAME_EVENTS = (EventType.address, EventType.top, EventType.impact, EventType.finish)
KEYFRAME_MAX_SIDE = 640
FRAME_REF = re.compile(r"\[frame (\d+)\]")


# --- Context -----------------------------------------------------------------------------------


def current_metrics(db: Session, swing_id) -> list[MetricRow]:
    rows = db.scalars(
        select(Metric).where(Metric.swing_id == swing_id, Metric.pipeline_version == PIPELINE_VERSION)
    ).all()
    return [MetricRow(m.metric_name, m.event_ref.value if m.event_ref else None, m.value, m.unit, m.is_estimate)
            for m in rows]


def current_events(db: Session, swing_id) -> list[dict]:
    preds = predicted_events(db, swing_id)
    corrs = event_corrections(db, swing_id)
    out = []
    for et in EventType:
        p, c = preds.get(et), corrs.get(et)
        if p is None and c is None:
            continue
        out.append({
            "event": et.value,
            "frame": int(c.corrected_value["frame_index"]) if c else p.frame_index,
            "confidence": round(p.confidence, 2) if p else None,
            "corrected_by_golfer": c is not None,
        })
    return out


def inputs_hash(metrics: list[MetricRow], events: list[dict]) -> str:
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "metrics": sorted([m.metric_name, m.event_ref or "", round(m.value, 3)] for m in metrics),
        "events": sorted([e["event"], e["frame"]] for e in events),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _draw_skeleton(img: np.ndarray, pose: PoseData, frame: int) -> None:
    kp = pose.kp2d[frame]
    if np.isnan(kp[0, 0]):
        return
    for a, b in CONNECTIONS:
        pa, pb = kp[a], kp[b]
        cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (90, 230, 160), 2, cv2.LINE_AA)


def extract_keyframes(proxy_path: Path, pose: PoseData | None, frames: dict[str, int]) -> list[dict]:
    """JPEG key frames (skeleton drawn on) as base64, labelled with event + frame index."""
    cap = cv2.VideoCapture(str(proxy_path))
    out = []
    try:
        for label, f in frames.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            if pose is not None and 0 <= f < pose.num_frames:
                _draw_skeleton(img, pose, f)
            h, w = img.shape[:2]
            scale = KEYFRAME_MAX_SIDE / max(h, w)
            if scale < 1:
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                out.append({"event": label, "frame": f, "jpeg_b64": base64.b64encode(buf.tobytes()).decode()})
    finally:
        cap.release()
    return out


def build_content(context: dict, keyframes: list[dict], symptom: str) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": "Swing data:\n" + json.dumps(context, indent=1)}]
    for kf in keyframes:
        content.append({"type": "text", "text": f"Key frame: {kf['event']} [frame {kf['frame']}]"})
        content.append({"type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": kf["jpeg_b64"]}})
    content.append({"type": "text", "text": "What the golfer describes:\n" + (symptom.strip() or "(nothing given - "
                    "review the swing for the most likely faults)")})
    return content


# --- Verification ------------------------------------------------------------------------------


def verify(output: claude_mod.DiagnosisOutput, metrics: list[MetricRow], num_frames: int) -> dict:
    """Check every citation against the stored data. Returns the output as a dict with an
    `unverified` list added to each fault and `narrative_unverified_frames` at the top level."""
    index = {(m.metric_name, m.event_ref): m for m in metrics}
    result = output.model_dump()
    for fault in result["faults"]:
        problems = []
        for ev in fault["evidence"]:
            m = index.get((ev["metric"], ev["event"]))
            if m is None:
                problems.append(f"{ev['metric']}@{ev['event'] or 'swing'}: no such metric")
            elif abs(m.value - ev["value"]) > max(0.5, 0.02 * abs(m.value)):
                problems.append(f"{ev['metric']}@{ev['event'] or 'swing'}: cited {ev['value']}, measured {m.value}")
        for f in fault["frames"]:
            if not 0 <= f < num_frames:
                problems.append(f"frame {f} is outside the clip (0-{num_frames - 1})")
        fault["unverified"] = problems
    result["narrative_unverified_frames"] = sorted(
        {int(n) for n in FRAME_REF.findall(output.narrative) if not 0 <= int(n) < num_frames}
    )
    return result


# --- Entry point -------------------------------------------------------------------------------


def run_diagnosis(
    db: Session, swing: Swing, symptom: str,
    call: Callable[..., claude_mod.ClaudeResult] | None = None,
) -> Diagnosis:
    call = call or claude_mod.call_claude
    s = get_settings()
    video = db.get(Video, swing.video_ids[0])
    metrics = current_metrics(db, swing.id)
    events = current_events(db, swing.id)
    hits = evaluate(metrics)
    num_frames = video.num_frames or 0

    seq = latest_pose_sequence(db, swing.id)
    pose = load_pose(seq) if seq else None
    frames = {e["event"]: e["frame"] for e in events if e["event"] in {k.value for k in KEYFRAME_EVENTS}}
    keyframes: list[dict] = []
    if video.proxy_uri and frames:
        with tempfile.TemporaryDirectory(prefix="hitfar-dx-") as tmp:
            proxy = Path(tmp) / "proxy.mp4"
            get_storage().download_file(video.proxy_uri, proxy)
            keyframes = extract_keyframes(proxy, pose, frames)

    context = {
        "camera_role": video.camera_role.value,
        "golfer_handedness": s.golfer_handedness,
        "fps": video.fps,
        "num_frames": num_frames,
        "events": events,
        "metrics": [m.__dict__ for m in metrics],
        "rule_hits": [h.to_dict() for h in hits],
        "units_note": "'% shoulder width' displacements are relative to shoulder width at address; "
                      "positive sway = toward the target.",
    }
    model = get_or_create_model(db, "claude-diagnosis", f"{s.diagnosis_model}+prompt-v{PROMPT_VERSION}", TASK_FAULTS,
                                notes="Proposes faults from metrics + key frames; golfer confirms or rejects")
    dx = Diagnosis(
        swing_id=swing.id, user_symptom_text=symptom, model_id=model.id,
        rule_hits=[h.to_dict() for h in hits],
        inputs_snapshot={
            "pipeline_version": PIPELINE_VERSION,
            "inputs_hash": inputs_hash(metrics, events),
            "events": events,
            "metrics": [m.__dict__ for m in metrics],
            "keyframes": [{"event": k["event"], "frame": k["frame"]} for k in keyframes],
        },
        meta={"requested_model": s.diagnosis_model, "prompt_version": PROMPT_VERSION,
              "catalog_version": CATALOG_VERSION},
    )
    try:
        res = call(system_prompt(), build_content(context, keyframes, symptom), output_schema())
    except claude_mod.NotConfigured:
        raise
    except claude_mod.DiagnosisError as e:
        dx.error = str(e)
        db.add(dx)
        db.commit()
        return dx

    checked = verify(res.output, metrics, num_frames)
    dx.output = checked
    dx.narrative = res.output.narrative
    dx.predicted_faults = {
        f["fault"]: {"likelihood": f["likelihood"], "unverified": bool(f["unverified"])}
        for f in checked["faults"] if f["fault"] in BY_NAME
    }
    dx.meta = {**dx.meta, "served_model": res.model, "usage": res.usage}
    db.add(dx)
    db.commit()
    return dx
