"""Prompt + output schema for the diagnosis call. Bump PROMPT_VERSION on any change."""

import json

from app.diagnosis.catalog import FAULTS

PROMPT_VERSION = "1"

SYSTEM_PROMPT = """You help one golfer understand their own swing. You receive measured data from their swing video \
(pose-derived metrics, detected swing events, rule-engine hits), a few annotated key frames, and a description of \
the problem they're having. You propose which faults from a fixed catalog are most likely involved. The golfer then \
confirms or rejects each proposal, so be specific and checkable rather than exhaustive.

Ground rules:
- Only cite metrics and frame numbers that appear in the data you were given, with the values as given. Never \
invent measurements.
- Metrics marked is_estimate=true come from a monocular 3D pose estimate (depth is inferred, not measured). Say \
"estimated" when you rely on them and weigh them less than 2D measurements and timings.
- Events with low confidence that were not corrected by the golfer may be misplaced; metrics at those events are \
less reliable. Say so when it matters.
- Keep what you can see in the key frames (visual_observation) separate from what was measured (evidence).
- Ball flight, clubface angle, and club path are not measured. Never state them as fact. For symptoms driven by \
them (a slice is mostly face-to-path), say what can't be determined and what would determine it.
- A rule-engine hit means a seed threshold was crossed, not that the fault is present. Use judgment.
- Only propose faults that the data supports at least somewhat, and give likelihoods that reflect the uncertainty. \
It is fine to propose none.
- Faults whose catalog entry lists `needs` cannot be confirmed from this data; you may raise them as possibilities \
with low likelihood and must list what's needed in cannot_assess.
- In the narrative, cite frames as [frame N] and metrics by name with their values. Plain, direct language; no \
padding; a few short paragraphs at most.

Fault catalog:
"""


def catalog_text() -> str:
    rows = []
    for f in FAULTS:
        entry = {"name": f.name, "title": f.title, "description": f.description}
        if f.rules:
            entry["seed_rules"] = [
                f"{r.metric}{'@' + r.event if r.event else ''} {r.op} {r.threshold}" for r in f.rules
            ]
        if f.needs:
            entry["needs"] = list(f.needs)
        if f.partial:
            entry["note"] = "face-on metrics are only a weak proxy for this fault"
        rows.append(entry)
    return json.dumps(rows, indent=1)


def system_prompt() -> str:
    return SYSTEM_PROMPT + catalog_text()


def output_schema() -> dict:
    fault_names = [f.name for f in FAULTS]
    evidence = {
        "type": "object",
        "properties": {
            "metric": {"type": "string"},
            "event": {"type": ["string", "null"]},
            "value": {"type": "number"},
            "note": {"type": "string"},
        },
        "required": ["metric", "event", "value", "note"],
        "additionalProperties": False,
    }
    fault = {
        "type": "object",
        "properties": {
            "fault": {"type": "string", "enum": fault_names},
            "likelihood": {"type": "number"},
            "evidence": {"type": "array", "items": evidence},
            "frames": {"type": "array", "items": {"type": "integer"}},
            "explanation": {"type": "string"},
            "visual_observation": {"type": ["string", "null"]},
        },
        "required": ["fault", "likelihood", "evidence", "frames", "explanation", "visual_observation"],
        "additionalProperties": False,
    }
    cannot = {
        "type": "object",
        "properties": {"topic": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["topic", "reason"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "faults": {"type": "array", "items": fault},
            "cannot_assess": {"type": "array", "items": cannot},
            "suggested_checks": {"type": "array", "items": {"type": "string"}},
            "narrative": {"type": "string"},
        },
        "required": ["summary", "faults", "cannot_assess", "suggested_checks", "narrative"],
        "additionalProperties": False,
    }
