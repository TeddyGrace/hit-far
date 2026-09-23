"""Optional: put the outcome model's findings into plain words with Claude.

Off by default (OUTCOME_LLM_EXPLAIN). The model output is the finding; Claude only rephrases it
and is told not to add swing advice the data doesn't support.
"""

import json

from pydantic import BaseModel

from app.diagnosis.claude import call_claude

SYSTEM = """You translate a statistical model's findings about one golfer's own swings into plain
language for that golfer. You are given JSON from models trained only on their tagged shots:
cross-validated AUC with a confidence interval, whether the model passed its reliability gate,
the factors that both the model and a model-free check agree on (with the golfer's good-shot and
bad-shot medians), and what-if estimates for their latest swing.

Rules:
- Say only what the JSON supports. Do not add generic swing tips, drills or causes not in it.
- If the model is not reliable, say so plainly and say what more data would help; do not
  interpret factors.
- These are correlations in the golfer's own data, not proven causes. Say "moving toward your
  good-shot range" rather than prescribing positions.
- Metrics marked is_estimate come from single-camera 3D estimates; mention that when relevant.
- 120-250 words, second person, no headings."""


class Explanation(BaseModel):
    explanation: str


SCHEMA = {
    "type": "object",
    "properties": {"explanation": {"type": "string"}},
    "required": ["explanation"],
    "additionalProperties": False,
}


def explain(payload: dict, client=None) -> tuple[str, str]:
    content = [{"type": "text", "text": json.dumps(payload, default=str)}]
    res = call_claude(SYSTEM, content, SCHEMA, client=client, output_model=Explanation)
    return res.output.explanation, res.model
