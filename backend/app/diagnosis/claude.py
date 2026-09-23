"""The one Claude API call. Kept behind `call_claude` so tests can swap it out."""

import json
import os
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.config import get_settings


class DiagnosisError(Exception):
    pass


class NotConfigured(DiagnosisError):
    pass


class Evidence(BaseModel):
    metric: str
    event: str | None
    value: float
    note: str


class ProposedFault(BaseModel):
    fault: str
    likelihood: float = Field(ge=0, le=1)
    evidence: list[Evidence]
    frames: list[int]
    explanation: str
    visual_observation: str | None


class CannotAssess(BaseModel):
    topic: str
    reason: str


class DiagnosisOutput(BaseModel):
    summary: str
    faults: list[ProposedFault]
    cannot_assess: list[CannotAssess]
    suggested_checks: list[str]
    narrative: str


@dataclass
class ClaudeResult:
    output: DiagnosisOutput
    model: str  # the model that actually served the request (may be a fallback)
    usage: dict


# Models documented to accept server-side refusal fallback (`fallbacks: "default"`). Other
# models get the plain request; a refusal is still detected and reported.
FALLBACK_MODELS = {"claude-opus-5", "claude-fable-5-1"}


def is_configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def call_claude(system: str, content: list[dict], schema: dict, client=None) -> ClaudeResult:
    import anthropic

    if client is None and not is_configured():
        raise NotConfigured("ANTHROPIC_API_KEY is not set on the API service")
    s = get_settings()
    client = client or anthropic.Anthropic(timeout=s.diagnosis_timeout_s, max_retries=2)
    extra = {}
    if s.diagnosis_model in FALLBACK_MODELS:
        # Route a (rare, false-positive) policy decline to Anthropic's recommended fallback model.
        extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    try:
        response = client.beta.messages.create(
            model=s.diagnosis_model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            output_config={"effort": s.diagnosis_effort, "format": {"type": "json_schema", "schema": schema}},
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": content}],
            **extra,
        )
    except anthropic.AuthenticationError as e:
        raise DiagnosisError("Claude API key was rejected") from e
    except anthropic.RateLimitError as e:
        raise DiagnosisError("Claude API rate limit hit; try again in a minute") from e
    except anthropic.APIStatusError as e:
        raise DiagnosisError(f"Claude API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise DiagnosisError("could not reach the Claude API") from e

    if response.stop_reason == "refusal":
        detail = response.stop_details.explanation if response.stop_details else None
        raise DiagnosisError(f"Claude declined this request{': ' + detail if detail else ''}")
    if response.stop_reason == "max_tokens":
        raise DiagnosisError("Claude's response was cut off (max_tokens)")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise DiagnosisError("Claude returned no text output")
    try:
        output = DiagnosisOutput.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValueError) as e:
        raise DiagnosisError(f"Claude returned malformed output: {e}") from e
    usage = response.usage.model_dump(exclude_none=True) if response.usage else {}
    return ClaudeResult(output=output, model=response.model, usage=usage)
