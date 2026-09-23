"""Deterministic rule engine: which catalog signatures does this swing's metrics trip?"""

from dataclasses import asdict, dataclass

from app.diagnosis.catalog import FAULTS, Rule


@dataclass
class MetricRow:
    metric_name: str
    event_ref: str | None
    value: float
    unit: str
    is_estimate: bool


@dataclass
class RuleHit:
    fault: str
    metric: str
    event: str | None
    value: float
    unit: str
    op: str
    threshold: float
    margin: float  # how far past the threshold, in the metric's unit (always >= 0)
    is_estimate: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _check(rule: Rule, value: float) -> float | None:
    """Returns the margin past the threshold if the rule fires, else None."""
    if rule.op == "<":
        return rule.threshold - value if value < rule.threshold else None
    if rule.op == ">":
        return value - rule.threshold if value > rule.threshold else None
    if rule.op == "abs>":
        return abs(value) - rule.threshold if abs(value) > rule.threshold else None
    raise ValueError(f"unknown op {rule.op!r}")


def evaluate(metrics: list[MetricRow]) -> list[RuleHit]:
    index = {(m.metric_name, m.event_ref): m for m in metrics}
    hits: list[RuleHit] = []
    for fault in FAULTS:
        for rule in fault.rules:
            m = index.get((rule.metric, rule.event))
            if m is None:
                continue
            margin = _check(rule, m.value)
            if margin is not None:
                hits.append(RuleHit(fault.name, rule.metric, rule.event, m.value, m.unit, rule.op,
                                    rule.threshold, round(margin, 3), m.is_estimate))
    return hits
