"""Fault catalog (versioned code, upserted into `fault_labels` at API startup).

Thresholds are seed values for a face-on camera, meant to be tuned against confirmed diagnoses -
they are not ground truth. Metric names/units match `app/pipeline/metrics.py`.

Faults with no rules are still listed so the diagnosis can name them as possibilities while being
explicit that the current capture/pipeline can't measure them.
"""

from dataclasses import dataclass, field

CATALOG_VERSION = "1"


@dataclass(frozen=True)
class Rule:
    metric: str
    event: str | None  # EventType value, or None for swing-level metrics (tempo)
    op: str  # "<", ">", "abs>"
    threshold: float


@dataclass(frozen=True)
class Fault:
    name: str
    title: str
    description: str
    rules: tuple[Rule, ...] = ()
    views: tuple[str, ...] = ("face_on",)  # camera roles that can show it
    needs: tuple[str, ...] = ()  # what's missing to assess it properly
    partial: bool = False  # face-on metrics are only a weak proxy

    @property
    def assessable(self) -> bool:
        return bool(self.rules)

    def signature(self) -> dict:
        return {
            "catalog_version": CATALOG_VERSION,
            "title": self.title,
            "rules": [r.__dict__ for r in self.rules],
            "views": list(self.views),
            "needs": list(self.needs),
            "partial": self.partial,
        }


FAULTS: tuple[Fault, ...] = (
    Fault(
        "lateral_sway", "Lateral sway (backswing)",
        "Hips drift away from the target during the backswing instead of rotating.",
        rules=(Rule("hip_sway_toward_target", "top", "<", -15.0),),
    ),
    Fault(
        "excess_slide", "Excess slide (downswing)",
        "Hips slide too far toward the target through impact instead of rotating open.",
        rules=(Rule("hip_sway_toward_target", "impact", ">", 25.0),),
    ),
    Fault(
        "head_movement", "Head movement",
        "Head moves laterally or rises noticeably between address and impact.",
        rules=(Rule("head_sway_toward_target", "top", "abs>", 15.0), Rule("head_rise", "impact", ">", 10.0)),
    ),
    Fault(
        "reverse_spine_angle", "Reverse spine angle",
        "Upper body leans toward the target at the top of the backswing.",
        rules=(Rule("spine_tilt_away_from_target", "top", "<", 0.0),),
    ),
    Fault(
        "early_extension", "Early extension",
        "Hips move toward the ball and the torso stands up before impact.",
        rules=(Rule("head_rise", "impact", ">", 10.0),),
        views=("down_the_line",), needs=("down-the-line video",), partial=True,
    ),
    Fault(
        "restricted_shoulder_turn", "Restricted shoulder turn",
        "Shoulders turn noticeably less than ~90 degrees by the top.",
        rules=(Rule("shoulder_turn", "top", "<", 75.0),),
    ),
    Fault(
        "collapsed_lead_arm", "Collapsed lead arm at top",
        "Lead elbow bends significantly at the top, shortening the swing arc.",
        rules=(Rule("lead_elbow_angle", "top", "<", 150.0),),
    ),
    Fault(
        "chicken_wing", "Chicken wing",
        "Lead elbow bends and pulls up through impact.",
        rules=(Rule("lead_elbow_angle", "impact", "<", 160.0),),
    ),
    Fault(
        "quick_tempo", "Quick tempo",
        "Backswing rushed relative to downswing (tour players average roughly 3:1).",
        rules=(Rule("tempo_ratio", None, "<", 2.5),),
    ),
    # Not measurable with the current pipeline - named so the diagnosis can say so explicitly.
    Fault(
        "over_the_top", "Over-the-top swing path",
        "Club moves outside the hand path early in the downswing, producing an out-to-in path.",
        views=("down_the_line",), needs=("down-the-line video", "club tracking"),
    ),
    Fault(
        "casting", "Casting / early release",
        "Wrist angle released early in the downswing, losing lag.",
        needs=("club tracking",),
    ),
    Fault(
        "open_clubface", "Open clubface at impact",
        "Clubface open relative to path at impact; the main driver of a slice.",
        needs=("club tracking", "launch monitor or ball-flight data"),
    ),
    Fault(
        "closed_clubface", "Closed clubface at impact",
        "Clubface closed relative to path at impact; the main driver of a hook.",
        needs=("club tracking", "launch monitor or ball-flight data"),
    ),
)

BY_NAME: dict[str, Fault] = {f.name: f for f in FAULTS}


def upsert_catalog(db) -> None:
    """Idempotently sync FAULTS into the fault_labels table."""
    from sqlalchemy import select

    from app.models import FaultLabel

    existing = {f.name: f for f in db.scalars(select(FaultLabel)).all()}
    for f in FAULTS:
        row = existing.get(f.name)
        if row is None:
            db.add(FaultLabel(name=f.name, description=f.description, metric_signature=f.signature()))
        else:
            row.description = f.description
            row.metric_signature = f.signature()
    db.commit()
