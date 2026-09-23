"""The shot outcomes the golfer tags, and the binary problems the outcome models learn."""

from dataclasses import dataclass

SHAPES = ("slice", "fade", "straight", "draw", "hook")
START_LINES = ("left", "straight", "right")
CONTACTS = ("fat", "solid", "thin")


@dataclass(frozen=True)
class Problem:
    key: str
    title: str
    field: str  # which ShotOutcome column the label comes from
    positive: frozenset[str]
    negative: frozenset[str]
    description: str

    def label(self, outcome: dict) -> int | None:
        v = outcome.get(self.field)
        if v in self.positive:
            return 1
        if v in self.negative:
            return 0
        return None  # untagged for this problem


PROBLEMS: dict[str, Problem] = {
    p.key: p
    for p in (
        Problem("slice", "Slice", "shape", frozenset({"slice", "fade"}), frozenset({"straight", "draw", "hook"}),
                "Ball curves away from you (slice or fade) vs. straight or curving in"),
        Problem("hook", "Hook", "shape", frozenset({"hook", "draw"}), frozenset({"straight", "fade", "slice"}),
                "Ball curves in toward you (hook or draw) vs. straight or curving away"),
        Problem("fat", "Fat", "contact", frozenset({"fat"}), frozenset({"solid", "thin"}),
                "Ground before the ball vs. solid or thin"),
        Problem("thin", "Thin", "contact", frozenset({"thin"}), frozenset({"solid", "fat"}),
                "Bottom of the club into the ball's middle vs. solid or fat"),
    )
}
