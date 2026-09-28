"""The shot outcomes the golfer tags, and the binary problems the outcome models learn."""

from dataclasses import dataclass

SHAPES = ("slice", "fade", "straight", "draw", "hook")
START_LINES = ("left", "straight", "right")
CONTACTS = ("fat", "solid", "thin")

# Bump whenever a problem's positive or negative set changes: models trained under another version
# answer a different question, so start-up retrains them and they are never compared like-for-like.
PROBLEMS_VERSION = 2


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
        # A controlled fade or draw is a good shot; only the hard curve is the miss.
        Problem("slice", "Slice", "shape", frozenset({"slice"}), frozenset({"fade", "straight", "draw", "hook"}),
                "Ball curves hard away from you (slice) vs. anything else, including a controlled fade"),
        Problem("hook", "Hook", "shape", frozenset({"hook"}), frozenset({"draw", "straight", "fade", "slice"}),
                "Ball curves hard in toward you (hook) vs. anything else, including a controlled draw"),
        Problem("fat", "Fat", "contact", frozenset({"fat"}), frozenset({"solid", "thin"}),
                "Ground before the ball vs. solid or thin"),
        Problem("thin", "Thin", "contact", frozenset({"thin"}), frozenset({"solid", "fat"}),
                "Bottom of the club into the ball's middle vs. solid or fat"),
    )
}
