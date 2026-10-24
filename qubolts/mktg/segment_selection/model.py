"""Problem definition for audience segment selection.

The decision: which audience segments to activate for a campaign.

Why it is hard: segments overlap. Two "auto intender" segments from different
data vendors may share 70% of their users, so buying both pays data fees twice
for the same people. The usual "top-N by value per dollar" rule ignores this.

Objective (maximize), a second-order inclusion-exclusion estimate of unique value::

    sum_i value_i * x_i  -  sum_{i<j} overlap_ij * min(vpu_i, vpu_j) * x_i * x_j

subject to ``sum_i cost_i * x_i <= budget`` and optionally ``sum_i x_i <= max_segments``.

``value_i = size_i * vpu_i`` where ``vpu`` is expected value per reached user. The
pairwise term removes the double-counted users once, keeping the more valuable
of the two segments' per-user value. Three-way overlaps are not modeled (the
estimate is conservative when many segments overlap the same users).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Segment:
    id: str
    size: int  # reachable users
    value_per_user: float  # expected value of reaching one user (e.g. conv. rate * AOV)
    cost: float  # data fee + expected media cost to activate the segment

    @property
    def value(self) -> float:
        return self.size * self.value_per_user


@dataclass
class SegmentSelectionProblem:
    segments: list[Segment]
    budget: float
    overlap: dict[tuple[str, str], int] = field(default_factory=dict)
    max_segments: int | None = None

    def __post_init__(self) -> None:
        ids = [s.id for s in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("segment ids must be unique")
        self.overlap = {_key(a, b): int(v) for (a, b), v in self.overlap.items() if v and a != b}
        self._by_id = {s.id: s for s in self.segments}

    def segment(self, sid: str) -> Segment:
        return self._by_id[sid]

    def pair_penalty(self, a: str, b: str) -> float:
        """Double-counted value when both ``a`` and ``b`` are active."""
        o = self.overlap.get(_key(a, b), 0)
        if not o:
            return 0.0
        return o * min(self._by_id[a].value_per_user, self._by_id[b].value_per_user)

    # --- serialization: this is the contract between Spark jobs and the resolver ---

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [s.__dict__ for s in self.segments],
            "budget": self.budget,
            "max_segments": self.max_segments,
            "overlap": [{"a": a, "b": b, "users": v} for (a, b), v in sorted(self.overlap.items())],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SegmentSelectionProblem:
        return cls(
            segments=[Segment(**s) for s in d["segments"]],
            budget=float(d["budget"]),
            max_segments=d.get("max_segments"),
            overlap={(o["a"], o["b"]): int(o["users"]) for o in d.get("overlap", [])},
        )


def problem_from_memberships(
    memberships: Iterable[tuple[str, str]],
    segment_meta: Mapping[str, Mapping[str, float]],
    budget: float,
    max_segments: int | None = None,
) -> SegmentSelectionProblem:
    """Build a problem from raw ``(user_id, segment_id)`` rows.

    Pure-Python reference implementation of what ``fabric/jobs/segment_inputs.py``
    does in Spark. Use it for tests and small data; use the Spark job at scale.

    ``segment_meta`` maps segment id -> ``{"value_per_user": ..., "cost": ...}``.
    """
    users_by_segment: dict[str, set[str]] = {}
    for user, seg in memberships:
        if seg in segment_meta:
            users_by_segment.setdefault(seg, set()).add(user)
    ids = sorted(users_by_segment)
    overlap: dict[tuple[str, str], int] = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            n = len(users_by_segment[a] & users_by_segment[b])
            if n:
                overlap[(a, b)] = n
    segments = [
        Segment(
            id=s,
            size=len(users_by_segment[s]),
            value_per_user=float(segment_meta[s]["value_per_user"]),
            cost=float(segment_meta[s]["cost"]),
        )
        for s in ids
    ]
    return SegmentSelectionProblem(segments, budget, overlap, max_segments)


def _key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)
