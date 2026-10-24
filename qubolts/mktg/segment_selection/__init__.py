"""Segment selection: activate the audience set with the most *unique* value per dollar.

from qubolts import solve
from qubolts.mktg.segment_selection import SegmentSelection, SegmentSelectionProblem

problem = SegmentSelectionProblem.from_dict(json.load(open("inputs.json")))
solution = solve(SegmentSelection(), problem)
print(sorted(solution.decision), solution.objective, solution.lift_vs_baseline)
"""

from .model import Segment, SegmentSelectionProblem, problem_from_memberships
from .resolver import SegmentSelection
from .synthetic import synthetic_problem

__all__ = [
    "Segment",
    "SegmentSelection",
    "SegmentSelectionProblem",
    "problem_from_memberships",
    "synthetic_problem",
]
