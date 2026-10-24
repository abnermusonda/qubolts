"""Synthetic segment selection problems for tests, demos and benchmarks.

Segments come in *families* (think: the same "in-market for SUVs" audience sold
by several data vendors). Segments in a family overlap heavily; across families
they overlap a little. This is exactly the structure that defeats
"rank by value per dollar".
"""

from __future__ import annotations

import numpy as np

from .model import Segment, SegmentSelectionProblem


def synthetic_problem(
    n: int = 30,
    families: int | None = None,
    budget_share: float = 0.3,
    within_family_overlap: tuple[float, float] = (0.4, 0.8),
    cross_family_overlap: tuple[float, float] = (0.0, 0.08),
    max_segments: int | None = None,
    seed: int | None = None,
) -> SegmentSelectionProblem:
    """Generate a reproducible problem.

    Args:
        n: number of segments.
        families: number of overlapping families (default: about n/3).
        budget_share: budget as a share of the cost of activating everything.
        within_family_overlap / cross_family_overlap: overlap as a share of the
            smaller segment, drawn uniformly from the given range.
    """
    rng = np.random.default_rng(seed)
    families = families or max(1, n // 3)
    family = rng.integers(0, families, size=n)
    sizes = rng.lognormal(mean=12.0, sigma=0.8, size=n).astype(int) + 1_000
    vpu = rng.uniform(0.002, 0.02, size=n)  # expected value per reached user, $
    cpm_fee = rng.uniform(0.5, 3.0, size=n)  # data fee, $ per 1000 users
    costs = sizes / 1000.0 * cpm_fee

    segments = [
        Segment(
            id=f"seg_{i:03d}_f{family[i]}",
            size=int(sizes[i]),
            value_per_user=round(float(vpu[i]), 5),
            cost=round(float(costs[i]), 2),
        )
        for i in range(n)
    ]
    overlap: dict[tuple[str, str], int] = {}
    for i in range(n):
        for j in range(i + 1, n):
            lo, hi = within_family_overlap if family[i] == family[j] else cross_family_overlap
            share = rng.uniform(lo, hi)
            o = int(share * min(sizes[i], sizes[j]))
            if o > 0:
                overlap[(segments[i].id, segments[j].id)] = o

    budget = round(float(costs.sum()) * budget_share, 2)
    return SegmentSelectionProblem(segments, budget, overlap, max_segments)
