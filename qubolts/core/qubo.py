"""Helpers for building QUBOs from business objectives and constraints.

The hard part of a QUBO pilot is rarely the solver: it is choosing penalty
weights. Too small and the annealer ignores the budget; too large and the
objective is drowned out. These helpers make that choice explicit and tunable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import dimod


def objective_scale(bqm: dimod.BinaryQuadraticModel) -> float:
    """Largest total absolute bias touching any single variable.

    A constraint penalty bigger than this guarantees that violating the
    constraint by one unit can never be "paid for" by the objective alone.
    """
    if bqm.num_variables == 0:
        return 1.0
    totals = {v: abs(b) for v, b in bqm.linear.items()}
    for (u, v), b in bqm.quadratic.items():
        totals[u] += abs(b)
        totals[v] += abs(b)
    return max(totals.values()) or 1.0


def default_penalty(bqm: dimod.BinaryQuadraticModel, factor: float = 1.5) -> float:
    """A safe starting penalty weight: ``factor`` x :func:`objective_scale`."""
    return factor * objective_scale(bqm)


def add_budget_constraint(
    bqm: dimod.BinaryQuadraticModel,
    costs: Mapping[str, float],
    budget: float,
    *,
    penalty: float,
    method: str = "unbalanced",
    resolution: float | None = None,
    label: str = "budget",
    unbalanced_weights: tuple[float, float] = (0.96, 0.0371),
) -> list:
    """Add ``sum(cost_i * x_i) <= budget`` to ``bqm`` as a penalty.

    Two encodings:

    ``"unbalanced"`` (default)
        No extra variables. With ``h = sum(c_i x_i)/budget - 1`` (negative when
        under budget) adds ``penalty * (l0 * h + l1 * h^2)``. Cheap and
        annealer-friendly; it slightly discourages spending, which the repair
        step undoes by greedily filling leftover budget. See Montanez-Barrera et
        al., "Unbalanced penalization", arXiv:2211.13914.

    ``"slack"``
        Exact equality ``sum + slack = budget`` with binary slack bits. Costs are
        rounded to ``resolution`` units (default 1/200 of budget) because dimod's
        slack encoding needs integers. Exact, but the extra variables and dense
        penalty make the landscape rougher for heuristic samplers.

    Returns the slack terms added (empty for ``"unbalanced"``).
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    if sum(c for c in costs.values() if c > 0) <= budget:
        return []  # constraint can never bind
    if method == "unbalanced":
        l0, l1 = unbalanced_weights
        # h = sum(a_i x_i) - 1 with a_i = c_i / budget. Expand l0*h + l1*h^2 (x_i^2 = x_i).
        items = [(v, c / budget) for v, c in costs.items() if c > 0]
        for v, a in items:
            bqm.add_linear(v, penalty * (l0 * a + l1 * (a * a - 2 * a)))
        for i, (u, a) in enumerate(items):
            for v, b in items[i + 1 :]:
                bqm.add_quadratic(u, v, penalty * l1 * 2 * a * b)
        bqm.offset += penalty * (l1 - l0)
        return []
    if method != "slack":
        raise ValueError(f"unknown method {method!r}; use 'unbalanced' or 'slack'")
    resolution = resolution or budget / 200.0
    terms = [(v, max(1, math.ceil(c / resolution))) for v, c in costs.items() if c > 0]
    ub = math.floor(budget / resolution)
    total = sum(c for _, c in terms)
    if total <= ub:  # constraint can never bind; skip the slack bits entirely
        return []
    # Normalize so `penalty` is, in objective units, the cost of overshooting the
    # budget by the cheapest item's cost -- independent of the chosen resolution.
    min_units = min(c for _, c in terms)
    lagrange = penalty / (min_units**2)
    return list(
        bqm.add_linear_inequality_constraint(
            terms, lagrange_multiplier=lagrange, label=label, lb=0, ub=ub
        )
    )


def is_slack(variable: object) -> bool:
    """True for slack variables added by :func:`add_budget_constraint`."""
    return isinstance(variable, str) and variable.startswith("slack_")
