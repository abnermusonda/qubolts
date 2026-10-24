"""Qubolts: quantum-ready decision engines for adtech and martech.

Each *qubolt* is a resolver that encodes one marketing decision (which audiences
to activate, how to split a budget, which creative goes where) as a QUBO and
solves it on a swappable backend: exact search, simulated annealing, tabu
search, or a cloud quantum/hybrid solver.

    >>> from qubolts import solve
    >>> from qubolts.mktg.segment_selection import SegmentSelection, synthetic_problem
    >>> sol = solve(SegmentSelection(), synthetic_problem(n=12, seed=1))
    >>> sol.feasible
    True
"""

from qubolts import backends  # noqa: F401  (registers built-in backends)
from qubolts.core import Resolver, RoutingPolicy, RunLog, Solution, solve

__version__ = "0.1.0"

__all__ = ["Resolver", "RoutingPolicy", "RunLog", "Solution", "solve", "__version__"]
