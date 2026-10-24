"""The resolver contract every Qubolt implements.

A *resolver* turns one adtech/martech decision problem into a QUBO, hands it to a
backend, and turns the backend's answer back into a feasible business decision.

    problem --build_bqm--> BQM --backend--> sample --decode--> assignment
            --repair--> feasible assignment --score/compare--> Solution

Resolvers only describe the *problem*. Routing, sampling, fallback and logging
live in :mod:`qubolts.core.solve`, so every resolver gets them for free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Generic, TypeVar

import dimod

P = TypeVar("P")  # the resolver's problem type
A = TypeVar("A")  # the resolver's decision type (e.g. a set of selected segment ids)


@dataclass
class Solution(Generic[A]):
    """The answer returned by :func:`qubolts.solve`.

    ``objective`` is always the *business* objective (higher is better), never the
    raw QUBO energy, so solutions from different backends compare directly.
    """

    decision: A
    objective: float
    feasible: bool
    backend: str
    runtime_s: float
    energy: float | None = None
    fell_back: bool = False
    baseline_objective: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def lift_vs_baseline(self) -> float | None:
        """Relative improvement over the classical baseline (0.05 == +5%)."""
        if self.baseline_objective in (None, 0):
            return None
        return (self.objective - self.baseline_objective) / abs(self.baseline_objective)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if isinstance(self.decision, (set, frozenset)):
            d["decision"] = sorted(self.decision)
        d["lift_vs_baseline"] = self.lift_vs_baseline
        return d


class Resolver(ABC, Generic[P, A]):
    """Base class for a Qubolt.

    Subclasses implement six small methods. Keep them pure functions of
    ``problem`` so resolvers stay easy to test and to run in parallel from Spark.
    """

    #: short stable identifier, e.g. ``"mktg.segment_selection"``
    name: str = "resolver"

    @abstractmethod
    def build_bqm(self, problem: P) -> dimod.BinaryQuadraticModel:
        """Encode the problem as a QUBO (minimization). Constraints become penalties."""

    @abstractmethod
    def decode(self, problem: P, sample: dict) -> A:
        """Map a raw sample (variable -> 0/1) to a business decision. Ignore slack vars."""

    @abstractmethod
    def objective(self, problem: P, decision: A) -> float:
        """Business value of a decision. Higher is better."""

    @abstractmethod
    def is_feasible(self, problem: P, decision: A) -> bool:
        """Hard-constraint check (budget, caps, ...)."""

    @abstractmethod
    def baseline(self, problem: P) -> A:
        """A fast classical heuristic. Used as the fallback and as the benchmark bar."""

    def repair(self, problem: P, decision: A) -> A:
        """Fix small constraint violations an annealer may leave behind.

        The default does nothing; override when the problem has a cheap repair
        (e.g. drop the least valuable item until the budget fits).
        """
        return decision
