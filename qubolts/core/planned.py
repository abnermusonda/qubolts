"""Base class for resolvers that are specified but not yet implemented."""

from __future__ import annotations

from typing import Any

import dimod

from qubolts.core.contract import Resolver


class PlannedResolver(Resolver[Any, Any]):
    """A resolver whose formulation is documented but whose code is open for contribution.

    Calling any method raises ``NotImplementedError`` pointing at the docs, so a
    pipeline wired against the interface fails loudly rather than silently.
    """

    status = "planned"

    def _todo(self) -> NotImplementedError:
        return NotImplementedError(
            f"{self.name} is planned but not implemented yet. The formulation is in this "
            "module's docstring; see CONTRIBUTING.md to pick it up."
        )

    def build_bqm(self, problem: Any) -> dimod.BinaryQuadraticModel:
        raise self._todo()

    def decode(self, problem: Any, sample: dict) -> Any:
        raise self._todo()

    def objective(self, problem: Any, decision: Any) -> float:
        raise self._todo()

    def is_feasible(self, problem: Any, decision: Any) -> bool:
        raise self._todo()

    def baseline(self, problem: Any) -> Any:
        raise self._todo()
