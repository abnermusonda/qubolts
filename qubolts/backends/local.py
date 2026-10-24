"""Backends that run on the local machine: exact, simulated annealing, tabu, descent.

These are the "quantum-inspired" workhorses: they solve the same QUBO a quantum
annealer would, on commodity CPUs, at zero marginal cost.
"""

from __future__ import annotations

from typing import Any

import dimod

from .base import Backend


class ExactBackend(Backend):
    """Brute force over all 2^n assignments. Ground truth for small problems."""

    name = "exact"
    max_variables = 22

    def sample(self, bqm: dimod.BinaryQuadraticModel, **kwargs: Any) -> dimod.SampleSet:
        if bqm.num_variables > self.max_variables:
            raise ValueError(
                f"exact backend limited to {self.max_variables} variables, got {bqm.num_variables}"
            )
        # return the lowest-energy states so the resolver can repair several candidates
        return dimod.ExactSolver().sample(bqm).truncate(kwargs.get("num_reads", 20))


class SimulatedAnnealingBackend(Backend):
    """Classical simulated annealing (dwave-samplers). The default for mid-size problems."""

    name = "sa"
    max_variables = 20_000

    def sample(self, bqm: dimod.BinaryQuadraticModel, **kwargs: Any) -> dimod.SampleSet:
        from dwave.samplers import SimulatedAnnealingSampler

        params = {"num_reads": 100, "num_sweeps": 2000}
        params.update(kwargs)
        return SimulatedAnnealingSampler().sample(bqm, **params)


class TabuBackend(Backend):
    """Multistart tabu search (dwave-samplers). Strong on dense QUBOs."""

    name = "tabu"
    max_variables = 20_000

    def sample(self, bqm: dimod.BinaryQuadraticModel, **kwargs: Any) -> dimod.SampleSet:
        from dwave.samplers import TabuSampler

        params = {"num_reads": 10, "timeout": 100}  # timeout in ms per read
        params.update(kwargs)
        return TabuSampler().sample(bqm, **params)


class SteepestDescentBackend(Backend):
    """Greedy local descent from random starts. Fast, weak; a sanity floor."""

    name = "descent"
    max_variables = 100_000

    def sample(self, bqm: dimod.BinaryQuadraticModel, **kwargs: Any) -> dimod.SampleSet:
        from dwave.samplers import SteepestDescentSolver

        params = {"num_reads": 100}
        params.update(kwargs)
        return SteepestDescentSolver().sample(bqm, **params)
