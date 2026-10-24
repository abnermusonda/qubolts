"""Remote backends: cloud hybrid solvers and QPUs.

These cost money per call, so the router never picks them unless asked
explicitly (``backend="leap_hybrid"``) or allowed via ``allow_remote=True``.
"""

from __future__ import annotations

import os
from typing import Any

import dimod

from .base import Backend


class LeapHybridBackend(Backend):
    """D-Wave Leap hybrid BQM solver. Requires ``pip install qubolts[leap]`` and a token.

    Set ``DWAVE_API_TOKEN`` (or configure ``dwave config create``).
    """

    name = "leap_hybrid"
    max_variables = 1_000_000
    remote = True

    def available(self) -> bool:
        try:
            import dwave.system  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("DWAVE_API_TOKEN")) or os.path.exists(
            os.path.expanduser("~/.config/dwave/dwave.conf")
        )

    def sample(self, bqm: dimod.BinaryQuadraticModel, **kwargs: Any) -> dimod.SampleSet:
        from dwave.system import LeapHybridSampler

        return LeapHybridSampler().sample(bqm, **kwargs)


class LeapQPUBackend(Backend):
    """D-Wave Advantage QPU with automatic minor-embedding. Small problems only."""

    name = "leap_qpu"
    max_variables = 150  # dense QUBOs embed poorly; keep subproblems small
    remote = True

    def available(self) -> bool:
        return LeapHybridBackend().available()

    def sample(self, bqm: dimod.BinaryQuadraticModel, **kwargs: Any) -> dimod.SampleSet:
        from dwave.system import DWaveSampler, EmbeddingComposite

        params = {"num_reads": 200}
        params.update(kwargs)
        return EmbeddingComposite(DWaveSampler()).sample(bqm, **params)


class BraketBackend(Backend):
    """Placeholder for Amazon Braket (QAOA on gate-model devices or simulators).

    Contributions welcome: wrap a QAOA routine in a dimod.Sampler and return a
    SampleSet. See docs/ARCHITECTURE.md#adding-a-backend.
    """

    name = "braket"
    remote = True

    def available(self) -> bool:
        return False

    def sample(self, bqm: dimod.BinaryQuadraticModel, **kwargs: Any) -> dimod.SampleSet:
        raise NotImplementedError("Braket backend not implemented yet; contributions welcome")
