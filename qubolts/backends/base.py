"""Backend interface and registry.

A backend is anything that can sample a :class:`dimod.BinaryQuadraticModel`.
Every backend speaks dimod's ``Sampler`` interface, so any dimod-compatible
solver (D-Wave Leap, BEIT, Pasqal via adapters, in-house GPU annealers, ...) can
be plugged in with a few lines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import dimod


class Backend(ABC):
    """A named, capability-described wrapper around a dimod sampler."""

    #: registry key, e.g. ``"sa"``
    name: str = "backend"
    #: rough upper bound on variables this backend handles well
    max_variables: int = 10_000
    #: True if calls cost money or leave the machine (QPU / cloud hybrid)
    remote: bool = False

    def available(self) -> bool:
        """Whether dependencies and credentials are present."""
        return True

    @abstractmethod
    def sample(self, bqm: dimod.BinaryQuadraticModel, **kwargs: Any) -> dimod.SampleSet:
        """Return a SampleSet; the lowest-energy sample is used first."""


_REGISTRY: dict[str, Backend] = {}


def register(backend: Backend) -> Backend:
    """Add a backend to the registry (overwrites an existing name)."""
    _REGISTRY[backend.name] = backend
    return backend


def get(name: str) -> Backend:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown backend {name!r}; registered: {sorted(_REGISTRY)}") from None


def registered() -> dict[str, Backend]:
    return dict(_REGISTRY)
