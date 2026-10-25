"""Solver backends. Import this package to register all built-in backends."""

from .base import Backend, get, register, registered
from .local import (
    ExactBackend,
    SimulatedAnnealingBackend,
    SteepestDescentBackend,
    TabuBackend,
    TorchAnnealingBackend,
)
from .remote import BraketBackend, LeapHybridBackend, LeapQPUBackend

for _b in (
    ExactBackend(),
    SimulatedAnnealingBackend(),
    TorchAnnealingBackend(),
    TabuBackend(),
    SteepestDescentBackend(),
    LeapHybridBackend(),
    LeapQPUBackend(),
    BraketBackend(),
):
    register(_b)

__all__ = [
    "Backend",
    "BraketBackend",
    "ExactBackend",
    "LeapHybridBackend",
    "LeapQPUBackend",
    "SimulatedAnnealingBackend",
    "SteepestDescentBackend",
    "TabuBackend",
    "TorchAnnealingBackend",
    "get",
    "register",
    "registered",
]
