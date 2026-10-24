"""Solver backends. Import this package to register all built-in backends."""

from .base import Backend, get, register, registered
from .local import ExactBackend, SimulatedAnnealingBackend, SteepestDescentBackend, TabuBackend
from .remote import BraketBackend, LeapHybridBackend, LeapQPUBackend

for _b in (
    ExactBackend(),
    SimulatedAnnealingBackend(),
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
    "get",
    "register",
    "registered",
]
