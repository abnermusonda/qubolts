"""Pick a backend for a QUBO based on size, deadline and cost policy."""

from __future__ import annotations

from dataclasses import dataclass

import dimod

from qubolts import backends
from qubolts.backends.base import Backend


@dataclass
class RoutingPolicy:
    """How ``backend="auto"`` chooses.

    Defaults are deliberately conservative: local solvers only, exact search for
    anything tiny enough to brute-force, simulated annealing otherwise.
    """

    exact_max_variables: int = 18
    allow_remote: bool = False
    remote_min_variables: int = 2_000  # only pay for cloud when local struggles
    remote_backend: str = "leap_hybrid"
    default_backend: str = "sa"


def choose_backend(bqm: dimod.BinaryQuadraticModel, policy: RoutingPolicy | None = None) -> Backend:
    policy = policy or RoutingPolicy()
    n = bqm.num_variables
    if n <= policy.exact_max_variables:
        return backends.get("exact")
    if policy.allow_remote and n >= policy.remote_min_variables:
        remote = backends.get(policy.remote_backend)
        if remote.available():
            return remote
    return backends.get(policy.default_backend)
