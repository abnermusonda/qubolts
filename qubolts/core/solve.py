"""The shared pipeline: encode -> route -> sample -> decode -> repair -> validate -> fall back."""

from __future__ import annotations

import time
from typing import Any, TypeVar

from qubolts import backends
from qubolts.core.contract import Resolver, Solution
from qubolts.core.router import RoutingPolicy, choose_backend
from qubolts.core.runlog import RunLog

P = TypeVar("P")
A = TypeVar("A")


def solve(
    resolver: Resolver[P, A],
    problem: P,
    *,
    backend: str = "auto",
    policy: RoutingPolicy | None = None,
    fallback: bool = True,
    candidates: int = 10,
    log: RunLog | None = None,
    tags: dict[str, Any] | None = None,
    **sampler_kwargs: Any,
) -> Solution[A]:
    """Solve ``problem`` with ``resolver`` and return the best feasible decision found.

    Args:
        backend: a registered backend name, or ``"auto"`` to let the router decide.
        fallback: if True, return the classical baseline whenever it beats (or the
            backend fails to produce) a feasible answer. You never ship a worse plan.
        candidates: how many of the lowest-energy samples to decode and repair.
            Annealers return many reads; the best *repaired* one is often not the
            lowest-energy raw one.
        log: optional :class:`RunLog` to record the run for benchmarking.
        sampler_kwargs: passed through to the backend (``num_reads``, ``seed``, ...).
    """
    t0 = time.perf_counter()
    baseline = resolver.baseline(problem)
    baseline_obj = resolver.objective(problem, baseline)
    t_baseline = time.perf_counter() - t0

    bqm = resolver.build_bqm(problem)
    chosen = backends.get(backend) if backend != "auto" else choose_backend(bqm, policy)

    error: str | None = None
    best: tuple[float, A, float] | None = None  # (objective, decision, energy)
    t1 = time.perf_counter()
    try:
        sampleset = chosen.sample(bqm, **sampler_kwargs)
        for i, rec in enumerate(sampleset.data(["sample", "energy"], sorted_by="energy")):
            if i >= candidates:
                break
            decision = resolver.repair(problem, resolver.decode(problem, dict(rec.sample)))
            if not resolver.is_feasible(problem, decision):
                continue
            obj = resolver.objective(problem, decision)
            if best is None or obj > best[0]:
                best = (obj, decision, float(rec.energy))
    except Exception as exc:  # backend outages must not break a pipeline
        if not fallback:
            raise
        error = f"{type(exc).__name__}: {exc}"
    t_backend = time.perf_counter() - t1

    use_baseline = best is None or (fallback and baseline_obj > best[0])
    if use_baseline:
        decision, obj, energy = baseline, baseline_obj, None
    else:
        obj, decision, energy = best  # type: ignore[misc]

    solution: Solution[A] = Solution(
        decision=decision,
        objective=obj,
        feasible=resolver.is_feasible(problem, decision),
        backend=chosen.name,
        runtime_s=t_backend,
        energy=energy,
        fell_back=use_baseline,
        baseline_objective=baseline_obj,
        metadata={
            "resolver": resolver.name,
            "num_variables": bqm.num_variables,
            "num_interactions": bqm.num_interactions,
            "baseline_runtime_s": t_baseline,
            "backend_objective": None if best is None else best[0],
            "error": error,
        },
    )
    if log is not None:
        log.record(**solution.to_dict(), **(tags or {}))
    return solution
