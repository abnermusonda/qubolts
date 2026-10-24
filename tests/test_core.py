import dimod
import pytest

from qubolts import RoutingPolicy, backends, solve
from qubolts.ads import Allocation
from qubolts.core import add_budget_constraint, choose_backend, objective_scale
from qubolts.mktg.segment_selection import SegmentSelection, synthetic_problem


def bqm_with(n):
    return dimod.BinaryQuadraticModel({f"x{i}": -1.0 for i in range(n)}, {}, 0.0, "BINARY")


def test_router_picks_exact_for_tiny_and_sa_otherwise():
    assert choose_backend(bqm_with(5)).name == "exact"
    assert choose_backend(bqm_with(50)).name == "sa"


def test_router_never_picks_remote_without_permission():
    policy = RoutingPolicy(allow_remote=False, remote_min_variables=1)
    assert not choose_backend(bqm_with(5000), policy).remote


@pytest.mark.parametrize("method", ["unbalanced", "slack"])
def test_budget_constraint_ground_state_is_feasible(method):
    costs = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 5.0}
    bqm = dimod.BinaryQuadraticModel({k: -1.0 for k in costs}, {}, 0.0, "BINARY")
    add_budget_constraint(bqm, costs, budget=6.0, penalty=4 * objective_scale(bqm), method=method)
    best = dimod.ExactSolver().sample(bqm).first.sample
    assert sum(costs[k] for k in costs if best[k]) <= 6.0


def test_budget_constraint_skipped_when_it_cannot_bind():
    bqm = bqm_with(3)
    before = bqm.num_variables
    add_budget_constraint(bqm, {"x0": 1, "x1": 1, "x2": 1}, budget=10, penalty=1.0, method="slack")
    assert bqm.num_variables == before


def test_backend_failure_falls_back_to_baseline():
    class Broken(backends.Backend):
        name = "broken"

        def sample(self, bqm, **kw):
            raise RuntimeError("QPU queue unavailable")

    backends.register(Broken())
    p = synthetic_problem(n=15, seed=0)
    sol = solve(SegmentSelection(), p, backend="broken")
    assert sol.fell_back and sol.feasible
    assert "QPU queue unavailable" in sol.metadata["error"]
    with pytest.raises(RuntimeError):
        solve(SegmentSelection(), p, backend="broken", fallback=False)


def test_planned_resolvers_fail_loudly():
    with pytest.raises(NotImplementedError, match="planned"):
        Allocation().build_bqm(None)


def test_remote_backends_report_unavailable_without_credentials(monkeypatch):
    monkeypatch.delenv("DWAVE_API_TOKEN", raising=False)
    assert backends.get("braket").available() is False
