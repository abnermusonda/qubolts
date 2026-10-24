import itertools

import pytest

from qubolts import RunLog, solve
from qubolts.mktg.segment_selection import (
    Segment,
    SegmentSelection,
    SegmentSelectionProblem,
    problem_from_memberships,
    synthetic_problem,
)


def brute_force(resolver, problem):
    ids = [s.id for s in problem.segments]
    best = (float("-inf"), frozenset())
    for k in range(len(ids) + 1):
        for combo in itertools.combinations(ids, k):
            d = frozenset(combo)
            if resolver.is_feasible(problem, d):
                best = max(best, (resolver.objective(problem, d), d), key=lambda t: t[0])
    return best


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("backend", ["exact", "sa", "tabu"])
def test_near_optimal_on_small_problems(seed, backend):
    # The default unbalanced budget encoding is approximate, so even the exact
    # backend solves a slightly different problem; repair + polish close most of
    # the gap. Measured worst case on these instances: 0.4%.
    r = SegmentSelection()
    p = synthetic_problem(n=12, seed=seed)
    opt, _ = brute_force(r, p)
    sol = solve(r, p, backend=backend, fallback=False)
    assert sol.feasible
    assert sol.objective >= opt * 0.99


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_slack_encoding_with_exact_backend_is_optimal(seed):
    r = SegmentSelection(budget_method="slack")
    p = synthetic_problem(n=12, seed=seed)
    opt, _ = brute_force(r, p)
    sol = solve(r, p, backend="exact", fallback=False)
    assert sol.objective == pytest.approx(opt, rel=1e-6)


def test_never_worse_than_baseline_with_fallback():
    r = SegmentSelection()
    for seed in range(3):
        p = synthetic_problem(n=40, seed=seed)
        sol = solve(r, p, backend="descent", num_reads=5)
        assert sol.feasible
        assert sol.objective >= sol.baseline_objective - 1e-9


def test_beats_naive_when_segments_overlap():
    r = SegmentSelection()
    p = synthetic_problem(n=30, seed=4)
    naive = r.objective(p, r.naive_baseline(p))
    sol = solve(r, p, backend="sa")
    assert sol.objective > naive


def test_max_segments_respected():
    r = SegmentSelection()
    p = synthetic_problem(n=20, seed=5, max_segments=3)
    for backend in ("sa", "exact" if False else "tabu"):
        sol = solve(r, p, backend=backend)
        assert len(sol.decision) <= 3
        assert sol.feasible


def test_slack_encoding_also_feasible():
    r = SegmentSelection(budget_method="slack")
    p = synthetic_problem(n=14, seed=2)
    sol = solve(r, p, backend="sa", fallback=False)
    assert sol.feasible


def test_overlap_is_penalized_once():
    segs = [Segment("a", 100, 1.0, 10), Segment("b", 100, 2.0, 10)]
    p = SegmentSelectionProblem(segs, budget=100, overlap={("b", "a"): 40})
    r = SegmentSelection()
    # 100*1 + 100*2 - 40 * min(1, 2)
    assert r.objective(p, frozenset({"a", "b"})) == pytest.approx(260)


def test_repair_restores_feasibility():
    r = SegmentSelection()
    p = synthetic_problem(n=20, seed=6)
    everything = frozenset(s.id for s in p.segments)
    assert not r.is_feasible(p, everything)
    assert r.is_feasible(p, r.repair(p, everything))


def test_problem_roundtrip_and_memberships():
    rows = [("u1", "a"), ("u2", "a"), ("u2", "b"), ("u3", "b"), ("u3", "c"), ("u2", "c")]
    meta = {k: {"value_per_user": 1.0, "cost": 1.0} for k in "abc"}
    p = problem_from_memberships(rows, meta, budget=2)
    assert {s.id: s.size for s in p.segments} == {"a": 2, "b": 2, "c": 2}
    assert p.overlap == {("a", "b"): 1, ("a", "c"): 1, ("b", "c"): 2}
    again = SegmentSelectionProblem.from_dict(p.to_dict())
    assert again.overlap == p.overlap and again.segments == p.segments


def test_run_log(tmp_path):
    log = RunLog(tmp_path / "runs.jsonl")
    solve(SegmentSelection(), synthetic_problem(n=10, seed=0), log=log, tags={"exp": "t"})
    rows = log.read()
    assert len(rows) == 1 and rows[0]["exp"] == "t" and rows[0]["feasible"]
