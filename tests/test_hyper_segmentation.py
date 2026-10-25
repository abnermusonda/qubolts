from examples.hyper_segmentation import (
    HyperSegmentation,
    greedy_selection,
    synthetic_problem,
)


def test_objective_counts_people_once_across_overlapping_audiences():
    problem = synthetic_problem(n=12, seed=21)
    resolver = HyperSegmentation()

    broad_only = resolver.objective(problem, frozenset({"broad_0"}))
    broad_and_person = resolver.objective(problem, frozenset({"broad_0", "person_000"}))

    assert broad_and_person == broad_only


def test_baseline_preserves_the_best_single_tier_plan():
    problem = synthetic_problem(n=60, seed=21)
    resolver = HyperSegmentation()
    tier_plans = [
        greedy_selection(problem, resolver, {tier}) for tier in ("broad", "micro", "singleton")
    ]

    baseline = resolver.baseline(problem)

    assert resolver.is_feasible(problem, baseline)
    assert resolver.objective(problem, baseline) == max(
        resolver.objective(problem, plan) for plan in tier_plans
    )


def test_repair_reduces_an_overselected_audience_set():
    problem = synthetic_problem(n=60, seed=21)
    resolver = HyperSegmentation()

    repaired = resolver.repair(problem, frozenset(problem.audiences_by_id))

    assert resolver.is_feasible(problem, repaired)
    assert len(repaired) <= problem.max_audiences
