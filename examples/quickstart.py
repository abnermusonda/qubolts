"""Compare common practice, greedy, and Qubolts on one problem.

Run from the repository root:

    PYTHONPATH=. python examples/quickstart.py
"""

from time import perf_counter

from qubolts import solve
from qubolts.mktg.segment_selection import SegmentSelection, synthetic_problem

problem = synthetic_problem(n=60, seed=7, max_segments=15)
resolver = SegmentSelection()

naive_decision = resolver.naive_baseline(problem)
greedy_decision = resolver.baseline(problem)
naive = resolver.objective(problem, naive_decision)
greedy = resolver.objective(problem, greedy_decision)

started = perf_counter()
solution = solve(resolver, problem, candidates=10, seed=7, num_reads=100, num_sweeps=2000)
solve_wall_time = perf_counter() - started

print("Qubolts quickstart: audience segment selection")
print("=" * 52)
print("Synthetic problem seed: 7 | sampler seed: 7")
print(
    f"Problem   {len(problem.segments)} segments | {len(problem.overlap):,} "
    f"overlapping pairs | budget ${problem.budget:,.2f} | "
    f"limit {problem.max_segments} segments"
)
print(
    f"QUBO      {solution.metadata['num_variables']} variables | "
    f"{solution.metadata['num_interactions']:,} interactions"
)

print("\nResults")
print("-" * 52)
print(f"{'Approach':<25} {'Objective':>13} {'Selected':>9} {'Spend':>13} {'vs. naive':>11}")
print("-" * 75)
for label, objective, decision in (
    ("Naive (value / $)", naive, naive_decision),
    ("Greedy (overlap-aware)", greedy, greedy_decision),
    (f"Qubolts ({solution.backend})", solution.objective, solution.decision),
):
    spend = resolver.cost(problem, decision)
    lift = (objective - naive) / abs(naive) if naive else 0.0
    objective_text = f"${objective:,.2f}"
    spend_text = f"${spend:,.2f}"
    print(f"{label:<25} {objective_text:>13} {len(decision):>9} {spend_text:>13} {lift:>+10.1%}")

print("\nExecution")
print("-" * 52)
print(f"Feasible                         {'yes' if solution.feasible else 'no'}")
print(f"Used baseline fallback           {'yes' if solution.fell_back else 'no'}")
print("Sampler configuration            100 reads x 2,000 sweeps")
print("Candidate decode/repair limit    10")
print(f"Sampler + candidate processing   {solution.runtime_s:>8.3f} s")
print(f"Total solve wall time            {solve_wall_time:>8.3f} s")
energy = "n/a" if solution.energy is None else f"{solution.energy:.6f}"
print(f"Selected sample QUBO energy      {energy:>12}")
if solution.metadata["error"]:
    print(f"Backend error                    {solution.metadata['error']}")

print("\nSelected segments")
print("-" * 80)
print(
    f"{'ID':<16} {'Reachable users':>16} {'Value / user':>14} {'Cost':>12} {'Standalone value':>18}"
)
print("-" * 80)
for segment in problem.segments:
    if segment.id in solution.decision:
        print(
            f"{segment.id:<16} {segment.size:>16,} "
            f"${segment.value_per_user:>13.5f} ${segment.cost:>11,.2f} "
            f"${segment.value:>17,.2f}"
        )

print("-" * 80)
print(
    f"Total spend: ${resolver.cost(problem, solution.decision):,.2f} / "
    f"${problem.budget:,.2f} budget"
)
