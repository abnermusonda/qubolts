"""Quickstart: compare common practice, greedy, and Qubolts on one problem.

python examples/quickstart.py
"""

from qubolts import solve
from qubolts.mktg.segment_selection import SegmentSelection, synthetic_problem

problem = synthetic_problem(n=60, seed=7, max_segments=15)
resolver = SegmentSelection()

naive = resolver.objective(problem, resolver.naive_baseline(problem))
solution = solve(resolver, problem)  # backend="auto" -> simulated annealing at this size

print(f"segments considered: {len(problem.segments)}   budget: ${problem.budget:,.0f}")
print(f"naive (rank by value/$):  {naive:>12,.0f}")
print(f"greedy (overlap-aware):   {solution.baseline_objective:>12,.0f}")
print(f"qubolts [{solution.backend}]:          {solution.objective:>12,.0f}")
print(f"lift vs naive: {(solution.objective - naive) / naive:+.1%}")
spend = resolver.cost(problem, solution.decision)
print(f"activate {len(solution.decision)} segments, spend ${spend:,.0f}")
