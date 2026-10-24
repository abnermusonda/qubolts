"""The segment selection qubolt."""

from __future__ import annotations

import dimod

from qubolts.core.contract import Resolver
from qubolts.core.qubo import add_budget_constraint, default_penalty, is_slack

from .model import SegmentSelectionProblem

Decision = frozenset[str]


class SegmentSelection(Resolver[SegmentSelectionProblem, Decision]):
    """Choose a set of audience segments that maximizes unique value within budget.

    Args:
        penalty_factor: multiplier on the automatic penalty weight for the budget
            and segment-count constraints. Raise it if the backend returns many
            infeasible samples; lower it if solutions look overly conservative.
        budget_resolution: cost granularity for the slack encoding, as a fraction
            of the budget. 1/200 = 0.5% of budget.
    """

    name = "mktg.segment_selection"

    def __init__(
        self,
        penalty_factor: float = 0.5,
        budget_method: str = "unbalanced",
        budget_resolution: float = 1 / 200,
        polish: bool = True,
    ):
        self.penalty_factor = penalty_factor
        self.budget_method = budget_method
        self.budget_resolution = budget_resolution
        self.polish = polish

    # --- encoding ---------------------------------------------------------------

    def build_bqm(self, problem: SegmentSelectionProblem) -> dimod.BinaryQuadraticModel:
        scale = max((s.value for s in problem.segments), default=1.0) or 1.0
        bqm = dimod.BinaryQuadraticModel(vartype="BINARY")
        for s in problem.segments:
            bqm.add_linear(s.id, -s.value / scale)  # minimize negative value
        ids = [s.id for s in problem.segments]
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                w = problem.pair_penalty(a, b)
                if w:
                    bqm.add_quadratic(a, b, w / scale)

        penalty = default_penalty(bqm, self.penalty_factor)
        add_budget_constraint(
            bqm,
            {s.id: s.cost for s in problem.segments},
            problem.budget,
            penalty=penalty,
            method=self.budget_method,
            resolution=problem.budget * self.budget_resolution,
        )
        if problem.max_segments is not None and problem.max_segments < len(ids):
            bqm.add_linear_inequality_constraint(
                [(sid, 1) for sid in ids],
                lagrange_multiplier=penalty,
                label="max_segments",
                lb=0,
                ub=problem.max_segments,
            )
        return bqm

    def decode(self, problem: SegmentSelectionProblem, sample: dict) -> Decision:
        return frozenset(v for v, x in sample.items() if x and not is_slack(v))

    # --- business logic ---------------------------------------------------------

    def objective(self, problem: SegmentSelectionProblem, decision: Decision) -> float:
        ids = sorted(decision)
        total = sum(problem.segment(s).value for s in ids)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                total -= problem.pair_penalty(a, b)
        return total

    def cost(self, problem: SegmentSelectionProblem, decision: Decision) -> float:
        return sum(problem.segment(s).cost for s in decision)

    def is_feasible(self, problem: SegmentSelectionProblem, decision: Decision) -> bool:
        if self.cost(problem, decision) > problem.budget + 1e-9:
            return False
        return problem.max_segments is None or len(decision) <= problem.max_segments

    def marginal(self, problem: SegmentSelectionProblem, sid: str, others: Decision) -> float:
        """Value added by ``sid`` given the other active segments."""
        v = problem.segment(sid).value
        return v - sum(problem.pair_penalty(sid, o) for o in others if o != sid)

    def repair(self, problem: SegmentSelectionProblem, decision: Decision) -> Decision:
        """Drop the least efficient segments until feasible, then greedily fill spare budget."""
        chosen = set(decision)
        while chosen and not self.is_feasible(problem, frozenset(chosen)):
            worst = min(
                chosen,
                key=lambda s: (
                    self.marginal(problem, s, frozenset(chosen))
                    / max(problem.segment(s).cost, 1e-9)
                ),
            )
            chosen.remove(worst)
        filled = self._greedy_fill(problem, chosen)
        return self.local_search(problem, set(filled)) if self.polish else filled

    def local_search(
        self, problem: SegmentSelectionProblem, chosen: set[str], max_rounds: int = 50
    ) -> Decision:
        """First-improvement hill climbing on the true objective.

        Moves: drop one segment; swap one in for one out; insert one segment and
        evict the least efficient ones until the budget fits. Standard
        post-processing for annealer output: it cannot make the answer worse, and
        it turns "near-optimal" samples into local optima.
        """
        chosen = set(chosen)
        for _ in range(max_rounds):
            before = chosen
            chosen = set(self._drop_or_swap(problem, chosen))
            if chosen == before:
                chosen = set(self._insert_with_eviction(problem, chosen))
            if chosen == before:
                break
        return frozenset(chosen)

    def _insert_with_eviction(self, problem: SegmentSelectionProblem, chosen: set[str]) -> Decision:
        current = self.objective(problem, frozenset(chosen))
        for s in problem.segments:
            if s.id in chosen:
                continue
            cand = set(chosen) | {s.id}
            while not self.is_feasible(problem, frozenset(cand)):
                others = cand - {s.id}
                if not others:
                    break
                worst = min(
                    others,
                    key=lambda o: (
                        self.marginal(problem, o, frozenset(cand))
                        / max(problem.segment(o).cost, 1e-9)
                    ),
                )
                cand.remove(worst)
            if not self.is_feasible(problem, frozenset(cand)):
                continue
            filled = self._greedy_fill(problem, cand)
            if self.objective(problem, filled) > current + 1e-9:
                return filled
        return frozenset(chosen)

    def _drop_or_swap(self, problem: SegmentSelectionProblem, chosen: set[str]) -> Decision:
        """Apply the first improving drop or 1-for-1 swap, then refill. No move: unchanged."""
        spend = self.cost(problem, frozenset(chosen))
        frozen = frozenset(chosen)
        for out in sorted(chosen):
            loss = self.marginal(problem, out, frozen)  # value lost by dropping `out`
            rest = frozenset(chosen - {out})
            if loss < -1e-9:  # dropping alone helps
                return self._greedy_fill(problem, set(rest))
            room = problem.budget - spend + problem.segment(out).cost
            for s in problem.segments:
                if s.id in chosen or s.cost > room + 1e-9:
                    continue
                if self.marginal(problem, s.id, rest) > loss + 1e-9:
                    return self._greedy_fill(problem, set(rest) | {s.id})
        return frozen

    def baseline(self, problem: SegmentSelectionProblem) -> Decision:
        """Overlap-aware greedy: repeatedly add the best marginal value per dollar."""
        return self._greedy_fill(problem, set())

    def naive_baseline(self, problem: SegmentSelectionProblem) -> Decision:
        """What most teams do today: rank by value per dollar, ignore overlap."""
        chosen: set[str] = set()
        spend = 0.0
        ranked = sorted(problem.segments, key=lambda s: s.value / max(s.cost, 1e-9), reverse=True)
        for s in ranked:
            if problem.max_segments is not None and len(chosen) >= problem.max_segments:
                break
            if spend + s.cost <= problem.budget:
                chosen.add(s.id)
                spend += s.cost
        return frozenset(chosen)

    def _greedy_fill(self, problem: SegmentSelectionProblem, chosen: set[str]) -> Decision:
        chosen = set(chosen)
        spend = self.cost(problem, frozenset(chosen))
        while problem.max_segments is None or len(chosen) < problem.max_segments:
            best, best_ratio = None, 0.0
            for s in problem.segments:
                if s.id in chosen or spend + s.cost > problem.budget:
                    continue
                gain = self.marginal(problem, s.id, frozenset(chosen))
                ratio = gain / max(s.cost, 1e-9)
                if gain > 0 and ratio > best_ratio:
                    best, best_ratio = s, ratio
            if best is None:
                break
            chosen.add(best.id)
            spend += best.cost
        return frozenset(chosen)
