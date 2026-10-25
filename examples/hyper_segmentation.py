"""Simulate hyper-segmentation from broad audiences down to one-person audiences.

Run from the repository root:

    PYTHONPATH=. python examples/hyper_segmentation.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from itertools import combinations
from time import perf_counter

import dimod

from qubolts import solve
from qubolts.core.contract import Resolver
from qubolts.core.qubo import add_budget_constraint, default_penalty


@dataclass(frozen=True)
class Person:
    id: str
    family: int
    intent: float
    value: float
    media_cost: float


@dataclass(frozen=True)
class Audience:
    id: str
    tier: str
    members: frozenset[str]
    cost: float


@dataclass
class HyperSegmentationProblem:
    people: list[Person]
    audiences: list[Audience]
    budget: float
    max_audiences: int

    def __post_init__(self) -> None:
        self.people_by_id = {person.id: person for person in self.people}
        self.audiences_by_id = {audience.id: audience for audience in self.audiences}
        self.person_indices = {person.id: i for i, person in enumerate(self.people)}
        self.person_values = tuple(person.value for person in self.people)
        self.audience_masks = {
            audience.id: sum(1 << self.person_indices[person_id] for person_id in audience.members)
            for audience in self.audiences
        }
        self.audiences_by_person: dict[str, list[str]] = defaultdict(list)
        for audience in self.audiences:
            for person_id in audience.members:
                self.audiences_by_person[person_id].append(audience.id)

    def value_for_mask(self, mask: int) -> float:
        total = 0.0
        while mask:
            bit = mask & -mask
            total += self.person_values[bit.bit_length() - 1]
            mask ^= bit
        return total

    def reached_mask(self, decision: frozenset[str] | set[str]) -> int:
        mask = 0
        for audience_id in decision:
            mask |= self.audience_masks[audience_id]
        return mask


class HyperSegmentation(Resolver[HyperSegmentationProblem, frozenset[str]]):
    name = "mktg.hyper_segmentation"

    def build_bqm(self, problem: HyperSegmentationProblem) -> dimod.BinaryQuadraticModel:
        polynomial: dict[tuple[str, ...], float] = defaultdict(float)
        for person in problem.people:
            audience_ids = sorted(problem.audiences_by_person[person.id])
            for degree in range(1, len(audience_ids) + 1):
                coefficient = person.value * (-1) ** degree
                for term in combinations(audience_ids, degree):
                    polynomial[term] += coefficient

        strength = 2 * sum(person.value for person in problem.people)
        bqm = dimod.make_quadratic(polynomial, strength, vartype=dimod.BINARY)
        penalty = default_penalty(bqm)
        add_budget_constraint(
            bqm,
            {a.id: a.cost for a in problem.audiences},
            problem.budget,
            penalty=penalty,
        )
        bqm.add_linear_inequality_constraint(
            [(a.id, 1) for a in problem.audiences],
            lagrange_multiplier=penalty,
            label="max_audiences",
            lb=0,
            ub=problem.max_audiences,
        )
        return bqm

    def decode(self, problem: HyperSegmentationProblem, sample: dict) -> frozenset[str]:
        return frozenset(aid for aid in problem.audiences_by_id if sample.get(aid, 0))

    def objective(self, problem: HyperSegmentationProblem, decision: frozenset[str]) -> float:
        return problem.value_for_mask(problem.reached_mask(decision))

    def cost(self, problem: HyperSegmentationProblem, decision: frozenset[str]) -> float:
        return sum(problem.audiences_by_id[aid].cost for aid in decision)

    def is_feasible(self, problem: HyperSegmentationProblem, decision: frozenset[str]) -> bool:
        return (
            self.cost(problem, decision) <= problem.budget + 1e-9
            and len(decision) <= problem.max_audiences
        )

    def baseline(self, problem: HyperSegmentationProblem) -> frozenset[str]:
        candidates = [
            greedy_selection(problem, self, {tier}) for tier in ("broad", "micro", "singleton")
        ]
        return max(candidates, key=lambda decision: self.objective(problem, decision))

    def repair(self, problem: HyperSegmentationProblem, decision: frozenset[str]) -> frozenset[str]:
        chosen = set(decision)
        coverage = [0] * len(problem.people)
        for audience_id in chosen:
            mask = problem.audience_masks[audience_id]
            while mask:
                bit = mask & -mask
                coverage[bit.bit_length() - 1] += 1
                mask ^= bit

        marginal_loss = {audience_id: 0.0 for audience_id in chosen}
        for person in problem.people:
            index = problem.person_indices[person.id]
            if coverage[index] == 1:
                covering_audience = next(
                    aid for aid in problem.audiences_by_person[person.id] if aid in chosen
                )
                marginal_loss[covering_audience] += person.value

        versions = {audience_id: 0 for audience_id in chosen}
        removal_queue = [
            (
                loss / max(problem.audiences_by_id[aid].cost, 1e-12),
                aid,
                0,
            )
            for aid, loss in marginal_loss.items()
        ]
        heapify(removal_queue)
        spend = self.cost(problem, frozenset(chosen))
        while chosen and (spend > problem.budget + 1e-9 or len(chosen) > problem.max_audiences):
            while True:
                _, least_efficient, version = heappop(removal_queue)
                if least_efficient in chosen and versions[least_efficient] == version:
                    break
            chosen.remove(least_efficient)
            spend -= problem.audiences_by_id[least_efficient].cost
            mask = problem.audience_masks[least_efficient]
            while mask:
                bit = mask & -mask
                index = bit.bit_length() - 1
                if coverage[index] == 2:
                    person_id = problem.people[index].id
                    survivor = next(
                        aid for aid in problem.audiences_by_person[person_id] if aid in chosen
                    )
                    marginal_loss[survivor] += problem.people[index].value
                    versions[survivor] += 1
                    heappush(
                        removal_queue,
                        (
                            marginal_loss[survivor]
                            / max(problem.audiences_by_id[survivor].cost, 1e-12),
                            survivor,
                            versions[survivor],
                        ),
                    )
                coverage[index] -= 1
                mask ^= bit
        return greedy_selection(problem, self, initially_selected=chosen)


def greedy_selection(
    problem: HyperSegmentationProblem,
    resolver: HyperSegmentation,
    tiers: set[str] | None = None,
    initially_selected: set[str] | None = None,
) -> frozenset[str]:
    selected = set(initially_selected or ())
    candidates = [
        audience
        for audience in problem.audiences
        if audience.id not in selected and (tiers is None or audience.tier in tiers)
    ]
    reached = problem.reached_mask(selected)
    spend = resolver.cost(problem, frozenset(selected))
    while len(selected) < problem.max_audiences:
        best = None
        best_ratio = 0.0
        for audience in candidates:
            if spend + audience.cost > problem.budget + 1e-9:
                continue
            gain = problem.value_for_mask(problem.audience_masks[audience.id] & ~reached)
            ratio = gain / audience.cost
            if ratio > best_ratio or (
                ratio == best_ratio and best is not None and audience.id > best.id
            ):
                best, best_ratio = audience, ratio
        if best is None:
            break
        gain = problem.value_for_mask(problem.audience_masks[best.id] & ~reached)
        if gain <= 0:
            break
        selected.add(best.id)
        reached |= problem.audience_masks[best.id]
        spend += best.cost
        candidates.remove(best)
    return frozenset(selected)


def synthetic_problem(n: int = 60, seed: int = 21) -> HyperSegmentationProblem:
    import numpy as np

    rng = np.random.default_rng(seed)
    family_count = 6
    people = []
    for i in range(n):
        family = i % family_count
        intent = float(rng.beta(2.2, 2.0))
        family_effect = (family - (family_count - 1) / 2) * 0.6
        value = round(max(1.0, 5.0 + 10.0 * intent + family_effect), 2)
        people.append(
            Person(
                id=f"person_{i:03d}",
                family=family,
                intent=intent,
                value=value,
                media_cost=round(float(rng.uniform(0.35, 0.75)), 2),
            )
        )

    audiences = []
    for family in range(family_count):
        members = [person for person in people if person.family == family]
        audiences.append(
            Audience(
                id=f"broad_{family}",
                tier="broad",
                members=frozenset(person.id for person in members),
                cost=round(3.0 + sum(person.media_cost for person in members), 2),
            )
        )
        median_intent = sorted(person.intent for person in members)[len(members) // 2]
        for band in ("high", "low"):
            subset = [
                person for person in members if (person.intent >= median_intent) == (band == "high")
            ]
            audiences.append(
                Audience(
                    id=f"micro_{family}_{band}",
                    tier="micro",
                    members=frozenset(person.id for person in subset),
                    cost=round(1.25 + sum(person.media_cost for person in subset), 2),
                )
            )

    audiences.extend(
        Audience(
            id=person.id,
            tier="singleton",
            members=frozenset({person.id}),
            cost=round(0.20 + person.media_cost, 2),
        )
        for person in people
    )
    return HyperSegmentationProblem(
        people=people,
        audiences=audiences,
        budget=30.0,
        max_audiences=12,
    )


def print_result(
    label: str,
    problem: HyperSegmentationProblem,
    resolver: HyperSegmentation,
    decision: frozenset[str],
) -> None:
    reached = sum(
        any(aid in decision for aid in problem.audiences_by_person[person.id])
        for person in problem.people
    )
    print(
        f"{label:<22} ${resolver.objective(problem, decision):>9,.2f} "
        f"{reached:>7}/{len(problem.people):<7} {len(decision):>5} "
        f"${resolver.cost(problem, decision):>8,.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--people", type=int, default=60, help="synthetic population size")
    parser.add_argument("--seed", type=int, default=21, help="population and sampler seed")
    parser.add_argument(
        "--budget-per-person",
        type=float,
        default=0.5,
        help="campaign budget per simulated person",
    )
    parser.add_argument("--max-audiences", type=int, default=12, help="activation cap")
    parser.add_argument(
        "--backend",
        choices=("auto", "sa", "torch_sa"),
        default="auto",
        help="auto/CPU simulated annealing or optional CUDA/Metal PyTorch backend",
    )
    parser.add_argument("--device", default="auto", help="torch_sa device: auto, mps, or cuda")
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="CPU SA workers; 0 selects a memory-aware default",
    )
    args = parser.parse_args()

    problem = synthetic_problem(n=args.people, seed=args.seed)
    problem.budget = args.budget_per_person * args.people
    problem.max_audiences = args.max_audiences
    resolver = HyperSegmentation()
    tiers = (
        ("Broad-only greedy", {"broad"}),
        ("Micro-only greedy", {"micro"}),
        ("Singleton-only greedy", {"singleton"}),
    )
    tier_results = [
        (label, greedy_selection(problem, resolver, allowed_tiers))
        for label, allowed_tiers in tiers
    ]

    started = perf_counter()
    sampler_options: dict[str, object] = {
        "seed": args.seed,
        "candidates": args.candidates,
        "num_reads": args.num_reads,
        "num_sweeps": args.num_sweeps,
    }
    if args.backend == "torch_sa":
        sampler_options["device"] = args.device
    elif args.backend == "sa":
        sampler_options["num_workers"] = args.num_workers
    solution = solve(
        resolver,
        problem,
        backend=args.backend,
        **sampler_options,
    )
    wall_time = perf_counter() - started

    print("Qubolts example: hyper-segmentation")
    print("=" * 58)
    print(f"Population seed: {args.seed} | sampler seed: {args.seed}")
    print(
        f"Population {len(problem.people)} people | {len(problem.audiences)} candidate "
        f"audiences (6 broad, 12 micro, {len(problem.people)} singleton)"
    )
    print(
        f"Constraints budget ${problem.budget:.2f} | max {problem.max_audiences} active audiences"
    )
    print(
        f"Solver request {args.backend} | reads {args.num_reads} | sweeps {args.num_sweeps}"
        + (f" | workers {args.num_workers or 'auto'}" if args.backend == "sa" else "")
        + (f" | device {args.device}" if args.backend == "torch_sa" else "")
    )
    print(
        f"QUBO       {solution.metadata['num_variables']} variables | "
        f"{solution.metadata['num_interactions']:,} interactions"
    )
    print("\nResults")
    print("-" * 58)
    print(f"{'Strategy':<22} {'Value':>10} {'Unique reach':>15} {'Audiences':>9} {'Spend':>10}")
    print("-" * 72)
    for label, decision in tier_results:
        print_result(label, problem, resolver, decision)
    print_result(f"Qubolts ({solution.backend})", problem, resolver, solution.decision)
    print("-" * 72)
    print(
        f"Feasible: {solution.feasible} | fallback: {solution.fell_back} | "
        f"backend: {solution.runtime_s:.3f}s | end-to-end: {wall_time:.3f}s"
    )
    if solution.fell_back:
        print("Returned the best feasible tier baseline because no sampled candidate beat it.")
    energy = f"{solution.energy:.6f}" if solution.energy is not None else "n/a"
    print(f"QUBO energy: {energy}")
    print("\nSelected audiences")
    for audience in problem.audiences:
        if audience.id in solution.decision:
            print(
                f"  {audience.id:<20} {audience.tier:<9} "
                f"{len(audience.members):>2} people  ${audience.cost:>6.2f}"
            )


if __name__ == "__main__":
    main()
