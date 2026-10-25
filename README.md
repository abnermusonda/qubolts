# Qubolts

**Quantum-ready decision engines for adtech and martech.**

Qubolts turns everyday marketing decisions (which audiences to activate, how to
split a budget, which creative goes where) into QUBO problems and solves them on
swappable backends: exact search, simulated annealing, tabu search, or a cloud
quantum/hybrid solver. You get a better decision today on a laptop or a Spark
driver, and a clean seam to try quantum hardware when it pays off.

Each solver is a **qubolt**: a small resolver that snaps onto a shared core.

```
problem ─► build_bqm ─► route ─► sample ─► decode ─► repair ─► validate ─► best of (backend, baseline)
             (QUBO)     (exact / SA / tabu / Leap / …)                       └─ never ships a worse plan
```

## Why

* **Adtech decisions are combinatorial.** Picking 12 of 80 overlapping audience
  segments under a budget has more combinations than anyone can check. Teams fall
  back on "rank by value per dollar", which pays twice for overlapping users.
* **QUBO is the common language** of quantum annealers, gate-model QAOA, and fast
  classical "quantum-inspired" solvers. Write the problem once, run it anywhere.
* **Nobody publishes adtech resolvers or honest benchmarks.** Agency pilots (IPG,
  Recruit) are proprietary and compare against greedy baselines. Qubolts is open
  and benchmarks every backend against a strong classical bar.

## Install

Until the first PyPI release, install from source: `pip install -e ".[dev]"`.

```bash
pip install qubolts            # core + local backends
pip install "qubolts[leap]"    # + D-Wave Leap hybrid / QPU backends
pip install "qubolts[spark]"   # + PySpark jobs (already present on Fabric / Databricks)
```

## Quickstart

```python
from qubolts import solve
from qubolts.mktg.segment_selection import SegmentSelection, synthetic_problem

problem = synthetic_problem(n=60, seed=7)  # or SegmentSelectionProblem.from_dict(...)
solution = solve(
    SegmentSelection(), problem
)  # backend="auto" | "exact" | "sa" | "tabu" | "leap_hybrid"

print(sorted(solution.decision))  # segments to activate
print(solution.objective, solution.lift_vs_baseline, solution.backend, solution.fell_back)
```

On Microsoft Fabric or any Spark runtime, see [`fabric/README.md`](fabric/README.md):
Spark computes segment sizes and pairwise overlaps, the driver solves, and the
decision lands in a Delta table.

## The qubolts

| Qubolt | Module | Decision | Status |
|---|---|---|---|
| Segment selection | `qubolts.mktg.segment_selection` | Audience segments to activate, overlap-aware, under budget and count caps | **Working** |
| Allocation | `qubolts.ads.allocation` | Budget split across line items, channels, dayparts | Planned |
| Coverage | `qubolts.ads.coverage` | Publisher mix maximizing unique reach under frequency caps | Planned |
| Assignment | `qubolts.ads.assignment` | Creative × audience × placement matching | Planned |
| Identity | `qubolts.ads.identity` | Hard identity clusters as graph partitioning | Planned |
| Scheduling | `qubolts.mktg.scheduling` | Syncs, refreshes and sends under rate limits and SLAs | Planned |
| Uncertainty | `qubolts.mktg.uncertainty` | MMM / attribution uncertainty, feature selection | Future |

Every planned module documents its QUBO formulation in its docstring, ready to implement.

## Backends

| Name | Where it runs | Notes |
|---|---|---|
| `exact` | local | brute force, ≤ 22 variables; ground truth for tests |
| `sa` | local | simulated annealing (dwave-samplers); default above 18 variables |
| `tabu` | local | multistart tabu search |
| `descent` | local | steepest descent from random starts; fast floor |
| `leap_hybrid` | D-Wave Leap | hybrid solver; needs `qubolts[leap]` + `DWAVE_API_TOKEN` |
| `leap_qpu` | D-Wave Leap | Advantage QPU with auto-embedding; small subproblems |
| `braket` | Amazon Braket | placeholder, contributions welcome |

Any dimod-compatible sampler plugs in with a ~10-line `Backend` subclass.
Remote (paid) backends are never chosen automatically unless you allow it with
`RoutingPolicy(allow_remote=True)`.

## Benchmarks (honest ones)

```bash
qubolts-bench segment-selection --sizes 12 40 100 --seeds 3 --backends exact sa tabu descent
```

Results on synthetic overlapping-family problems, 3 seeds each, laptop CPU:

| n segments | naive (rank by value/$) | greedy (overlap-aware) | greedy + local search | SA / tabu / descent + repair |
|---|---|---|---|---|
| 12 | baseline | +18.6% | +22.9% | +22.9% (= optimum) |
| 40 | baseline | +15.9% | +16.5% | +16.5% |
| 100 | baseline | +43.6% | +45.8% | +45.8% |

With a cap of 12 activated segments (`--max-segments 12`, 5 seeds), plain greedy
struggles and the samplers start to edge past the classical bar:

| n segments | greedy vs greedy+LS | SA vs greedy+LS | tabu vs greedy+LS | descent vs greedy+LS |
|---|---|---|---|---|
| 60 | −30.5% | +0.3% | +0.3% | +0.4% |
| 100 | −36.3% | +0.1% | +0.1% | +0.1% |

### 1,000-segment scaling example

A single seeded run with a 15-segment activation cap exercised a much denser
problem than the multi-seed benchmark above. It generated 499,408 overlapping
pairs and a QUBO with 1,004 variables and 503,506 interactions. The local
simulated-annealing solver returned a feasible, non-fallback solution:

| Method | Objective | Segments selected | Spend | Lift vs. naive |
|---|---:|---:|---:|---:|
| Naive (value / $) | $41,165.47 | 15 | $1,492.17 | baseline |
| Greedy (overlap-aware) | $88,207.13 | 15 | $3,510.16 | +114.3% |
| Qubolts (SA) | $212,447.55 | 15 | $31,811.29 | +416.1% |

The budget was $119,274.25. The run took 33.7 seconds in the backend and
35.7 seconds end-to-end on an Apple Silicon Mac; timings depend on the machine.
This is one synthetic instance, not an aggregate benchmark, so the result should
not be interpreted as a general performance claim.

Reproduce the problem and solver settings with:

```python
from qubolts import solve
from qubolts.mktg.segment_selection import SegmentSelection, synthetic_problem

problem = synthetic_problem(n=1000, seed=7, max_segments=15)
solution = solve(
    SegmentSelection(),
    problem,
    seed=7,
    num_reads=100,
    num_sweeps=2000,
)
print(solution.to_dict())
```

Selected segment IDs in that run:
`seg_206_f26`, `seg_218_f300`, `seg_256_f35`, `seg_339_f320`,
`seg_514_f114`, `seg_608_f107`, `seg_609_f240`, `seg_667_f63`,
`seg_841_f183`, `seg_865_f187`, `seg_893_f249`, `seg_903_f206`,
`seg_937_f148`, `seg_967_f332`, and `seg_979_f296`.

What this says, plainly:

1. **Modeling overlap is the big win**: +16% to +46% unique value over the common
   practice at the same budget, more under count caps. No quantum hardware needed.
2. **Annealers roughly tie a well-tuned classical heuristic** on these instances
   (0% to +1.7% better per instance). Every backend run gets the same repair and
   local search, and "greedy + local search" is the bar a backend has to beat. We
   report it so nobody has to take quantum claims on faith.
3. The interesting next step is harder instances (many interacting constraints,
   coupled allocation + selection) and real QPU/hybrid runs. Run logs are JSONL so
   results from any machine or backend merge into one table.

## Project layout

```
qubolts/
├── core/            contract, QUBO helpers, router, solve pipeline, run log
├── backends/        local (exact, sa, tabu, descent) and remote (Leap, Braket)
├── ads/             allocation, coverage, assignment, identity
├── mktg/            segment_selection, scheduling, uncertainty
├── integrations/
│   └── spark/       Spark/Fabric jobs: build inputs, write decisions
└── bench.py         benchmark harness (qubolts-bench)
fabric/              Fabric notebooks and setup notes
tests/               pytest suite (Spark tests skip without a JVM)
docs/                architecture and design notes
```

## Contributing

Contributions are very welcome, especially new resolvers, backends and real-world
(anonymized) benchmark instances. Start with [CONTRIBUTING.md](CONTRIBUTING.md)
and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Built on

[dimod](https://github.com/dwavesystems/dimod) and
[dwave-samplers](https://github.com/dwavesystems/dwave-samplers) (Apache-2.0) for
the QUBO model and classical samplers, NumPy, and optionally PySpark and the
D-Wave Ocean SDK. The default budget encoding uses the unbalanced penalization of
Montañez-Barrera et al. ([arXiv:2211.13914](https://arxiv.org/abs/2211.13914)).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
