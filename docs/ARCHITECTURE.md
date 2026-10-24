# Architecture

## The idea in one paragraph

Spark (or any data platform) does the heavy lifting and reduces raw data to a small
table of coefficients. A **resolver** (a qubolt) encodes those coefficients as a QUBO.
The **router** picks a backend. The **solve pipeline** decodes and repairs the
backend's samples, checks feasibility, compares against a classical **baseline**, and
returns the better answer. Every run can be appended to a **run log**, so backends are
compared on evidence.

## Components

| Module | Responsibility |
|---|---|
| `core/contract.py` | `Resolver` base class and `Solution` result type |
| `core/qubo.py` | penalty helpers: `objective_scale`, `default_penalty`, `add_budget_constraint` |
| `core/router.py` | `RoutingPolicy` and `choose_backend` |
| `core/solve.py` | the pipeline: encode → route → sample → decode → repair → validate → fallback |
| `core/runlog.py` | append-only JSONL run log |
| `core/planned.py` | `PlannedResolver` for documented-but-unimplemented qubolts |
| `backends/` | dimod-sampler wrappers with a registry |
| `integrations/spark/` | Spark jobs that build inputs and write decisions back |

Rules that keep the codebase easy to grow:

* Resolvers import from `core/` only, never from each other.
* Resolver methods are pure functions of the problem, so they run safely in
  parallel (e.g. one solve per advertiser from a Spark `mapInPandas`).
* `objective` is always the business objective (higher is better), never QUBO
  energy, so solutions from different encodings and backends compare directly.

## Writing a resolver

Implement six methods:

```python
class MyQubolt(Resolver[MyProblem, MyDecision]):
    name = "ads.my_qubolt"

    def build_bqm(self, problem):      # QUBO to minimize; constraints as penalties
    def decode(self, problem, sample):  # 0/1 sample -> decision (skip slack vars)
    def objective(self, problem, d):    # business value, higher is better
    def is_feasible(self, problem, d):  # hard constraints
    def baseline(self, problem):        # fast classical heuristic (fallback + benchmark bar)
    def repair(self, problem, d):       # optional: fix small violations, then polish
```

Checklist for a new resolver PR:

1. Problem dataclass with `to_dict` / `from_dict` (this is the Spark contract).
2. A synthetic generator that produces realistic structure (see
   `mktg/segment_selection/synthetic.py`).
3. Tests: feasibility, a brute-force comparison on small instances, fallback safety.
4. A benchmark entry in `qubolts/bench.py` with a strong classical reference.
5. Docstring explaining the formulation in business terms.

## Encoding constraints

`add_budget_constraint` offers two encodings for `sum(c_i x_i) <= B`:

* **`unbalanced`** (default): no extra variables; adds `λ(l0·h + l1·h²)` with
  `h = Σc_i x_i / B − 1`. Annealer-friendly and approximate. It slightly
  discourages spending, which `repair` undoes by greedily filling leftover budget.
* **`slack`**: exact equality with binary slack bits; costs rounded to a
  resolution. Exact under the `exact` backend, but a rougher landscape for heuristic
  samplers. In our tuning runs it produced 5–30% worse raw samples than unbalanced.

Penalty weights are expressed relative to `objective_scale(bqm)` so the same
`penalty_factor` works across problem sizes. Tuning results for segment selection
are recorded in the resolver's defaults (`penalty_factor=0.5`, unbalanced).

## Adding a backend

```python
from qubolts.backends import Backend, register


class MySolver(Backend):
    name = "my_solver"
    remote = True  # if it costs money or leaves the machine

    def available(self):
        return has_credentials()

    def sample(self, bqm, **kwargs):
        return my_dimod_compatible_sampler().sample(bqm, **kwargs)


register(MySolver())
```

Return a `dimod.SampleSet`. The pipeline decodes the lowest-energy `candidates`
samples (default 10), so returning many reads helps.

## Data flow on Fabric

```
Lakehouse tables ──► Spark: sizes, overlaps, response curves ──► problem dict (small)
                                                                     │
                                                       driver: solve(resolver, problem)
                                                                     │
Delta: decisions (run_id, selected, objective, backend) ◄────────────┘
JSONL run log in OneLake Files ◄──── every solve, for benchmarking
```

Decompose large plans (by advertiser, region or flight) and solve the pieces in
parallel; each piece should stay in the tens to low thousands of variables.
