# Contributing to Qubolts

Thanks for helping. The most valuable contributions right now:

1. **Implement a planned qubolt.** Each module under `qubolts/ads/` and
   `qubolts/mktg/` that subclasses `PlannedResolver` has its formulation in the
   docstring. Open an issue saying which one you are taking.
2. **Add a backend.** Amazon Braket (QAOA), Pasqal, Fujitsu Digital Annealer,
   GPU annealers, OR-Tools CP-SAT as a classical reference. See
   `docs/ARCHITECTURE.md#adding-a-backend`.
3. **Contribute benchmark instances.** Anonymized real problems (segment sizes,
   overlaps, costs) are worth more than any new algorithm. Strip identifiers and
   round numbers; never submit user-level data.
4. **Report results.** Ran `qubolts-bench` on a QPU or hybrid solver? Open a PR
   adding your JSONL run log under `benchmarks/community/`.

## Development setup

```bash
git clone https://github.com/qubolts/qubolts && cd qubolts
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ",spark" to run the Spark tests (needs Java 17+)
pytest
ruff check . && ruff format --check .
```

## Ground rules

* Keep resolvers independent: import from `qubolts.core`, never from another resolver.
* Every resolver ships with a classical baseline and a test that `solve()` with
  fallback is never worse than that baseline.
* Benchmark claims must include the strongest classical reference you have, not
  just a naive one.
* No user-level or personal data in tests, fixtures or benchmark instances.

## Pull requests

* One logical change per PR, with tests.
* Describe the formulation change in plain business terms in the PR description.
* By contributing you agree your work is licensed under Apache-2.0.

Be kind and constructive. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
