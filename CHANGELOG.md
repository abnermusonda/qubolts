# Changelog

## 0.1.0 (unreleased)

* Core: resolver contract, QUBO penalty helpers (unbalanced and slack budget
  encodings), router, solve pipeline with repair, validation and baseline
  fallback, JSONL run log.
* Backends: exact, simulated annealing, tabu, steepest descent (dwave-samplers);
  D-Wave Leap hybrid and QPU (optional); Braket placeholder.
* Segment selection qubolt with overlap-aware objective, greedy baselines,
  repair and local search, synthetic generator, Spark/Fabric job.
* Planned qubolts with documented formulations: allocation, coverage,
  assignment, identity, scheduling, uncertainty.
* `qubolts-bench` benchmark harness.
