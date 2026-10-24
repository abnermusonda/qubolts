# Running Qubolts on Microsoft Fabric (or any Spark)

Spark does the heavy data work; Qubolts solves the small decision problem on the driver.

```
memberships (user_id, segment_id)  ─┐
                                    ├─ build_problem()  →  sizes + pairwise overlaps (Spark)
segment_meta (segment_id, vpu, cost)┘                         │ collect: n rows + ≤ n²/2 pairs
                                                              ▼
                                         solve(SegmentSelection(), problem)   (driver)
                                                              │
                                                              ▼
                               segment_selection_decisions (Delta): segment_id, selected, run_id, ...
```

## Quick start in a Fabric notebook

```python
%pip install qubolts

from qubolts.integrations.spark.segment_selection import run

solution = run(
    spark,
    memberships=spark.table("audience_memberships"),
    segment_meta=spark.table("segment_meta"),
    budget=50_000,
    max_segments=12,
    output_table="segment_selection_decisions",
    runlog_path="/lakehouse/default/Files/qubolts/runs.jsonl",
)
print(sorted(solution.decision), solution.objective, solution.lift_vs_baseline)
```

## Git integration

Point the Fabric workspace's Git integration at this `fabric/` folder (or a separate
repo that depends on the `qubolts` package) so Fabric's serialized notebook and
pipeline files stay separate from the Python package.

## Scaling

The exact overlap self-join is fine for tens of millions of membership rows. Beyond
that, compute a HyperLogLog sketch per segment and estimate pairwise overlap as
`|A| + |B| - |A ∪ B|`. The resolver only needs the resulting numbers.
