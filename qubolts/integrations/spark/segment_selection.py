"""Segment selection on Spark: build inputs, solve, write the decision back.

Runs on Microsoft Fabric (notebook or Spark job definition), Databricks, or any
Spark 3.4+ runtime. The heavy lifting (sizes, pairwise overlaps) happens in
Spark; only the small coefficient table (n segments, at most n^2/2 pairs) is
collected to the driver for the solve.

Inputs (Delta tables or any DataFrame)
    memberships:   user_id STRING, segment_id STRING
    segment_meta:  segment_id STRING, value_per_user DOUBLE, cost DOUBLE

Outputs
    a problem dict (also written as JSON if ``problem_path`` is given), and a
    Delta table of the decision: segment_id, selected, run_id, objective, backend

In a Fabric notebook::

    %pip install qubolts
    from qubolts.integrations.spark.segment_selection import run
    sol = run(spark, spark.table("memberships"), spark.table("segment_meta"),
              budget=50_000, output_table="segment_selection_decisions")

Scaling note
    The self-join on user_id is O(sum over users of segments_per_user^2). For
    very large audiences, replace exact overlaps with sketch estimates (e.g.
    HyperLogLog union counts or MinHash) computed per segment, then derive
    pairwise overlap as |A| + |B| - |A u B|.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

from qubolts import RunLog, solve
from qubolts.mktg.segment_selection import SegmentSelection, SegmentSelectionProblem

if TYPE_CHECKING:  # pragma: no cover
    from pyspark.sql import DataFrame, SparkSession


def build_problem(
    memberships: DataFrame,
    segment_meta: DataFrame,
    budget: float,
    max_segments: int | None = None,
    min_overlap: int = 1,
) -> SegmentSelectionProblem:
    """Compute segment sizes and pairwise overlaps in Spark; return the problem."""
    from pyspark.sql import functions as F

    m = memberships.select("user_id", "segment_id").dropDuplicates()
    m = m.join(segment_meta.select("segment_id"), "segment_id", "inner")

    sizes = m.groupBy("segment_id").agg(F.count("*").alias("size"))
    seg_rows = (
        sizes.join(segment_meta, "segment_id", "inner")
        .select("segment_id", "size", "value_per_user", "cost")
        .collect()
    )

    a, b = m.alias("a"), m.alias("b")
    pairs = (
        a.join(
            b,
            (F.col("a.user_id") == F.col("b.user_id"))
            & (F.col("a.segment_id") < F.col("b.segment_id")),
        )
        .groupBy(F.col("a.segment_id").alias("a"), F.col("b.segment_id").alias("b"))
        .agg(F.count("*").alias("users"))
        .where(F.col("users") >= min_overlap)
        .collect()
    )

    return SegmentSelectionProblem.from_dict(
        {
            "segments": [
                {
                    "id": r["segment_id"],
                    "size": int(r["size"]),
                    "value_per_user": float(r["value_per_user"]),
                    "cost": float(r["cost"]),
                }
                for r in seg_rows
            ],
            "budget": budget,
            "max_segments": max_segments,
            "overlap": [{"a": r["a"], "b": r["b"], "users": int(r["users"])} for r in pairs],
        }
    )


def write_decision(
    spark: SparkSession,
    problem: SegmentSelectionProblem,
    solution: Any,
    output_table: str,
    run_id: str,
    mode: str = "append",
) -> DataFrame:
    """Write one row per candidate segment with a ``selected`` flag."""
    rows = [
        (s.id, s.id in solution.decision, run_id, float(solution.objective), solution.backend)
        for s in problem.segments
    ]
    df = spark.createDataFrame(
        rows, "segment_id STRING, selected BOOLEAN, run_id STRING, objective DOUBLE, backend STRING"
    )
    df.write.mode(mode).format("delta").saveAsTable(output_table)
    return df


def run(
    spark: SparkSession,
    memberships: DataFrame,
    segment_meta: DataFrame,
    budget: float,
    *,
    max_segments: int | None = None,
    backend: str = "auto",
    output_table: str | None = None,
    problem_path: str | None = None,
    runlog_path: str | None = None,
    **solve_kwargs: Any,
):
    """End to end: build inputs in Spark, solve on the driver, write the decision."""
    run_id = str(uuid.uuid4())
    problem = build_problem(memberships, segment_meta, budget, max_segments)
    if problem_path:
        with open(problem_path, "w", encoding="utf-8") as fh:
            json.dump(problem.to_dict(), fh)
    log = RunLog(runlog_path) if runlog_path else None
    solution = solve(
        SegmentSelection(),
        problem,
        backend=backend,
        log=log,
        tags={"run_id": run_id},
        **solve_kwargs,
    )
    if output_table:
        write_decision(spark, problem, solution, output_table, run_id)
    return solution
