# Fabric notebook source: paste cells into a Fabric notebook, or import this file.
# Runs end to end on synthetic data, so you can try Qubolts before wiring real tables.

# %% [markdown]
# # Qubolts: segment selection on Fabric
# 1. Build synthetic `memberships` and `segment_meta` tables (swap in your own).
# 2. Spark computes sizes and pairwise overlaps; the driver solves the QUBO.
# 3. The decision is written to a Delta table and every run to a JSONL log.

# %%
# %pip install qubolts

# %%
import random

from pyspark.sql import SparkSession

from qubolts.integrations.spark.segment_selection import run

spark = SparkSession.builder.getOrCreate()

# %%
# Synthetic data: 30 segments in 8 "families" of heavily overlapping vendor audiences.
random.seed(7)
n_users, n_segments, n_families = 50_000, 30, 8
family = {f"seg_{i:02d}": i % n_families for i in range(n_segments)}
rows = []
for u in range(n_users):
    fam = random.randrange(n_families)
    for seg, f in family.items():
        p = 0.35 if f == fam else 0.01  # users mostly belong to one family's segments
        if random.random() < p:
            rows.append((f"u{u}", seg))
memberships = spark.createDataFrame(rows, "user_id STRING, segment_id STRING")
segment_meta = spark.createDataFrame(
    [(s, random.uniform(0.002, 0.02), random.uniform(20, 120)) for s in family],
    "segment_id STRING, value_per_user DOUBLE, cost DOUBLE",
)

# %%
solution = run(
    spark,
    memberships,
    segment_meta,
    budget=600.0,
    max_segments=10,
    # output_table="segment_selection_decisions",                    # Delta table in your lakehouse
    # runlog_path="/lakehouse/default/Files/qubolts/runs.jsonl",     # benchmark log in OneLake
)
print("selected:", sorted(solution.decision))
print(f"objective {solution.objective:,.0f}  lift vs greedy {solution.lift_vs_baseline:+.1%}")
print("backend:", solution.backend, "| fell back:", solution.fell_back)
