"""Runs only where PySpark and Java are available (CI installs them; skipped otherwise)."""

import pytest

pyspark = pytest.importorskip("pyspark")

from qubolts.integrations.spark.segment_selection import build_problem  # noqa: E402
from qubolts.mktg.segment_selection import problem_from_memberships  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    try:
        s = SparkSession.builder.master("local[1]").appName("qubolts-test").getOrCreate()
    except Exception as exc:  # no JVM
        pytest.skip(f"Spark unavailable: {exc}")
    yield s
    s.stop()


def test_spark_build_matches_reference(spark):
    rows = [
        ("u1", "a"),
        ("u2", "a"),
        ("u2", "b"),
        ("u3", "b"),
        ("u3", "c"),
        ("u2", "c"),
        ("u2", "c"),
    ]
    meta = {"a": (0.5, 3.0), "b": (1.0, 2.0), "c": (2.0, 4.0)}
    mem = spark.createDataFrame(rows, "user_id STRING, segment_id STRING")
    seg = spark.createDataFrame(
        [(k, v, c) for k, (v, c) in meta.items()],
        "segment_id STRING, value_per_user DOUBLE, cost DOUBLE",
    )
    got = build_problem(mem, seg, budget=5.0)
    ref = problem_from_memberships(
        rows, {k: {"value_per_user": v, "cost": c} for k, (v, c) in meta.items()}, budget=5.0
    )
    assert sorted(got.segments, key=lambda s: s.id) == sorted(ref.segments, key=lambda s: s.id)
    assert got.overlap == ref.overlap
