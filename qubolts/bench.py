"""Benchmark harness: compare backends against classical baselines on the same problems.

    qubolts-bench segment-selection --sizes 20 40 80 --seeds 5 --backends sa tabu descent

Every run is appended to a JSONL log, so results accumulate across machines and
backends (including paid QPU runs) into one comparable table.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import defaultdict

from qubolts import RunLog, backends, solve
from qubolts.mktg.segment_selection import SegmentSelection, synthetic_problem


def bench_segment_selection(
    sizes: list[int],
    seeds: int,
    backend_names: list[str],
    log: RunLog | None = None,
    max_segments: int | None = None,
) -> list[dict]:
    """Run every backend on every synthetic problem. Returns one row per run.

    Reference points per problem:
      * ``naive``   - rank by value per dollar, ignore overlap (common practice)
      * ``greedy``  - overlap-aware greedy (the fallback baseline)
      * ``greedy+ls`` - greedy followed by the same local search the backends get;
        the honest classical bar. A backend is only interesting if it beats this.
    """
    resolver = SegmentSelection()
    rows: list[dict] = []
    for n in sizes:
        for seed in range(seeds):
            p = synthetic_problem(n=n, seed=seed, max_segments=max_segments)
            greedy = resolver.baseline(p)
            t = time.perf_counter()
            gls = resolver.local_search(p, set(greedy))
            t_gls = time.perf_counter() - t
            refs = {
                "naive": (resolver.objective(p, resolver.naive_baseline(p)), 0.0),
                "greedy": (resolver.objective(p, greedy), 0.0),
                "greedy+ls": (resolver.objective(p, gls), t_gls),
            }
            for name, (obj, rt) in refs.items():
                rows.append(
                    {"n": n, "seed": seed, "method": name, "objective": obj, "runtime_s": rt}
                )
            for b in backend_names:
                if b == "exact" and n > 18:
                    continue
                sol = solve(
                    resolver,
                    p,
                    backend=b,
                    fallback=False,
                    log=log,
                    tags={"bench": "segment_selection", "n": n, "seed": seed},
                )
                rows.append(
                    {
                        "n": n,
                        "seed": seed,
                        "method": b,
                        "objective": sol.objective,
                        "runtime_s": sol.runtime_s,
                        "feasible": sol.feasible,
                    }
                )
    return rows


def summarize(rows: list[dict]) -> str:
    """Markdown table: mean lift of each method vs naive and vs greedy+ls, per size."""
    by = defaultdict(dict)
    for r in rows:
        by[(r["n"], r["seed"])][r["method"]] = r
    methods = [m for m in dict.fromkeys(r["method"] for r in rows)]
    sizes = sorted({r["n"] for r in rows})
    lines = [
        "| n | method | vs naive | vs greedy+ls | mean runtime (s) |",
        "|---|---|---|---|---|",
    ]
    for n in sizes:
        for m in methods:
            lv_naive, lv_gls, rts = [], [], []
            for (nn, _), d in by.items():
                if nn != n or m not in d:
                    continue
                o = d[m]["objective"]
                lv_naive.append((o - d["naive"]["objective"]) / d["naive"]["objective"])
                lv_gls.append((o - d["greedy+ls"]["objective"]) / d["greedy+ls"]["objective"])
                rts.append(d[m]["runtime_s"])
            if not lv_naive:
                continue
            lines.append(
                f"| {n} | {m} | {statistics.mean(lv_naive):+.1%} | "
                f"{statistics.mean(lv_gls):+.1%} | {statistics.mean(rts):.2f} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="qubolts-bench", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="problem", required=True)
    ss = sub.add_parser("segment-selection", help="audience segment selection")
    ss.add_argument("--sizes", type=int, nargs="+", default=[16, 40, 80])
    ss.add_argument("--seeds", type=int, default=3)
    ss.add_argument(
        "--backends",
        nargs="+",
        default=["exact", "sa", "tabu", "descent"],
        help=f"registered: {', '.join(sorted(backends.registered()))}",
    )
    ss.add_argument("--max-segments", type=int, default=None, help="cap on segments activated")
    ss.add_argument("--log", default="benchmarks/results/runs.jsonl")
    args = ap.parse_args(argv)

    rows = bench_segment_selection(
        args.sizes, args.seeds, args.backends, RunLog(args.log), args.max_segments
    )
    print(summarize(rows))
    print(f"\nRun log: {args.log}")


if __name__ == "__main__":
    main()
