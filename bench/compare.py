#!/usr/bin/env python3
"""Compare two aggregated bench JSON files; flag regressions.

Regression policy (PLAN.md §벤치마크):
  - Latency p50/p90/p99: tolerate ±15% drift.
  - Peak RSS: monotonic non-increasing across milestones (5% noise allowed).
  - Major page faults: explicit per-milestone direction (M3→M4 increase OK,
    M4→M5 per-item decrease required). Currently report-only.
"""

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List


LATENCY_TOL = 0.15   # ±15%
RSS_TOL     = 0.05   # 5% noise allowance for peak RSS


def fmt_pct(p: float) -> str:
    return f"{p * 100:+.1f}%"


def metric_delta(base: float, cur: float) -> float:
    if base <= 0:
        return 0.0
    return (cur - base) / base


def compare_scenario(name: str, base: Dict[str, Any], cur: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    bm = base["metrics"]
    cm = cur["metrics"]

    for key in ("latency_p50_ms", "latency_p90_ms", "latency_p99_ms"):
        b, c = bm.get(key, 0.0), cm.get(key, 0.0)
        if b <= 0.0:
            continue
        d = metric_delta(b, c)
        if abs(d) > LATENCY_TOL:
            issues.append(
                f"{name} {key}: {b:.3f} -> {c:.3f} ({fmt_pct(d)}, "
                f"> ±{fmt_pct(LATENCY_TOL)})"
            )

    b, c = bm.get("rss_peak_mb", 0.0), cm.get("rss_peak_mb", 0.0)
    if b > 0.0:
        d = metric_delta(b, c)
        if d > RSS_TOL:
            issues.append(
                f"{name} rss_peak_mb: {b:.1f} -> {c:.1f} ({fmt_pct(d)}, "
                f"peak RSS must be non-increasing within ±{fmt_pct(RSS_TOL)})"
            )

    return issues


METRICS_FOR_TABLE = (
    "rss_peak_mb",
    "rss_avg_mb",
    "latency_p50_ms",
    "latency_p90_ms",
    "latency_p99_ms",
    "throughput_items_per_sec",
    "page_faults_major",
    "page_faults_minor",
)


def print_table(base: Dict[str, Any], cur: Dict[str, Any]) -> None:
    header = f"{'scenario':<28} {'metric':<26} {'base':>14} {'cur':>14} {'delta':>10}"
    print(header)
    print("-" * len(header))
    for name in cur.get("scenarios", {}):
        if name not in base.get("scenarios", {}):
            continue
        bm = base["scenarios"][name]["metrics"]
        cm = cur["scenarios"][name]["metrics"]
        for key in METRICS_FOR_TABLE:
            b, c = bm.get(key, 0), cm.get(key, 0)
            d = metric_delta(b, c) if isinstance(b, (int, float)) and b > 0 else 0.0
            delta_str = fmt_pct(d) if isinstance(b, (int, float)) and b > 0 else "—"
            print(f"{name:<28} {key:<26} {b:>14.4g} {c:>14.4g} {delta_str:>10}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=pathlib.Path)
    parser.add_argument("cur",  type=pathlib.Path)
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero on any regression")
    args = parser.parse_args()

    base = json.loads(args.base.read_text())
    cur  = json.loads(args.cur.read_text())

    print(f"base: {base.get('milestone','?')} ({base.get('git_sha','?')}) {args.base}")
    print(f"cur : {cur.get('milestone','?')} ({cur.get('git_sha','?')}) {args.cur}")
    print()
    print_table(base, cur)
    print()

    issues: List[str] = []
    for name in cur.get("scenarios", {}):
        if name in base.get("scenarios", {}):
            issues += compare_scenario(name, base["scenarios"][name],
                                              cur["scenarios"][name])

    if issues:
        print("REGRESSIONS:")
        for i in issues:
            print(f"  - {i}")
        return 1 if args.strict else 0

    print("no regressions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
