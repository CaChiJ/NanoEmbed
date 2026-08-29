#!/usr/bin/env python3
"""Compare two aggregated bench JSON files; flag regressions.

Regression policy (PLAN.md §벤치마크):
  - Latency p50/p90/p99: tolerate ±15% drift. Checked both ways (a large
    unexplained speedup is also worth a look) but only slowdowns are
    regressions; speedups are reported under IMPROVEMENTS.
  - Peak RSS: monotonic non-increasing across milestones (5% noise allowed).
    Gated on rss_peak_lifetime_mb — the whole-worker number the M4 budget is
    stated against, and exact because the kernel tracks VmHWM for us.
  - PSS / USS: reported only. They have no kernel high-water mark, so their
    peaks come from sampling and are too noisy for a 5% gate. They earn their
    keep at M4, when mmap'ing the GGUF makes the three diverge.
  - Major page faults: explicit per-milestone direction (M3→M4 increase OK,
    M4→M5 per-item decrease required). Currently report-only.

Environment: bench numbers are machine-specific, so a baseline recorded
elsewhere is not comparable. Runs carry an `environment` fingerprint and a
mismatch is surfaced loudly (fatal under --strict) rather than silently
producing a diff that really measures two different CPUs.
"""

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Tuple


LATENCY_TOL = 0.15   # ±15%
RSS_TOL     = 0.05   # 5% noise allowance for peak RSS

# Fields that must agree for two runs to be comparable at all.
ENV_KEYS = ("kernel", "cpu_model", "nproc", "page_size_bytes")
SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3)

# Schema v1 was emitted without a schema_version field. Keep aliases here so
# committed M3/M3.5 baselines remain useful after names become explicit in v2.
METRIC_ALIASES = {
    "single_request_items_per_sec": ("throughput_items_per_sec",),
}


def fmt_pct(p: float) -> str:
    return f"{p * 100:+.1f}%"


def metric_delta(base: float, cur: float) -> float:
    if base <= 0:
        return 0.0
    return (cur - base) / base


def schema_version(run: Dict[str, Any]) -> int:
    """Treat the pre-versioned M3/M3.5 contract as schema v1."""
    version = run.get("schema_version", 1)
    return version if isinstance(version, int) and not isinstance(version, bool) else -1


def metric_value(metrics: Dict[str, Any], key: str) -> Any:
    """Read a v2 metric, falling back to its v1 spelling when necessary."""
    value = metrics.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    for alias in METRIC_ALIASES.get(key, ()):
        value = metrics.get(alias)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def compare_scenario(name: str, base: Dict[str, Any], cur: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return (regressions, improvements) for one scenario.

    Latency drift is checked in both directions on purpose — a large
    unexplained speedup usually means the workload or the measurement changed,
    not that the code got 6x faster. But an improvement is not a regression, so
    the two are reported apart and only regressions fail --strict.
    """
    regressions:  List[str] = []
    improvements: List[str] = []
    bm = base["metrics"]
    cm = cur["metrics"]

    for key in ("latency_p50_ms", "latency_p90_ms", "latency_p99_ms"):
        b, c = metric_value(bm, key), metric_value(cm, key)
        if b is None or c is None or b <= 0.0:
            continue
        d = metric_delta(b, c)
        if abs(d) > LATENCY_TOL:
            msg = (f"{name} {key}: {b:.3f} -> {c:.3f} ({fmt_pct(d)}, "
                   f"> ±{fmt_pct(LATENCY_TOL)})")
            (regressions if d > 0 else improvements).append(msg)

    b = metric_value(bm, "rss_peak_lifetime_mb")
    c = metric_value(cm, "rss_peak_lifetime_mb")
    if b is not None and c is not None and b > 0.0:
        d = metric_delta(b, c)
        if d > RSS_TOL:
            regressions.append(
                f"{name} rss_peak_lifetime_mb: {b:.1f} -> {c:.1f} ({fmt_pct(d)}, "
                f"peak RSS must be non-increasing within ±{fmt_pct(RSS_TOL)})"
            )
        elif -d > RSS_TOL:
            improvements.append(
                f"{name} rss_peak_lifetime_mb: {b:.1f} -> {c:.1f} ({fmt_pct(d)})"
            )

    return regressions, improvements


METRICS_FOR_TABLE = (
    "rss_peak_lifetime_mb",
    "rss_peak_window_mb",
    "rss_baseline_mb",
    "rss_avg_mb",
    "pss_avg_mb",
    "pss_peak_sampled_mb",
    "uss_avg_mb",
    "uss_peak_sampled_mb",
    "latency_p50_ms",
    "latency_p90_ms",
    "latency_p99_ms",
    "single_request_items_per_sec",
    "page_faults_major",
    "page_faults_minor",
)


def env_of(run: Dict[str, Any]) -> Dict[str, Any]:
    """Environment fingerprint of a run, taken from its first scenario.

    runner.py executes every scenario on one machine in one pass, so the
    fingerprint is per-run in practice even though the bench tool stamps it
    per-scenario.
    """
    for sc in run.get("scenarios", {}).values():
        env = sc.get("environment")
        if env:
            return env
    return {}


def compare_env(base: Dict[str, Any], cur: Dict[str, Any]) -> List[str]:
    be, ce = env_of(base), env_of(cur)
    if not be or not ce:
        return ["environment fingerprint missing (run predates the fingerprint "
                "or was produced by the pre-fork harness) — not comparable"]
    return [
        f"environment {k}: {be.get(k)!r} -> {ce.get(k)!r}"
        for k in ENV_KEYS
        if be.get(k) != ce.get(k)
    ]


def _nested(data: Dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def fingerprint_differences(
    base: Dict[str, Any], cur: Dict[str, Any]
) -> List[str]:
    """Explain identity differences without changing regression gate policy."""
    base_fp = base.get("fingerprints")
    cur_fp = cur.get("fingerprints")
    if not isinstance(base_fp, dict) and not isinstance(cur_fp, dict):
        return []
    if not isinstance(base_fp, dict) or not isinstance(cur_fp, dict):
        missing = "base" if not isinstance(base_fp, dict) else "current"
        return [
            f"{missing} extended fingerprint unavailable (legacy or pre-A5 result)"
        ]

    differences: List[str] = []
    run_fields = (
        ("code.nanoembed.sha", "NanoEmbed git SHA"),
        ("code.nanoembed.dirty", "NanoEmbed dirty status"),
        ("code.ggml.sha", "ggml git SHA"),
        ("benchmark_binary.sha256", "benchmark binary SHA-256"),
        ("build.build_type", "CMake build type"),
        ("build.compiler.id", "C++ compiler ID"),
        ("build.compiler.version", "C++ compiler version"),
        ("build.cmake_options", "relevant CMake options"),
        ("scenario_config.sha256", "scenario config SHA-256"),
        ("corpus_manifest.sha256", "corpus manifest SHA-256"),
        ("environment.cpu_frequency_policy.governors", "CPU governors"),
        ("environment.cpu_frequency_policy.drivers", "CPU frequency drivers"),
        ("environment.cpu_frequency_policy.policy_count", "CPU policy count"),
        ("environment.numa.online_nodes", "NUMA online nodes"),
        ("environment.numa.node_count", "NUMA node count"),
        ("environment.total_ram_bytes", "total RAM"),
        (
            "environment.working_directory_filesystem.fs_type",
            "working filesystem type",
        ),
        (
            "environment.working_directory_filesystem.source",
            "working filesystem source",
        ),
        (
            "environment.working_directory_filesystem.uuid",
            "working filesystem UUID",
        ),
    )
    for path, label in run_fields:
        before, after = _nested(base_fp, path), _nested(cur_fp, path)
        if before != after:
            differences.append(f"run {label}: {before!r} -> {after!r}")

    base_scenarios = base.get("scenarios", {})
    cur_scenarios = cur.get("scenarios", {})
    scenario_fields = (
        ("fingerprints.model.sha256", "model SHA-256"),
        ("fingerprints.model.size_bytes", "model size"),
        ("corpus_selection.group", "corpus group"),
        ("corpus_selection.selection_sha256", "selected-input SHA-256"),
        ("settings.requested", "requested settings"),
        ("settings.resolved.pooling", "resolved pooling"),
        ("settings.resolved.normalize", "resolved normalization"),
        ("settings.resolved.max_seq_len", "resolved max sequence length"),
        ("settings.resolved.cache_regime", "resolved cache regime"),
        ("requested_execution_mode", "requested execution mode"),
        ("resolved_execution_mode", "resolved execution mode"),
        ("execution_mode_resolution.contract_version",
         "execution-mode contract version"),
        ("execution_mode_resolution.strict_no_fallback",
         "strict execution-mode resolution"),
    )
    if isinstance(base_scenarios, dict) and isinstance(cur_scenarios, dict):
        for name in sorted(set(base_scenarios) & set(cur_scenarios)):
            base_scenario = base_scenarios[name]
            cur_scenario = cur_scenarios[name]
            if not isinstance(base_scenario, dict) or not isinstance(cur_scenario, dict):
                continue
            # A pre-A5 v2 scenario has no model fingerprint. One concise line
            # is more actionable than a mismatch for every absent field.
            base_has_fp = isinstance(base_scenario.get("fingerprints"), dict)
            cur_has_fp = isinstance(cur_scenario.get("fingerprints"), dict)
            if base_has_fp != cur_has_fp:
                missing = "base" if not base_has_fp else "current"
                differences.append(
                    f"scenario {name} {missing} artifact fingerprint unavailable"
                )
            for path, label in scenario_fields:
                if path.startswith("fingerprints.") and not (
                    base_has_fp and cur_has_fp
                ):
                    continue
                before = _nested(base_scenario, path)
                after = _nested(cur_scenario, path)
                if before != after:
                    differences.append(
                        f"scenario {name} {label}: {before!r} -> {after!r}"
                    )
    return differences


def print_table(base: Dict[str, Any], cur: Dict[str, Any]) -> None:
    header = f"{'scenario':<28} {'metric':<36} {'base':>14} {'cur':>14} {'delta':>10}"
    print(header)
    print("-" * len(header))
    for name in cur.get("scenarios", {}):
        if name not in base.get("scenarios", {}):
            continue
        bm = base["scenarios"][name]["metrics"]
        cm = cur["scenarios"][name]["metrics"]
        for key in METRICS_FOR_TABLE:
            b, c = metric_value(bm, key), metric_value(cm, key)
            comparable = b is not None and c is not None and b > 0
            delta_str = fmt_pct(metric_delta(b, c)) if comparable else "—"
            base_str = f"{b:.4g}" if b is not None else "—"
            cur_str = f"{c:.4g}" if c is not None else "—"
            print(f"{name:<28} {key:<36} {base_str:>14} {cur_str:>14} {delta_str:>10}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=pathlib.Path)
    parser.add_argument("cur",  type=pathlib.Path)
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero on any regression")
    args = parser.parse_args()

    base = json.loads(args.base.read_text())
    cur  = json.loads(args.cur.read_text())

    base_schema = schema_version(base)
    cur_schema = schema_version(cur)
    unsupported = [
        ("base", base_schema), ("current", cur_schema)
    ]
    unsupported = [item for item in unsupported
                   if item[1] not in SUPPORTED_SCHEMA_VERSIONS]
    if unsupported:
        for label, version in unsupported:
            print(f"error: {label} result has unsupported schema_version "
                  f"{version!r}; supported versions are "
                  f"{SUPPORTED_SCHEMA_VERSIONS}", file=sys.stderr)
        return 2

    print(f"base: {base.get('milestone','?')} ({base.get('git_sha','?')}) {args.base}")
    print(f"cur : {cur.get('milestone','?')} ({cur.get('git_sha','?')}) {args.cur}")
    if base_schema == 1 or cur_schema == 1:
        print("schema: using the v1 compatibility adapter for unversioned "
              "M3/M3.5 result data")
    if base_schema != cur_schema:
        print(f"schema: comparing v{base_schema} to v{cur_schema}; compatible "
              "metric aliases are applied")

    env_issues = compare_env(base, cur)
    fingerprint_issues = fingerprint_differences(base, cur)
    if env_issues:
        # Not a code regression, so it is reported separately: comparing across
        # machines measures the hardware, not the change under test.
        print()
        print("ENVIRONMENT MISMATCH — these runs are not comparable:")
        for i in env_issues:
            print(f"  - {i}")
        if not args.strict:
            print("  (continuing anyway; deltas below reflect the machines as much "
                  "as the code)")

    if fingerprint_issues:
        print()
        print("FINGERPRINT DIFFERENCES — review before interpreting deltas:")
        for issue in fingerprint_issues:
            print(f"  - {issue}")
        print("  (diagnostic only; existing regression gates are unchanged)")

    if env_issues and args.strict:
        return 2

    print()
    print_table(base, cur)
    print()

    regressions:  List[str] = []
    improvements: List[str] = []
    for name in cur.get("scenarios", {}):
        if name in base.get("scenarios", {}):
            r, i = compare_scenario(name, base["scenarios"][name],
                                          cur["scenarios"][name])
            regressions  += r
            improvements += i

    # A vanished scenario is a regression too; otherwise an incomplete run
    # could compare cleanly simply because it produced no result.
    for name in base.get("scenarios", {}):
        if name not in cur.get("scenarios", {}):
            regressions.append(
                f"{name}: present in baseline, missing from current run")

    if improvements:
        print("IMPROVEMENTS (beyond tolerance — confirm they are real, then "
              "commit a new baseline):")
        for i in improvements:
            print(f"  - {i}")
        print()

    if regressions:
        print("REGRESSIONS:")
        for i in regressions:
            print(f"  - {i}")
        return 1 if args.strict else 0

    print("no regressions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
