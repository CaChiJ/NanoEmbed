#!/usr/bin/env python3
"""Run all bench scenarios for a given milestone, write an aggregated JSON.

Each scenario shells out to nanoembed-bench (a single self-contained tool
that prints one scenario's JSON). This script collects them into a single
file under bench/results/.
"""

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import pathlib
import subprocess
import sys
import tempfile

from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__:
    from . import corpus_selection, fingerprint, model_source
else:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import corpus_selection  # type: ignore  # noqa: E402
    import fingerprint  # type: ignore  # noqa: E402
    import model_source  # type: ignore  # noqa: E402


RESULT_SCHEMA_VERSION = 3
AB_CONTROLLED_FIELDS = (
    "model", "corpus_group", "pooling", "normalize", "threads",
    "cache_state", "warmup", "iter", "max_seq_len", "memory_profile",
    "memory_profile_interval_ms", "partition", "batch_size", "max_batch",
    "batch_control", "samples",
)


@dataclass(frozen=True)
class BenchRun:
    scenario: Dict[str, Any]
    result_key: str
    selection: corpus_selection.CorpusSelection


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def git_sha() -> str:
    try:
        return (subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=repo_root())
                .decode().strip())
    except Exception:
        return "unknown"


def build_cmd(
    bench: pathlib.Path,
    sc: Dict[str, Any],
    root: pathlib.Path,
    inputs_path: pathlib.Path,
    scenario_name: str,
    memory_profile: bool = False,
    memory_profile_interval_ms: Optional[int] = None,
    cache_state: Optional[str] = None,
    strict_cold: bool = False,
    resolved_model_path: Optional[pathlib.Path] = None,
    raw_samples_out: Optional[pathlib.Path] = None,
) -> List[str]:
    resolved_cache_state = cache_state or sc.get("cache_state", "warm")
    if resolved_cache_state not in ("cold", "warm"):
        raise SystemExit(
            f"scenario {sc['name']}: cache_state must be cold or warm"
        )
    if strict_cold and resolved_cache_state != "cold":
        raise SystemExit(
            f"scenario {sc['name']}: strict_cold requires cache_state cold"
        )
    for field, default in (("batch_size", 1), ("max_batch", 64)):
        value = sc.get(field, default)
        if (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise SystemExit(
                f"scenario {sc['name']}: {field} must be a positive integer"
            )
    if not isinstance(sc.get("batch_control", False), bool):
        raise SystemExit(
            f"scenario {sc['name']}: batch_control must be true or false"
        )

    cmd: List[str] = [
        str(bench),
        "--model",     str(resolved_model_path or model_source.resolve(
            sc["model"], root
        )),
        "--inputs",    str(inputs_path),
        "--scenario",  scenario_name,
        # A native cold invocation represents exactly one first request. The
        # Python runner repeats that invocation once per selected corpus item.
        "--warmup",    str(0 if resolved_cache_state == "cold"
                            else sc.get("warmup", 5)),
        "--iter",      str(1 if resolved_cache_state == "cold"
                            else sc.get("iter", 50)),
        "--threads",   str(sc.get("threads", 0)),
        "--batch-size", str(sc.get("batch_size", 1)),
        "--max-batch", str(sc.get("max_batch", 64)),
        "--cache-state", resolved_cache_state,
    ]
    if strict_cold:
        cmd.append("--strict-cold")
    if raw_samples_out is not None:
        cmd += ["--raw-samples-out", str(raw_samples_out)]
    # Omitting `pooling` means the model's own, which is what the library does
    # by default. Naming it is for pinning a comparison, not a requirement.
    pooling = sc.get("pooling")
    if pooling is not None:
        if pooling not in ("mean", "cls", "last"):
            raise SystemExit(
                f"scenario {sc['name']}: unknown pooling {pooling!r} "
                "(expected mean, cls or last)")
        cmd.append("--" + pooling)
    if not sc.get("normalize", True):
        cmd.append("--no-normalize")
    streaming = sc.get("streaming", False)
    if not isinstance(streaming, bool):
        raise SystemExit(
            f"scenario {sc['name']}: streaming must be true or false"
        )
    if streaming:
        cmd.append("--streaming")
    if sc.get("batch_control", False):
        cmd.append("--batch-control")

    partition = sc.get("partition")
    if partition is not None:
        if not isinstance(partition, str) or not partition:
            raise SystemExit(
                f"scenario {sc['name']}: partition must be a non-empty string"
            )
        cmd += ["--partition", partition]
    if sc.get("max_seq_len", 0) > 0:
        cmd += ["--max-seq-len", str(sc["max_seq_len"])]
    scenario_profile = sc.get("memory_profile", False)
    if not isinstance(scenario_profile, bool):
        raise SystemExit(
            f"scenario {sc['name']}: memory_profile must be true or false"
        )
    if memory_profile or scenario_profile:
        cmd.append("--memory-profile")

    interval = memory_profile_interval_ms
    if interval is None:
        interval = sc.get("memory_profile_interval_ms")
    if interval is not None:
        if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
            raise SystemExit(
                f"scenario {sc['name']}: memory_profile_interval_ms "
                "must be a positive integer"
            )
        cmd += ["--memory-profile-interval-ms", str(interval)]

    return cmd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _numeric(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    return None


def _aggregate_result_bucket(
    entries: Sequence[Tuple[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    total_items = 0
    total_batches = 0
    total_wall_sec = 0.0
    latency_count = 0
    weighted_latency_sum = 0.0
    latency_mins: List[float] = []
    latency_maxes: List[float] = []
    model_hashes = set()
    cache_regimes = set()
    profile_states = set()
    selection_hashes = set()
    execution_modes = set()

    for _, result in entries:
        metrics = result.get("metrics", {})
        items = result.get("total_items")
        batches = result.get("total_batches")
        wall = _numeric(metrics.get("wall_sec"))
        count = metrics.get("latency_count")
        mean = _numeric(metrics.get("latency_mean_ms"))
        minimum = _numeric(metrics.get("latency_min_ms"))
        maximum = _numeric(metrics.get("latency_max_ms"))
        if isinstance(items, int) and not isinstance(items, bool) and items >= 0:
            total_items += items
        if isinstance(batches, int) and not isinstance(batches, bool) and batches >= 0:
            total_batches += batches
        if wall is not None and wall >= 0.0:
            total_wall_sec += wall
        if (isinstance(count, int) and not isinstance(count, bool) and count > 0
                and mean is not None):
            latency_count += count
            weighted_latency_sum += count * mean
        if minimum is not None:
            latency_mins.append(minimum)
        if maximum is not None:
            latency_maxes.append(maximum)

        model_sha = result.get("fingerprints", {}).get("model", {}).get("sha256")
        if isinstance(model_sha, str):
            model_hashes.add(model_sha)
        cache = result.get("measurement", {}).get("cache_regime")
        if isinstance(cache, str):
            cache_regimes.add(cache)
        profile = result.get("measurement", {}).get("memory_profile_enabled")
        if isinstance(profile, bool):
            profile_states.add(profile)
        selection_sha = result.get("corpus_selection", {}).get(
            "selection_sha256"
        )
        if isinstance(selection_sha, str):
            selection_hashes.add(selection_sha)
        execution_mode = result.get("resolved_execution_mode")
        if isinstance(execution_mode, str):
            execution_modes.add(execution_mode)

    dimensions = {
        "model_sha256s": sorted(model_hashes),
        "cache_regimes": sorted(cache_regimes),
        "memory_profile_states": sorted(profile_states),
        "selection_sha256s": sorted(selection_hashes),
        "resolved_execution_modes": sorted(execution_modes),
    }
    homogeneous = all(len(values) <= 1 for values in dimensions.values())
    return {
        "scenario_keys": [name for name, _ in entries],
        "scenario_count": len(entries),
        "homogeneous_dimensions": homogeneous,
        "dimensions": dimensions,
        "total_items": total_items,
        "total_batches": total_batches,
        "total_timed_wall_sec": total_wall_sec,
        "single_request_items_per_sec": (
            total_items / total_wall_sec if total_items > 0 and total_wall_sec > 0.0
            else None
        ),
        "items_per_sec": (
            total_items / total_wall_sec if total_items > 0 and total_wall_sec > 0.0
            else None
        ),
        "batches_per_sec": (
            total_batches / total_wall_sec
            if total_batches > 0 and total_wall_sec > 0.0 else None
        ),
        "latency_count": latency_count,
        "latency_min_ms": min(latency_mins) if latency_mins else None,
        "latency_max_ms": max(latency_maxes) if latency_maxes else None,
        "latency_weighted_mean_ms": (
            weighted_latency_sum / latency_count if latency_count else None
        ),
        "latency_percentiles": None,
        "confidence_interval": None,
    }


def aggregate_result_summaries(
    scenarios: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build bounded descriptive group/overall summaries.

    Percentiles cannot be reconstructed from scenario summaries and remain
    null. Individual scenario results stay canonical; an aggregate mixing
    models or cache/profile states calls that out through its dimensions.
    """
    grouped: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    entries = list(scenarios.items())
    for name, result in entries:
        group = result.get("corpus_selection", {}).get("group")
        group_name = group if isinstance(group, str) and group else "unknown"
        grouped.setdefault(group_name, []).append((name, result))
    return {
        "aggregation_role": "descriptive; individual scenario results are canonical",
        "throughput_method": "sum(total_items) / sum(canonical timed wall seconds)",
        "latency_method": (
            "count-weighted mean plus extrema; percentiles require raw samples "
            "and are not reconstructed"
        ),
        "independent_runs": 1,
        "confidence_interval": None,
        "by_group": {
            group: _aggregate_result_bucket(grouped[group])
            for group in sorted(grouped)
        },
        "overall": _aggregate_result_bucket(entries),
    }


_MEMORY_SAMPLE_VALUE_FIELDS = ("rss_bytes", "pss_bytes", "uss_bytes")
_MEMORY_BREAKDOWN_FIELDS = (
    "pss_anon_bytes", "pss_file_bytes", "anonymous_bytes",
    "private_clean_bytes", "private_dirty_bytes", "shared_clean_bytes",
    "shared_dirty_bytes",
)
_MEMORY_PERCENTILES = (
    ("p50", 0.50), ("p75", 0.75), ("p90", 0.90),
    ("p95", 0.95), ("p99", 0.99),
)


def _load_native_raw_payload(path: pathlib.Path, label: str) -> Dict[str, Any]:
    """Validate native raw schema 1/2 and its memory-profile extension."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}: cannot read native raw samples: {exc}") from exc
    values = payload.get("latency_ms") if isinstance(payload, dict) else None
    if (not isinstance(payload, dict) or
            payload.get("schema_version") not in (1, 2)):
        raise ValueError(f"{label}: native raw samples have an unsupported schema")
    if payload.get("latency_unit") != "ms":
        raise ValueError(f"{label}: native raw samples have an unexpected unit")
    if not isinstance(values, list):
        raise ValueError(f"{label}: native raw samples omitted latency_ms")
    latency_samples: List[float] = []
    for value in values:
        sample = _numeric(value)
        if sample is None or sample < 0.0:
            raise ValueError(f"{label}: native raw samples contain an invalid value")
        latency_samples.append(sample)
    if payload.get("schema_version") == 2:
        counts = payload.get("batch_item_counts")
        if (not isinstance(counts, list) or len(counts) != len(latency_samples) or
                any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                    for value in counts)):
            raise ValueError(f"{label}: native raw samples contain invalid batch counts")

    memory = payload.get("memory_profile")
    if memory is None:
        # Native schema 1 predates memory timelines. Keep accepting it so old
        # harnesses remain usable, but do not pretend that profiling was off.
        memory_profile: Dict[str, Any] = {
            "schema_version": None,
            "collection_status": "unavailable_legacy",
            "samples": None,
        }
    elif not isinstance(memory, dict) or memory.get("schema_version") != 1:
        raise ValueError(
            f"{label}: native memory profile has an unsupported schema"
        )
    else:
        status = memory.get("collection_status")
        samples = memory.get("samples")
        if status == "disabled":
            if samples is not None or memory.get("sample_count") not in (None, 0):
                raise ValueError(
                    f"{label}: disabled native memory profile contains samples"
                )
            for marker in (
                "go_monotonic_timestamp_ns",
                "done_marker_monotonic_timestamp_ns",
                "done_marker_elapsed_ms_from_go",
            ):
                if memory.get(marker) is not None:
                    raise ValueError(
                        f"{label}: disabled native memory profile contains {marker}"
                    )
        elif status == "collected":
            if not isinstance(samples, list) or not samples:
                raise ValueError(
                    f"{label}: collected native memory profile omitted samples"
                )
            if memory.get("sample_count") != len(samples):
                raise ValueError(
                    f"{label}: native memory profile sample_count mismatch"
                )
            go_ns = _numeric(memory.get("go_monotonic_timestamp_ns"))
            done_ns = _numeric(memory.get("done_marker_monotonic_timestamp_ns"))
            done_elapsed = _numeric(
                memory.get("done_marker_elapsed_ms_from_go")
            )
            if (go_ns is None or done_ns is None or done_elapsed is None or
                    done_ns < go_ns or done_elapsed < 0.0):
                raise ValueError(
                    f"{label}: native memory profile has invalid GO/DONE markers"
                )

            previous_ns: Optional[float] = None
            roles: List[str] = []
            for index, entry in enumerate(samples):
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"{label}: native memory sample {index} is not an object"
                    )
                role = entry.get("sample_role")
                valid = entry.get("valid")
                timestamp = _numeric(entry.get("monotonic_timestamp_ns"))
                elapsed = _numeric(entry.get("elapsed_ms_from_go"))
                duration = _numeric(entry.get("read_duration_ms"))
                if role not in ("baseline", "periodic", "final"):
                    raise ValueError(
                        f"{label}: native memory sample {index} has invalid role"
                    )
                if not isinstance(valid, bool):
                    raise ValueError(
                        f"{label}: native memory sample {index} has invalid validity"
                    )
                if (timestamp is None or elapsed is None or duration is None or
                        duration < 0.0 or
                        (previous_ns is not None and timestamp < previous_ns)):
                    raise ValueError(
                        f"{label}: native memory sample {index} has invalid timing"
                    )
                previous_ns = timestamp
                roles.append(role)
                if role == "baseline" and elapsed >= 0.0:
                    raise ValueError(
                        f"{label}: baseline memory sample is not before GO"
                    )
                if role in ("periodic", "final") and elapsed < 0.0:
                    raise ValueError(
                        f"{label}: {role} memory sample is before GO"
                    )
                for field in _MEMORY_SAMPLE_VALUE_FIELDS:
                    value = entry.get(field)
                    if value is not None and (
                        not isinstance(value, int) or isinstance(value, bool) or
                        value < 0
                    ):
                        raise ValueError(
                            f"{label}: native memory sample {index} has invalid {field}"
                        )
                breakdown = entry.get("breakdown_bytes")
                if not isinstance(breakdown, dict):
                    raise ValueError(
                        f"{label}: native memory sample {index} omitted breakdown"
                    )
                for field in _MEMORY_BREAKDOWN_FIELDS:
                    value = breakdown.get(field)
                    if value is not None and (
                        not isinstance(value, int) or isinstance(value, bool) or
                        value < 0
                    ):
                        raise ValueError(
                            f"{label}: native memory sample {index} has invalid {field}"
                        )
            if roles.count("baseline") != 1 or roles.count("final") != 1:
                raise ValueError(
                    f"{label}: native memory profile requires one baseline and final"
                )
        else:
            raise ValueError(
                f"{label}: native memory profile has invalid collection_status"
            )
        memory_profile = copy.deepcopy(memory)

    return {
        "schema_version": 1,
        "scenario": payload.get("scenario"),
        "latency_unit": "ms",
        "latency_ms": latency_samples,
        "memory_profile": memory_profile,
    }


def _load_native_raw_samples(path: pathlib.Path, label: str) -> List[float]:
    """Backward-compatible latency-only view used by existing callers/tests."""
    return _load_native_raw_payload(path, label)["latency_ms"]


def _validate_raw_memory_summary(
    payload: Dict[str, Any], result: Dict[str, Any], label: str
) -> None:
    """Ensure profile-on raw observations exactly back the native summary counts."""
    profile = payload.get("memory_profile", {})
    measurement = result.get("measurement", {})
    enabled = measurement.get("memory_profile_enabled") is True
    status = profile.get("collection_status")
    if not enabled:
        if status not in ("disabled", "unavailable_legacy"):
            raise ValueError(
                f"{label}: profile-off result contains collected raw memory samples"
            )
        return
    if status != "collected":
        raise ValueError(
            f"{label}: profile-on result omitted the raw memory timeline"
        )
    samples = profile.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"{label}: raw memory timeline omitted samples")
    population = [
        sample for sample in samples
        if isinstance(sample, dict) and
        sample.get("sample_role") in ("periodic", "final")
    ]
    effective = sum(sample.get("valid") is True for sample in population)
    if len(population) != measurement.get("memory_profile_samples_attempted"):
        raise ValueError(
            f"{label}: raw memory attempt count does not match native summary"
        )
    if effective != measurement.get("memory_profile_samples_effective"):
        raise ValueError(
            f"{label}: raw memory effective count does not match native summary"
        )


def _memory_sample_quantiles(
    profiles: Sequence[Dict[str, Any]], field: str
) -> Dict[str, Optional[float]]:
    """Merge periodic+final values across workers, excluding each baseline."""
    values: List[float] = []
    for profile in profiles:
        if profile.get("collection_status") != "collected":
            continue
        samples = profile.get("samples")
        if not isinstance(samples, list):
            continue
        for sample in samples:
            if (not isinstance(sample, dict) or
                    sample.get("sample_role") not in ("periodic", "final") or
                    sample.get("valid") is not True):
                continue
            value = sample.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                values.append(float(value) / (1024.0 * 1024.0))
    if not values:
        return {name: None for name, _ in _MEMORY_PERCENTILES}
    ordered = sorted(values)
    return {
        name: ordered[math.floor(q * (len(ordered) - 1))]
        for name, q in _MEMORY_PERCENTILES
    }


def _memory_sample_mean(
    profiles: Sequence[Dict[str, Any]], field: str
) -> Optional[float]:
    """Count-weighted mean over the same periodic+final population as quantiles."""
    values: List[float] = []
    for profile in profiles:
        if profile.get("collection_status") != "collected":
            continue
        samples = profile.get("samples")
        if not isinstance(samples, list):
            continue
        for sample in samples:
            if (not isinstance(sample, dict) or
                    sample.get("sample_role") not in ("periodic", "final") or
                    sample.get("valid") is not True):
                continue
            value = sample.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                values.append(float(value) / (1024.0 * 1024.0))
    return math.fsum(values) / len(values) if values else None


def _sidecar_reference(sidecar: pathlib.Path, result_path: pathlib.Path) -> str:
    try:
        return str(sidecar.resolve().relative_to(result_path.resolve().parent))
    except (OSError, ValueError):
        return str(sidecar)



def _describe_samples(values: Sequence[float]) -> Dict[str, Any]:
    """Match the native lower-percentile/population-statistics contract."""
    samples = [float(value) for value in values]
    if not samples or not all(math.isfinite(value) for value in samples):
        return {
            "count": 0,
            **{name: None for name in (
                "min_ms", "max_ms", "mean_ms", "p50_ms", "p90_ms",
                "p95_ms", "p99_ms", "stddev_ms", "mad_ms",
            )},
        }
    ordered = sorted(samples)

    def lower(q: float, source: Sequence[float] = ordered) -> float:
        return source[math.floor(q * (len(source) - 1))]

    mean = math.fsum(samples) / len(samples)
    stddev = math.sqrt(
        math.fsum((value - mean) ** 2 for value in samples) / len(samples)
    )
    median = lower(0.50)
    deviations = sorted(abs(value - median) for value in samples)
    return {
        "count": len(samples),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "mean_ms": mean,
        "p50_ms": lower(0.50),
        "p90_ms": lower(0.90),
        "p95_ms": lower(0.95),
        "p99_ms": lower(0.99),
        "stddev_ms": stddev,
        "mad_ms": deviations[math.floor(0.50 * (len(deviations) - 1))],
    }


def _mean_available(results: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    values = [result["metrics"].get(key) for result in results]
    available = [float(value) for value in values
                 if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return math.fsum(available) / len(available) if available else None


def _max_available(results: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
    values = [result["metrics"].get(key) for result in results]
    available = [float(value) for value in values
                 if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return max(available) if available else None


def aggregate_cold_results(
    results: Sequence[Dict[str, Any]],
    selected_ids: Sequence[Sequence[str]],
    memory_profiles: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Aggregate N one-batch/one-worker cold runs without hiding provenance.

    Each worker is a fresh process against a verified-cold page cache that
    performs exactly one timed request. That request carries one sub-batch of
    inputs, so `selected_ids` is one id list per worker. At batch size 1 every
    list holds a single id and this is the historical per-item cold shape.
    """
    if not results or len(results) != len(selected_ids):
        raise ValueError("cold aggregation requires one result per cold worker")

    phase_names = (
        "model_load_ms",
        "context_create_ms",
        "first_request_latency_ms",
        "startup_to_first_result_ms",
    )
    phase_values: Dict[str, List[float]] = {name: [] for name in phase_names}
    for index, result in enumerate(results):
        measurement = result.get("measurement", {})
        metrics = result.get("metrics", {})
        phases = metrics.get("cold_phase_timings", {})
        if measurement.get("cache_regime") not in ("cold", "cold-unverified"):
            raise ValueError(
                f"cold worker {index} did not report a cold cache regime"
            )
        expected_items = len(selected_ids[index])
        if metrics.get("latency_count") != 1:
            raise ValueError(
                f"cold worker {index} must report exactly one timed first request"
            )
        if result.get("total_items") != expected_items:
            raise ValueError(
                f"cold worker {index} ran {result.get('total_items')} items but "
                f"was given {expected_items}"
            )
        for name in phase_names:
            value = phases.get(name, {}).get("mean_ms")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"cold worker {index} omitted {name}")
            phase_values[name].append(float(value))

    worker_item_counts = [len(ids) for ids in selected_ids]
    total_cold_items = sum(worker_item_counts)

    out = copy.deepcopy(results[0])
    out["total_items"] = total_cold_items
    out["total_batches"] = len(results)
    out["warmup"] = 0
    out["iter"] = 1

    measurement = out["measurement"]
    measurement["execution_shape"] = (
        "selected-inputs-split-into-sub-batches-each-using-one-fresh-"
        "native-and-worker-process"
    )
    measurement["cold_worker_invocations"] = len(results)
    measurement["warmup_items_executed"] = 0
    measurement["cold_aggregation"] = {
        "worker_count": len(results),
        "items_per_worker": worker_item_counts,
        "latency_population": "one first inference from each fresh worker",
        "latency_unit": (
            "one sub-batch; item latency divides each worker's batch latency "
            "by that worker's item count"
        ),
        "throughput_denominator": (
            "sum of worker GO-to-first-result durations; excludes cache eviction "
            "and process launch"
        ),
        "rss_peak_fields": "maximum across workers",
        "rss_pss_uss_baseline_final_fields": "mean across available workers",
        "rss_pss_uss_sampled_mean": (
            "count-weighted mean over every valid periodic/final raw sample across "
            "workers; baseline excluded"
        ),
        "rss_pss_uss_sampled_percentiles": (
            "lower percentiles over every valid periodic/final raw sample across "
            "workers; baseline excluded"
        ),
        "resource_counters": "sum across workers from model load through first result",
    }

    cache_controls = [result["measurement"]["cache_control"] for result in results]
    per_worker_cache = []
    for worker_ids, cache in zip(selected_ids, cache_controls):
        per_worker_cache.append({
            "selected_ids": list(worker_ids),
            "eviction_call_succeeded": cache.get("eviction_call_succeeded"),
            "cold_cache_verified": cache.get("cold_cache_verified"),
            "verification_status": cache.get("verification_status"),
            "before_eviction": cache.get("before_eviction"),
            "after_eviction_before_worker": cache.get(
                "after_eviction_before_worker"
            ),
            "after_worker_load_and_first_result": cache.get(
                "after_worker_load_and_first_result"
            ),
        })
    all_eviction_succeeded = all(
        cache.get("eviction_call_succeeded") is True for cache in cache_controls
    )
    all_verified = all(
        cache.get("cold_cache_verified") is True for cache in cache_controls
    )
    measurement["cache_regime_requested"] = "cold"
    measurement["cache_regime"] = "cold" if all_verified else "cold-unverified"
    out.get("settings", {}).get("resolved", {})["cache_regime"] = measurement[
        "cache_regime"
    ]
    measurement["cache_control"] = {
        "cold_cache_requested": True,
        "strict_cold": all(cache.get("strict_cold") is True
                           for cache in cache_controls),
        "platform_supported": all(cache.get("platform_supported") is True
                                  for cache in cache_controls),
        "eviction_call_succeeded": all_eviction_succeeded,
        "cold_cache_verified": all_verified,
        "verified_worker_count": sum(
            cache.get("cold_cache_verified") is True for cache in cache_controls
        ),
        "worker_count": len(results),
        "verification_definition": (
            "every worker's pre-fork mincore observation saw zero resident "
            "model-file pages after fadvise"
        ),
        "verification_status": (
            "all_workers_verified_cold" if all_verified
            else "one_or_more_workers_not_verified_cold"
        ),
        "per_worker": per_worker_cache,
    }

    # Sampling attempts and resource counters are additive across disjoint
    # workers. A profiled run remains diagnostic, exactly as in native output.
    for key in (
        "memory_profile_samples_requested",
        "memory_profile_samples_attempted",
        "memory_profile_samples_effective",
    ):
        measurement[key] = sum(int(result["measurement"].get(key, 0))
                               for result in results)
    attempted = measurement["memory_profile_samples_attempted"]
    measurement["memory_profile_valid_sample_ratio"] = (
        measurement["memory_profile_samples_effective"] / attempted
        if attempted else None
    )
    measurement["memory_profile_final_sample_collected"] = all(
        result["measurement"].get("memory_profile_final_sample_collected") is True
        for result in results
    )
    measurement["memory_profile_final_in_aggregates"] = all(
        result["measurement"].get("memory_profile_final_in_aggregates") is True
        for result in results
    )
    measurement["hwm_reset"] = all(
        result["measurement"].get("hwm_reset") is True for result in results
    )
    measurement["rss_samples"] = sum(
        int(result["measurement"].get("rss_samples", 0)) for result in results
    )
    measurement["rollup_samples"] = sum(
        int(result["measurement"].get("rollup_samples", 0)) for result in results
    )

    metrics = out["metrics"]
    first_request_stats = _describe_samples(
        phase_values["first_request_latency_ms"]
    )
    metrics["latency_count"] = first_request_stats["count"]
    for target, source in (
        ("latency_min_ms", "min_ms"),
        ("latency_max_ms", "max_ms"),
        ("latency_mean_ms", "mean_ms"),
        ("latency_p50_ms", "p50_ms"),
        ("latency_p90_ms", "p90_ms"),
        ("latency_p95_ms", "p95_ms"),
        ("latency_p99_ms", "p99_ms"),
        ("latency_stddev_ms", "stddev_ms"),
        ("latency_mad_ms", "mad_ms"),
    ):
        metrics[target] = first_request_stats[source]
    for percentile in ("p50", "p90", "p95", "p99"):
        metrics[f"batch_latency_{percentile}_ms"] = first_request_stats[
            f"{percentile}_ms"
        ]
    # Each worker's timed request covers that worker's whole sub-batch, so the
    # per-item view divides by the items that worker actually ran.
    item_latency_stats = _describe_samples([
        latency / count
        for latency, count in zip(
            phase_values["first_request_latency_ms"], worker_item_counts
        )
    ])
    for percentile in ("p50", "p90", "p95", "p99"):
        metrics[f"item_latency_{percentile}_ms"] = item_latency_stats[
            f"{percentile}_ms"
        ]

    startup_total_ms = math.fsum(phase_values["startup_to_first_result_ms"])
    metrics["wall_sec"] = startup_total_ms / 1000.0
    metrics["single_request_items_per_sec"] = (
        1000.0 * total_cold_items / startup_total_ms
        if startup_total_ms > 0.0 else None
    )
    metrics["items_per_sec"] = metrics["single_request_items_per_sec"]
    metrics["batches_per_sec"] = (
        1000.0 * len(results) / startup_total_ms
        if startup_total_ms > 0.0 else None
    )
    metrics["cold_phase_timings"] = {
        "collection_status": "collected",
        "startup_boundary": (
            "worker GO through first embed result; excludes cache eviction, "
            "process launch, input parsing and pipe synchronization"
        ),
        "fresh_worker_count": len(results),
        **{name: _describe_samples(values)
           for name, values in phase_values.items()},
    }
    metrics["fixed_item_window_throughput"] = {
        "collection_status": "not_applicable_cold_start",
        "canonical": False,
        "source": "not_applicable_cold_start",
        "window_size_items": None,
        "minimum_windows_for_statistics": None,
        "complete_windows": 0,
        "dropped_tail_items": len(results),
        "count": 0,
        **{name: None for name in (
            "min_items_per_sec", "max_items_per_sec", "mean_items_per_sec",
            "p50_items_per_sec", "p90_items_per_sec", "p95_items_per_sec",
            "p99_items_per_sec", "stddev_items_per_sec", "mad_items_per_sec",
        )},
    }
    metrics["collection_status"]["fixed_item_window_throughput"] = (
        "not_applicable_cold_start"
    )

    for key in ("cpu_user_sec", "cpu_sys_sec", "page_faults_major",
                "page_faults_minor", "io_read_bytes"):
        metrics[key] = sum(result["metrics"].get(key, 0) for result in results)
    for key in ("page_faults_major", "page_faults_minor", "io_read_bytes"):
        metrics[f"{key}_per_item"] = metrics[key] / total_cold_items
        metrics[f"{key}_per_batch"] = metrics[key] / len(results)

    peak_keys = (
        "rss_peak_lifetime_mb", "rss_peak_window_mb", "rss_max_sampled_mb",
        "pss_peak_sampled_mb", "uss_peak_sampled_mb",
    )
    mean_keys = (
        "rss_baseline_mb", "rss_final_mb", "rss_avg_mb",
        "pss_baseline_mb", "pss_final_mb", "pss_avg_mb",
        "uss_baseline_mb", "uss_final_mb", "uss_avg_mb",
    )
    for key in peak_keys:
        metrics[key] = _max_available(results, key)
    for key in mean_keys:
        metrics[key] = _mean_available(results, key)

    profiles = list(memory_profiles or [])
    if profiles and len(profiles) != len(results):
        raise ValueError(
            "cold memory-profile aggregation requires one profile per worker"
        )
    for prefix in ("rss", "pss", "uss"):
        sampled_mean = _memory_sample_mean(profiles, f"{prefix}_bytes")
        if sampled_mean is not None:
            metrics[f"{prefix}_avg_mb"] = sampled_mean
        quantiles = _memory_sample_quantiles(profiles, f"{prefix}_bytes")
        for percentile, _ in _MEMORY_PERCENTILES:
            metrics[f"{prefix}_sampled_{percentile}_mb"] = quantiles[
                percentile
            ]

    for field_name, field in metrics.get("memory_breakdown", {}).items():
        source_fields = [result["metrics"].get("memory_breakdown", {}).get(
            field_name, {}) for result in results]
        for key in ("baseline_mb", "average_mb", "final_mb"):
            values = [source.get(key) for source in source_fields]
            available = [float(value) for value in values
                         if isinstance(value, (int, float)) and
                         not isinstance(value, bool)]
            field[key] = math.fsum(available) / len(available) if available else None
        values = [source.get("peak_sampled_mb") for source in source_fields]
        available = [float(value) for value in values
                     if isinstance(value, (int, float)) and
                     not isinstance(value, bool)]
        field["peak_sampled_mb"] = max(available) if available else None

    metrics["cold_worker_runs"] = [
        {
            "selected_ids": list(worker_ids),
            "item_count": len(worker_ids),
            "cold_cache_verified": result["measurement"]["cache_control"].get(
                "cold_cache_verified"
            ),
            "phases_ms": {
                name: result["metrics"]["cold_phase_timings"][name]["mean_ms"]
                for name in phase_names
            },
            "rss_peak_lifetime_mb": result["metrics"].get(
                "rss_peak_lifetime_mb"
            ),
            "rss_peak_window_mb": result["metrics"].get("rss_peak_window_mb"),
            "page_faults_major": result["metrics"].get("page_faults_major"),
            "page_faults_minor": result["metrics"].get("page_faults_minor"),
            "io_read_bytes": result["metrics"].get("io_read_bytes"),
        }
        for worker_ids, result in zip(selected_ids, results)
    ]
    return out


def validate_execution_mode_claim(
    result: Dict[str, Any], expected_mode: str, label: str
) -> None:
    """Reject a native result that could silently relabel execution mode."""
    if expected_mode not in ("eager", "streaming"):
        raise ValueError(f"{label}: invalid expected execution mode {expected_mode!r}")
    resolution = result.get("execution_mode_resolution")
    settings = result.get("settings")
    settings = settings if isinstance(settings, dict) else {}
    requested_settings = settings.get("requested")
    requested_settings = (
        requested_settings if isinstance(requested_settings, dict) else {}
    )
    resolved_settings = settings.get("resolved")
    resolved_settings = (
        resolved_settings if isinstance(resolved_settings, dict) else {}
    )
    requested = requested_settings.get("execution_mode")
    resolved = resolved_settings.get("execution_mode")
    valid = (
        result.get("requested_execution_mode") == expected_mode
        and result.get("resolved_execution_mode") == expected_mode
        and requested == expected_mode
        and resolved == expected_mode
        and isinstance(resolution, dict)
        and resolution.get("contract_version") == 1
        and resolution.get("requested_execution_mode") == expected_mode
        and resolution.get("resolved_execution_mode") == expected_mode
        and resolution.get("strict_no_fallback") is True
        and resolution.get("context_creation_succeeded_with_exact_request") is True
    )
    if not valid:
        raise ValueError(
            f"{label}: native execution-mode claim does not prove strict "
            f"{expected_mode} resolution"
        )


def validate_streaming_ab_pairs(scenarios: Sequence[Dict[str, Any]]) -> None:
    """Keep every *_streaming scenario controlled except for mode and name."""
    by_name = {
        scenario.get("name"): scenario
        for scenario in scenarios
        if isinstance(scenario.get("name"), str)
    }
    for name, streaming in by_name.items():
        if not name.endswith("_streaming"):
            continue
        eager_name = name.removesuffix("_streaming")
        eager = by_name.get(eager_name)
        if eager is None:
            raise ValueError(
                f"streaming A/B scenario {name!r} has no eager pair {eager_name!r}"
            )
        if streaming.get("streaming") is not True:
            raise ValueError(f"streaming A/B scenario {name!r} must set streaming: true")
        if eager.get("streaming", False) is not False:
            raise ValueError(f"eager A/B scenario {eager_name!r} must set streaming: false")
        mismatches = [
            field for field in AB_CONTROLLED_FIELDS
            if streaming.get(field) != eager.get(field)
        ]
        if mismatches:
            raise ValueError(
                f"streaming A/B pair {eager_name!r}/{name!r} differs in "
                f"controlled fields: {', '.join(mismatches)}"
            )


def prepare_runs(
    scenarios: Sequence[Dict[str, Any]],
    requested_groups: Sequence[Tuple[str, Optional[int]]],
    samples_per_group: Optional[int],
    selection_seed: int,
    manifest: Dict[str, Any],
    root: pathlib.Path,
) -> List[BenchRun]:
    """Validate and select every input before any native process is launched."""
    groups = manifest["groups"]
    selection_cache: Dict[
        Tuple[str, Optional[int], int], corpus_selection.CorpusSelection
    ] = {}
    runs: List[BenchRun] = []
    result_keys = set()

    for sc in scenarios:
        scenario_name = sc.get("name")
        default_group = sc.get("corpus_group")
        if not isinstance(scenario_name, str) or not scenario_name:
            raise corpus_selection.CorpusSelectionError(
                "each selected scenario must have a non-empty name"
            )
        if not isinstance(default_group, str) or not default_group:
            raise corpus_selection.CorpusSelectionError(
                f"scenario {scenario_name!r} must name a corpus_group"
            )

        scenario_samples = sc.get("samples")
        if scenario_samples is not None and (
            not isinstance(scenario_samples, int) or isinstance(scenario_samples, bool)
            or scenario_samples <= 0
        ):
            raise corpus_selection.CorpusSelectionError(
                f"scenario {scenario_name!r} samples must be a positive integer"
            )
        default_count = samples_per_group
        if not requested_groups and default_count is None:
            default_count = scenario_samples
        resolved = corpus_selection.resolve_group_requests(
            requested_groups,
            default_group,
            default_count,
            groups,
        )
        for group_name, count in resolved:
            cache_key = (group_name, count, selection_seed)
            selection = selection_cache.get(cache_key)
            if selection is None:
                selection = corpus_selection.select_group(
                    root,
                    group_name,
                    groups[group_name],
                    count,
                    selection_seed,
                )
                selection_cache[cache_key] = selection

            # Keep historical scenario keys for their declared default corpus,
            # while making alternate workloads distinct and non-overwriting.
            result_key = (
                scenario_name
                if group_name == default_group
                else f"{scenario_name}::{group_name}"
            )
            if result_key in result_keys:
                raise corpus_selection.CorpusSelectionError(
                    f"duplicate benchmark result key {result_key!r}"
                )
            result_keys.add(result_key)
            runs.append(BenchRun(sc, result_key, selection))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone", required=True, help="e.g. M3, M4 ...")
    parser.add_argument("--scenarios", type=pathlib.Path,
                        default=pathlib.Path("bench/scenarios.yaml"))
    parser.add_argument("--corpus-manifest", type=pathlib.Path,
                        default=pathlib.Path("bench/corpus_groups.json"))
    parser.add_argument("--bench", type=pathlib.Path,
                        default=pathlib.Path("build/bin/nanoembed-bench"))
    parser.add_argument("--out", type=pathlib.Path,
                        help="defaults to bench/results/<git_sha>.json")
    parser.add_argument("--filter",
                        help="run only the scenario with this name")
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="NAME[:N]",
        help=("run a corpus group, optionally bounded to N samples; repeat for "
              "multiple groups"),
    )
    parser.add_argument(
        "--samples-per-group",
        type=int,
        help="global sample bound; an explicit NAME:N takes precedence",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=0,
        help="deterministic corpus selection seed (default: 0)",
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help=("enable detailed RSS/PSS/USS sampling; latency from this run is "
              "diagnostic"),
    )
    parser.add_argument(
        "--memory-profile-interval-ms",
        type=int,
        help=("override the detailed sampling interval for all selected "
              "scenarios; does not enable profiling by itself"),
    )
    parser.add_argument(
        "--cache-state",
        choices=("cold", "warm"),
        help=("override every selected scenario's cache state; default is "
              "scenario value or warm"),
    )
    parser.add_argument(
        "--strict-cold",
        action="store_true",
        help=("fail before a cold worker starts unless post-fadvise mincore "
              "verification sees zero resident model pages"),
    )
    parser.add_argument(
        "--raw-samples-out",
        type=pathlib.Path,
        help=("write per-request latency samples to a separate hashed JSON "
              "sidecar; normal result JSON remains bounded"),
    )
    args = parser.parse_args()

    if (args.memory_profile_interval_ms is not None and
            args.memory_profile_interval_ms <= 0):
        parser.error("--memory-profile-interval-ms must be positive")
    if args.strict_cold and args.cache_state == "warm":
        parser.error("--strict-cold cannot be combined with --cache-state warm")

    try:
        import yaml  # type: ignore
    except ImportError:
        print("error: pip install -r requirements-dev.txt (need pyyaml)",
              file=sys.stderr)
        return 2

    root = repo_root()
    cfg_path = (args.scenarios if args.scenarios.is_absolute()
                else root / args.scenarios)
    cfg = yaml.safe_load(cfg_path.read_text())
    scenarios: List[Dict[str, Any]] = cfg.get("scenarios", [])
    try:
        validate_streaming_ab_pairs(scenarios)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    selected_scenarios = [
        sc for sc in scenarios
        if args.milestone in sc.get("milestones", [])
        and (not args.filter or sc.get("name") == args.filter)
    ]

    manifest_path = (
        args.corpus_manifest if args.corpus_manifest.is_absolute()
        else root / args.corpus_manifest
    )
    try:
        manifest = corpus_selection.load_manifest(manifest_path)
        requested_groups = corpus_selection.parse_group_requests(args.group)
        runs = prepare_runs(
            selected_scenarios,
            requested_groups,
            args.samples_per_group,
            args.selection_seed,
            manifest,
            root,
        )
    except corpus_selection.CorpusSelectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Cache-state mistakes must fail during plan preparation, before the first
    # model page is evicted or native process is launched.
    for run in runs:
        cache_state = args.cache_state or run.scenario.get("cache_state", "warm")
        if cache_state not in ("cold", "warm"):
            print(
                f"error: scenario {run.scenario['name']}: cache_state must be "
                "cold or warm",
                file=sys.stderr,
            )
            return 2
        streaming = run.scenario.get("streaming", False)
        if not isinstance(streaming, bool):
            print(
                f"error: scenario {run.scenario['name']}: streaming must be "
                "true or false",
                file=sys.stderr,
            )
            return 2
        scenario_profile = run.scenario.get("memory_profile", False)
        scenario_profile_interval = run.scenario.get(
            "memory_profile_interval_ms"
        )
        if not isinstance(scenario_profile, bool):
            print(
                f"error: scenario {run.scenario['name']}: memory_profile must "
                "be true or false", file=sys.stderr,
            )
            return 2
        if scenario_profile_interval is not None and (
            not isinstance(scenario_profile_interval, int)
            or isinstance(scenario_profile_interval, bool)
            or scenario_profile_interval <= 0
        ):
            print(
                f"error: scenario {run.scenario['name']}: "
                "memory_profile_interval_ms must be a positive integer",
                file=sys.stderr,
            )
            return 2
        if args.strict_cold and cache_state != "cold":
            print(
                f"error: scenario {run.scenario['name']}: --strict-cold "
                "requires cold cache state",
                file=sys.stderr,
            )
            return 2

    bench_bin = (args.bench if args.bench.is_absolute()
                 else root / args.bench)
    if not bench_bin.exists():
        print(f"error: bench binary not found at {bench_bin}; "
              "build the project first (cmake --build build)",
              file=sys.stderr)
        return 2

    started_at_utc = utc_now()
    sha = git_sha()
    out_path = args.out or (root / "bench/results" / f"{sha}.json")
    raw_sidecar_path = args.raw_samples_out
    artifact_paths = [out_path]
    if raw_sidecar_path is not None:
        artifact_paths.append(raw_sidecar_path)
    resolved_artifact_paths = [path.resolve() for path in artifact_paths]
    if len(set(resolved_artifact_paths)) != len(resolved_artifact_paths):
        print(
            "error: --out and --raw-samples-out must be distinct",
            file=sys.stderr,
        )
        return 2

    # Required identities are collected before any native process starts.
    # This is especially important for multi-gigabyte GGUF hashes: hashing is
    # setup work, never part of model-load or inference timing.
    hashes = fingerprint.FileHashCache()
    model_paths: Dict[str, pathlib.Path] = {}
    model_identities: Dict[str, Dict[str, Any]] = {}
    try:
        benchmark_identity = fingerprint.file_identity(
            bench_bin, root, hashes, str(args.bench)
        )
        manifest_identity = fingerprint.file_identity(
            manifest_path, root, hashes, str(args.corpus_manifest)
        )
        scenario_config_identity = fingerprint.file_identity(
            cfg_path, root, hashes, str(args.scenarios)
        )
        for run in runs:
            model_spec = run.scenario.get("model")
            if not isinstance(model_spec, str) or not model_spec:
                raise fingerprint.FingerprintError(
                    f"scenario {run.scenario.get('name')!r} has no model identity"
                )
            if model_spec not in model_paths:
                resolved_model = model_source.resolve(model_spec, root)
                model_paths[model_spec] = resolved_model
                model_identities[model_spec] = fingerprint.file_identity(
                    resolved_model, root, hashes, model_spec
                )
    except (fingerprint.FingerprintError, SystemExit) as exc:
        print(f"error: benchmark fingerprint failed: {exc}", file=sys.stderr)
        return 2

    environment_identity = fingerprint.collect_environment(root)
    code_identity = {
        "nanoembed": fingerprint.collect_git_identity(root),
        "ggml": fingerprint.collect_ggml_identity(root),
    }
    build_identity = fingerprint.collect_build_identity(bench_bin, root)

    aggregated: Dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "milestone": args.milestone,
        "git_sha":   sha,
        "started_at_utc": started_at_utc,
        "independent_runs": 1,
        "confidence_interval": None,
        "run_settings": {
            "requested": {
                "milestone": args.milestone,
                "scenario_filter": args.filter,
                "groups": list(args.group),
                "samples_per_group": args.samples_per_group,
                "selection_seed": args.selection_seed,
                "memory_profile": args.memory_profile,
                "memory_profile_interval_ms": args.memory_profile_interval_ms,
                "cache_state_override": args.cache_state,
                "strict_cold": args.strict_cold,
                "raw_samples_out": (
                    str(args.raw_samples_out)
                    if args.raw_samples_out is not None else None
                ),
                "scenarios_path": str(args.scenarios),
                "corpus_manifest_path": str(args.corpus_manifest),
                "benchmark_binary_path": str(args.bench),
                "output_path": str(out_path),
            },
            "resolved": {
                "selected_scenario_definitions": len(selected_scenarios),
                "planned_scenario_group_runs": len(runs),
            },
        },
        "fingerprints": {
            "code": code_identity,
            "benchmark_binary": benchmark_identity,
            "build": build_identity,
            "environment": environment_identity,
            "scenario_config": scenario_config_identity,
            "corpus_manifest": manifest_identity,
        },
        "corpus_manifest": {
            "path": str(args.corpus_manifest),
            "schema_version": manifest["schema_version"],
            "size_bytes": manifest_identity["size_bytes"],
            "sha256": manifest_identity["sha256"],
            "selection_seed": args.selection_seed,
            "samples_per_group": args.samples_per_group,
            "duplicate_policy": manifest["duplicate_policy"],
        },
        "cache_state_override": args.cache_state,
        "strict_cold": args.strict_cold,
        "raw_samples": {
            "collection_status": "not_requested",
            "format": None,
            "schema_version": None,
            "memory_profile_schema_version": None,
            "extensions": [],
            "path": None,
            "size_bytes": None,
            "sha256": None,
            "scenario_count": 0,
        },
        "scenarios": {},
    }

    raw_sidecar: Optional[Dict[str, Any]] = None
    if raw_sidecar_path is not None:
        raw_sidecar = {
            "schema_version": 1,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "started_at_utc": started_at_utc,
            "latency_unit": "ms",
            "memory_profile_schema_version": 1,
            "memory_profile_time_axis": (
                "per-native-invocation GO=0ms; cold worker clocks are independent"
            ),
            "scenarios": {},
        }

    # The native tool accepts one file, so each group is one isolated native
    # invocation. TemporaryDirectory guarantees cleanup on success, validation
    # errors after entry, native failures, and Python exceptions.
    with tempfile.TemporaryDirectory(
        prefix="nanoembed-bench-inputs-"
    ) as temp_dir:
        temp_root = pathlib.Path(temp_dir)
        input_files: Dict[str, pathlib.Path] = {}

        def execute_native(
            cmd: List[str], label: str, expected_mode: str
        ) -> Optional[Dict[str, Any]]:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"FAIL: {label}\n{proc.stderr}", file=sys.stderr)
                return None
            try:
                result = json.loads(proc.stdout)
            except json.JSONDecodeError as exc:
                print(f"FAIL: {label} emitted invalid JSON: {exc}", file=sys.stderr)
                return None
            scenario_schema = result.get("schema_version")
            if scenario_schema != RESULT_SCHEMA_VERSION:
                print(
                    f"FAIL: {label} emitted schema_version "
                    f"{scenario_schema!r}; runner expects {RESULT_SCHEMA_VERSION}. "
                    "Rebuild nanoembed-bench before collecting a baseline.",
                    file=sys.stderr,
                )
                return None
            try:
                validate_execution_mode_claim(result, expected_mode, label)
            except ValueError as exc:
                print(f"FAIL: {exc}", file=sys.stderr)
                return None
            return result

        for index, run in enumerate(runs):
            selection = run.selection
            cache_state = args.cache_state or run.scenario.get(
                "cache_state", "warm"
            )
            requested_mode = (
                "streaming" if run.scenario.get("streaming", False) else "eager"
            )
            run_profile_enabled = (
                args.memory_profile or
                run.scenario.get("memory_profile", False) is True
            )
            print(
                f"==> {run.result_key} "
                f"[{selection.selected_size}/{selection.group_size} samples, "
                f"{cache_state}]",
                file=sys.stderr,
            )

            if cache_state == "cold":
                cold_results: List[Dict[str, Any]] = []
                cold_raw_workers: List[Dict[str, Any]] = []
                cold_memory_profiles: List[Dict[str, Any]] = []
                # One fresh worker per sub-batch. Batch size 1 reproduces the
                # historical one-item-per-worker cold shape exactly.
                cold_batch_size = int(run.scenario.get("batch_size", 1) or 1)
                cold_chunks = [
                    list(selection.items[begin:begin + cold_batch_size])
                    for begin in range(0, len(selection.items), cold_batch_size)
                ]
                for item_index, chunk in enumerate(cold_chunks):
                    input_path = temp_root / (
                        f"cold-{index:04d}-{item_index:04d}-{selection.group}.txt"
                    )
                    input_path.write_text(
                        "".join(entry.text + "\n" for entry in chunk),
                        encoding="utf-8",
                    )
                    native_raw_path = (
                        temp_root / f"raw-cold-{index:04d}-{item_index:04d}.json"
                        if raw_sidecar is not None or run_profile_enabled else None
                    )
                    cmd = build_cmd(
                        bench_bin,
                        run.scenario,
                        root,
                        input_path,
                        run.result_key,
                        args.memory_profile,
                        args.memory_profile_interval_ms,
                        cache_state,
                        args.strict_cold,
                        model_paths[run.scenario["model"]],
                        native_raw_path,
                    )
                    result = execute_native(
                        cmd,
                        f"{run.result_key} cold worker {item_index + 1}/"
                        f"{len(cold_chunks)}",
                        requested_mode,
                    )
                    if result is None:
                        return 1
                    cold_results.append(result)
                    if native_raw_path is not None:
                        try:
                            worker_payload = _load_native_raw_payload(
                                native_raw_path,
                                f"{run.result_key} cold worker {item_index + 1}",
                            )
                        except ValueError as exc:
                            print(f"FAIL: {exc}", file=sys.stderr)
                            return 1
                        worker_samples = worker_payload["latency_ms"]
                        if len(worker_samples) != result.get("metrics", {}).get(
                            "latency_count"
                        ):
                            print(
                                f"FAIL: {run.result_key} cold worker "
                                f"{item_index + 1}: raw latency count does not "
                                "match native summary",
                                file=sys.stderr,
                            )
                            return 1
                        try:
                            _validate_raw_memory_summary(
                                worker_payload,
                                result,
                                f"{run.result_key} cold worker {item_index + 1}",
                            )
                        except ValueError as exc:
                            print(f"FAIL: {exc}", file=sys.stderr)
                            return 1
                        cold_raw_workers.append({
                            "selected_ids": [entry.text_id for entry in chunk],
                            "latency_ms": worker_samples,
                            "memory_profile": worker_payload["memory_profile"],
                        })
                        cold_memory_profiles.append(
                            worker_payload["memory_profile"]
                        )
                try:
                    result = aggregate_cold_results(
                        cold_results,
                        [[entry.text_id for entry in chunk]
                         for chunk in cold_chunks],
                        cold_memory_profiles,
                    )
                except ValueError as exc:
                    print(f"FAIL: {run.result_key}: {exc}", file=sys.stderr)
                    return 1
            else:
                input_path = input_files.get(selection.selection_sha256)
                if input_path is None:
                    input_path = temp_root / (
                        f"selection-{index:04d}-{selection.group}.txt"
                    )
                    corpus_selection.write_selection(input_path, selection)
                    input_files[selection.selection_sha256] = input_path
                native_raw_path = (
                    temp_root / f"raw-warm-{index:04d}.json"
                    if raw_sidecar is not None or run_profile_enabled else None
                )
                cmd = build_cmd(
                    bench_bin,
                    run.scenario,
                    root,
                    input_path,
                    run.result_key,
                    args.memory_profile,
                    args.memory_profile_interval_ms,
                    cache_state,
                    args.strict_cold,
                    model_paths[run.scenario["model"]],
                    native_raw_path,
                )
                result = execute_native(cmd, run.result_key, requested_mode)
                if result is None:
                    return 1
                if native_raw_path is not None:
                    try:
                        warm_raw_payload = _load_native_raw_payload(
                            native_raw_path, run.result_key
                        )
                    except ValueError as exc:
                        print(f"FAIL: {exc}", file=sys.stderr)
                        return 1
                    warm_raw_samples = warm_raw_payload["latency_ms"]
                    if len(warm_raw_samples) != result.get("metrics", {}).get(
                        "latency_count"
                    ):
                        print(
                            f"FAIL: {run.result_key}: raw latency count does not "
                            "match native summary",
                            file=sys.stderr,
                        )
                        return 1
                    try:
                        _validate_raw_memory_summary(
                            warm_raw_payload, result, run.result_key
                        )
                    except ValueError as exc:
                        print(f"FAIL: {exc}", file=sys.stderr)
                        return 1

            # Never persist a path that is deleted when this context exits.
            result["inputs"] = f"corpus-group:{selection.group}"
            selection_metadata = selection.metadata(
                str(args.corpus_manifest),
                manifest["duplicate_policy"],
            )
            selection_metadata["manifest_sha256"] = manifest_identity["sha256"]
            selection_metadata["selected_input_sha256"] = (
                selection.selection_sha256
            )
            result["corpus_selection"] = selection_metadata
            result["independent_runs"] = 1
            result["confidence_interval"] = None
            result["scenario_definition"] = copy.deepcopy(run.scenario)
            result.setdefault("settings", {}).setdefault("requested", {}).update({
                "corpus_group": selection.group,
                "selected_input_count": selection.selected_size,
                "selection_seed": selection.seed,
            })
            result.setdefault("settings", {}).setdefault("resolved", {}).update({
                "corpus_group": selection.group,
                "selected_input_count": selection.selected_size,
            })
            native_environment = result.get("environment")
            result["environment"] = copy.deepcopy(environment_identity)
            if isinstance(native_environment, dict):
                for key in ("kernel", "cpu_model", "nproc", "page_size_bytes"):
                    if result["environment"].get(key) is None:
                        result["environment"][key] = native_environment.get(key)
            result["environment"]["native_worker_report"] = native_environment
            result["fingerprints"] = {
                "model": copy.deepcopy(
                    model_identities[run.scenario["model"]]
                ),
                "selected_input": {
                    "collection_status": "collected",
                    "sha256": selection.selection_sha256,
                    "selected_ids": [item.text_id for item in selection.items],
                },
                "corpus_manifest_sha256": manifest_identity["sha256"],
                "benchmark_binary_sha256": benchmark_identity["sha256"],
            }
            aggregated["scenarios"][run.result_key] = result

            if raw_sidecar is not None:
                if cache_state == "cold":
                    worker_profile_statuses = {
                        worker["memory_profile"].get("collection_status")
                        for worker in cold_raw_workers
                    }
                    if worker_profile_statuses == {"collected"}:
                        aggregate_profile_status = "collected"
                    elif worker_profile_statuses == {"disabled"}:
                        aggregate_profile_status = "disabled"
                    elif worker_profile_statuses == {"unavailable_legacy"}:
                        aggregate_profile_status = "unavailable_legacy"
                    else:
                        aggregate_profile_status = "mixed"
                    raw_entry = {
                        "corpus_group": selection.group,
                        "cache_state": cache_state,
                        "selected_ids": [item.text_id for item in selection.items],
                        "latency_ms": [
                            sample
                            for worker in cold_raw_workers
                            for sample in worker["latency_ms"]
                        ],
                        "cold_workers": cold_raw_workers,
                        "memory_profile": {
                            "schema_version": 1,
                            "collection_status": aggregate_profile_status,
                            "time_axis": (
                                "independent GO=0ms origin in each cold_workers entry"
                            ),
                            "worker_count": len(cold_raw_workers),
                            "samples_location": (
                                "cold_workers[].memory_profile.samples"
                            ),
                        },
                    }
                else:
                    raw_entry = {
                        "corpus_group": selection.group,
                        "cache_state": cache_state,
                        "selected_ids": [item.text_id for item in selection.items],
                        "latency_ms": warm_raw_samples,
                        "memory_profile": warm_raw_payload["memory_profile"],
                    }
                raw_sidecar["scenarios"][run.result_key] = raw_entry

    aggregated["aggregates"] = aggregate_result_summaries(
        aggregated["scenarios"]
    )

    if raw_sidecar is not None and raw_sidecar_path is not None:
        try:
            raw_sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            raw_sidecar_path.write_text(
                json.dumps(raw_sidecar, indent=2) + "\n", encoding="utf-8"
            )
        except OSError as exc:
            print(f"error: cannot write raw sample sidecar: {exc}",
                  file=sys.stderr)
            return 1
        try:
            sidecar_identity = fingerprint.file_identity(
                raw_sidecar_path, root, hashes, str(args.raw_samples_out)
            )
        except fingerprint.FingerprintError as exc:
            print(f"error: raw sample sidecar fingerprint failed: {exc}",
                  file=sys.stderr)
            return 1
        aggregated["raw_samples"] = {
            "collection_status": "collected",
            "format": "nanoembed-benchmark-raw-latency-json",
            "schema_version": 1,
            "memory_profile_schema_version": 1,
            "extensions": ["memory-profile-timeline-v1"],
            "path": _sidecar_reference(raw_sidecar_path, out_path),
            "size_bytes": sidecar_identity["size_bytes"],
            "sha256": sidecar_identity["sha256"],
            "scenario_count": len(raw_sidecar["scenarios"]),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregated, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path} with {len(aggregated['scenarios'])} scenarios",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
