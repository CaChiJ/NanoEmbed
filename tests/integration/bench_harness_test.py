"""Cross-feature contract tests for the benchmark orchestration layer.

The native benchmark is replaced with a short-lived executable fixture so this
suite can exercise runner.py as a real CLI on every development platform.  The
Linux /proc implementation has separate selftests; this suite focuses on the
contract between corpus selection, process topology, profiling flags, schema
validation, raw samples, and the bounded aggregate result.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "bench" / "runner.py"
COMPARE = ROOT / "bench" / "compare.py"


FAKE_BENCH = r'''#!/usr/bin/env python3
import json
import os
import pathlib
import sys


def option(name, default=None):
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except ValueError:
        return default


inputs = pathlib.Path(option("--inputs")).read_text(encoding="utf-8").splitlines()
warmup = int(option("--warmup", "0"))
iterations = int(option("--iter", "1"))
cache_state = option("--cache-state", "warm")
profile = "--memory-profile" in sys.argv
execution_mode = "streaming" if "--streaming" in sys.argv else "eager"
scenario = option("--scenario", "fixture")
latency_count = len(inputs) * iterations

log_path = pathlib.Path(os.environ["NANOEMBED_FAKE_BENCH_LOG"])
with log_path.open("a", encoding="utf-8") as log:
    log.write(json.dumps({
        "pid": os.getpid(),
        "argv": sys.argv[1:],
        "inputs": inputs,
        "profile": profile,
        "cache_state": cache_state,
    }) + "\n")

metrics = {
    "collection_status": {
        "latency": "collected",
        "fixed_item_window_throughput": "insufficient_samples",
    },
    "wall_sec": max(latency_count, 1) * 0.01,
    "latency_count": latency_count,
    "latency_min_ms": 10.0,
    "latency_max_ms": 10.0 + max(latency_count - 1, 0),
    "latency_mean_ms": 10.0,
    "latency_p50_ms": 10.0,
    "latency_p90_ms": 10.0,
    "latency_p95_ms": 10.0,
    "latency_p99_ms": 10.0,
    "latency_stddev_ms": 0.0,
    "latency_mad_ms": 0.0,
    "single_request_items_per_sec": 100.0,
    "fixed_item_window_throughput": {},
    "cpu_user_sec": 0.01,
    "cpu_sys_sec": 0.0,
    "page_faults_major": 0,
    "page_faults_minor": 1,
    "io_read_bytes": 0,
    "rss_peak_lifetime_mb": 20.0,
    "rss_peak_window_mb": 19.0,
    "rss_baseline_mb": 5.0,
    "rss_final_mb": 18.0,
    "rss_avg_mb": 15.0 if profile else None,
    "rss_max_sampled_mb": 19.0 if profile else None,
    "pss_baseline_mb": 4.0 if profile else None,
    "pss_final_mb": 16.0 if profile else None,
    "pss_avg_mb": 13.0 if profile else None,
    "pss_peak_sampled_mb": 17.0 if profile else None,
    "uss_baseline_mb": 3.0 if profile else None,
    "uss_final_mb": 15.0 if profile else None,
    "uss_avg_mb": 12.0 if profile else None,
    "uss_peak_sampled_mb": 16.0 if profile else None,
    **{
        f"{prefix}_sampled_{percentile}_mb": value if profile else None
        for prefix, value in (("rss", 15.0), ("pss", 13.0), ("uss", 12.0))
        for percentile in ("p50", "p75", "p90", "p95", "p99")
    },
    "memory_breakdown": {},
}

measurement = {
    "cache_regime_requested": cache_state,
    "cache_regime": cache_state,
    "execution_shape": "one-worker-warmup-then-timed-repetitions",
    "warmup_items_executed": len(inputs) * warmup,
    "latency_scope": "timed-inference-window-warmup-excluded",
    "memory_profile_enabled": profile,
    "memory_profile_samples_requested": 2 if profile else 0,
    "memory_profile_samples_attempted": 2 if profile else 0,
    "memory_profile_samples_effective": 2 if profile else 0,
    "memory_profile_valid_sample_ratio": 1.0 if profile else None,
    "memory_profile_final_sample_collected": profile,
    "memory_profile_final_in_aggregates": profile,
    "latency_result_role": "diagnostic" if profile else "authoritative",
    "hwm_reset": True,
    "rss_samples": 2 if profile else 0,
    "rollup_samples": 2 if profile else 0,
}

if cache_state == "cold":
    assert len(inputs) == 1 and warmup == 0 and iterations == 1
    measurement.update({
        "execution_shape": "one-fresh-worker-one-first-inference",
        "warmup_items_executed": 0,
        "latency_scope": "first-inference-only-after-model-and-context-creation",
        "cache_control": {
            "cold_cache_requested": True,
            "strict_cold": "--strict-cold" in sys.argv,
            "platform_supported": True,
            "eviction_call_succeeded": True,
            "cold_cache_verified": True,
            "verification_status": "verified",
            "before_eviction": {"resident_percent": 80.0},
            "after_eviction_before_worker": {"resident_percent": 0.0},
            "after_worker_load_and_first_result": {"resident_percent": 100.0},
        },
    })
    metrics["cold_phase_timings"] = {
        "collection_status": "collected",
        "startup_boundary": "fixture",
        "fresh_worker_count": 1,
        "model_load_ms": {"count": 1, "mean_ms": 40.0},
        "context_create_ms": {"count": 1, "mean_ms": 5.0},
        "first_request_latency_ms": {"count": 1, "mean_ms": 10.0},
        "startup_to_first_result_ms": {"count": 1, "mean_ms": 55.0},
    }

raw_path = option("--raw-samples-out")
if raw_path is not None:
    if profile:
        def memory_sample(role, timestamp_ns, elapsed_ms, rss, pss, uss):
            return {
                "sample_role": role,
                "monotonic_timestamp_ns": timestamp_ns,
                "elapsed_ms_from_go": elapsed_ms,
                "read_duration_ms": 0.25,
                "valid": True,
                "rss_bytes": rss << 20,
                "pss_bytes": pss << 20,
                "uss_bytes": uss << 20,
                "breakdown_bytes": {
                    "pss_anon_bytes": pss << 20,
                    "pss_file_bytes": 0,
                    "anonymous_bytes": uss << 20,
                    "private_clean_bytes": 0,
                    "private_dirty_bytes": uss << 20,
                    "shared_clean_bytes": 0,
                    "shared_dirty_bytes": 0,
                },
            }
        memory_samples = [
            memory_sample("baseline", 999_000_000, -1.0, 5, 4, 3),
            memory_sample("periodic", 1_001_000_000, 1.0, 15, 13, 12),
            memory_sample("final", 1_002_000_000, 2.0, 18, 16, 15),
        ]
        memory_profile = {
            "schema_version": 1,
            "collection_status": "collected",
            "clock_source": "std::chrono::steady_clock",
            "go_monotonic_timestamp_ns": 1_000_000_000,
            "done_marker_monotonic_timestamp_ns": 1_001_500_000,
            "done_marker_elapsed_ms_from_go": 1.5,
            "sample_count": len(memory_samples),
            "samples": memory_samples,
        }
    else:
        memory_profile = {
            "schema_version": 1,
            "collection_status": "disabled",
            "go_monotonic_timestamp_ns": None,
            "done_marker_monotonic_timestamp_ns": None,
            "done_marker_elapsed_ms_from_go": None,
            "sample_count": 0,
            "samples": None,
        }
    pathlib.Path(raw_path).write_text(json.dumps({
        "schema_version": 1,
        "scenario": scenario,
        "latency_unit": "ms",
        "latency_ms": [10.0 + index for index in range(latency_count)],
        "memory_profile": memory_profile,
    }), encoding="utf-8")

print(json.dumps({
    "schema_version": 2,
    "scenario": scenario,
    "inputs": option("--inputs"),
    "warmup": warmup,
    "iter": iterations,
    "total_items": latency_count,
    "settings": {
        "requested": {"warmup": warmup, "iter": iterations,
                      "execution_mode": execution_mode},
        "resolved": {"pooling": "cls", "normalize": True,
                     "execution_mode": execution_mode},
    },
    "requested_execution_mode": execution_mode,
    "resolved_execution_mode": execution_mode,
    "execution_mode_resolution": {
        "contract_version": 1,
        "requested_execution_mode": execution_mode,
        "resolved_execution_mode": execution_mode,
        "strict_no_fallback": True,
        "context_creation_succeeded_with_exact_request": True,
    },
    "environment": {
        "kernel": "fixture", "cpu_model": "fixture", "nproc": 1,
        "page_size_bytes": 4096,
    },
    "measurement": measurement,
    "metrics": metrics,
}))
'''


class BenchHarnessIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.temp = pathlib.Path(self._temp.name)
        self.fake_bench = self.temp / "fake-bench"
        self.fake_bench.write_text(textwrap.dedent(FAKE_BENCH), encoding="utf-8")
        self.fake_bench.chmod(0o755)
        self.model = self.temp / "model.gguf"
        self.model.write_bytes(b"fixture-model")
        self.log = self.temp / "invocations.jsonl"

        (self.temp / "alpha.txt").write_text(
            "alpha one\nalpha two\nalpha three\n", encoding="utf-8"
        )
        (self.temp / "beta.txt").write_text(
            "beta one\nbeta two\n", encoding="utf-8"
        )
        self.manifest = self.temp / "manifest.json"
        self.manifest.write_text(json.dumps({
            "schema_version": 1,
            "duplicate_policy": {
                "within_group": "error", "across_groups": "preserve",
            },
            "groups": {
                "alpha": {"sources": [str(self.temp / "alpha.txt")]},
                "beta": {"sources": [str(self.temp / "beta.txt")]},
            },
        }), encoding="utf-8")
        self.scenarios = self.temp / "scenarios.yaml"
        self.scenarios.write_text(json.dumps({"scenarios": [{
            "name": "contract",
            "model": str(self.model),
            "corpus_group": "alpha",
            "pooling": "cls",
            "normalize": True,
            "threads": 1,
            "cache_state": "warm",
            "warmup": 2,
            "iter": 3,
            "milestones": ["M-contract"],
        }]}), encoding="utf-8")

    def tearDown(self) -> None:
        self._temp.cleanup()

    def run_runner(self, *extra: str) -> tuple[dict, list[dict]]:
        output = self.temp / f"result-{len(list(self.temp.glob('result-*')))}.json"
        env = os.environ.copy()
        env["NANOEMBED_FAKE_BENCH_LOG"] = str(self.log)
        command = [
            sys.executable, str(RUNNER),
            "--milestone", "M-contract",
            "--scenarios", str(self.scenarios),
            "--corpus-manifest", str(self.manifest),
            "--bench", str(self.fake_bench),
            "--out", str(output),
            *extra,
        ]
        completed = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(output.read_text(encoding="utf-8"))
        invocations = [
            json.loads(line)
            for line in self.log.read_text(encoding="utf-8").splitlines()
        ]
        self.log.unlink()
        return result, invocations

    def test_warm_profile_off_uses_full_default_group_and_excludes_warmup(self) -> None:
        result, invocations = self.run_runner()

        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["independent_runs"], 1)
        self.assertIsNone(result["confidence_interval"])
        self.assertEqual(len(invocations), 1)
        invocation = invocations[0]
        self.assertEqual(len(invocation["inputs"]), 3)
        self.assertNotIn("--memory-profile", invocation["argv"])
        self.assertNotIn("--raw-samples-out", invocation["argv"])
        scenario = result["scenarios"]["contract"]
        self.assertEqual(scenario["corpus_selection"]["selected_size"], 3)
        self.assertEqual(scenario["measurement"]["warmup_items_executed"], 6)
        self.assertEqual(scenario["metrics"]["latency_count"], 9)
        self.assertEqual(
            scenario["measurement"]["latency_scope"],
            "timed-inference-window-warmup-excluded",
        )
        self.assertEqual(
            scenario["measurement"]["latency_result_role"], "authoritative"
        )
        self.assertIsNone(scenario["metrics"]["pss_peak_sampled_mb"])

    def test_profile_on_multiple_groups_respects_global_and_per_group_n(self) -> None:
        raw = self.temp / "profile.samples.json"
        result, invocations = self.run_runner(
            "--group", "alpha", "--group", "beta:1",
            "--samples-per-group", "2", "--selection-seed", "17",
            "--memory-profile", "--memory-profile-interval-ms", "7",
            "--raw-samples-out", str(raw),
        )

        self.assertEqual(len(invocations), 2)
        self.assertTrue(all(invocation["profile"] for invocation in invocations))
        for invocation in invocations:
            self.assertIn("--memory-profile", invocation["argv"])
            interval = invocation["argv"].index("--memory-profile-interval-ms")
            self.assertEqual(invocation["argv"][interval + 1], "7")
        scenarios = result["scenarios"]
        self.assertEqual(scenarios["contract"]["corpus_selection"]["selected_size"], 2)
        self.assertEqual(
            scenarios["contract::beta"]["corpus_selection"]["selected_size"], 1
        )
        self.assertEqual(set(result["aggregates"]["by_group"]), {"alpha", "beta"})
        self.assertTrue(all(
            scenario["measurement"]["latency_result_role"] == "diagnostic"
            for scenario in scenarios.values()
        ))
        self.assertTrue(all(
            scenario["metrics"]["pss_peak_sampled_mb"] == 17.0
            for scenario in scenarios.values()
        ))
        self.assertTrue(all(
            scenario["metrics"]["rss_sampled_p75_mb"] == 15.0
            for scenario in scenarios.values()
        ))
        sidecar = json.loads(raw.read_text(encoding="utf-8"))
        for raw_scenario in sidecar["scenarios"].values():
            profile = raw_scenario["memory_profile"]
            self.assertEqual(profile["collection_status"], "collected")
            self.assertEqual(
                [sample["sample_role"] for sample in profile["samples"]],
                ["baseline", "periodic", "final"],
            )
            self.assertLess(profile["samples"][0]["elapsed_ms_from_go"], 0)
            self.assertEqual(
                profile["done_marker_monotonic_timestamp_ns"], 1_001_500_000
            )
            self.assertIn(
                "private_dirty_bytes",
                profile["samples"][1]["breakdown_bytes"],
            )

    def test_cold_creates_one_fresh_native_and_worker_per_selected_input(self) -> None:
        raw = self.temp / "cold.samples.json"
        result, invocations = self.run_runner(
            "--group", "alpha:2", "--selection-seed", "9",
            "--cache-state", "cold", "--strict-cold",
            "--raw-samples-out", str(raw),
        )

        self.assertEqual(len(invocations), 2)
        self.assertEqual(len({invocation["pid"] for invocation in invocations}), 2)
        self.assertTrue(all(len(invocation["inputs"]) == 1 for invocation in invocations))
        self.assertTrue(all(invocation["cache_state"] == "cold" for invocation in invocations))
        for invocation in invocations:
            warmup = invocation["argv"].index("--warmup")
            iterations = invocation["argv"].index("--iter")
            self.assertEqual(invocation["argv"][warmup + 1], "0")
            self.assertEqual(invocation["argv"][iterations + 1], "1")
        scenario = result["scenarios"]["contract"]
        self.assertEqual(scenario["measurement"]["cold_worker_invocations"], 2)
        self.assertEqual(
            scenario["metrics"]["cold_phase_timings"]["fresh_worker_count"], 2
        )
        self.assertTrue(
            scenario["measurement"]["cache_control"]["cold_cache_verified"]
        )
        self.assertEqual(scenario["measurement"]["warmup_items_executed"], 0)
        sidecar = json.loads(raw.read_text(encoding="utf-8"))
        self.assertEqual(
            len(sidecar["scenarios"]["contract"]["cold_workers"]), 2
        )
        raw_scenario = sidecar["scenarios"]["contract"]
        self.assertEqual(
            raw_scenario["memory_profile"]["collection_status"], "disabled"
        )
        self.assertTrue(all(
            worker["memory_profile"]["samples"] is None
            for worker in raw_scenario["cold_workers"]
        ))
        for prefix in ("rss", "pss", "uss"):
            for percentile in ("p50", "p75", "p90", "p95", "p99"):
                self.assertIsNone(
                    scenario["metrics"][
                        f"{prefix}_sampled_{percentile}_mb"
                    ]
                )

    def test_compare_cli_accepts_v1_baseline_and_v2_result(self) -> None:
        environment = {
            "kernel": "fixture", "cpu_model": "fixture", "nproc": 1,
            "page_size_bytes": 4096,
        }
        v1 = {
            "milestone": "M-old",
            "scenarios": {"contract": {
                "environment": environment,
                "metrics": {
                    "latency_p50_ms": 10.0,
                    "latency_p90_ms": 10.0,
                    "latency_p99_ms": 10.0,
                    "rss_peak_lifetime_mb": 20.0,
                    "throughput_items_per_sec": 100.0,
                },
            }},
        }
        v2 = {
            "schema_version": 2,
            "milestone": "M-new",
            "scenarios": {"contract": {
                "environment": environment,
                "metrics": {
                    "latency_p50_ms": 10.0,
                    "latency_p90_ms": 10.0,
                    "latency_p99_ms": 10.0,
                    "rss_peak_lifetime_mb": 20.0,
                    "single_request_items_per_sec": 100.0,
                },
            }},
        }
        old_path = self.temp / "v1.json"
        new_path = self.temp / "v2.json"
        old_path.write_text(json.dumps(v1), encoding="utf-8")
        new_path.write_text(json.dumps(v2), encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(COMPARE), str(old_path), str(new_path), "--strict"],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("comparing v1 to v2", completed.stdout)
        self.assertIn("no regressions", completed.stdout)


if __name__ == "__main__":
    unittest.main()
