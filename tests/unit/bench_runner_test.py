import json
import hashlib
import pathlib
import tempfile
import types
import unittest
from unittest import mock

from bench import corpus_selection
from bench import runner


def raw_memory_profile(start_mb: int) -> dict[str, object]:
    def sample(role: str, elapsed_ms: float, rss_mb: int) -> dict[str, object]:
        return {
            "sample_role": role,
            "monotonic_timestamp_ns": 1_000_000_000 + int(elapsed_ms * 1e6),
            "elapsed_ms_from_go": elapsed_ms,
            "read_duration_ms": 0.25,
            "valid": True,
            "rss_bytes": rss_mb << 20,
            "pss_bytes": (rss_mb // 2) << 20,
            "uss_bytes": (rss_mb // 4) << 20,
            "breakdown_bytes": {
                name: 0 for name in runner._MEMORY_BREAKDOWN_FIELDS
            },
        }

    samples = [
        sample("baseline", -1.0, 1),
        sample("periodic", 1.0, start_mb),
        sample("final", 2.0, start_mb + 10),
    ]
    return {
        "schema_version": 1,
        "collection_status": "collected",
        "go_monotonic_timestamp_ns": 1_000_000_000,
        "done_marker_monotonic_timestamp_ns": 1_001_500_000,
        "done_marker_elapsed_ms_from_go": 1.5,
        "sample_count": len(samples),
        "samples": samples,
    }


def cold_native_result(index: int, verified: bool = True,
                       total_items: int = 1) -> dict[str, object]:
    phase_values = {
        "model_load_ms": 100.0 + index,
        "context_create_ms": 20.0 + index,
        "first_request_latency_ms": 10.0 + index,
        "startup_to_first_result_ms": 140.0 + index,
    }


    phases = {
        "collection_status": "collected",
        "startup_boundary": "test",
        "fresh_worker_count": 1,
        **{name: {"count": 1, "mean_ms": value}
           for name, value in phase_values.items()},
    }
    cache = {
        "cold_cache_requested": True,
        "strict_cold": False,
        "platform_supported": True,
        "eviction_call_succeeded": True,
        "cold_cache_verified": verified,
        "verification_status": "verified" if verified else "resident",
        "before_eviction": {"resident_percent": 80.0},
        "after_eviction_before_worker": {
            "resident_percent": 0.0 if verified else 5.0,
        },
        "after_worker_load_and_first_result": {"resident_percent": 100.0},
    }
    return {
        "schema_version": 3,
        "scenario": "scenario",
        "inputs": "temporary",
        "warmup": 0,
        "iter": 1,
        "total_items": total_items,
        "settings": {
            "requested": {"execution_mode": "eager"},
            "resolved": {"execution_mode": "eager"},
        },
        "requested_execution_mode": "eager",
        "resolved_execution_mode": "eager",
        "execution_mode_resolution": {
            "contract_version": 1,
            "requested_execution_mode": "eager",
            "resolved_execution_mode": "eager",
            "strict_no_fallback": True,
            "context_creation_succeeded_with_exact_request": True,
        },
        "environment": {"kernel": "test"},
        "measurement": {
            "cache_regime": "cold" if verified else "cold-unverified",
            "execution_shape": "one-fresh-worker-one-first-inference",
            "warmup_items_executed": 0,
            "memory_profile_enabled": False,
            "memory_profile_samples_requested": 0,
            "memory_profile_samples_attempted": 0,
            "memory_profile_samples_effective": 0,
            "memory_profile_valid_sample_ratio": None,
            "memory_profile_final_sample_collected": False,
            "memory_profile_final_in_aggregates": False,
            "hwm_reset": True,
            "rss_samples": 0,
            "rollup_samples": 0,
            "cache_control": cache,
        },
        "metrics": {
            "collection_status": {
                "latency": "collected",
                "fixed_item_window_throughput": "insufficient_samples",
            },
            "latency_count": 1,
            "cold_phase_timings": phases,
            "fixed_item_window_throughput": {},
            "cpu_user_sec": 1.0,
            "cpu_sys_sec": 0.5,
            "page_faults_major": 2,
            "page_faults_minor": 3,
            "io_read_bytes": 4096,
            "rss_peak_lifetime_mb": 50.0 + index,
            "rss_peak_window_mb": 45.0 + index,
            "rss_baseline_mb": 2.0,
            "rss_final_mb": 40.0,
            "rss_avg_mb": None,
            "pss_baseline_mb": None,
            "pss_final_mb": None,
            "pss_avg_mb": None,
            "uss_baseline_mb": None,
            "uss_final_mb": None,
            "uss_avg_mb": None,
            "rss_max_sampled_mb": None,
            "pss_peak_sampled_mb": None,
            "uss_peak_sampled_mb": None,
            **{
                f"{prefix}_sampled_{percentile}_mb": None
                for prefix in ("rss", "pss", "uss")
                for percentile, _ in runner._MEMORY_PERCENTILES
            },
            "memory_breakdown": {},
        },
    }



class BenchRunnerColdTest(unittest.TestCase):
    def test_streaming_ab_pair_allows_only_mode_and_name_to_differ(self) -> None:
        eager = {
            "name": "model", "model": "m.gguf", "corpus_group": "english",
            "threads": 4, "streaming": False, "iter": 10,
        }
        streaming = {**eager, "name": "model_streaming", "streaming": True}
        runner.validate_streaming_ab_pairs([eager, streaming])

        biased = {**streaming, "threads": 1}
        with self.assertRaisesRegex(ValueError, "controlled fields: threads"):
            runner.validate_streaming_ab_pairs([eager, biased])

    def test_execution_mode_claim_must_match_strict_request(self) -> None:
        result = cold_native_result(0)
        runner.validate_execution_mode_claim(result, "eager", "scenario")
        result["resolved_execution_mode"] = "streaming"
        with self.assertRaisesRegex(ValueError, "strict eager resolution"):
            runner.validate_execution_mode_claim(result, "eager", "scenario")

    def test_cold_command_forces_one_first_request_and_can_be_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            (root / "model.gguf").write_bytes(b"model")
            scenario = {
                "name": "scenario",
                "model": "model.gguf",
                "warmup": 99,
                "iter": 88,
            }
            command = runner.build_cmd(
                pathlib.Path("bench-bin"), scenario, root,
                pathlib.Path("one.txt"), "scenario",
                cache_state="cold", strict_cold=True,
            )
        self.assertEqual(command[command.index("--warmup") + 1], "0")
        self.assertEqual(command[command.index("--iter") + 1], "1")
        self.assertEqual(command[command.index("--cache-state") + 1], "cold")
        self.assertIn("--strict-cold", command)

    def test_batch_layout_is_validated_and_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            (root / "model.gguf").write_bytes(b"model")
            scenario = {
                "name": "scenario",
                "model": "model.gguf",
                "batch_layout": "padded",
            }
            command = runner.build_cmd(
                pathlib.Path("bench-bin"), scenario, root,
                pathlib.Path("one.txt"), "scenario",
            )
            self.assertEqual(
                command[command.index("--batch-layout") + 1], "padded"
            )

            scenario["batch_layout"] = "unknown"
            with self.assertRaisesRegex(SystemExit, "batch_layout"):
                runner.build_cmd(
                    pathlib.Path("bench-bin"), scenario, root,
                    pathlib.Path("one.txt"), "scenario",
                )

    def test_cold_aggregation_uses_one_sample_per_worker(self) -> None:
        results = [cold_native_result(0), cold_native_result(1, verified=False)]
        aggregated = runner.aggregate_cold_results(results, [["id-0"], ["id-1"]])
        self.assertEqual(aggregated["total_items"], 2)
        self.assertEqual(aggregated["metrics"]["latency_count"], 2)
        self.assertEqual(aggregated["metrics"]["latency_p50_ms"], 10.0)
        self.assertEqual(aggregated["metrics"]["latency_p99_ms"], 10.0)
        self.assertEqual(
            aggregated["metrics"]["cold_phase_timings"]
                      ["startup_to_first_result_ms"]["mean_ms"],
            140.5,
        )
        self.assertEqual(aggregated["metrics"]["page_faults_major"], 4)
        self.assertEqual(aggregated["metrics"]["io_read_bytes"], 8192)
        self.assertEqual(aggregated["metrics"]["rss_peak_lifetime_mb"], 51.0)
        cache = aggregated["measurement"]["cache_control"]
        self.assertEqual(
            aggregated["measurement"]["cache_regime"], "cold-unverified"
        )
        self.assertFalse(cache["cold_cache_verified"])
        self.assertEqual(cache["verified_worker_count"], 1)
        self.assertEqual(len(cache["per_worker"]), 2)

    def test_cold_runner_gives_each_worker_one_sub_batch(self) -> None:
        # Five inputs at batch_size 2 must become three workers holding
        # 2, 2 and 1 inputs -- never five workers, and never one worker
        # holding all five.
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            (root / "group.txt").write_text(
                "a\nb\nc\nd\ne\n", encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "duplicate_policy": corpus_selection.EXPECTED_DUPLICATE_POLICY,
                "groups": {"example": {"sources": ["group.txt"]}},
            }), encoding="utf-8")
            (root / "scenarios.yaml").write_text("fake\n", encoding="utf-8")
            (root / "model.gguf").write_bytes(b"model")
            (root / "fake-bench").write_text("x\n", encoding="utf-8")
            output = root / "result.json"
            config = {"scenarios": [{
                "name": "scenario",
                "model": "model.gguf",
                "corpus_group": "example",
                "batch_size": 2,
                "max_batch": 2,
                "milestones": ["M-test"],
            }]}
            per_worker_lines: list[list[str]] = []

            def fake_run(cmd: list[str], **_: object) -> types.SimpleNamespace:
                input_path = pathlib.Path(cmd[cmd.index("--inputs") + 1])
                lines = input_path.read_text(encoding="utf-8").splitlines()
                per_worker_lines.append(lines)
                self.assertEqual(cmd[cmd.index("--batch-size") + 1], "2")
                return types.SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(cold_native_result(
                        len(per_worker_lines) - 1, total_items=len(lines))),
                    stderr="",
                )

            argv = [
                "runner.py", "--milestone", "M-test",
                "--scenarios", "scenarios.yaml",
                "--corpus-manifest", "manifest.json",
                "--bench", "fake-bench",
                "--samples-per-group", "5",
                "--cache-state", "cold",
                "--out", str(output),
            ]
            fake_yaml = types.SimpleNamespace(safe_load=lambda _: config)
            with (
                mock.patch.object(runner, "repo_root", return_value=root),
                mock.patch.object(runner, "git_sha", return_value="test-sha"),
                mock.patch.object(
                    runner.fingerprint, "collect_git_identity",
                    return_value={"collection_status": "collected",
                                  "sha": "full-test-sha", "dirty": False},
                ),
                mock.patch.object(
                    runner.fingerprint, "collect_ggml_identity",
                    return_value={"collection_status": "collected",
                                  "sha": "ggml-test-sha"},
                ),
                mock.patch.object(runner.subprocess, "run", side_effect=fake_run),
                mock.patch.object(runner.sys, "argv", argv),
                mock.patch.dict(runner.sys.modules, {"yaml": fake_yaml}),
            ):
                self.assertEqual(runner.main(), 0)

            self.assertEqual(
                [len(lines) for lines in per_worker_lines], [2, 2, 1])
            self.assertCountEqual(
                [line for lines in per_worker_lines for line in lines],
                ["a", "b", "c", "d", "e"],
            )
            scenario = json.loads(output.read_text(encoding="utf-8"))[
                "scenarios"]["scenario"]
            self.assertEqual(scenario["total_items"], 5)
            self.assertEqual(scenario["total_batches"], 3)
            self.assertEqual(
                scenario["measurement"]["cold_worker_invocations"], 3)

    def test_cold_aggregation_divides_batch_latency_by_items(self) -> None:
        # One fresh cold worker per sub-batch. Its single timed request covers
        # every item it was given, so throughput and per-item counters must use
        # the item total while batch counters keep using the worker count.
        results = [
            cold_native_result(0, total_items=3),
            cold_native_result(1, total_items=2),
        ]
        aggregated = runner.aggregate_cold_results(
            results, [["id-0", "id-1", "id-2"], ["id-3", "id-4"]]
        )
        metrics = aggregated["metrics"]
        self.assertEqual(aggregated["total_items"], 5)
        self.assertEqual(aggregated["total_batches"], 2)
        # Batch latency stays the raw first-request duration.
        self.assertEqual(metrics["batch_latency_p50_ms"], 10.0)
        # Item latency divides each worker by its own item count:
        # 10.0/3 = 3.333... and 11.0/2 = 5.5. The harness uses lower
        # percentiles (floor(q * (count - 1))), so with two samples every
        # quantile below 1.0 selects the smaller value.
        self.assertAlmostEqual(metrics["item_latency_p50_ms"], 10.0 / 3.0)
        self.assertAlmostEqual(metrics["item_latency_p99_ms"], 10.0 / 3.0)
        # 5 items and 2 batches over the same wall clock.
        startup_total_ms = 140.0 + 141.0
        self.assertAlmostEqual(
            metrics["items_per_sec"], 1000.0 * 5 / startup_total_ms
        )
        self.assertAlmostEqual(
            metrics["batches_per_sec"], 1000.0 * 2 / startup_total_ms
        )
        # page_faults_major is 2 per worker in the fixture.
        self.assertEqual(metrics["page_faults_major"], 4)
        self.assertAlmostEqual(metrics["page_faults_major_per_item"], 4 / 5)
        self.assertAlmostEqual(metrics["page_faults_major_per_batch"], 4 / 2)
        self.assertEqual(
            aggregated["measurement"]["cold_aggregation"]["items_per_worker"],
            [3, 2],
        )

    def test_cold_aggregation_rejects_worker_item_count_mismatch(self) -> None:
        results = [cold_native_result(0, total_items=2)]
        with self.assertRaisesRegex(ValueError, "was given 3"):
            runner.aggregate_cold_results(results, [["a", "b", "c"]])

    def test_cold_profile_aggregation_uses_means_peaks_and_summed_samples(self) -> None:
        results = [cold_native_result(0), cold_native_result(1)]
        for index, result in enumerate(results):
            measurement = result["measurement"]
            metrics = result["metrics"]
            measurement.update({
                "memory_profile_enabled": True,
                "memory_profile_samples_requested": 4,
                "memory_profile_samples_attempted": 3,
                "memory_profile_samples_effective": 2,
                "memory_profile_final_sample_collected": True,
                "memory_profile_final_in_aggregates": True,
                "rss_samples": 2,
                "rollup_samples": 2,
            })
            metrics.update({
                "rss_avg_mb": 30.0 + 10.0 * index,
                "rss_max_sampled_mb": 40.0 + 20.0 * index,
                "pss_avg_mb": 20.0 + 10.0 * index,
                "pss_peak_sampled_mb": 30.0 + 20.0 * index,
                "uss_avg_mb": 10.0 + 10.0 * index,
                "uss_peak_sampled_mb": 20.0 + 20.0 * index,
            })

        profiles = [raw_memory_profile(10), raw_memory_profile(30)]
        # Give the second worker one additional periodic sample so the exact
        # merged mean proves it is count-weighted rather than mean-of-means.
        extra = dict(profiles[1]["samples"][1])
        extra.update({
            "monotonic_timestamp_ns": 1_001_500_000,
            "elapsed_ms_from_go": 1.5,
            "rss_bytes": 50 << 20,
            "pss_bytes": 40 << 20,
            "uss_bytes": 9 << 20,
        })
        profiles[1]["samples"].insert(2, extra)
        profiles[1]["sample_count"] += 1
        invalid = dict(profiles[0]["samples"][1])
        invalid.update({"valid": False, "rss_bytes": 999 << 20})
        profiles[0]["samples"].insert(2, invalid)
        aggregated = runner.aggregate_cold_results(
            results, [["id-0"], ["id-1"]], profiles
        )
        measurement = aggregated["measurement"]
        metrics = aggregated["metrics"]
        self.assertEqual(measurement["memory_profile_samples_attempted"], 6)
        self.assertEqual(measurement["memory_profile_samples_effective"], 4)
        self.assertAlmostEqual(
            measurement["memory_profile_valid_sample_ratio"], 4 / 6
        )
        self.assertEqual(measurement["rss_samples"], 4)
        self.assertEqual(metrics["rss_avg_mb"], 30.0)
        self.assertEqual(metrics["rss_max_sampled_mb"], 60.0)
        self.assertEqual(metrics["pss_avg_mb"], 18.0)
        self.assertEqual(metrics["pss_peak_sampled_mb"], 50.0)
        # Lower percentiles over [10, 20, 30, 40, 50], baseline excluded.
        self.assertEqual(metrics["rss_sampled_p50_mb"], 30.0)
        self.assertEqual(metrics["rss_sampled_p75_mb"], 40.0)
        self.assertEqual(metrics["rss_sampled_p99_mb"], 40.0)
        self.assertEqual(metrics["pss_sampled_p50_mb"], 15.0)
        self.assertEqual(metrics["uss_sampled_p75_mb"], 9.0)

    def test_native_raw_loader_accepts_legacy_and_validates_memory_timeline(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "raw.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "scenario": "legacy",
                "latency_unit": "ms",
                "latency_ms": [1.0],
            }), encoding="utf-8")
            legacy = runner._load_native_raw_payload(path, "legacy")
            self.assertEqual(
                legacy["memory_profile"]["collection_status"],
                "unavailable_legacy",
            )
            with self.assertRaisesRegex(ValueError, "omitted.*raw memory"):
                runner._validate_raw_memory_summary(
                    legacy,
                    {"measurement": {
                        "memory_profile_enabled": True,
                        "memory_profile_samples_attempted": 2,
                        "memory_profile_samples_effective": 2,
                    }},
                    "legacy-profile-on",
                )

            memory = raw_memory_profile(10)
            path.write_text(json.dumps({
                "schema_version": 1,
                "scenario": "current",
                "latency_unit": "ms",
                "latency_ms": [1.0],
                "memory_profile": memory,
            }), encoding="utf-8")
            current = runner._load_native_raw_payload(path, "current")
            self.assertEqual(
                [sample["sample_role"] for sample in
                 current["memory_profile"]["samples"]],
                ["baseline", "periodic", "final"],
            )

            memory["samples"][0]["elapsed_ms_from_go"] = 0.0
            path.write_text(json.dumps({
                "schema_version": 1,
                "scenario": "bad",
                "latency_unit": "ms",
                "latency_ms": [1.0],
                "memory_profile": memory,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "baseline.*not before GO"):
                runner._load_native_raw_payload(path, "bad")

    def test_runner_launches_one_native_process_per_selected_cold_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            (root / "group.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({
                "schema_version": 1,
                "duplicate_policy": corpus_selection.EXPECTED_DUPLICATE_POLICY,
                "groups": {"example": {"sources": ["group.txt"]}},
            }), encoding="utf-8")
            (root / "scenarios.yaml").write_text("fake\n", encoding="utf-8")
            (root / "model.gguf").write_bytes(b"model")
            (root / "fake-bench").write_text("placeholder\n", encoding="utf-8")
            output = root / "result.json"
            raw_output = root / "result.samples.json"
            config = {"scenarios": [{
                "name": "scenario",
                "model": "model.gguf",
                "corpus_group": "example",
                "milestones": ["M-test"],
            }]}
            captured_lines: list[str] = []

            def fake_run(cmd: list[str], **_: object) -> types.SimpleNamespace:
                input_path = pathlib.Path(cmd[cmd.index("--inputs") + 1])
                lines = input_path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(lines), 1)
                self.assertEqual(cmd[cmd.index("--cache-state") + 1], "cold")
                self.assertEqual(cmd[cmd.index("--warmup") + 1], "0")
                self.assertEqual(cmd[cmd.index("--iter") + 1], "1")
                captured_lines.extend(lines)
                raw_path = pathlib.Path(cmd[cmd.index("--raw-samples-out") + 1])
                raw_path.write_text(json.dumps({
                    "schema_version": 1,
                    "scenario": "scenario",
                    "latency_unit": "ms",
                    "latency_ms": [10.0 + len(captured_lines) - 1],
                }), encoding="utf-8")
                return types.SimpleNamespace(
                    returncode=0,
                    stdout=json.dumps(cold_native_result(len(captured_lines) - 1)),
                    stderr="",
                )

            argv = [
                "runner.py", "--milestone", "M-test",
                "--scenarios", "scenarios.yaml",
                "--corpus-manifest", "manifest.json",
                "--bench", "fake-bench",
                "--samples-per-group", "3",
                "--cache-state", "cold",
                "--out", str(output),
                "--raw-samples-out", str(raw_output),
            ]
            fake_yaml = types.SimpleNamespace(safe_load=lambda _: config)
            with (
                mock.patch.object(runner, "repo_root", return_value=root),
                mock.patch.object(runner, "git_sha", return_value="test-sha"),
                mock.patch.object(
                    runner.fingerprint,
                    "collect_git_identity",
                    return_value={
                        "collection_status": "collected",
                        "sha": "full-test-sha",
                        "dirty": False,
                    },
                ),
                mock.patch.object(
                    runner.fingerprint,
                    "collect_ggml_identity",
                    return_value={
                        "collection_status": "collected",
                        "sha": "ggml-test-sha",
                    },
                ),
                mock.patch.object(runner.subprocess, "run", side_effect=fake_run),
                mock.patch.object(runner.sys, "argv", argv),
                mock.patch.dict(runner.sys.modules, {"yaml": fake_yaml}),
            ):
                self.assertEqual(runner.main(), 0)

            self.assertCountEqual(captured_lines, ["alpha", "beta", "gamma"])
            scenario = json.loads(output.read_text(encoding="utf-8"))[
                "scenarios"
            ]["scenario"]
            self.assertEqual(scenario["metrics"]["latency_count"], 3)
            self.assertEqual(
                scenario["measurement"]["cold_worker_invocations"], 3
            )
            main_result = json.loads(output.read_text(encoding="utf-8"))
            sidecar_bytes = raw_output.read_bytes()
            self.assertEqual(
                main_result["raw_samples"]["sha256"],
                hashlib.sha256(sidecar_bytes).hexdigest(),
            )
            self.assertEqual(
                main_result["raw_samples"]["memory_profile_schema_version"], 1
            )
            self.assertIn(
                "memory-profile-timeline-v1",
                main_result["raw_samples"]["extensions"],
            )
            sidecar = json.loads(sidecar_bytes)
            self.assertEqual(sidecar["memory_profile_schema_version"], 1)
            raw_scenario = sidecar["scenarios"]["scenario"]
            self.assertEqual(len(raw_scenario["cold_workers"]), 3)
            self.assertEqual(raw_scenario["latency_ms"], [10.0, 11.0, 12.0])
            self.assertEqual(
                raw_scenario["memory_profile"]["collection_status"],
                "unavailable_legacy",
            )
            self.assertTrue(all(
                worker["memory_profile"]["collection_status"] ==
                "unavailable_legacy"
                for worker in raw_scenario["cold_workers"]
            ))
            # A5 sidecar collection must not replace A4's native audit trail.
            self.assertEqual(
                len(scenario["measurement"]["cache_control"]["per_worker"]), 3
            )


class BenchRunnerAggregationTest(unittest.TestCase):
    @staticmethod
    def _scenario(group: str, model_sha: str, items: int, wall: float,
                  mean: float) -> dict[str, object]:
        return {
            "total_items": items,
            "corpus_selection": {
                "group": group,
                "selection_sha256": f"selection-{group}",
            },
            "fingerprints": {"model": {"sha256": model_sha}},
            "measurement": {
                "cache_regime": "warm",
                "memory_profile_enabled": False,
            },
            "metrics": {
                "wall_sec": wall,
                "latency_count": items,
                "latency_min_ms": mean - 1.0,
                "latency_max_ms": mean + 1.0,
                "latency_mean_ms": mean,
            },
        }

    def test_aggregates_by_group_and_overall_without_losing_scenarios(self) -> None:
        scenarios = {
            "bert::english": self._scenario("english", "m1", 2, 1.0, 10.0),
            "harrier::english": self._scenario(
                "english", "m2", 3, 2.0, 20.0
            ),
            "bert::unicode": self._scenario("unicode", "m1", 1, 1.0, 30.0),
        }
        summary = runner.aggregate_result_summaries(scenarios)

        self.assertEqual(set(summary["by_group"]), {"english", "unicode"})
        english = summary["by_group"]["english"]
        self.assertEqual(
            english["scenario_keys"], ["bert::english", "harrier::english"]
        )
        self.assertEqual(english["total_items"], 5)
        self.assertAlmostEqual(english["single_request_items_per_sec"], 5 / 3)
        self.assertAlmostEqual(english["latency_weighted_mean_ms"], 16.0)
        self.assertFalse(english["homogeneous_dimensions"])
        self.assertEqual(summary["overall"]["total_items"], 6)
        self.assertIsNone(summary["overall"]["latency_percentiles"])


if __name__ == "__main__":
    unittest.main()
