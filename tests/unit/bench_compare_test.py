import contextlib
import io
import unittest

from bench import compare


class BenchCompareCompatibilityTest(unittest.TestCase):
    def test_unversioned_result_is_schema_v1(self) -> None:
        self.assertEqual(compare.schema_version({}), 1)
        self.assertEqual(compare.schema_version({"schema_version": 2}), 2)
        self.assertEqual(compare.schema_version({"schema_version": "2"}), -1)

    def test_v1_throughput_alias_is_visible_under_v2_name(self) -> None:
        metrics = {"throughput_items_per_sec": 12.5}
        self.assertEqual(
            compare.metric_value(metrics, "single_request_items_per_sec"),
            12.5,
        )

    def test_null_metric_is_not_coerced_to_zero(self) -> None:
        self.assertIsNone(compare.metric_value(
            {"single_request_items_per_sec": None},
            "single_request_items_per_sec",
        ))

    def test_table_displays_unavailable_metric(self) -> None:
        base = {"scenarios": {"s": {"metrics": {
            "rss_peak_lifetime_mb": 10.0,
        }}}}
        current = {"scenarios": {"s": {"metrics": {
            "rss_peak_lifetime_mb": None,
        }}}}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            compare.print_table(base, current)
        self.assertIn("—", output.getvalue())

    def test_fingerprint_mismatch_names_run_and_scenario_identity(self) -> None:
        def result(code_sha: str, model_sha: str, selection_sha: str) -> dict:
            return {
                "fingerprints": {
                    "code": {
                        "nanoembed": {"sha": code_sha, "dirty": False},
                        "ggml": {"sha": "ggml"},
                    },
                    "benchmark_binary": {"sha256": f"binary-{code_sha}"},
                    "build": {
                        "build_type": "Release",
                        "compiler": {"id": "GNU", "version": "14"},
                        "cmake_options": {"GGML_BLAS": "ON"},
                    },
                    "scenario_config": {"sha256": "scenarios"},
                    "corpus_manifest": {"sha256": "manifest"},
                    "environment": {
                        "cpu_frequency_policy": {
                            "governors": ["performance"],
                            "drivers": ["intel_pstate"],
                            "policy_count": 1,
                        },
                        "numa": {"online_nodes": "0", "node_count": 1},
                        "total_ram_bytes": 1024,
                        "working_directory_filesystem": {
                            "fs_type": "ext4", "source": "/dev/vda", "uuid": "u"
                        },
                    },
                },
                "scenarios": {
                    "s": {
                        "fingerprints": {"model": {
                            "sha256": model_sha, "size_bytes": 10,
                        }},
                        "corpus_selection": {
                            "group": "english",
                            "selection_sha256": selection_sha,
                        },
                        "settings": {
                            "requested": {"threads": 4},
                            "resolved": {
                                "pooling": "cls", "normalize": True,
                                "max_seq_len": 512, "cache_regime": "warm",
                            },
                        },
                    },
                },
            }

        issues = compare.fingerprint_differences(
            result("old", "model-a", "input-a"),
            result("new", "model-b", "input-b"),
        )
        self.assertTrue(any("NanoEmbed git SHA" in issue for issue in issues))
        self.assertTrue(any("benchmark binary SHA-256" in issue for issue in issues))
        self.assertTrue(any("scenario s model SHA-256" in issue for issue in issues))
        self.assertTrue(any("selected-input SHA-256" in issue for issue in issues))

    def test_legacy_fingerprint_absence_is_one_diagnostic(self) -> None:
        issues = compare.fingerprint_differences(
            {"scenarios": {}}, {"fingerprints": {}, "scenarios": {}}
        )
        self.assertEqual(
            issues,
            ["base extended fingerprint unavailable (legacy or pre-A5 result)"],
        )

    def test_execution_mode_mismatch_is_a_comparability_diagnostic(self) -> None:
        common = {
            "fingerprints": {},
            "settings": {"requested": {}, "resolved": {}},
            "corpus_selection": {},
        }
        eager = {**common,
                 "requested_execution_mode": "eager",
                 "resolved_execution_mode": "eager",
                 "execution_mode_resolution": {
                     "contract_version": 1, "strict_no_fallback": True,
                 }}
        streaming = {**common,
                     "requested_execution_mode": "streaming",
                     "resolved_execution_mode": "streaming",
                     "execution_mode_resolution": {
                         "contract_version": 1, "strict_no_fallback": True,
                     }}
        issues = compare.fingerprint_differences(
            {"fingerprints": {}, "scenarios": {"s": eager}},
            {"fingerprints": {}, "scenarios": {"s": streaming}},
        )
        self.assertTrue(any("requested execution mode" in issue for issue in issues))
        self.assertTrue(any("resolved execution mode" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
