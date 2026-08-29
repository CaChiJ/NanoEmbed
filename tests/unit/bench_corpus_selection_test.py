import json
import pathlib
import tempfile
import types
import unittest
from unittest import mock

from bench import corpus_selection
from bench import runner


class CorpusSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_lines(self, relative_path: str, lines: list[str]) -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")

    @staticmethod
    def spec(path: str = "group.txt") -> dict[str, object]:
        return {"sources": [path]}

    def test_same_seed_and_count_are_repeatable_and_not_source_ordered(self) -> None:
        lines = [f"input number {index}" for index in range(20)]
        self.write_lines("group.txt", lines)
        first = corpus_selection.select_group(
            self.root, "example", self.spec(), 6, 17
        )

        self.write_lines("group.txt", list(reversed(lines)))
        second = corpus_selection.select_group(
            self.root, "example", self.spec(), 6, 17
        )

        self.assertEqual(
            [item.text_id for item in first.items],
            [item.text_id for item in second.items],
        )
        self.assertEqual(first.selection_sha256, second.selection_sha256)
        self.assertNotEqual([item.text for item in first.items], lines[:6])

    def test_seed_changes_bounded_selection(self) -> None:
        self.write_lines("group.txt", [f"line {index}" for index in range(30)])
        first = corpus_selection.select_group(
            self.root, "example", self.spec(), 5, 1
        )
        second = corpus_selection.select_group(
            self.root, "example", self.spec(), 5, 2
        )
        self.assertNotEqual(
            {item.text_id for item in first.items},
            {item.text_id for item in second.items},
        )

    def test_explicit_count_precedes_global_count_and_large_count_clamps(self) -> None:
        groups = {"first": {}, "second": {}}
        requested = corpus_selection.parse_group_requests(["first:2", "second"])
        self.assertEqual(
            corpus_selection.resolve_group_requests(
                requested, "first", 4, groups
            ),
            [("first", 2), ("second", 4)],
        )

        self.write_lines("group.txt", ["one", "two", "three"])
        selected = corpus_selection.select_group(
            self.root, "example", self.spec(), 100, 0
        )
        self.assertEqual(selected.group_size, 3)
        self.assertEqual(selected.selected_size, 3)
        self.assertEqual([item.text for item in selected.items], ["one", "two", "three"])

    def test_unknown_group_is_rejected_during_plan_preparation(self) -> None:
        manifest = {
            "groups": {"known": self.spec()},
            "duplicate_policy": corpus_selection.EXPECTED_DUPLICATE_POLICY,
        }
        scenarios = [{"name": "scenario", "corpus_group": "known"}]
        with self.assertRaisesRegex(
            corpus_selection.CorpusSelectionError, "unknown corpus group 'missing'"
        ):
            runner.prepare_runs(
                scenarios,
                [("missing", None)],
                None,
                0,
                manifest,
                self.root,
            )

    def test_empty_group_is_rejected_during_plan_preparation(self) -> None:
        self.write_lines("empty.txt", [])
        manifest = {
            "groups": {"empty": self.spec("empty.txt")},
            "duplicate_policy": corpus_selection.EXPECTED_DUPLICATE_POLICY,
        }
        scenarios = [{"name": "scenario", "corpus_group": "empty"}]
        with self.assertRaisesRegex(
            corpus_selection.CorpusSelectionError, "corpus group 'empty' is empty"
        ):
            runner.prepare_runs(
                scenarios, [], None, 0, manifest, self.root
            )

    def test_each_requested_group_keeps_a_distinct_result_identity(self) -> None:
        self.write_lines("first.txt", ["one", "two", "three"])
        self.write_lines("second.txt", ["four", "five", "six"])
        manifest = {
            "groups": {
                "first": self.spec("first.txt"),
                "second": self.spec("second.txt"),
            },
            "duplicate_policy": corpus_selection.EXPECTED_DUPLICATE_POLICY,
        }
        scenarios = [{"name": "scenario", "corpus_group": "first"}]
        runs = runner.prepare_runs(
            scenarios,
            [("first", 1), ("second", 2)],
            3,
            5,
            manifest,
            self.root,
        )
        self.assertEqual(
            [(run.result_key, run.selection.group, run.selection.selected_size)
             for run in runs],
            [("scenario", "first", 1), ("scenario::second", "second", 2)],
        )

    def test_runner_memory_profile_is_opt_in_and_cli_interval_overrides(self) -> None:
        (self.root / "model.gguf").write_bytes(b"model")
        scenario = {"name": "scenario", "model": "model.gguf"}
        base = runner.build_cmd(
            pathlib.Path("bench-bin"), scenario, self.root,
            pathlib.Path("inputs.txt"), "scenario"
        )
        self.assertNotIn("--memory-profile", base)
        self.assertNotIn("--memory-profile-interval-ms", base)

        scenario["memory_profile"] = True
        scenario["memory_profile_interval_ms"] = 50
        profiled = runner.build_cmd(
            pathlib.Path("bench-bin"), scenario, self.root,
            pathlib.Path("inputs.txt"), "scenario",
            memory_profile_interval_ms=10,
        )
        self.assertIn("--memory-profile", profiled)
        interval_index = profiled.index("--memory-profile-interval-ms")
        self.assertEqual(profiled[interval_index + 1], "10")

    def test_runner_streaming_is_explicit_and_strictly_typed(self) -> None:
        (self.root / "model.gguf").write_bytes(b"model")
        scenario = {
            "name": "scenario", "model": "model.gguf", "streaming": True,
        }
        command = runner.build_cmd(
            pathlib.Path("bench-bin"), scenario, self.root,
            pathlib.Path("inputs.txt"), "scenario",
        )
        self.assertIn("--streaming", command)

        scenario["streaming"] = "yes"
        with self.assertRaisesRegex(SystemExit, "streaming"):
            runner.build_cmd(
                pathlib.Path("bench-bin"), scenario, self.root,
                pathlib.Path("inputs.txt"), "scenario",
            )

    def test_runner_rejects_invalid_memory_profile_scenario_values(self) -> None:
        (self.root / "model.gguf").write_bytes(b"model")
        scenario = {
            "name": "scenario",
            "model": "model.gguf",
            "memory_profile": "yes",
        }
        with self.assertRaisesRegex(SystemExit, "memory_profile"):
            runner.build_cmd(
                pathlib.Path("bench-bin"), scenario, self.root,
                pathlib.Path("inputs.txt"), "scenario"
            )

        scenario["memory_profile"] = True
        scenario["memory_profile_interval_ms"] = 0
        with self.assertRaisesRegex(SystemExit, "positive integer"):
            runner.build_cmd(
                pathlib.Path("bench-bin"), scenario, self.root,
                pathlib.Path("inputs.txt"), "scenario"
            )

    def test_duplicate_policy_rejects_within_group_but_preserves_across_groups(self) -> None:
        self.write_lines("duplicate.txt", ["same", "same"])
        with self.assertRaisesRegex(
            corpus_selection.CorpusSelectionError, "duplicate text"
        ):
            corpus_selection.select_group(
                self.root, "duplicate", self.spec("duplicate.txt"), None, 0
            )

        self.write_lines("first.txt", ["shared", "first only"])
        self.write_lines("second.txt", ["shared", "second only"])
        first = corpus_selection.select_group(
            self.root, "first", self.spec("first.txt"), None, 0
        )
        second = corpus_selection.select_group(
            self.root, "second", self.spec("second.txt"), None, 0
        )
        shared_id = corpus_selection.stable_text_id("shared")
        self.assertIn(shared_id, [item.text_id for item in first.items])
        self.assertIn(shared_id, [item.text_id for item in second.items])

    def test_metadata_and_materialized_bytes_match_selection_digest(self) -> None:
        self.write_lines("group.txt", ["alpha", "한글", "emoji 🚀"])
        selection = corpus_selection.select_group(
            self.root, "example", self.spec(), 2, 9
        )
        output = self.root / "selected.txt"
        corpus_selection.write_selection(output, selection)

        import hashlib

        self.assertEqual(
            hashlib.sha256(output.read_bytes()).hexdigest(),
            selection.selection_sha256,
        )
        metadata = selection.metadata(
            "bench/corpus_groups.json",
            corpus_selection.EXPECTED_DUPLICATE_POLICY,
        )
        self.assertEqual(metadata["group_size"], 3)
        self.assertEqual(metadata["selected_size"], 2)
        self.assertEqual(metadata["seed"], 9)
        self.assertEqual(len(metadata["selected_ids"]), 2)

    def test_only_lf_splits_records_like_the_native_reader(self) -> None:
        self.write_lines("group.txt", ["alpha\u2028beta", "gamma"])
        items, _ = corpus_selection.load_group(
            self.root, "example", self.spec()
        )
        self.assertEqual([item.text for item in items], ["alpha\u2028beta", "gamma"])

    def test_tracked_manifest_has_all_non_empty_initial_groups(self) -> None:
        root = pathlib.Path(__file__).resolve().parents[2]
        manifest = corpus_selection.load_manifest(root / "bench/corpus_groups.json")
        expected = {
            "english_short",
            "multilingual_short",
            "mixed_short",
            "unicode_edge",
            "vocab_spread",
            "medium",
            "long_context",
            "uniform_len",
        }
        self.assertEqual(set(manifest["groups"]), expected)
        for group_name, spec in manifest["groups"].items():
            items, _ = corpus_selection.load_group(root, group_name, spec)
            self.assertGreater(len(items), 0, group_name)

    def test_manifest_duplicate_policy_is_explicit(self) -> None:
        path = self.root / "manifest.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "duplicate_policy": {
                "within_group": "deduplicate",
                "across_groups": "preserve",
            },
            "groups": {"example": self.spec()},
        }), encoding="utf-8")
        with self.assertRaisesRegex(
            corpus_selection.CorpusSelectionError, "duplicate_policy"
        ):
            corpus_selection.load_manifest(path)

    def test_runner_cleans_temporary_selection_and_persists_group_identity(self) -> None:
        self.write_lines("group.txt", ["alpha", "beta", "gamma"])
        (self.root / "manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "duplicate_policy": corpus_selection.EXPECTED_DUPLICATE_POLICY,
            "groups": {"example": self.spec()},
        }), encoding="utf-8")
        (self.root / "scenarios.yaml").write_text("ignored by fake yaml\n", encoding="utf-8")
        (self.root / "model.gguf").write_bytes(b"model")
        (self.root / "fake-bench").write_text("placeholder\n", encoding="utf-8")
        output = self.root / "result.json"
        config = {"scenarios": [{
            "name": "scenario",
            "model": "model.gguf",
            "corpus_group": "example",
            "milestones": ["M-test"],
        }]}
        captured_inputs: list[pathlib.Path] = []

        def fake_run(cmd: list[str], **_: object) -> types.SimpleNamespace:
            input_path = pathlib.Path(cmd[cmd.index("--inputs") + 1])
            self.assertTrue(input_path.exists())
            self.assertEqual(len(input_path.read_text(encoding="utf-8").splitlines()), 2)
            captured_inputs.append(input_path)
            mode = "streaming" if "--streaming" in cmd else "eager"
            return types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps({
                    "schema_version": 3,
                    "settings": {
                        "requested": {"execution_mode": mode},
                        "resolved": {"execution_mode": mode},
                    },
                    "requested_execution_mode": mode,
                    "resolved_execution_mode": mode,
                    "execution_mode_resolution": {
                        "contract_version": 1,
                        "requested_execution_mode": mode,
                        "resolved_execution_mode": mode,
                        "strict_no_fallback": True,
                        "context_creation_succeeded_with_exact_request": True,
                    },
                    "metrics": {},
                }),
                stderr="",
            )

        argv = [
            "runner.py",
            "--milestone", "M-test",
            "--scenarios", "scenarios.yaml",
            "--corpus-manifest", "manifest.json",
            "--bench", "fake-bench",
            "--samples-per-group", "2",
            "--selection-seed", "23",
            "--out", str(output),
        ]
        fake_yaml = types.SimpleNamespace(safe_load=lambda _: config)
        with (
            mock.patch.object(runner, "repo_root", return_value=self.root),
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

        self.assertEqual(len(captured_inputs), 1)
        self.assertFalse(captured_inputs[0].exists())
        result = json.loads(output.read_text(encoding="utf-8"))
        scenario = result["scenarios"]["scenario"]
        self.assertEqual(scenario["inputs"], "corpus-group:example")
        self.assertEqual(scenario["corpus_selection"]["group_size"], 3)
        self.assertEqual(scenario["corpus_selection"]["selected_size"], 2)
        self.assertEqual(scenario["corpus_selection"]["seed"], 23)
        self.assertEqual(len(scenario["corpus_selection"]["selected_ids"]), 2)


if __name__ == "__main__":
    unittest.main()
