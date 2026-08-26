import importlib.util
import json
import pathlib
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("dump_golden", ROOT / "tools/dump_golden.py")
assert SPEC and SPEC.loader
dump_golden = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dump_golden)


class FakeArray:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __iter__(self):
        for row in self.rows:
            yield FakeEmbedding(row)


class FakeEmbedding:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tobytes(self) -> bytes:
        return struct.pack(f"<{len(self.values)}f", *self.values)


class FakePooling:
    def __init__(self) -> None:
        self.pooling_mode_cls_token = True
        self.pooling_mode_mean_tokens = False
        self.pooling_mode_lasttoken = False
        self.pooling_mode_max_tokens = False


class FakeModel:
    def __init__(self, pooling: FakePooling) -> None:
        self.pooling = pooling

    def modules(self):
        return [self, self.pooling]


class DumpGoldenTest(unittest.TestCase):
    def test_committed_lock_is_exact_and_matches_reference_environment(self) -> None:
        pins = dump_golden.read_exact_lock(ROOT / "requirements-golden.lock")
        self.assertEqual(pins["torch"], "2.11.0")
        with mock.patch.object(
            dump_golden.importlib.metadata,
            "version",
            side_effect=lambda package: pins[package],
        ):
            self.assertEqual(
                dump_golden.verify_exact_environment(ROOT / "requirements-golden.lock"), pins
            )

    def test_lock_rejects_mutable_minimum_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "bad.lock"
            path.write_text("torch>=2.2\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exact == pins"):
                dump_golden.read_exact_lock(path)

    def test_snapshot_revision_is_taken_from_exact_snapshot_directory(self) -> None:
        exact = "a" * 40
        fake_hub = types.SimpleNamespace(
            snapshot_download=mock.Mock(return_value=f"/cache/snapshots/{exact}")
        )
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            snapshot, revision = dump_golden.resolve_snapshot(
                "org/model", "main", local_files_only=True
            )
        self.assertEqual(snapshot.name, exact)
        self.assertEqual(revision, exact)
        fake_hub.snapshot_download.assert_called_once_with(
            repo_id="org/model", revision="main", local_files_only=True
        )

    def test_snapshot_rejects_non_commit_identity(self) -> None:
        fake_hub = types.SimpleNamespace(
            snapshot_download=mock.Mock(return_value="/cache/snapshots/main")
        )
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            with self.assertRaisesRegex(RuntimeError, "exact 40-character"):
                dump_golden.resolve_snapshot("org/model", "main", local_files_only=True)

    def test_pooling_is_resolved_and_forced_to_one_mode(self) -> None:
        pooling = FakePooling()
        model = FakeModel(pooling)
        self.assertEqual(dump_golden.force_pooling(model, "model-default"), "cls")
        self.assertEqual(dump_golden.force_pooling(model, "last-token"), "last-token")
        self.assertFalse(pooling.pooling_mode_cls_token)
        self.assertFalse(pooling.pooling_mode_mean_tokens)
        self.assertTrue(pooling.pooling_mode_lasttoken)
        self.assertFalse(pooling.pooling_mode_max_tokens)

    def test_negd_and_provenance_sidecars_are_self_consistent(self) -> None:
        binary = dump_golden.encode_negd(
            ["first", "second"], FakeArray([[1.0, 2.0], [3.0, 4.0]])
        )
        self.assertEqual(binary[:4], b"NEGD")
        self.assertEqual(struct.unpack_from("<III", binary, 4), (1, 2, 2))

        with tempfile.TemporaryDirectory() as directory:
            fixture = pathlib.Path(directory) / "fixture.bin"
            fixture.write_bytes(binary)
            manifest = {
                "schema_version": 1,
                "provenance_status": "verified",
                "fixture": {"fixture_sha256": dump_golden.sha256_file(fixture)},
            }
            manifest_path, integrity_path = dump_golden.write_provenance(fixture, manifest)
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded, manifest)
            entries = dict(
                line.split() for line in integrity_path.read_text(encoding="ascii").splitlines()
            )
            self.assertEqual(entries["fixture"], dump_golden.sha256_file(fixture))
            self.assertEqual(entries["manifest"], dump_golden.sha256_file(manifest_path))

    def test_canonical_command_replaces_mutable_revision_and_pooling(self) -> None:
        command = dump_golden.canonical_command(
            model_id="org/model",
            resolved_revision="b" * 40,
            corpora=[pathlib.Path("corpus.txt")],
            output=pathlib.Path("golden.bin"),
            batch_size=8,
            max_length=256,
            pooling="mean",
            normalize=True,
            seed=7,
            torch_threads=1,
            lock_path=pathlib.Path("requirements-golden.lock"),
        )
        self.assertIn(f"--revision {'b' * 40}", command)
        self.assertIn("--pooling mean", command)
        self.assertIn("--normalize", command)
        self.assertNotIn(" main", command)


if __name__ == "__main__":
    unittest.main()
