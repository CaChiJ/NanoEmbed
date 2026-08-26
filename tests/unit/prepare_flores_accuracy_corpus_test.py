import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "prepare_flores_accuracy_corpus",
    ROOT / "tools/prepare_flores_accuracy_corpus.py",
)
assert SPEC and SPEC.loader
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class PrepareFloresAccuracyCorpusTest(unittest.TestCase):
    REVISION = "a" * 40

    def write_source(
        self,
        snapshot: pathlib.Path,
        split: str,
        languoid: str,
        rows: list[dict],
    ) -> None:
        directory = snapshot / split
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{languoid}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def row(self, identifier: str, text: str, split: str, *, script: str) -> dict:
        return {
            "id": identifier,
            "iso_639_3": "eng" if script == "Latn" else "kor",
            "iso_15924": script,
            "glottocode": "test1234",
            "variant": "",
            "text": text,
            "domain": "wikinews",
            "topic": "science",
            "split": split,
        }

    def make_spec(self, directory: pathlib.Path, shard_size: int = 2) -> tuple[pathlib.Path, dict]:
        path = directory / "source.json"
        spec = {
            "schema_version": 1,
            "dataset_id": "org/flores",
            "dataset_revision": self.REVISION,
            "dataset_version": "test",
            "license": "CC-BY-SA-4.0",
            "splits": ["dev", "devtest"],
            "shard_size": shard_size,
            "access_url": "https://example.invalid/flores",
        }
        path.write_text(json.dumps(spec), encoding="utf-8")
        return path, spec

    def test_builds_deterministic_shards_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            snapshot = directory / self.REVISION
            self.write_source(
                snapshot,
                "dev",
                "eng_Latn",
                [
                    self.row("1", "first", "dev", script="Latn"),
                    self.row("2", "second", "dev", script="Latn"),
                ],
            )
            self.write_source(
                snapshot,
                "devtest",
                "kor_Hang",
                [self.row("1", "세 번째", "devtest", script="Hang")],
            )
            spec_path, spec = self.make_spec(directory)
            output = directory / "output"

            manifest = prepare.build_corpus(
                spec_path, spec, snapshot, output, invoked_command="test"
            )

            self.assertEqual(manifest["corpus"]["sample_count"], 3)
            self.assertEqual(manifest["corpus"]["languoid_count"], 2)
            self.assertEqual(manifest["corpus"]["split_counts"], {"dev": 2, "devtest": 1})
            self.assertEqual(len(manifest["corpus"]["shards"]), 2)
            verified = prepare.verify_corpus(output)
            self.assertEqual(
                verified["corpus"]["canonical_sha256"],
                manifest["corpus"]["canonical_sha256"],
            )

            records = []
            for shard in sorted((output / "corpus").glob("*.jsonl")):
                records.extend(json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines())
            self.assertEqual([record["text"] for record in records], ["first", "second", "세 번째"])
            self.assertEqual(records[-1]["languoid"], "kor_Hang")

    def test_rejects_duplicate_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            snapshot = directory / self.REVISION
            duplicate = self.row("1", "same id", "dev", script="Latn")
            self.write_source(snapshot, "dev", "eng_Latn", [duplicate, duplicate])
            self.write_source(
                snapshot,
                "devtest",
                "eng_Latn",
                [self.row("1", "other split", "devtest", script="Latn")],
            )
            spec_path, spec = self.make_spec(directory)

            with self.assertRaisesRegex(prepare.CorpusPreparationError, "duplicate identity"):
                prepare.build_corpus(
                    spec_path, spec, snapshot, directory / "output", invoked_command="test"
                )

    def test_verifier_rejects_tampered_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            snapshot = directory / self.REVISION
            self.write_source(
                snapshot,
                "dev",
                "eng_Latn",
                [self.row("1", "first", "dev", script="Latn")],
            )
            self.write_source(
                snapshot,
                "devtest",
                "eng_Latn",
                [self.row("1", "second", "devtest", script="Latn")],
            )
            spec_path, spec = self.make_spec(directory)
            output = directory / "output"
            prepare.build_corpus(spec_path, spec, snapshot, output, invoked_command="test")
            shard = next((output / "corpus").glob("*.jsonl"))
            shard.write_text(shard.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

            with self.assertRaisesRegex(prepare.CorpusPreparationError, "mismatch"):
                prepare.verify_corpus(output)


if __name__ == "__main__":
    unittest.main()
