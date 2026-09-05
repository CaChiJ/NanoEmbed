#!/usr/bin/env python3
"""Prepare a pinned, auditable FLORES+ corpus for accuracy certification.

The source dataset is gated to protect evaluation integrity. This command
therefore never falls back to another revision, split, language subset, or
cached partial snapshot. It writes deterministic JSONL shards plus a manifest
that records every source file and output hash.

The generated corpus is intentionally kept outside Git. Commit the small
source specification, not the protected FLORES+ text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import shlex
import shutil
import sys
import tempfile
from collections import Counter
from typing import Any, Iterable, TextIO


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "tests/certification/flores_plus_v4_6.source.json"
DEFAULT_OUTPUT = ROOT / "certification-data/flores-plus-v4.6"
CORPUS_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
COMMIT_SHA_LENGTH = 40


class CorpusPreparationError(RuntimeError):
    """A deterministic corpus could not be prepared or verified."""


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: pathlib.Path, *, name: str) -> dict[str, Any]:
    return {
        "path": name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_spec(path: pathlib.Path) -> dict[str, Any]:
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CorpusPreparationError(f"cannot read source specification {path}: {error}") from error

    required = {
        "schema_version",
        "dataset_id",
        "dataset_revision",
        "dataset_version",
        "license",
        "splits",
        "shard_size",
        "access_url",
    }
    missing = sorted(required - spec.keys())
    if missing:
        raise CorpusPreparationError(f"source specification is missing: {', '.join(missing)}")
    if spec["schema_version"] != 1:
        raise CorpusPreparationError(
            f"unsupported source specification schema: {spec['schema_version']}"
        )
    revision = spec["dataset_revision"]
    if (
        not isinstance(revision, str)
        or len(revision) != COMMIT_SHA_LENGTH
        or any(ch not in "0123456789abcdef" for ch in revision.lower())
    ):
        raise CorpusPreparationError("dataset_revision must be an exact 40-character commit SHA")
    if not isinstance(spec["splits"], list) or not spec["splits"]:
        raise CorpusPreparationError("splits must be a non-empty list")
    if len(set(spec["splits"])) != len(spec["splits"]):
        raise CorpusPreparationError("splits must not contain duplicates")
    if not isinstance(spec["shard_size"], int) or spec["shard_size"] <= 0:
        raise CorpusPreparationError("shard_size must be a positive integer")
    return spec


def resolve_snapshot(spec: dict[str, Any], *, local_files_only: bool) -> pathlib.Path:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except ImportError as error:
        raise CorpusPreparationError(
            "huggingface-hub is required; install requirements-golden.lock"
        ) from error

    patterns = [f"{split}/*.jsonl" for split in spec["splits"]]
    patterns.extend(["README.md", "CHANGELOG.md", "bibliography.bib"])
    try:
        snapshot = pathlib.Path(
            snapshot_download(
                repo_id=spec["dataset_id"],
                repo_type="dataset",
                revision=spec["dataset_revision"],
                allow_patterns=patterns,
                local_files_only=local_files_only,
            )
        ).resolve()
    except GatedRepoError as error:
        raise CorpusPreparationError(
            "FLORES+ access is not authorized. Accept the dataset conditions at "
            f"{spec['access_url']} and run `hf auth login`, then retry."
        ) from error
    except HfHubHTTPError as error:
        raise CorpusPreparationError(f"cannot download pinned FLORES+ snapshot: {error}") from error
    except OSError as error:
        raise CorpusPreparationError(f"cannot resolve pinned FLORES+ snapshot: {error}") from error

    if snapshot.name.lower() != spec["dataset_revision"].lower():
        raise CorpusPreparationError(
            "resolved dataset snapshot does not match the requested commit: "
            f"expected {spec['dataset_revision']}, found {snapshot.name}"
        )
    return snapshot


def source_files(snapshot: pathlib.Path, splits: Iterable[str]) -> list[tuple[str, pathlib.Path]]:
    files: list[tuple[str, pathlib.Path]] = []
    for split in splits:
        split_dir = snapshot / split
        split_files = sorted(split_dir.glob("*.jsonl"))
        if not split_files:
            raise CorpusPreparationError(f"snapshot contains no {split} JSONL files")
        files.extend((split, path) for path in split_files)
    return files


def canonical_record(
    source: dict[str, Any],
    *,
    dataset_revision: str,
    split: str,
    languoid: str,
    source_path: str,
    source_line: int,
) -> dict[str, Any]:
    required = ["id", "text", "iso_639_3", "iso_15924", "glottocode", "variant", "split"]
    missing = [field for field in required if field not in source]
    if missing:
        raise CorpusPreparationError(
            f"{source_path}:{source_line}: missing fields: {', '.join(missing)}"
        )
    if source["split"] != split:
        raise CorpusPreparationError(
            f"{source_path}:{source_line}: row split {source['split']!r} does not match {split!r}"
        )
    if not isinstance(source["text"], str) or not source["text"].strip():
        raise CorpusPreparationError(f"{source_path}:{source_line}: text must be non-empty")

    return {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "dataset_revision": dataset_revision,
        "split": split,
        "languoid": languoid,
        "id": str(source["id"]),
        "iso_639_3": source["iso_639_3"],
        "iso_15924": source["iso_15924"],
        "glottocode": source["glottocode"],
        "variant": source["variant"],
        "domain": source.get("domain"),
        "topic": source.get("topic"),
        "text": source["text"],
        "source_file": source_path,
        "source_line": source_line,
    }


def encode_record(record: dict[str, Any]) -> bytes:
    return (
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


class ShardWriter:
    def __init__(self, directory: pathlib.Path, shard_size: int) -> None:
        self.directory = directory
        self.shard_size = shard_size
        self.stream: TextIO | None = None
        self.path: pathlib.Path | None = None
        self.index = -1
        self.shard_count = 0
        self.total_count = 0
        self.shards: list[dict[str, Any]] = []
        self.canonical_digest = hashlib.sha256()

    def _open(self) -> None:
        self.index += 1
        self.path = self.directory / f"part-{self.index:05d}.jsonl"
        self.stream = self.path.open("w", encoding="utf-8", newline="\n")
        self.shard_count = 0

    def _close(self) -> None:
        if self.stream is None or self.path is None:
            return
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        first = self.total_count - self.shard_count
        self.shards.append(
            {
                **artifact(self.path, name=self.path.name),
                "sample_count": self.shard_count,
                "first_sample_index": first,
                "last_sample_index": self.total_count - 1,
            }
        )
        self.stream = None
        self.path = None

    def write(self, record: dict[str, Any]) -> None:
        if self.stream is None:
            self._open()
        assert self.stream is not None
        encoded = encode_record(record)
        self.stream.write(encoded.decode("utf-8"))
        self.canonical_digest.update(encoded)
        self.shard_count += 1
        self.total_count += 1
        if self.shard_count == self.shard_size:
            self._close()

    def finish(self) -> tuple[list[dict[str, Any]], str]:
        self._close()
        return self.shards, self.canonical_digest.hexdigest()


def build_corpus(
    spec_path: pathlib.Path,
    spec: dict[str, Any],
    snapshot: pathlib.Path,
    output: pathlib.Path,
    *,
    invoked_command: str,
) -> dict[str, Any]:
    if output.exists():
        raise CorpusPreparationError(
            f"output already exists: {output}; verify it with --verify-only or choose a new path"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = pathlib.Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    corpus_dir = temporary / "corpus"
    corpus_dir.mkdir()

    split_counts: Counter[str] = Counter()
    script_counts: Counter[str] = Counter()
    languoid_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    source_artifacts: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str]] = set()
    writer = ShardWriter(corpus_dir, spec["shard_size"])

    try:
        for split, path in source_files(snapshot, spec["splits"]):
            relative = path.relative_to(snapshot).as_posix()
            languoid = path.stem
            rows = 0
            with path.open(encoding="utf-8") as stream:
                for line_number, raw in enumerate(stream, 1):
                    try:
                        source = json.loads(raw)
                    except json.JSONDecodeError as error:
                        raise CorpusPreparationError(f"{relative}:{line_number}: invalid JSON: {error}") from error
                    record = canonical_record(
                        source,
                        dataset_revision=spec["dataset_revision"],
                        split=split,
                        languoid=languoid,
                        source_path=relative,
                        source_line=line_number,
                    )
                    identity = (split, languoid, record["id"])
                    if identity in identities:
                        raise CorpusPreparationError(
                            f"{relative}:{line_number}: duplicate identity {identity!r}"
                        )
                    identities.add(identity)
                    writer.write(record)
                    rows += 1
                    split_counts[split] += 1
                    script_counts[str(record["iso_15924"])] += 1
                    languoid_counts[languoid] += 1
                    if record["domain"] is not None:
                        domain_counts[str(record["domain"])] += 1
            source_artifacts.append(
                {
                    **artifact(path, name=relative),
                    "split": split,
                    "languoid": languoid,
                    "sample_count": rows,
                }
            )

        shards, canonical_sha256 = writer.finish()
        if not shards:
            raise CorpusPreparationError("FLORES+ corpus is empty")

        metadata_artifacts = []
        for name in ["README.md", "CHANGELOG.md", "bibliography.bib"]:
            path = snapshot / name
            if path.is_file():
                metadata_artifacts.append(artifact(path, name=name))

        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "source_dataset": {
                "dataset_id": spec["dataset_id"],
                "requested_revision": spec["dataset_revision"],
                "resolved_revision": snapshot.name,
                "dataset_version": spec["dataset_version"],
                "license": spec["license"],
                "splits": spec["splits"],
                "access_url": spec["access_url"],
                "source_spec": artifact(spec_path, name=spec_path.name),
                "metadata_artifacts": metadata_artifacts,
                "files": source_artifacts,
            },
            "corpus": {
                "format": "nanoembed-certification-jsonl",
                "format_version": CORPUS_SCHEMA_VERSION,
                "ordering": "spec split order, source filename byte order, source line order",
                "sample_count": writer.total_count,
                "languoid_count": len(languoid_counts),
                "shard_size": spec["shard_size"],
                "canonical_sha256": canonical_sha256,
                "split_counts": dict(sorted(split_counts.items())),
                "script_counts": dict(sorted(script_counts.items())),
                "domain_counts": dict(sorted(domain_counts.items())),
                "languoid_counts": dict(sorted(languoid_counts.items())),
                "shards": shards,
            },
            "generator": {
                "script": artifact(pathlib.Path(__file__).resolve(), name="tools/prepare_flores_accuracy_corpus.py"),
                "invoked_command": invoked_command,
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with manifest_path.open("rb") as stream:
            os.fsync(stream.fileno())
        integrity_path = temporary / "manifest.sha256"
        integrity_path.write_text(
            f"{sha256_file(manifest_path)}  manifest.json\n",
            encoding="ascii",
        )
        with integrity_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        return manifest
    except BaseException:
        writer._close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_corpus(output: pathlib.Path) -> dict[str, Any]:
    manifest_path = output / "manifest.json"
    integrity_path = output / "manifest.sha256"
    try:
        integrity_parts = integrity_path.read_text(encoding="ascii").split()
    except OSError as error:
        raise CorpusPreparationError(f"cannot read manifest integrity file {integrity_path}: {error}") from error
    if len(integrity_parts) != 2 or integrity_parts[1] != "manifest.json":
        raise CorpusPreparationError(f"invalid manifest integrity file: {integrity_path}")
    if sha256_file(manifest_path) != integrity_parts[0]:
        raise CorpusPreparationError("manifest SHA-256 mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CorpusPreparationError(f"cannot read corpus manifest {manifest_path}: {error}") from error

    digest = hashlib.sha256()
    count = 0
    for shard in manifest.get("corpus", {}).get("shards", []):
        path = output / "corpus" / shard["path"]
        if not path.is_file():
            raise CorpusPreparationError(f"missing corpus shard: {path}")
        actual = artifact(path, name=shard["path"])
        for field in ["size_bytes", "sha256"]:
            if actual[field] != shard[field]:
                raise CorpusPreparationError(
                    f"corpus shard {path} {field} mismatch: expected {shard[field]}, found {actual[field]}"
                )
        shard_count = 0
        with path.open("rb") as stream:
            for raw in stream:
                digest.update(raw)
                shard_count += 1
        if shard_count != shard["sample_count"]:
            raise CorpusPreparationError(
                f"corpus shard {path} count mismatch: expected {shard['sample_count']}, found {shard_count}"
            )
        count += shard_count

    corpus = manifest.get("corpus", {})
    if count != corpus.get("sample_count"):
        raise CorpusPreparationError(
            f"corpus count mismatch: expected {corpus.get('sample_count')}, found {count}"
        )
    if digest.hexdigest() != corpus.get("canonical_sha256"):
        raise CorpusPreparationError("canonical corpus SHA-256 mismatch")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="require the pinned dataset snapshot to already exist in the HF cache",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an existing derived corpus without accessing Hugging Face",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.verify_only:
            manifest = verify_corpus(args.out_dir)
            print(
                f"verified {manifest['corpus']['sample_count']} samples in "
                f"{len(manifest['corpus']['shards'])} shards at {args.out_dir}"
            )
            return 0

        spec = load_spec(args.spec)
        snapshot = resolve_snapshot(spec, local_files_only=args.local_files_only)
        command = shlex.join([sys.executable, *sys.argv])
        manifest = build_corpus(args.spec, spec, snapshot, args.out_dir, invoked_command=command)
        print(
            f"prepared {manifest['corpus']['sample_count']} samples from "
            f"{manifest['corpus']['languoid_count']} language varieties in "
            f"{len(manifest['corpus']['shards'])} shards at {args.out_dir}"
        )
        print(f"canonical SHA-256: {manifest['corpus']['canonical_sha256']}")
        return 0
    except CorpusPreparationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
