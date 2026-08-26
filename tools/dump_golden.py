#!/usr/bin/env python3
"""Generate an auditable PyTorch/sentence-transformers embedding oracle.

The binary NEGD format stays at version 1 so existing C++ readers remain
compatible. Reproducibility metadata lives beside the fixture:

    <fixture>.provenance.json    complete generation provenance
    <fixture>.provenance.sha256  SHA-256 of both fixture and manifest

Unlike the historical generator, this command refuses mutable model identity:
``--revision`` is required and is resolved to the exact Hugging Face commit
whose local snapshot is loaded. The reference path is CPU-only, FP32 and uses
PyTorch deterministic algorithms.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import random
import shlex
import struct
import sys
import tempfile
from typing import Any, Iterable


SCHEMA_VERSION = 1
DEFAULT_LOCK = pathlib.Path(__file__).resolve().parents[1] / "requirements-golden.lock"
COMMIT_SHA_LENGTH = 40


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: pathlib.Path, *, name: str | None = None) -> dict[str, Any]:
    return {
        "path": name if name is not None else str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _normalise_distribution_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def read_exact_lock(path: pathlib.Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise RuntimeError(
                f"{path}:{line_number}: golden lock entries must use exact == pins"
            )
        package, version = (part.strip() for part in line.split("==", 1))
        if not package or not version or any(ch in package for ch in "[;<>=!~"):
            raise RuntimeError(f"{path}:{line_number}: unsupported lock entry: {line!r}")
        pins[_normalise_distribution_name(package)] = version
    if not pins:
        raise RuntimeError(f"golden dependency lock is empty: {path}")
    return pins


def verify_exact_environment(path: pathlib.Path) -> dict[str, str]:
    pins = read_exact_lock(path)
    installed: dict[str, str] = {}
    mismatches: list[str] = []
    for package, expected in sorted(pins.items()):
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "not-installed"
        installed[package] = actual
        if actual != expected:
            mismatches.append(f"{package}: expected {expected}, found {actual}")
    if mismatches:
        raise RuntimeError(
            "golden environment differs from requirements-golden.lock:\n  "
            + "\n  ".join(mismatches)
        )
    return installed


def resolve_snapshot(
    model_id: str, revision: str, *, local_files_only: bool
) -> tuple[pathlib.Path, str]:
    """Return a pinned local snapshot and its exact Hugging Face commit SHA."""
    from huggingface_hub import snapshot_download

    snapshot = pathlib.Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
    ).resolve()
    resolved_revision = snapshot.name.lower()
    if (
        len(resolved_revision) != COMMIT_SHA_LENGTH
        or any(ch not in "0123456789abcdef" for ch in resolved_revision)
    ):
        raise RuntimeError(
            "Hugging Face snapshot did not resolve to an exact 40-character "
            f"commit SHA: {snapshot}"
        )
    return snapshot, resolved_revision


def model_artifacts(snapshot: pathlib.Path) -> list[dict[str, Any]]:
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"resolved model snapshot has no files: {snapshot}")
    return [artifact(path, name=path.relative_to(snapshot).as_posix()) for path in files]


_POOL_ATTRS = {
    "cls": "pooling_mode_cls_token",
    "mean": "pooling_mode_mean_tokens",
    "last-token": "pooling_mode_lasttoken",
}


def force_pooling(model: Any, requested: str) -> str:
    modules = [
        module
        for module in model.modules()
        if all(hasattr(module, attribute) for attribute in _POOL_ATTRS.values())
    ]
    if len(modules) != 1:
        raise RuntimeError(
            "expected exactly one sentence-transformers pooling module, "
            f"found {len(modules)}"
        )
    pooling = modules[0]
    enabled = [
        name for name, attribute in _POOL_ATTRS.items() if bool(getattr(pooling, attribute))
    ]
    if requested == "model-default":
        if len(enabled) != 1:
            raise RuntimeError(
                "model-default pooling must resolve to exactly one supported mode "
                f"(cls, mean, last-token); found {enabled!r}"
            )
        resolved = enabled[0]
    else:
        resolved = requested

    # Assign every known Pooling mode explicitly. Sentence-transformers has
    # additional modes which must be disabled or the output concatenates more
    # than one pooled vector.
    all_mode_attributes = [
        name for name in vars(pooling) if name.startswith("pooling_mode_")
    ]
    for attribute in all_mode_attributes:
        setattr(pooling, attribute, attribute == _POOL_ATTRS[resolved])
    for name, attribute in _POOL_ATTRS.items():
        setattr(pooling, attribute, name == resolved)
    if hasattr(pooling, "pooling_mode"):
        pooling.pooling_mode = resolved
    return resolved


def canonical_command(
    *,
    model_id: str,
    resolved_revision: str,
    corpora: Iterable[pathlib.Path],
    output: pathlib.Path,
    batch_size: int,
    max_length: int,
    pooling: str,
    normalize: bool,
    seed: int,
    torch_threads: int,
    lock_path: pathlib.Path,
) -> str:
    args = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--model",
        model_id,
        "--revision",
        resolved_revision,
        "--corpus",
        *(str(path) for path in corpora),
        "--out",
        str(output),
        "--batch-size",
        str(batch_size),
        "--max-length",
        str(max_length),
        "--pooling",
        pooling,
        "--normalize" if normalize else "--no-normalize",
        "--seed",
        str(seed),
        "--torch-threads",
        str(torch_threads),
        "--requirements-lock",
        str(lock_path),
    ]
    return shlex.join(args)


def _write_bytes_atomically(path: pathlib.Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def encode_negd(texts: list[str], embeddings: Any) -> bytes:
    n_samples, n_embed = embeddings.shape
    result = bytearray(b"NEGD")
    result.extend(struct.pack("<III", 1, n_samples, n_embed))
    for text, embedding in zip(texts, embeddings, strict=True):
        text_bytes = text.encode("utf-8")
        result.extend(struct.pack("<I", len(text_bytes)))
        result.extend(text_bytes)
        result.extend(embedding.tobytes())
    return bytes(result)


def write_provenance(
    output: pathlib.Path,
    manifest: dict[str, Any],
) -> tuple[pathlib.Path, pathlib.Path]:
    manifest_path = pathlib.Path(f"{output}.provenance.json")
    integrity_path = pathlib.Path(f"{output}.provenance.sha256")
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_bytes_atomically(manifest_path, manifest_bytes)
    integrity = (
        f"fixture {sha256_file(output)}\n"
        f"manifest {hashlib.sha256(manifest_bytes).hexdigest()}\n"
    ).encode("ascii")
    _write_bytes_atomically(integrity_path, integrity)
    return manifest_path, integrity_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument(
        "--revision",
        required=True,
        help="HF revision to resolve; the manifest records and reuses its exact commit SHA",
    )
    parser.add_argument(
        "--corpus",
        type=pathlib.Path,
        nargs="+",
        default=[pathlib.Path("tests/corpus/eval.txt")],
        help="one or more corpus files, concatenated in order",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("tests/fixtures/golden/st-bge-small.bin"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--pooling", choices=["model-default", *_POOL_ATTRS], default="model-default"
    )
    normalize = parser.add_mutually_exclusive_group()
    normalize.add_argument("--normalize", dest="normalize", action="store_true")
    normalize.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--requirements-lock", type=pathlib.Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="resolve the requested revision from the local HF cache without network access",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_length <= 0 or args.torch_threads <= 0:
        parser.error("--batch-size, --max-length and --torch-threads must be positive")
    if args.seed < 0 or args.seed > 0xFFFFFFFF:
        parser.error("--seed must fit NumPy's deterministic uint32 seed range")
    return args


def main() -> int:
    args = parse_args()

    try:
        packages = verify_exact_environment(args.requirements_lock)
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer
    except (ImportError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        print(
            f"install the exact reference environment with: {sys.executable} -m pip "
            f"install -r {args.requirements_lock}",
            file=sys.stderr,
        )
        return 2

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)

    try:
        snapshot, resolved_revision = resolve_snapshot(
            args.model, args.revision, local_files_only=args.local_files_only
        )
        model = SentenceTransformer(
            str(snapshot),
            device="cpu",
            local_files_only=True,
            model_kwargs={"dtype": torch.float32, "attn_implementation": "eager"},
        )
        model.to(device="cpu", dtype=torch.float32)
        model.eval()
        model.max_seq_length = args.max_length
        resolved_pooling = force_pooling(model, args.pooling)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: cannot load exact model snapshot: {error}", file=sys.stderr)
        return 2

    texts: list[str] = []
    for corpus in args.corpus:
        with corpus.open(encoding="utf-8") as stream:
            texts.extend(line.rstrip("\n") for line in stream if line.strip())
    if not texts:
        print("error: corpus selection is empty", file=sys.stderr)
        return 2

    with torch.inference_mode():
        embeddings = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            normalize_embeddings=args.normalize,
            convert_to_numpy=True,
            device="cpu",
        ).astype("float32", copy=False)

    _write_bytes_atomically(args.out, encode_negd(texts, embeddings))
    command = canonical_command(
        model_id=args.model,
        resolved_revision=resolved_revision,
        corpora=args.corpus,
        output=args.out,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=resolved_pooling,
        normalize=args.normalize,
        seed=args.seed,
        torch_threads=args.torch_threads,
        lock_path=args.requirements_lock,
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "provenance_status": "verified",
        "fixture": {
            **artifact(args.out, name=args.out.name),
            "fixture_sha256": sha256_file(args.out),
            "format": "NEGD",
            "format_version": 1,
            "sample_count": int(embeddings.shape[0]),
            "embedding_dimension": int(embeddings.shape[1]),
        },
        "reference": {
            "framework": "sentence-transformers/pytorch",
            "model_id": args.model,
            "requested_revision": args.revision,
            "resolved_revision": resolved_revision,
            "device": "cpu",
            "dtype": "float32",
            "deterministic_algorithms": True,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "pooling": resolved_pooling,
            "normalize": args.normalize,
            "truncation": True,
            "max_length": args.max_length,
            "torch_threads": args.torch_threads,
            "attention_implementation": "eager",
        },
        "generator": {
            "command": command,
            "invoked_command": shlex.join([sys.executable, *sys.argv]),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "package_versions": packages,
            "script": artifact(
                pathlib.Path(__file__).resolve(), name="tools/dump_golden.py"
            ),
            "requirements_lock": artifact(
                args.requirements_lock.resolve(), name="requirements-golden.lock"
            ),
        },
        "inputs": [artifact(path.resolve(), name=str(path)) for path in args.corpus],
        "model_artifacts": model_artifacts(snapshot),
    }
    manifest_path, integrity_path = write_provenance(args.out, manifest)

    print(
        f"wrote {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]}, "
        f"revision={resolved_revision}, pooling={resolved_pooling}, "
        f"normalize={args.normalize}) to {args.out}"
    )
    print(f"provenance: {manifest_path}")
    print(f"integrity:  {integrity_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
