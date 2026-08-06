"""Resolve a scenario's `model:` field to a local GGUF path.

Two forms are accepted:

    models/bge-small-en-v1.5-f16.gguf          repo-relative path
    hf:<repo_id>:<filename.gguf>               Hugging Face repo file

The HF form is resolved against the local cache only. Downloading is a
side effect a bench run should not perform silently -- a missing file prints
the exact command to fetch it and stops, so a run never blocks on the network
or quietly measures a different quantization than the one that was asked for.
"""

import pathlib
import sys
from typing import Optional


HF_PREFIX = "hf:"


def _cache_root() -> pathlib.Path:
    import os
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return pathlib.Path(hub)
    home = os.environ.get("HF_HOME")
    if home:
        return pathlib.Path(home) / "hub"
    return pathlib.Path.home() / ".cache" / "huggingface" / "hub"


def _find_in_cache(repo_id: str, filename: str) -> Optional[pathlib.Path]:
    repo_dir = _cache_root() / ("models--" + repo_id.replace("/", "--"))
    if not repo_dir.is_dir():
        return None
    # snapshots/<revision>/<filename>; take whichever revision has the file.
    for snapshot in sorted((repo_dir / "snapshots").glob("*")):
        candidate = snapshot / filename
        if candidate.exists():
            # Not .resolve()d: the snapshot path names the actual file, while
            # the blob it links to is an opaque hash.
            return candidate
    return None


def resolve(spec: str, root: pathlib.Path) -> pathlib.Path:
    """Return a local path for `spec`, or raise SystemExit with instructions."""
    if not spec.startswith(HF_PREFIX):
        path = (root / spec) if not pathlib.Path(spec).is_absolute() else pathlib.Path(spec)
        if not path.exists():
            raise SystemExit(f"error: model not found: {path}")
        return path

    body = spec[len(HF_PREFIX):]
    if ":" not in body:
        raise SystemExit(
            f"error: bad model spec {spec!r}; expected hf:<repo_id>:<filename.gguf>")
    repo_id, filename = body.rsplit(":", 1)

    found = _find_in_cache(repo_id, filename)
    if found is not None:
        return found

    raise SystemExit(
        f"error: {filename} of {repo_id} is not in the local Hugging Face cache.\n"
        f"       Bench runs never download implicitly. Fetch it first:\n"
        f"         huggingface-cli download {repo_id} {filename}")


if __name__ == "__main__":
    # Debug helper: bench/model_source.py <spec>
    print(resolve(sys.argv[1], pathlib.Path(__file__).resolve().parent.parent))
