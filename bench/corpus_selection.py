"""Deterministic corpus-group loading and bounded selection.

The native benchmark consumes one line-oriented input file.  This module keeps
group semantics in the Python orchestration layer and produces the exact lines
that runner.py writes to a short-lived input file for one native invocation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import pathlib
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MANIFEST_SCHEMA_VERSION = 1
GROUP_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
EXPECTED_DUPLICATE_POLICY = {
    "within_group": "error",
    "across_groups": "preserve",
}


class CorpusSelectionError(ValueError):
    """The manifest or requested corpus selection is invalid."""


@dataclass(frozen=True)
class CorpusItem:
    text: str
    text_id: str
    source: str
    source_line: int


@dataclass(frozen=True)
class CorpusSelection:
    group: str
    group_size: int
    seed: int
    items: Tuple[CorpusItem, ...]
    selection_sha256: str
    sources: Tuple[str, ...]

    @property
    def selected_size(self) -> int:
        return len(self.items)

    def metadata(
        self,
        manifest_path: str,
        duplicate_policy: Mapping[str, str],
    ) -> Dict[str, Any]:
        return {
            "manifest": manifest_path,
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "group": self.group,
            "group_size": self.group_size,
            "selected_size": self.selected_size,
            "seed": self.seed,
            "selected_ids": [item.text_id for item in self.items],
            "selection_sha256": self.selection_sha256,
            "sources": list(self.sources),
            "duplicate_policy": dict(duplicate_policy),
        }


def stable_text_id(text: str) -> str:
    """Return an identity derived only from the exact UTF-8 input text."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _selection_digest(items: Sequence[CorpusItem]) -> str:
    # This is exactly the byte representation written to the native input file.
    data = "".join(item.text + "\n" for item in items).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _selection_rank(seed: int, item: CorpusItem) -> Tuple[bytes, str]:
    material = f"{seed}\0{item.text_id}".encode("utf-8")
    return hashlib.sha256(material).digest(), item.text_id


def load_manifest(path: pathlib.Path) -> Dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CorpusSelectionError(
            f"cannot read corpus manifest {path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CorpusSelectionError(
            f"invalid JSON in corpus manifest {path}: {exc}"
        ) from exc

    if not isinstance(manifest, dict):
        raise CorpusSelectionError("corpus manifest root must be an object")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise CorpusSelectionError(
            "unsupported corpus manifest schema_version "
            f"{manifest.get('schema_version')!r}; expected {MANIFEST_SCHEMA_VERSION}"
        )
    if manifest.get("duplicate_policy") != EXPECTED_DUPLICATE_POLICY:
        raise CorpusSelectionError(
            "corpus manifest duplicate_policy must be "
            f"{EXPECTED_DUPLICATE_POLICY!r}"
        )

    groups = manifest.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise CorpusSelectionError("corpus manifest groups must be a non-empty object")
    for name, spec in groups.items():
        if not isinstance(name, str) or not GROUP_NAME_RE.fullmatch(name):
            raise CorpusSelectionError(f"invalid corpus group name {name!r}")
        if not isinstance(spec, dict):
            raise CorpusSelectionError(f"corpus group {name!r} must be an object")
        sources = spec.get("sources")
        if (not isinstance(sources, list) or not sources or
                not all(isinstance(source, str) and source for source in sources)):
            raise CorpusSelectionError(
                f"corpus group {name!r} must contain a non-empty string sources list"
            )
    return manifest


def parse_group_requests(values: Iterable[str]) -> List[Tuple[str, Optional[int]]]:
    """Parse repeatable NAME[:N] values and reject ambiguous duplicates."""
    requests: List[Tuple[str, Optional[int]]] = []
    seen = set()
    for value in values:
        name, separator, count_text = value.partition(":")
        if not GROUP_NAME_RE.fullmatch(name):
            raise CorpusSelectionError(f"invalid corpus group selector {value!r}")
        if name in seen:
            raise CorpusSelectionError(
                f"corpus group {name!r} was requested more than once"
            )
        seen.add(name)

        count: Optional[int] = None
        if separator:
            if not count_text or ":" in count_text:
                raise CorpusSelectionError(f"invalid corpus group selector {value!r}")
            try:
                count = int(count_text)
            except ValueError as exc:
                raise CorpusSelectionError(
                    f"invalid sample count in corpus group selector {value!r}"
                ) from exc
            if count <= 0:
                raise CorpusSelectionError(
                    f"sample count for corpus group {name!r} must be positive"
                )
        requests.append((name, count))
    return requests


def resolve_group_requests(
    requested: Sequence[Tuple[str, Optional[int]]],
    default_group: str,
    samples_per_group: Optional[int],
    known_groups: Mapping[str, Any],
) -> List[Tuple[str, Optional[int]]]:
    """Resolve CLI requests, with NAME:N taking precedence over the global N."""
    if samples_per_group is not None and samples_per_group <= 0:
        raise CorpusSelectionError("--samples-per-group must be positive")
    effective = list(requested) if requested else [(default_group, None)]
    resolved: List[Tuple[str, Optional[int]]] = []
    for name, explicit_count in effective:
        if name not in known_groups:
            choices = ", ".join(sorted(known_groups))
            raise CorpusSelectionError(
                f"unknown corpus group {name!r}; available groups: {choices}"
            )
        count = explicit_count if explicit_count is not None else samples_per_group
        resolved.append((name, count))
    return resolved


def load_group(
    root: pathlib.Path,
    name: str,
    spec: Mapping[str, Any],
) -> Tuple[List[CorpusItem], Tuple[str, ...]]:
    items: List[CorpusItem] = []
    sources = tuple(spec["sources"])
    seen: Dict[str, CorpusItem] = {}
    for source in sources:
        source_path = root / source
        try:
            # C++ std::getline uses only LF as the delimiter. Avoid splitlines(),
            # which would silently treat several Unicode characters as extra
            # record boundaries and make Python select a different workload.
            lines = source_path.read_text(encoding="utf-8").split("\n")
        except OSError as exc:
            raise CorpusSelectionError(
                f"cannot read source {source!r} for corpus group {name!r}: {exc}"
            ) from exc
        for line_number, text in enumerate(lines, start=1):
            # Match nanoembed-bench: empty lines are ignored, whitespace-only
            # lines are real inputs.
            if not text:
                continue
            item = CorpusItem(text, stable_text_id(text), source, line_number)
            previous = seen.get(item.text_id)
            if previous is not None:
                raise CorpusSelectionError(
                    f"duplicate text in corpus group {name!r}: {source}:"
                    f"{line_number} duplicates {previous.source}:{previous.source_line}"
                )
            seen[item.text_id] = item
            items.append(item)

    if not items:
        raise CorpusSelectionError(f"corpus group {name!r} is empty")
    return items, sources


def select_group(
    root: pathlib.Path,
    name: str,
    spec: Mapping[str, Any],
    count: Optional[int],
    seed: int,
) -> CorpusSelection:
    if count is not None and count <= 0:
        raise CorpusSelectionError(
            f"sample count for corpus group {name!r} must be positive"
        )
    items, sources = load_group(root, name, spec)
    if count is None or count >= len(items):
        # There is no subset bias to remove when the entire group is requested.
        # Preserve the historical source order for full baseline runs.
        selected = tuple(items)
    else:
        ranked = sorted(items, key=lambda item: _selection_rank(seed, item))
        selected = tuple(ranked[:count])
    return CorpusSelection(
        group=name,
        group_size=len(items),
        seed=seed,
        items=selected,
        selection_sha256=_selection_digest(selected),
        sources=sources,
    )


def write_selection(path: pathlib.Path, selection: CorpusSelection) -> None:
    path.write_text(
        "".join(item.text + "\n" for item in selection.items),
        encoding="utf-8",
    )
