"""Benchmark artifact, build, and Linux environment fingerprints.

All collection in this module runs in the Python orchestrator before a native
benchmark is launched.  Hashing a multi-gigabyte model therefore cannot enter
the authoritative inference timing window.  Artifact hashes are mandatory;
best-effort host/build metadata is explicit when it cannot be discovered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
import pathlib
import platform
import re
import subprocess
from typing import Any, Dict, Iterable, Optional, Tuple


class FingerprintError(RuntimeError):
    """A required benchmark identity could not be collected safely."""


def sha256_file(path: pathlib.Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a required regular file, raising a stable benchmark error."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while True:
                chunk = source.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise FingerprintError(f"cannot hash required file {path}: {exc}") from exc
    return digest.hexdigest()


@dataclass
class FileHashCache:
    """Cache large-file hashes while detecting changes during collection."""

    _digests: Dict[Tuple[int, int, int, int], str] = field(default_factory=dict)

    def sha256(self, path: pathlib.Path) -> str:
        try:
            before = path.stat()
        except OSError as exc:
            raise FingerprintError(
                f"cannot stat required file {path}: {exc}"
            ) from exc
        if not path.is_file():
            raise FingerprintError(f"required fingerprint target is not a file: {path}")

        key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        digest = self._digests.get(key)
        if digest is None:
            digest = sha256_file(path)
            try:
                after = path.stat()
            except OSError as exc:
                raise FingerprintError(
                    f"cannot restat required file {path}: {exc}"
                ) from exc
            after_key = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            if after_key != key:
                raise FingerprintError(
                    f"required file changed while it was being hashed: {path}"
                )
            self._digests[key] = digest
        return digest


def _run_text(command: Iterable[str], cwd: Optional[pathlib.Path] = None) -> Optional[str]:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def collect_git_identity(root: pathlib.Path) -> Dict[str, Any]:
    sha = _run_text(("git", "rev-parse", "HEAD"), root)
    if sha is None:
        dirty: Optional[bool] = None
        collection_status = "unavailable"
    else:
        try:
            probe = subprocess.run(
                ("git", "status", "--porcelain", "--untracked-files=all"),
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            dirty = bool(probe.stdout) if probe.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            dirty = None
        collection_status = "collected" if dirty is not None else "partial"
    return {
        "collection_status": collection_status,
        "sha": sha,
        "dirty": dirty,
    }


def collect_ggml_identity(root: pathlib.Path) -> Dict[str, Any]:
    ggml_root = root / "third_party" / "ggml"
    sha = _run_text(("git", "rev-parse", "HEAD"), ggml_root)
    return {
        "collection_status": "collected" if sha is not None else "unavailable",
        "sha": sha,
    }


def _display_path(path: pathlib.Path, root: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except (OSError, ValueError):
        return str(path)


def _read_optional(path: pathlib.Path) -> Optional[str]:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return value or None


def filesystem_identity(path: pathlib.Path) -> Dict[str, Any]:
    """Best-effort mount identity; file content hashes remain authoritative."""
    try:
        result = subprocess.run(
            (
                "findmnt", "--json", "--target", str(path),
                "--output", "TARGET,SOURCE,FSTYPE,UUID",
            ),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            raise ValueError("findmnt failed")
        parsed = json.loads(result.stdout)
        if not isinstance(parsed, dict):
            raise ValueError("findmnt returned a non-object root")
        filesystems = parsed.get("filesystems", [])
        entry = filesystems[0] if filesystems else None
        if not isinstance(entry, dict):
            raise ValueError("findmnt returned no filesystem")
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        return {
            "collection_status": "unavailable",
            "mount_target": None,
            "source": None,
            "fs_type": None,
            "uuid": None,
        }
    return {
        "collection_status": "collected",
        "mount_target": entry.get("target"),
        "source": entry.get("source"),
        "fs_type": entry.get("fstype"),
        "uuid": entry.get("uuid"),
    }


def file_identity(
    path: pathlib.Path,
    root: pathlib.Path,
    hashes: FileHashCache,
    configured_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Identity for a required benchmark artifact."""
    digest = hashes.sha256(path)
    try:
        stat = path.stat()
    except OSError as exc:
        raise FingerprintError(f"cannot stat required file {path}: {exc}") from exc
    return {
        "collection_status": "collected",
        "configured_path": configured_path,
        "resolved_path": _display_path(path, root),
        "size_bytes": stat.st_size,
        "sha256": digest,
        "filesystem": filesystem_identity(path),
    }


def _parse_cmake_cache(path: pathlib.Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return values
    for line in lines:
        if not line or line.startswith(("//", "#")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key, separator, _ = key_and_type.partition(":")
        if separator and key:
            values[key] = value
    return values


def _find_cmake_cache(binary: pathlib.Path, root: pathlib.Path) -> Optional[pathlib.Path]:
    candidates = []
    current = binary.resolve().parent
    root_resolved = root.resolve()
    for _ in range(6):
        candidates.append(current / "CMakeCache.txt")
        if current == root_resolved or current.parent == current:
            break
        current = current.parent
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _cmake_set_value(path: pathlib.Path, name: str) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(
        rf'^set\({re.escape(name)}\s+"?([^"\)]+)"?\)', text, re.MULTILINE
    )
    return match.group(1).strip() if match else None


_RELEVANT_CMAKE_OPTIONS = (
    "BUILD_SHARED_LIBS",
    "BUILD_TESTING",
    "CMAKE_CXX_FLAGS",
    "CMAKE_CXX_FLAGS_RELEASE",
    "CMAKE_C_FLAGS",
    "CMAKE_C_FLAGS_RELEASE",
    "CMAKE_INTERPROCEDURAL_OPTIMIZATION",
    "GGML_AMX_BF16",
    "GGML_AMX_INT8",
    "GGML_ACCELERATE",
    "GGML_AVX",
    "GGML_AVX2",
    "GGML_AVX512",
    "GGML_AVX512_BF16",
    "GGML_AVX512_VNNI",
    "GGML_BLAS",
    "GGML_BLAS_VENDOR",
    "GGML_BMI2",
    "GGML_CPU",
    "GGML_CPU_ALL_VARIANTS",
    "GGML_CPU_REPACK",
    "GGML_CUDA",
    "GGML_HIP",
    "GGML_F16C",
    "GGML_FMA",
    "GGML_LTO",
    "GGML_METAL",
    "GGML_NATIVE",
    "GGML_OPENCL",
    "GGML_OPENMP",
    "GGML_RPC",
    "GGML_SSE42",
    "GGML_SYCL",
    "GGML_VULKAN",
    "NANOEMBED_BUILD_BENCH",
)


def collect_build_identity(binary: pathlib.Path, root: pathlib.Path) -> Dict[str, Any]:
    try:
        cache_path = _find_cmake_cache(binary, root)
    except (OSError, RuntimeError):
        cache_path = None
    if cache_path is None:
        return {
            "collection_status": "unavailable",
            "cmake_cache_path": None,
            "build_type": None,
            "compiler": {
                "id": None,
                "version": None,
                "path": None,
            },
            "cmake_options": None,
        }

    cache = _parse_cmake_cache(cache_path)
    compiler_files = sorted(
        cache_path.parent.glob("CMakeFiles/*/CMakeCXXCompiler.cmake")
    )
    compiler_file = compiler_files[-1] if compiler_files else None
    compiler_id = (
        _cmake_set_value(compiler_file, "CMAKE_CXX_COMPILER_ID")
        if compiler_file else None
    )
    compiler_version = (
        _cmake_set_value(compiler_file, "CMAKE_CXX_COMPILER_VERSION")
        if compiler_file else None
    )
    options = {
        key: cache[key]
        for key in _RELEVANT_CMAKE_OPTIONS
        if key in cache
    }
    discovered = any((compiler_id, compiler_version, cache.get("CMAKE_BUILD_TYPE")))
    return {
        "collection_status": "collected" if discovered else "partial",
        "cmake_cache_path": _display_path(cache_path, root),
        "build_type": cache.get("CMAKE_BUILD_TYPE"),
        "compiler": {
            "id": compiler_id,
            "version": compiler_version,
            "path": cache.get("CMAKE_CXX_COMPILER"),
        },
        "cmake_options": options,
    }


def _cpu_model() -> Optional[str]:
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip() or None
    except OSError:
        pass
    return None


def _cpu_frequency_policy() -> Dict[str, Any]:
    policy_root = pathlib.Path("/sys/devices/system/cpu/cpufreq")
    try:
        policies = sorted(policy_root.glob("policy*"))
    except OSError:
        policies = []
    if not policies:
        return {
            "collection_status": "unavailable",
            "policy_count": None,
            "governors": None,
            "drivers": None,
        }
    governors = sorted({value for policy in policies
                        if (value := _read_optional(policy / "scaling_governor"))})
    drivers = sorted({value for policy in policies
                      if (value := _read_optional(policy / "scaling_driver"))})
    return {
        "collection_status": (
            "collected" if governors or drivers else "partial"
        ),
        "policy_count": len(policies),
        "governors": governors or None,
        "drivers": drivers or None,
    }


def _numa_identity() -> Dict[str, Any]:
    node_root = pathlib.Path("/sys/devices/system/node")
    online = _read_optional(node_root / "online")
    possible = _read_optional(node_root / "possible")
    try:
        count = len(list(node_root.glob("node[0-9]*")))
    except OSError:
        count = 0
    available = online is not None or possible is not None or count > 0
    return {
        "collection_status": "collected" if available else "unavailable",
        "online_nodes": online,
        "possible_nodes": possible,
        "node_count": count if count > 0 else None,
    }


def _total_ram_bytes() -> Optional[int]:
    try:
        for line in pathlib.Path("/proc/meminfo").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            if line.startswith("MemTotal:"):
                fields = line.split()
                return int(fields[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def collect_environment(working_path: pathlib.Path) -> Dict[str, Any]:
    total_ram = _total_ram_bytes()
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        page_size = None
    cpu_frequency = _cpu_frequency_policy()
    numa = _numa_identity()
    filesystem = filesystem_identity(working_path)
    core_values = (
        platform.system() or None,
        platform.release() or None,
        platform.machine() or None,
        _cpu_model(),
        os.cpu_count(),
        page_size,
        total_ram,
    )
    fully_collected = (
        all(value is not None for value in core_values)
        and cpu_frequency["collection_status"] == "collected"
        and numa["collection_status"] == "collected"
        and filesystem["collection_status"] == "collected"
    )
    return {
        "collection_status": "collected" if fully_collected else "partial",
        "os": core_values[0],
        "kernel": core_values[1],
        "machine": core_values[2],
        "cpu_model": core_values[3],
        "nproc": core_values[4],
        "page_size_bytes": page_size,
        "cpu_frequency_policy": cpu_frequency,
        "numa": numa,
        "total_ram_bytes": total_ram,
        "total_ram_collection_status": (
            "collected" if total_ram is not None else "unavailable"
        ),
        "working_directory_filesystem": filesystem,
    }
