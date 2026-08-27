import hashlib
import pathlib
import tempfile
import unittest
from unittest import mock

from bench import fingerprint


class BenchFingerprintTest(unittest.TestCase):
    def test_sha256_file_and_cache_use_content_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "artifact.bin"
            path.write_bytes(b"nanoembed")
            alias = pathlib.Path(temp) / "same-model.gguf"
            alias.hardlink_to(path)
            expected = hashlib.sha256(b"nanoembed").hexdigest()
            cache = fingerprint.FileHashCache()
            with mock.patch.object(
                fingerprint, "sha256_file", wraps=fingerprint.sha256_file
            ) as hasher:
                self.assertEqual(cache.sha256(path), expected)
                self.assertEqual(cache.sha256(alias), expected)
                self.assertEqual(hasher.call_count, 1)

    def test_missing_required_file_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            missing = pathlib.Path(temp) / "missing.gguf"
            with self.assertRaises(fingerprint.FingerprintError):
                fingerprint.FileHashCache().sha256(missing)

    def test_optional_filesystem_identity_is_explicit_when_unavailable(self) -> None:
        with mock.patch.object(
            fingerprint.subprocess, "run", side_effect=FileNotFoundError()
        ):
            identity = fingerprint.filesystem_identity(pathlib.Path("/tmp"))
        self.assertEqual(identity["collection_status"], "unavailable")
        self.assertIsNone(identity["fs_type"])
        self.assertIsNone(identity["uuid"])

    def test_optional_git_identity_is_explicit_when_unavailable(self) -> None:
        with mock.patch.object(
            fingerprint.subprocess, "run", side_effect=FileNotFoundError()
        ):
            identity = fingerprint.collect_git_identity(pathlib.Path("/source"))
        self.assertEqual(identity["collection_status"], "unavailable")
        self.assertIsNone(identity["sha"])
        self.assertIsNone(identity["dirty"])

    def test_build_identity_reads_compiler_and_relevant_options(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            build = root / "build"
            binary = build / "bin" / "nanoembed-bench"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"binary")
            (build / "CMakeCache.txt").write_text(
                "CMAKE_BUILD_TYPE:STRING=Release\n"
                "CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++\n"
                "GGML_BLAS:BOOL=ON\n"
                "NANOEMBED_TEST_MODEL:FILEPATH=/secret/model.gguf\n",
                encoding="utf-8",
            )
            compiler_dir = build / "CMakeFiles" / "3.30.0"
            compiler_dir.mkdir(parents=True)
            (compiler_dir / "CMakeCXXCompiler.cmake").write_text(
                'set(CMAKE_CXX_COMPILER_ID "GNU")\n'
                'set(CMAKE_CXX_COMPILER_VERSION "14.2.0")\n',
                encoding="utf-8",
            )

            identity = fingerprint.collect_build_identity(binary, root)

        self.assertEqual(identity["build_type"], "Release")
        self.assertEqual(identity["compiler"]["id"], "GNU")
        self.assertEqual(identity["compiler"]["version"], "14.2.0")
        self.assertEqual(identity["cmake_options"], {"GGML_BLAS": "ON"})
        self.assertNotIn("NANOEMBED_TEST_MODEL", identity["cmake_options"])


if __name__ == "__main__":
    unittest.main()
