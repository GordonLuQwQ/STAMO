"""Unit tests for sampled model/checkpoint fingerprints."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stamo.renderer.utils.fingerprint import (
    sampled_file_fingerprint,
    sampled_tree_fingerprint,
)


class SampledFileFingerprintTests(unittest.TestCase):
    def test_is_deterministic_and_filename_is_part_of_the_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "weights-a.bin"
            path.write_bytes(b"deterministic checkpoint bytes")

            first = sampled_file_fingerprint(path, sample_bytes=8)
            second = sampled_file_fingerprint(path, sample_bytes=8)
            self.assertEqual(first, second)
            self.assertRegex(first, r"^[0-9a-f]{64}$")

            renamed = path.with_name("weights-b.bin")
            path.rename(renamed)
            self.assertNotEqual(
                first,
                sampled_file_fingerprint(renamed, sample_bytes=8),
            )

    def test_sampled_content_and_file_size_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "large.bin"
            file_size = 1024 * 1024
            path.write_bytes(b"\0" * file_size)
            baseline = sampled_file_fingerprint(path, sample_bytes=4096)

            with path.open("r+b") as handle:
                handle.seek(file_size // 2)
                handle.write(b"\x7f")
            changed_content = sampled_file_fingerprint(path, sample_bytes=4096)
            self.assertNotEqual(baseline, changed_content)

            with path.open("ab") as handle:
                handle.write(b"one-more-byte")
            changed_size = sampled_file_fingerprint(path, sample_bytes=4096)
            self.assertNotEqual(changed_content, changed_size)

    def test_large_file_reads_at_most_three_fixed_windows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "large.bin"
            file_size = 1024 * 1024
            sample_bytes = 1024
            path.write_bytes(b"x" * file_size)
            original_open = Path.open
            read_bytes = 0

            class CountingReader:
                def __init__(self, handle):
                    self._handle = handle

                def __enter__(self):
                    self._handle.__enter__()
                    return self

                def __exit__(self, *args):
                    return self._handle.__exit__(*args)

                def seek(self, *args):
                    return self._handle.seek(*args)

                def read(self, size=-1):
                    nonlocal read_bytes
                    self.assert_bounded_read(size)
                    result = self._handle.read(size)
                    read_bytes += len(result)
                    return result

                @staticmethod
                def assert_bounded_read(size):
                    if size < 0:
                        raise AssertionError("Fingerprint attempted an unbounded read")

            def counting_open(path_instance, *args, **kwargs):
                return CountingReader(
                    original_open(path_instance, *args, **kwargs)
                )

            with patch.object(Path, "open", new=counting_open):
                sampled_file_fingerprint(path, sample_bytes=sample_bytes)

            self.assertLess(read_bytes, file_size)
            self.assertLessEqual(read_bytes, 3 * sample_bytes)

    def test_missing_file_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing.bin"
            with self.assertRaises(FileNotFoundError):
                sampled_file_fingerprint(missing)


class SampledTreeFingerprintTests(unittest.TestCase):
    def test_traversal_and_explicit_input_order_are_stable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            nested = root / "nested"
            nested.mkdir()
            (root / "z.bin").write_bytes(b"z" * 100)
            (nested / "weights.bin").write_bytes(b"w" * 200)

            forward = sampled_tree_fingerprint(
                root,
                ["nested/weights.bin", "z.bin"],
                sample_bytes=16,
            )
            reverse = sampled_tree_fingerprint(
                root,
                ["z.bin", "nested\\weights.bin"],
                sample_bytes=16,
            )
            recursive = sampled_tree_fingerprint(root, sample_bytes=16)

            self.assertEqual(forward, reverse)
            self.assertEqual(forward, recursive)

    def test_content_size_and_relative_filename_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "transformer.bin"
            path.write_bytes(b"a" * 4096)
            baseline = sampled_tree_fingerprint(
                root,
                ["transformer.bin"],
                sample_bytes=64,
            )

            with path.open("r+b") as handle:
                handle.seek(2048)
                handle.write(b"b")
            changed_content = sampled_tree_fingerprint(
                root,
                ["transformer.bin"],
                sample_bytes=64,
            )
            self.assertNotEqual(baseline, changed_content)

            with path.open("ab") as handle:
                handle.write(b"larger")
            changed_size = sampled_tree_fingerprint(
                root,
                ["transformer.bin"],
                sample_bytes=64,
            )
            self.assertNotEqual(changed_content, changed_size)

            renamed = root / "renamed-transformer.bin"
            path.rename(renamed)
            changed_name = sampled_tree_fingerprint(
                root,
                ["renamed-transformer.bin"],
                sample_bytes=64,
            )
            self.assertNotEqual(changed_size, changed_name)

    def test_missing_required_tree_file_fails(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "present.bin").write_bytes(b"present")
            with self.assertRaises(FileNotFoundError):
                sampled_tree_fingerprint(
                    root,
                    ["present.bin", "missing.bin"],
                )


if __name__ == "__main__":
    unittest.main()
