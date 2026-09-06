import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path

from stamo.renderer.utils.metadata_index import (
    JsonlImagePathCollection,
    JsonlImagePathIndex,
)


class JsonlImagePathIndexTests(unittest.TestCase):
    def _write_manifest(self, path, image_paths):
        with path.open("w", encoding="utf-8", newline="") as stream:
            for image_path in image_paths:
                stream.write(json.dumps({"image": image_path}) + "\n")

    def test_random_access_relative_paths_and_compact_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "train.jsonl"
            absolute = root / "absolute.png"
            self._write_manifest(
                manifest,
                ["images/first.png", str(absolute), "images/third.png"],
            )

            paths = JsonlImagePathIndex(manifest)
            self.assertEqual(len(paths), 3)
            self.assertEqual(paths[0], str((root / "images/first.png").resolve()))
            self.assertEqual(paths[1], str(absolute.resolve()))
            self.assertEqual(paths[-1], str((root / "images/third.png").resolve()))
            with self.assertRaises(IndexError):
                _ = paths[3]

            # Open both mmaps, then prove that the spawn payload excludes them.
            _ = paths[0]
            payload = pickle.dumps(paths)
            self.assertLess(len(payload), 4096)
            restored = pickle.loads(payload)
            self.assertIsNone(restored._source_map)
            self.assertIsNone(restored._index_map)
            self.assertEqual(restored[2], paths[2])
            restored.close()
            paths.close()

    def test_collection_concatenates_manifests_without_materializing_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            self._write_manifest(first, ["a.png", "b.png"])
            self._write_manifest(second, ["c.png"])

            paths = JsonlImagePathCollection()
            paths.add(first)
            paths.add(second)
            self.assertEqual(len(paths), 3)
            self.assertEqual(
                list(paths),
                [str((root / name).resolve()) for name in ("a.png", "b.png", "c.png")],
            )
            self.assertLess(len(pickle.dumps(paths)), 8192)
            paths.close()

    def test_source_change_invalidates_and_rebuilds_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "changing.jsonl"
            self._write_manifest(manifest, ["one.png"])
            first = JsonlImagePathIndex(manifest)
            first_index_stat = os.stat(first.index_path)
            first.close()

            with manifest.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"image": "two.png"}) + "\n")
            second = JsonlImagePathIndex(manifest)
            self.assertEqual(len(second), 2)
            self.assertGreaterEqual(
                os.stat(second.index_path).st_mtime_ns,
                first_index_stat.st_mtime_ns,
            )
            self.assertEqual(second[1], str((root / "two.png").resolve()))
            second.close()

    def test_invalid_records_fail_during_one_time_index_build(self):
        invalid_records = (
            "\n",
            "not-json\n",
            json.dumps(["not", "an", "object"]) + "\n",
            json.dumps({"wrong": "field"}) + "\n",
        )
        for record in invalid_records:
            with self.subTest(record=record):
                with tempfile.TemporaryDirectory() as directory:
                    manifest = Path(directory) / "invalid.jsonl"
                    manifest.write_text(record, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        JsonlImagePathIndex(manifest)


if __name__ == "__main__":
    unittest.main()
