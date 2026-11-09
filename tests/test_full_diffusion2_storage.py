import errno
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from method.full_diffusion2.snapshot import (SensorSnapshot, DEFAULT_INDEX, DEFAULT_SNAPSHOT,
                                           build_route, finalize, index_identity, retry_read)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.index = self.root / "index"
        self.index.mkdir()
        for name in ("manifest.json", "routes.json", "train.npy", "val.npy"):
            (self.index / name).write_bytes(name.encode())
        self.cache = self.root / "snapshot"
        self.cache.mkdir()
        self.route = "Accident/example"
        self.payload = b"exact source bytes, not decoded or compressed"
        self.member = "rgb/0002.jpg"
        self.metadata = {"route": self.route, "files": {self.member: {
            "bytes": len(self.payload), "sha256": hashlib.sha256(self.payload).hexdigest()}}}
        self.write_archive()

    def write_archive(self):
        path = self.cache / "route_00000.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(self.member, self.payload)
            archive.writestr("__manifest__.json", json.dumps(self.metadata))
        complete = {"index_identity": index_identity(self.index), "fallback_to_source": False,
                    "verified_all_members": True,
                    "routes": {self.route: {"id": 0, "zip_bytes": path.stat().st_size}}}
        (self.cache / "complete.json").write_text(json.dumps(complete))

    def reader(self):
        reader = SensorSnapshot(self.cache, self.index)
        self.addCleanup(reader.close)
        return reader

    def test_stale_handle_reopens_archive_and_preserves_bytes(self):
        reader = self.reader()
        archive, _ = reader._open(self.route)
        with patch.object(archive, "read", side_effect=OSError(errno.ESTALE, "injected stale handle")), \
             patch("method.full_diffusion2.snapshot.time.sleep"):
            self.assertEqual(reader.read(self.route, self.member), self.payload)
        self.assertIsNone(archive.fp)

    def test_permanent_io_failure_has_finite_retries(self):
        operation = unittest.mock.Mock(side_effect=OSError(errno.ESTALE, "persistent"))
        with patch("method.full_diffusion2.snapshot.time.sleep") as sleep:
            with self.assertRaises(OSError):
                retry_read(operation, "injected", attempts=3)
        self.assertEqual(operation.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_missing_payload_has_no_source_fallback(self):
        reader = self.reader()
        self.assertFalse(reader.exists(self.route, "lidar/0001.laz"))
        with self.assertRaisesRegex(FileNotFoundError, "raw fallback is forbidden"):
            reader.read(self.route, "lidar/0001.laz")

    def test_sha256_mismatch_is_fatal(self):
        self.metadata["files"][self.member]["sha256"] = "0" * 64
        self.write_archive()
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.reader().read(self.route, self.member)

    def test_missing_completion_marker_rejects_partial_snapshot(self):
        (self.cache / "complete.json").rename(self.cache / "unfinished.json")
        with self.assertRaises(FileNotFoundError):
            SensorSnapshot(self.cache, self.index)

    def test_different_index_is_rejected(self):
        (self.index / "train.npy").write_bytes(b"changed training split")
        with self.assertRaisesRegex(ValueError, "index differs"):
            SensorSnapshot(self.cache, self.index)

    def test_spawn_serialization_does_not_share_open_handles(self):
        import pickle
        reader = self.reader()
        reader.read(self.route, self.member)
        restored = pickle.loads(pickle.dumps(reader))
        self.addCleanup(restored.close)
        self.assertEqual(len(restored.handles), 0)
        self.assertEqual(restored.read(self.route, self.member), self.payload)

    def test_builder_verifies_all_members_and_keeps_missing_history(self):
        source = self.root / "source"
        for kind, suffix in (("rgb", ".jpg"), ("rgb_augmented", ".jpg"),
                             ("lidar", ".laz"), ("measurements", ".json.gz")):
            folder = source / self.route / kind
            folder.mkdir(parents=True)
            (folder / ("0002" + suffix)).write_bytes(self.payload + kind.encode())
        destination = self.root / "built"
        destination.mkdir()
        record = {"id": 0, "route": self.route, "current": [2], "frames": [0, 1, 2]}
        plan = {"index_identity": index_identity(self.index), "source_root": str(source),
                "routes": [record], "frames": 3}
        receipt = build_route(record, plan, destination)
        self.assertEqual(receipt["file_count"], 4)
        self.assertEqual(receipt["missing_historical"], 8)
        self.assertTrue(receipt["verified_all_members"])
        finalize(plan, destination)
        reader = SensorSnapshot(destination, self.index)
        self.addCleanup(reader.close)
        self.assertEqual(reader.read(self.route, self.member), self.payload + b"rgb")
        self.assertFalse(reader.exists(self.route, "rgb/0001.jpg"))

    def test_snapshot_matches_raw_augmented_training_and_validation(self):
        if os.environ.get("WMMPC_SENSOR_PARITY_TEST") != "1":
            self.skipTest("Enable explicitly after production snapshot completion")
        from method.full_diffusion2.data import GarageTemporalDataset
        count = 0
        for split in ("train", "val"):
            raw = GarageTemporalDataset(DEFAULT_INDEX, split=split)
            cached = GarageTemporalDataset(DEFAULT_INDEX, split=split, sensor_snapshot=DEFAULT_SNAPSHOT)
            self.addCleanup(cached.sensor_snapshot.close)
            for epoch in (0, 4, 9):
                raw.epoch = cached.epoch = epoch
                for sample in np.linspace(0, len(raw) - 1, 8, dtype=int):
                    expected, actual = raw[int(sample)], cached[int(sample)]
                    self.assertEqual(expected.keys(), actual.keys())
                    for key, value in expected.items():
                        if torch.is_tensor(value):
                            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
                        else:
                            self.assertEqual(actual[key], value)
                    count += 1
        print(json.dumps({"sensor_parity_windows": count, "max_difference": 0,
                          "splits": ["train", "val"], "epochs": [0, 4, 9]}), flush=True)


if __name__ == "__main__":
    unittest.main()
