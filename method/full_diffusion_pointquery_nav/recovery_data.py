"""Bounded verified snapshot reopening and the original eight logical batches."""
import json
import socket
import time
import zipfile
import zlib

import torch

from method.full_diffusion2.snapshot import SensorSnapshot
from .data import PointDataset as OriginalDataset, ValidationBatches
from .utils import ROOT


class ReopeningSnapshot(SensorSnapshot):
    def _operation(self, route, name, exists):
        for attempt in range(4):
            try:
                return super()._operation(route, name, exists)
            except (zipfile.BadZipFile, EOFError, zlib.error, ValueError) as error:
                if isinstance(error, ValueError) and "SHA256 mismatch" not in str(error):
                    raise
                self._forget(route)
                if attempt == 3:
                    raise RuntimeError(f"Snapshot verification failed after 4 independent opens: {route}/{name}") from error
                print(json.dumps({"snapshot_integrity_retry": f"{route}/{name}",
                                  "attempt": attempt + 1, "host": socket.gethostname(),
                                  "error": repr(error)}), flush=True)
                time.sleep(.25 * 2 ** attempt)


class RecoveryDataset(OriginalDataset):
    def __init__(self, config, split="train"):
        super().__init__(config, split)
        original = self.loader.sensor_snapshot
        original.close()
        self.loader.sensor_snapshot = ReopeningSnapshot(original.directory, ROOT / config["retained_index"])


class VirtualBatches:
    def __init__(self, size, virtual_ranks, seed, epoch, offset=0):
        self.size, self.virtual_ranks = size, virtual_ranks
        self.seed, self.epoch, self.offset = seed, epoch, offset

    def __len__(self):
        return self.size // 256 - self.offset

    def __iter__(self):
        count = self.size // 256
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch))
        original = order[:count * 256].view(count, 8, 32)
        yield from original[self.offset:, self.virtual_ranks].flatten(1).tolist()
