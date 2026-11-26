"""Read the failed sample and check bounded CRC/SHA retries without skipping data."""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from .recovery_data import RecoveryDataset, ReopeningSnapshot, VirtualBatches
from .utils import ROOT, atomic_json, inspect_allocation, sha, verify_data


def integrity_checks(directory):
    directory.mkdir()
    payload = b"verified fixture payload"
    metadata = {"route": "fixture", "files": {"rgb/0069.jpg": {"sha256": hashlib.sha256(payload).hexdigest()}}}
    path = directory / "route_00001.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("rgb/0069.jpg", payload)
        archive.writestr("__manifest__.json", json.dumps(metadata))
    corrupted = directory / "corrupted.zip"
    raw = path.read_bytes()
    assert raw.count(payload) == 1
    corrupted.write_bytes(raw.replace(payload, b"X" + payload[1:]))

    class Injected(ReopeningSnapshot):
        def __init__(self, failures):
            self.handles = OrderedDict()
            self.directory = directory
            self.complete = {"routes": {"fixture": {"id": 1, "zip_bytes": path.stat().st_size}}}
            self.max_open_routes = 4
            self.failures, self.opens = failures, 0

        def _open(self, route):
            self.opens += 1
            if self.opens <= self.failures:
                archive = zipfile.ZipFile(corrupted)
                self.handles[route] = (archive, metadata)
                return archive, metadata
            return super()._open(route)

    transient = Injected(1)
    assert transient.read("fixture", "rgb/0069.jpg") == payload and transient.opens == 2
    transient.close()
    persistent = Injected(4)
    try:
        persistent.read("fixture", "rgb/0069.jpg")
    except RuntimeError:
        assert persistent.opens == 4
    else:
        raise AssertionError("Persistent corruption must fail, never skip/replace data")
    persistent.close()
    return {"one_crc_failure_reopened_and_verified": True, "persistent_corruption_fails_after_four_opens": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(1)
    proof = inspect_allocation(config, args.job_id, idle=True)
    metadata = verify_data(config)
    out = Path(config["output"])
    assert sha(config["resume_from"]) == config["resume_checkpoint_sha256"]
    state = torch.load(config["resume_from"], map_location="cpu", weights_only=False)
    assert state["dataset"] == metadata and state["optimizer"]["state"] and len(state["rank_states"]) == 8
    assert state["progress"]["updates"] == 2000
    expected = torch.randperm(237351, generator=torch.Generator().manual_seed(903))[:927 * 256].view(927, 8, 32)
    partition = []
    for group in config["virtual_rank_groups"]:
        batch = next(iter(VirtualBatches(237351, group, 901, 2, 146)))
        assert batch == expected[146, group].flatten().tolist()
        partition.extend(batch)
    assert len(partition) == 256 and sorted(partition) == sorted(expected[146].flatten().tolist())
    dataset = RecoveryDataset(config)
    assert tuple(map(int, dataset.samples[3347]))[2] == 69
    sample = dataset[3347]
    assert sample["rgb"].isfinite().all() and sample["path"].shape == (10, 2)
    snapshot = dataset.loader.sensor_snapshot
    route = dataset.routes[int(dataset.samples[3347, 0])]
    rereads = [hashlib.sha256(snapshot.read(route, "rgb/0069.jpg")).hexdigest() for _ in range(3)]
    assert len(set(rereads)) == 1
    checks = integrity_checks(out / "integrity_fixture")
    report = {"passed": True, "allocation": proof, "data_identity": metadata["protocol_sha256"],
              "resume_update": 2000, "optimizer_tensors": len(state["optimizer"]["state"]),
              "rng_streams": 8, "global_batch": 256, "physical_gpus": 6,
              "original_sample_partition_and_cursor_preserved": True,
              "failed_route": route, "failed_sample_rechecked": 3347, "member_sha256": rereads[0], **checks}
    atomic_json(out / "recovery_preflight.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
