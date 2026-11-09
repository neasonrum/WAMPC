"""Verified, immutable sensor snapshots on workspace storage; no source fallback."""

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import zipfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = ROOT / "data/full_diffusion/pdm_lite_seed901"
DEFAULT_SNAPSHOT = ROOT / "data/full_diffusion/pdm_lite_seed901_sensor_snapshot_v1"
KINDS = (("measurements", ".json.gz"), ("rgb", ".jpg"),
         ("rgb_augmented", ".jpg"), ("lidar", ".laz"))
TRANSIENT = {errno.ESTALE, errno.EIO, errno.ETIMEDOUT, errno.EAGAIN, errno.EINTR}


def retry_read(operation, label, attempts=6):
    """Reopen on transient failures, bounded to 7.75 s of backoff; never skip data."""
    for attempt in range(attempts):
        try:
            return operation()
        except OSError as error:
            if error.errno not in TRANSIENT or attempt + 1 == attempts:
                raise
            print(json.dumps({"io_retry": label, "errno": error.errno,
                              "attempt": attempt + 1, "host": socket.gethostname()}), flush=True)
            time.sleep(0.25 * 2**attempt)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha_file(path):
    def calculate():
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    return retry_read(calculate, str(path))


def index_identity(index):
    return {name: sha_file(Path(index) / name)
            for name in ("manifest.json", "routes.json", "train.npy", "val.npy")}


def prepare(index, destination):
    import numpy as np
    index, destination = Path(index), Path(destination)
    destination.resolve().relative_to(ROOT.resolve())
    destination.mkdir(parents=True, exist_ok=True)
    identity = index_identity(index)
    plan_path = destination / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["index_identity"] != identity:
            raise ValueError("Existing snapshot belongs to a different data index")
        return plan
    routes = json.loads((index / "routes.json").read_text())
    manifest = json.loads((index / "manifest.json").read_text())
    samples = np.concatenate([np.load(index / "train.npy"), np.load(index / "val.npy")])
    current = [set() for _ in routes]
    for rid, frame in samples:
        current[int(rid)].add(int(frame))
    records = []
    for rid, route in enumerate(routes):
        if not current[rid]:
            continue
        frames = sorted({frame - offset for frame in current[rid]
                         for offset in (0, 1, 2) if frame >= offset})
        records.append({"id": rid, "route": route, "current": sorted(current[rid]), "frames": frames})
    plan = {"format": 1, "index_identity": identity, "source_root": manifest["data_root"],
            "routes": records, "frames": sum(len(item["frames"]) for item in records),
            "history_frames": 2, "history_stride": 1}
    atomic_json(plan_path, plan)
    return plan


def build_route(record, plan, destination):
    import fcntl
    destination = Path(destination)
    rid = record["id"]
    path = destination / f"route_{rid:05}.zip"
    receipt_path = destination / f"route_{rid:05}.json"
    with (destination / f"route_{rid:05}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if (receipt["index_identity"] == plan["index_identity"] and path.is_file()
                    and path.stat().st_size == receipt["zip_bytes"]):
                return receipt
            raise ValueError(f"Incomplete or incompatible existing snapshot receipt: {receipt_path}")
        source = Path(plan["source_root"]) / record["route"]
        current = set(record["current"])
        temporary = path.with_name(path.name + f".{os.getpid()}.partial")
        files, missing = {}, []
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for frame in record["frames"]:
                for kind, extension in KINDS:
                    name = f"{kind}/{frame:04}{extension}"
                    try:
                        payload = retry_read(lambda: (source / name).read_bytes(), str(source / name))
                    except FileNotFoundError:
                        if frame in current:
                            raise
                        missing.append(name)
                        continue
                    files[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                    archive.writestr(name, payload)
            archive.writestr("__manifest__.json", json.dumps({"route": record["route"],
                             "files": files, "missing_historical": missing}, sort_keys=True))
        # Verify every copied member against the bytes read from the source before publishing.
        with zipfile.ZipFile(temporary) as archive:
            for name, metadata in files.items():
                payload = archive.read(name)
                if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
                    raise IOError(f"Snapshot copy verification failed: {temporary}:{name}")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        digest = sha_file(temporary)
        temporary.replace(path)
        receipt = {"route_id": rid, "route": record["route"], "file_count": len(files),
                   "source_bytes": sum(item["bytes"] for item in files.values()),
                   "missing_historical": len(missing), "zip_bytes": path.stat().st_size,
                   "sha256": digest, "index_identity": plan["index_identity"],
                   "verified_all_members": True, "host": socket.gethostname(),
                   "completed_at": datetime.now(timezone.utc).isoformat()}
        atomic_json(receipt_path, receipt)
        return receipt


def build(plan, destination, shard, shards, workers):
    started = time.monotonic()
    selected = [record for record in plan["routes"] if record["id"] % shards == shard]
    completed, source_bytes, failures = 0, 0, []
    status_path = Path(destination) / f"builder_{shard}.json"
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build_route, item, plan, destination): item for item in selected}
        for future in as_completed(futures):
            item = futures[future]
            try:
                receipt = future.result()
                completed += 1
                source_bytes += receipt["source_bytes"]
            except Exception as error:
                failures.append({"id": item["id"], "error": repr(error)})
                print(json.dumps({"snapshot_error": failures[-1]}), flush=True)
            if (completed + len(failures)) % 10 == 0 or completed + len(failures) == len(selected):
                status = {"shard": shard, "completed": completed, "total": len(selected),
                          "source_bytes": source_bytes, "failures": failures,
                          "elapsed_seconds": time.monotonic() - started, "host": socket.gethostname(),
                          "utc": datetime.now(timezone.utc).isoformat()}
                atomic_json(status_path, status)
                print(json.dumps({key: value for key, value in status.items() if key != "failures"}), flush=True)
    if failures:
        raise RuntimeError(f"Snapshot shard {shard} failed on {len(failures)} routes; preserve and retry")


def finalize(plan, destination):
    destination = Path(destination)
    receipts, source_bytes, file_count = {}, 0, 0
    for item in plan["routes"]:
        rid = item["id"]
        path = destination / f"route_{rid:05}.zip"
        receipt = json.loads((destination / f"route_{rid:05}.json").read_text())
        if (receipt["index_identity"] != plan["index_identity"] or not receipt["verified_all_members"]
                or receipt["route"] != item["route"] or path.stat().st_size != receipt["zip_bytes"]):
            raise ValueError(f"Invalid snapshot receipt for route {rid}")
        receipts[item["route"]] = {"id": rid, "zip_bytes": receipt["zip_bytes"], "sha256": receipt["sha256"]}
        source_bytes += receipt["source_bytes"]
        file_count += receipt["file_count"]
    complete = {"format": 1, "index_identity": plan["index_identity"], "routes": receipts,
                "frames": plan["frames"], "source_bytes": source_bytes, "file_count": file_count,
                "fallback_to_source": False, "verified_all_members": True,
                "completed_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(destination / "complete.json", complete)
    print(json.dumps({key: value for key, value in complete.items() if key != "routes"}), flush=True)


class SensorSnapshot:
    def __init__(self, directory, index, max_open_routes=4):
        self.directory = Path(directory)
        self.complete = json.loads(retry_read(lambda: (self.directory / "complete.json").read_text(),
                                              str(self.directory / "complete.json")))
        if self.complete["index_identity"] != index_identity(index):
            raise ValueError("Sensor snapshot index differs from the training/validation index")
        if self.complete.get("fallback_to_source") is not False or not self.complete.get("verified_all_members"):
            raise ValueError("The snapshot is incomplete or permits source fallback")
        self.max_open_routes = max_open_routes
        self.handles = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["handles"] = OrderedDict()
        return state

    def close(self):
        for archive, _ in self.handles.values():
            archive.close()
        self.handles.clear()

    def _forget(self, route):
        previous = self.handles.pop(route, None)
        if previous is not None:
            previous[0].close()

    def _open(self, route):
        if route not in self.handles:
            record = self.complete["routes"][route]
            path = self.directory / f"route_{record['id']:05}.zip"
            archive = zipfile.ZipFile(path)
            try:
                metadata = json.loads(archive.read("__manifest__.json"))
                if metadata["route"] != route or path.stat().st_size != record["zip_bytes"]:
                    raise ValueError(f"Snapshot route identity/size mismatch: {path}")
            except Exception:
                archive.close()
                raise
            self.handles[route] = (archive, metadata)
            while len(self.handles) > self.max_open_routes:
                _, (old, _) = self.handles.popitem(last=False)
                old.close()
        self.handles.move_to_end(route)
        return self.handles[route]

    def _operation(self, route, name, exists):
        def attempt():
            try:
                archive, metadata = self._open(route)
                if exists:
                    return name in metadata["files"]
                if name not in metadata["files"]:
                    raise FileNotFoundError(f"Sensor snapshot has no {route}/{name}; raw fallback is forbidden")
                payload = archive.read(name)
                if hashlib.sha256(payload).hexdigest() != metadata["files"][name]["sha256"]:
                    raise ValueError(f"Sensor snapshot SHA256 mismatch: {route}/{name}")
                return payload
            except OSError:
                self._forget(route)
                raise
        return retry_read(attempt, f"snapshot:{route}/{name}")

    def read(self, route, name):
        return self._operation(route, name, False)

    def exists(self, route, name):
        return self._operation(route, name, True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "build", "finalize"))
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--destination", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards or args.workers < 1:
        parser.error("Invalid snapshot builder parallelism")
    plan = prepare(args.index, args.destination)
    if args.phase == "prepare":
        print(json.dumps({"routes": len(plan["routes"]), "frames": plan["frames"]}), flush=True)
    elif args.phase == "build":
        build(plan, args.destination, args.shard, args.shards, args.workers)
    else:
        finalize(plan, args.destination)


if __name__ == "__main__":
    main()
