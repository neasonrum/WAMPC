"""Read-only Garage/Bench2Drive inventory; never starts CUDA or CARLA."""

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
MODALITIES = {"rgb": ".jpg", "lidar": ".laz", "measurements": ".json.gz", "boxes": ".json.gz"}


def read_json(path):
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def frame_ids(directory, suffix):
    return {path.name[:-len(suffix)] for path in directory.iterdir() if path.name.endswith(suffix)}


def quality_passes(results):
    return not (
        (results["scores"]["score_composed"] < 100.0
         and results["num_infractions"] != len(results["infractions"]["min_speed_infractions"]))
        or results["status"] in {
            "Failed - Agent couldn't be set up", "Failed", "Failed - Simulation crashed", "Failed - Agent crashed"
        }
    )


def audit_dataset(data_root, inventory_frames=False, decode=False):
    package_root = data_root.parent
    manifest = read_json(package_root / "manifest.json")["archives"]
    counts = Counter()
    problems = []
    samples = []
    recorded_files = 0
    recorded_bytes = 0
    for archive in manifest:
        scenario = archive["scenario"]
        archive_path = package_root / "archives" / f"{scenario}.tar"
        marker = read_json(package_root / "checks" / "extraction" / f"{scenario}.json")
        if not archive_path.is_file() or archive_path.stat().st_size != archive["bytes"]:
            problems.append(f"Archive missing or size mismatch: {archive_path}")
        if marker.get("verified") is not True or marker.get("archive_bytes") != archive["bytes"]:
            problems.append(f"Extraction marker mismatch: {scenario}")
        recorded_files += marker["files"]
        recorded_bytes += marker["bytes"]
        scenario_root = data_root / scenario
        if not scenario_root.is_dir():
            problems.append(f"Missing scenario: {scenario_root}")
            continue
        counts["scenarios"] += 1
        routes = sorted(path for path in scenario_root.iterdir() if path.is_dir())
        sample_route = None
        for route in routes:
            counts["routes"] += 1
            result_path = route / "results.json.gz"
            if not result_path.is_file():
                counts["routes_without_results"] += 1
                problems.append(f"Missing route results: {route}")
            else:
                passed = not route.name.startswith("FAILED_") and quality_passes(read_json(result_path))
                counts["garage_quality_pass_routes"] += int(passed)
                if passed and sample_route is None:
                    sample_route = route
            for modality in MODALITIES:
                if not (route / modality).is_dir():
                    problems.append(f"Missing modality: {route / modality}")
            if inventory_frames:
                frames = {
                    name: frame_ids(route / name, suffix) if (route / name).is_dir() else set()
                    for name, suffix in MODALITIES.items()
                }
                for name, identifiers in frames.items():
                    counts[f"{name}_frames"] += len(identifiers)
                if not frames["rgb"] or any(identifiers != frames["rgb"] for identifiers in frames.values()):
                    counts["frame_mismatch_routes"] += 1
                    problems.append(f"Empty or unaligned sensor frame IDs: {route}")
        if sample_route is None:
            problems.append(f"No Garage-quality route to sample: {scenario}")
            continue
        identifiers = sorted(frame_ids(sample_route / "rgb", ".jpg"))
        for identifier in dict.fromkeys(identifiers[:1] + identifiers[-1:]):
            sample = {"route": str(sample_route.relative_to(data_root)), "frame": identifier}
            measurement = read_json(sample_route / "measurements" / f"{identifier}.json.gz")
            sample["measurement_keys"] = sorted(measurement)
            sample["box_count"] = len(read_json(sample_route / "boxes" / f"{identifier}.json.gz"))
            for modality, suffix in MODALITIES.items():
                path = sample_route / modality / f"{identifier}{suffix}"
                if not path.is_file() or path.stat().st_size == 0:
                    problems.append(f"Missing or empty sampled frame: {path}")
            if decode:
                import cv2
                import laspy
                image = cv2.imread(str(sample_route / "rgb" / f"{identifier}.jpg"))
                if image is None:
                    problems.append(f"Cannot decode RGB: {sample}")
                else:
                    sample["image_shape"] = list(image.shape)
                cloud = laspy.read(sample_route / "lidar" / f"{identifier}.laz")
                sample["lidar_points"] = len(cloud.points)
                if not len(cloud.points):
                    problems.append(f"Empty LiDAR: {sample}")
            samples.append(sample)
    return {
        "root": str(data_root), "expected_scenarios": len(manifest),
        "expected_archive_bytes": sum(archive["bytes"] for archive in manifest),
        "extraction_record_files": recorded_files, "extraction_record_bytes": recorded_bytes,
        "counts": dict(counts), "inventory_all_frame_names": inventory_frames,
        "sample_decoding": decode, "samples": samples, "problems": problems,
        "limits": "Archive sizes/markers and optional frame-name inventory; not a new full byte or hash verification.",
    }


def audit_evaluation(carla_root, routes_path):
    routes = ET.parse(routes_path).getroot().findall("route")
    towns = sorted({route.attrib["town"] for route in routes})
    maps_root = carla_root / "CarlaUE4" / "Content" / "Carla" / "Maps"
    maps = {town: sorted(str(path) for path in maps_root.rglob(f"{town}.umap")) for town in towns}
    classes = set()
    for path in (ROOT / "scenario_runner" / "srunner" / "scenarios").glob("*.py"):
        classes.update(node.name for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))
    required = {scenario.attrib["type"] for route in routes for scenario in route.findall("scenarios/scenario")}
    problems = []
    if len(routes) != 220 or len({route.attrib["id"] for route in routes}) != 220:
        problems.append("Expected exactly 220 unique evaluation routes")
    for town, paths in maps.items():
        if not paths:
            problems.append(f"Missing CARLA map: {town}")
        else:
            with Path(paths[0]).open("rb") as handle:
                if handle.read(4) != b"\xc1\x83\x2a\x9e":
                    problems.append(f"Invalid Unreal map header: {paths[0]}")
    for name in sorted(required - classes):
        problems.append(f"Missing scenario class: {name}")
    if not (carla_root / "CarlaUE4.sh").is_file():
        problems.append("Missing CarlaUE4.sh")
    return {
        "route_count": len(routes), "towns": towns, "map_files": maps,
        "required_scenario_types": sorted(required), "problems": problems,
        "runtime_tested": False, "author_protocol_equivalence_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-frames", action="store_true")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = read_json(ROOT / "configs" / "paths.json")
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "target": "BridgeDrive Table 2 Full Diffusion geo, DS 83.85 / SR 67.27%, no anchors",
        "dataset": audit_dataset(Path(paths["garage_root"]), args.inventory_frames, args.decode),
        "evaluation": audit_evaluation(Path(paths["carla_root"]), ROOT / paths["routes"]),
    }
    report["resource_checks_pass"] = not (report["dataset"]["problems"] or report["evaluation"]["problems"])
    report["exact_reproduction_ready"] = False
    report["blocker"] = "Author PDM-Lite Full Diffusion training/config/controller pipeline is not publicly supplied."
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({
            "output": str(args.output), "resource_checks_pass": report["resource_checks_pass"],
            "dataset_counts": report["dataset"]["counts"],
            "samples": len(report["dataset"]["samples"]), "route_count": report["evaluation"]["route_count"],
            "problems": report["dataset"]["problems"] + report["evaluation"]["problems"],
        }, indent=2))
    else:
        print(json.dumps(report, indent=2))
    return 0 if report["resource_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())