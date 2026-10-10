                      
                                                                          
 
                                                                           
                                                                               
                                                                    
   

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from bocf_v1c_common import (
    FULL_MODE,
    PILOT_MODE,
    array_file,
    compute_bvec_projection,
    compute_bz_projection,
    compute_mig_curves_from_bz,
    default_workers,
    load_array,
    mode_paths,
    monotonic_storage_summary,
    now_utc,
    read_csv_rows,
    rel,
    save_array,
    truthy,
    write_csv_rows,
    write_json,
    write_mig_curves,
)
from magnetic_common import compute_element_gradients, generate_sensor_grid, load_mesh, load_vm_tn, triangle_geometry


MANIFEST_FIELDS = [
    "sample_id",
    "condition",
    "ep_output_dir",
    "magnetic_output_dir",
    "Bz_file",
    "B_vec_file",
    "mig_curves_file",
    "mig_summary_file",
    "success",
    "notes",
]


def debug_sample_ids(ep_rows: list[dict], max_per_condition: int = 10, max_total: int = 40) -> set[str]:
    out: set[str] = set()
    counts: dict[str, int] = defaultdict(int)
    for row in sorted(ep_rows, key=lambda r: r["sample_id"]):
        condition = row["condition"]
        if counts[condition] < max_per_condition and len(out) < max_total:
            out.add(row["sample_id"])
            counts[condition] += 1
    return out


def compute_source(ep_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, object]:
    mesh = load_mesh(ep_dir)
    centroids, areas, grad_coeffs = triangle_geometry(mesh)
    vm_tn, times, vm_debug = load_vm_tn(ep_dir / "vm.igb", mesh.pts_raw.shape[0])
    grad = compute_element_gradients(vm_tn, mesh.elements, grad_coeffs)
    j_eq = (-1.0 * grad).astype(np.float32)
    source_meta = {
        "source_definition": "J_eq_elem = -D_eff * grad(Vm), D_eff=1.0",
        "source_kind": "equivalent current proxy, not intracellular current reconstruction",
        "vm_debug": vm_debug,
        "num_elements": int(mesh.elements.shape[0]),
        "num_time_steps": int(j_eq.shape[0]),
        "num_nodes": int(mesh.pts_raw.shape[0]),
        "coord_scale_to_m": mesh.coord_scale_to_m,
        "coord_units": mesh.coord_units,
        "j_eq_min": float(np.min(j_eq)),
        "j_eq_max": float(np.max(j_eq)),
    }
    return j_eq, centroids, areas, times, mesh.tags, source_meta, mesh


def manifest_row(sample_id: str, condition: str, ep_dir: Path, mag_dir: Path, success: bool, notes: str) -> dict:
    bz = array_file(mag_dir, "Bz")
    bvec = array_file(mag_dir, "B_vec")
    return {
        "sample_id": sample_id,
        "condition": condition,
        "ep_output_dir": rel(ep_dir),
        "magnetic_output_dir": rel(mag_dir),
        "Bz_file": rel(bz) if bz else "",
        "B_vec_file": rel(bvec) if bvec else "",
        "mig_curves_file": rel(mag_dir / "mig_curves.csv") if (mag_dir / "mig_curves.csv").exists() else "",
        "mig_summary_file": rel(mag_dir / "mig_summary.json") if (mag_dir / "mig_summary.json").exists() else "",
        "success": success,
        "notes": notes,
    }


def process_row(row: dict, mag_base: Path, debug_base: Path, debug_ids: set[str], force: bool, compress: bool, store_heavy: bool) -> dict:
    sample_id = row["sample_id"]
    condition = row["condition"]
    ep_dir = Path(__file__).resolve().parents[1] / row["output_dir"]
    mag_dir = mag_base / sample_id
    meta_file = mag_dir / "magnetic_metadata.json"
    if meta_file.exists() and not force:
        try:
            meta = json.loads(meta_file.read_text())
            if meta.get("success"):
                return manifest_row(sample_id, condition, ep_dir, mag_dir, True, "")
        except Exception:
            pass
    mag_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        j_eq, centroids, areas, times, tags, source_meta, mesh = compute_source(ep_dir)
        sensors, sensor_meta = generate_sensor_grid(mesh, nx=8, ny=8, height_m=0.05, margin_fraction=0.1)
        bz = compute_bz_projection(j_eq, centroids, areas, sensors)
        save_array(mag_dir / "Bz.npy", bz, compress)
        save_array(mag_dir / "sensor_positions.npy", sensors, compress)
        save_array(mag_dir / "time_ms.npy", times, compress)
        write_json(mag_dir / "sensor_metadata.json", sensor_meta)
        curves, summary = compute_mig_curves_from_bz(bz, times)
        write_mig_curves(mag_dir / "mig_curves.csv", curves)
        write_json(mag_dir / "mig_summary.json", summary)
        heavy_path = None
        heavy_saved = bool(store_heavy or sample_id in debug_ids)
        if heavy_saved:
            out_heavy = debug_base / sample_id
            out_heavy.mkdir(parents=True, exist_ok=True)
            np.save(out_heavy / "J_eq_elem.npy", j_eq.astype(np.float32))
            np.save(out_heavy / "element_centroids.npy", centroids.astype(np.float64))
            np.save(out_heavy / "element_areas.npy", areas.astype(np.float64))
            np.save(out_heavy / "element_region_tags.npy", tags.astype(np.int32))
            b_vec = compute_bvec_projection(j_eq, centroids, areas, sensors)
            np.save(out_heavy / "B_vec.npy", b_vec.astype(np.float32))
            np.save(out_heavy / "sensor_positions.npy", sensors.astype(np.float64))
            np.save(out_heavy / "time_ms.npy", times.astype(np.float64))
            heavy_path = rel(out_heavy)
        current_meta = {
            **source_meta,
            "heavy_arrays_stored": heavy_saved,
            "heavy_array_output_dir": heavy_path,
            "compact_mode": True,
        }
        write_json(mag_dir / "current_proxy_metadata.json", current_meta)
        magnetic_meta = {
            "created_utc": now_utc(),
            "sample_id": sample_id,
            "condition": condition,
            "source_ep_dir": rel(ep_dir),
            "success": True,
            "runtime_seconds": time.perf_counter() - start,
            "storage_mode": "compact_bz_only",
            "B_fields_note": "Synthetic/proxy magnetic fields in relative units.",
            "scientific_note": "J_eq is an equivalent-current proxy, not intracellular current reconstruction.",
            "sensor_metadata": sensor_meta,
            "Bz_shape": list(bz.shape),
            "Bz_min": float(np.min(bz)),
            "Bz_max": float(np.max(bz)),
            "compressed_arrays": compress,
            "debug_heavy_arrays": heavy_path,
        }
        write_json(meta_file, magnetic_meta)
        return manifest_row(sample_id, condition, ep_dir, mag_dir, True, "")
    except Exception as exc:
        write_json(
            meta_file,
            {
                "created_utc": now_utc(),
                "sample_id": sample_id,
                "condition": condition,
                "source_ep_dir": rel(ep_dir),
                "success": False,
                "error": repr(exc),
                "runtime_seconds": time.perf_counter() - start,
            },
        )
        return manifest_row(sample_id, condition, ep_dir, mag_dir, False, repr(exc))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[PILOT_MODE, FULL_MODE], default=PILOT_MODE)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--compact", action="store_true", help="Compatibility flag; compact is the default.")
    parser.add_argument("--compress-arrays", action="store_true")
    parser.add_argument("--store-heavy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = mode_paths(args.mode)
    ep_rows = [r for r in read_csv_rows(paths["ep"] / "manifest.csv") if truthy(r.get("success"))]
    if args.conditions:
        keep = set(args.conditions)
        ep_rows = [r for r in ep_rows if r["condition"] in keep]
    if args.limit is not None:
        ep_rows = ep_rows[: args.limit]
    paths["mag"].mkdir(parents=True, exist_ok=True)
    debug_ids = debug_sample_ids(ep_rows)
    start = time.perf_counter()
    rows: list[dict] = []
    if args.workers <= 1:
        for row in ep_rows:
            out = process_row(row, paths["mag"], paths["mag_debug"], debug_ids, args.force, args.compress_arrays, args.store_heavy)
            rows.append(out)
            print(f"{out['sample_id']}: success={out['success']} {out['notes']}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(process_row, row, paths["mag"], paths["mag_debug"], debug_ids, args.force, args.compress_arrays, args.store_heavy)
                for row in ep_rows
            ]
            for fut in as_completed(futures):
                out = fut.result()
                rows.append(out)
                print(f"{out['sample_id']}: success={out['success']} {out['notes']}")
    rows.sort(key=lambda r: r["sample_id"])
    write_csv_rows(paths["mag"] / "manifest.csv", rows, MANIFEST_FIELDS)
    success_count = sum(1 for r in rows if truthy(r.get("success")))
    summary = {
        "created_utc": now_utc(),
        "mode": args.mode,
        "row_count": len(rows),
        "success_count": success_count,
        "failure_count": len(rows) - success_count,
        "workers": args.workers,
        "runtime_seconds": time.perf_counter() - start,
        "condition_counts": dict(Counter(r["condition"] for r in rows if truthy(r.get("success")))),
        "storage": monotonic_storage_summary(paths["mag"], len(rows)),
        "debug_full_output_dir": rel(paths["mag_debug"]),
        "debug_full_sample_count": len(debug_ids),
        "compact_storage": True,
    }
    write_json(paths["mag"] / "run_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ["success_count", "failure_count", "runtime_seconds"]}, indent=2))
    return 0 if success_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
