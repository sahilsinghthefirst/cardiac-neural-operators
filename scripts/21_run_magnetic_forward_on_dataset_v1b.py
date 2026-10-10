                      
                                                                  

                                                                            
                                                                         
                                                                       
   

from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path

import numpy as np

from magnetic_common import (
    ROOT,
    biot_savart_projection,
    compute_element_gradients,
    compute_mig_curves,
    generate_sensor_grid,
    load_mesh,
    load_vm_tn,
    now_utc,
    triangle_geometry,
    write_csv,
    write_json,
)


EP_DATASET = ROOT / "outputs" / "bocf_dataset_v1b_conditioned"
MAG_DATASET = ROOT / "outputs" / "magnetic_dataset_v1b"
CORE_OUTPUTS = [
    "J_eq_elem.npy",
    "element_centroids.npy",
    "element_areas.npy",
    "B_vec.npy",
    "Bz.npy",
    "sensor_positions.npy",
    "mig_curves.csv",
    "mig_summary.json",
    "current_proxy_metadata.json",
    "magnetic_metadata.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of successful EP samples. Balanced by condition when possible.")
    parser.add_argument("--conditions", nargs="+", default=None, help="Condition names to process.")
    parser.add_argument("--force", action="store_true", help="Recompute existing magnetic outputs.")
    parser.add_argument("--d-eff", type=float, default=1.0)
    parser.add_argument("--coord-scale-to-m", type=float, default=1e-6)
    parser.add_argument("--coord-units", default="um")
    parser.add_argument("--sensor-nx", type=int, default=8)
    parser.add_argument("--sensor-ny", type=int, default=8)
    parser.add_argument("--sensor-height-m", type=float, default=0.05)
    parser.add_argument("--margin-fraction", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-18)
    return parser.parse_args()


def read_ep_manifest() -> list[dict]:
    manifest = EP_DATASET / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing EP v1b manifest: {manifest}")
    rows = list(csv.DictReader(manifest.open()))
    return [r for r in rows if str(r.get("success")).lower() == "true"]


def select_rows(rows: list[dict], conditions: list[str] | None, limit: int | None) -> list[dict]:
    if conditions:
        allowed = set(conditions)
        rows = [r for r in rows if r["condition"] in allowed]
    if limit is None or limit >= len(rows):
        return rows

    order: list[str] = []
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        cond = row["condition"]
        if cond not in grouped:
            grouped[cond] = []
            order.append(cond)
        grouped[cond].append(row)

    selected: list[dict] = []
    while len(selected) < limit and any(grouped.values()):
        for cond in order:
            if grouped[cond] and len(selected) < limit:
                selected.append(grouped[cond].pop(0))
    return selected


def output_complete(out_dir: Path) -> bool:
    return all((out_dir / name).exists() and (out_dir / name).stat().st_size > 0 for name in CORE_OUTPUTS)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def process_sample(ep_row: dict, args: argparse.Namespace) -> dict:
    sample_id = ep_row["sample_id"]
    condition = ep_row["condition"]
    ep_dir = ROOT / ep_row["output_dir"]
    out_dir = MAG_DATASET / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_complete(out_dir) and not args.force:
        return manifest_row(sample_id, condition, ep_dir, out_dir, True, "skipped existing complete output")
    if args.force and out_dir.exists():
        for name in CORE_OUTPUTS + ["time_ms.npy", "element_region_tags.npy", "sensor_metadata.json", "error_metadata.json"]:
            path = out_dir / name
            if path.exists():
                path.unlink()

    try:
        ep_metadata = json.loads((ep_dir / "metadata.json").read_text()) if (ep_dir / "metadata.json").exists() else {}
        mesh = load_mesh(ep_dir, coord_scale_to_m=args.coord_scale_to_m, coord_units=args.coord_units)
        centroids, areas, grad_coeffs = triangle_geometry(mesh)
        vm_tn, times, vm_debug = load_vm_tn(ep_dir / "vm.igb", node_count=mesh.pts_m.shape[0])
        grad = compute_element_gradients(vm_tn, mesh.elements, grad_coeffs)
        j_eq = (-args.d_eff * grad).astype(np.float32)
        del grad

        np.save(out_dir / "J_eq_elem.npy", j_eq)
        np.save(out_dir / "element_centroids.npy", centroids.astype(np.float64))
        np.save(out_dir / "element_areas.npy", areas.astype(np.float64))
        np.save(out_dir / "time_ms.npy", times.astype(np.float64))
        np.save(out_dir / "element_region_tags.npy", mesh.tags.astype(np.int64))

        current_meta = {
            "created_utc": now_utc(),
            "sample_id": sample_id,
            "condition": condition,
            "source_ep_dir": rel(ep_dir),
            "num_nodes": int(mesh.pts_m.shape[0]),
            "num_elements": int(mesh.elements.shape[0]),
            "element_type": "Tr",
            "coordinate_units_observed": args.coord_units,
            "coord_scale_to_m": args.coord_scale_to_m,
            "mesh_bbox_raw": {"min": mesh.pts_raw.min(axis=0).tolist(), "max": mesh.pts_raw.max(axis=0).tolist()},
            "mesh_bbox_m": {"min": mesh.pts_m.min(axis=0).tolist(), "max": mesh.pts_m.max(axis=0).tolist()},
            "source_definition": "J_eq_elem = -D_eff * grad(Vm), element-centered",
            "D_eff": args.d_eff,
            "gradient_method": "triangle element plane linear reconstruction using mesh.pts and mesh.elem connectivity",
            "vm_debug": vm_debug,
            "outputs": {
                "J_eq_elem": rel(out_dir / "J_eq_elem.npy"),
                "element_centroids": rel(out_dir / "element_centroids.npy"),
                "element_areas": rel(out_dir / "element_areas.npy"),
                "time_ms": rel(out_dir / "time_ms.npy"),
                "element_region_tags": rel(out_dir / "element_region_tags.npy"),
            },
            "limitations": "Equivalent current proxy only; not true intracellular current reconstruction.",
            "success": True,
        }
        write_json(out_dir / "current_proxy_metadata.json", current_meta)

        sensors, sensor_meta = generate_sensor_grid(
            mesh,
            nx=args.sensor_nx,
            ny=args.sensor_ny,
            height_m=args.sensor_height_m,
            margin_fraction=args.margin_fraction,
        )
        b_vec = biot_savart_projection(j_eq, centroids, areas, sensors, eps=args.eps)
        b_z = b_vec[:, :, 2]
        np.save(out_dir / "sensor_positions.npy", sensors)
        np.save(out_dir / "B_vec.npy", b_vec)
        np.save(out_dir / "Bz.npy", b_z)
        write_json(out_dir / "sensor_metadata.json", sensor_meta)

        curves, mig_summary = compute_mig_curves(b_z, times)
        write_csv(out_dir / "mig_curves.csv", curves, ["time", "MI", "DI", "RMS"])
        mig_summary = {
            **mig_summary,
            "sample_id": sample_id,
            "condition": condition,
            "input_component": "Bz",
            "input_Bz_file": rel(out_dir / "Bz.npy"),
            "scientific_note": "MIG biomarkers from synthetic/proxy magnetic fields.",
        }
        write_json(out_dir / "mig_summary.json", mig_summary)

        magnetic_meta = {
            "created_utc": now_utc(),
            "sample_id": sample_id,
            "condition": condition,
            "source_ep_dir": rel(ep_dir),
            "magnetic_output_dir": rel(out_dir),
            "ep_conditioning": {
                "nbeats": ep_metadata.get("nbeats"),
                "BCL": ep_metadata.get("BCL"),
                "conditioning_used": ep_metadata.get("conditioning_used"),
                "final_beat_extraction": ep_metadata.get("final_beat_extraction"),
            },
            "source_current_file": rel(out_dir / "J_eq_elem.npy"),
            "biot_savart": {
                "formula": "B = mu0/(4*pi) sum_e cross(J_e, s - x_e) / (|s - x_e|^3 + eps) * area_e",
                "mu0_over_4pi": 1e-7,
                "eps": args.eps,
                "weight": "triangle area; no tissue thickness calibration in v1b magnetic dataset",
            },
            "sensor_grid": sensor_meta,
            "units_note": "Synthetic/proxy magnetic fields; D_eff/thickness/conductivity calibration is not finalized.",
            "B_vec_shape": list(b_vec.shape),
            "Bz_shape": list(b_z.shape),
            "outputs": {
                "B_vec": rel(out_dir / "B_vec.npy"),
                "Bz": rel(out_dir / "Bz.npy"),
                "sensor_positions": rel(out_dir / "sensor_positions.npy"),
                "mig_curves": rel(out_dir / "mig_curves.csv"),
                "mig_summary": rel(out_dir / "mig_summary.json"),
            },
            "scientific_note": "Synthetic magnetic-forward proxy for controlled benchmarking; no intracellular current reconstruction.",
            "success": bool(np.all(np.isfinite(b_vec)) and np.all(np.isfinite(j_eq))),
        }
        write_json(out_dir / "magnetic_metadata.json", magnetic_meta)
        return manifest_row(sample_id, condition, ep_dir, out_dir, magnetic_meta["success"], "")
    except Exception as exc:                                           
        error = {
            "created_utc": now_utc(),
            "sample_id": sample_id,
            "condition": condition,
            "source_ep_dir": rel(ep_dir),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "success": False,
        }
        write_json(out_dir / "error_metadata.json", error)
        return manifest_row(sample_id, condition, ep_dir, out_dir, False, repr(exc))


def manifest_row(sample_id: str, condition: str, ep_dir: Path, out_dir: Path, success: bool, notes: str) -> dict:
    return {
        "sample_id": sample_id,
        "condition": condition,
        "ep_output_dir": rel(ep_dir),
        "magnetic_output_dir": rel(out_dir),
        "Bz_file": rel(out_dir / "Bz.npy") if (out_dir / "Bz.npy").exists() else "",
        "B_vec_file": rel(out_dir / "B_vec.npy") if (out_dir / "B_vec.npy").exists() else "",
        "mig_curves_file": rel(out_dir / "mig_curves.csv") if (out_dir / "mig_curves.csv").exists() else "",
        "mig_summary_file": rel(out_dir / "mig_summary.json") if (out_dir / "mig_summary.json").exists() else "",
        "success": success,
        "notes": notes,
    }


def write_run_metadata(args: argparse.Namespace, selected_rows: list[dict], all_rows: list[dict]) -> None:
    data = {
        "created_utc": now_utc(),
        "source_ep_dataset": rel(EP_DATASET),
        "output_dataset": rel(MAG_DATASET),
        "selected_count": len(selected_rows),
        "total_successful_ep_samples": len(all_rows),
        "conditions": args.conditions,
        "limit": args.limit,
        "D_eff": args.d_eff,
        "coord_scale_to_m": args.coord_scale_to_m,
        "coord_units": args.coord_units,
        "sensor_grid": [args.sensor_nx, args.sensor_ny],
        "sensor_height_m": args.sensor_height_m,
        "margin_fraction": args.margin_fraction,
        "scientific_note": "J_eq is an equivalent current proxy; B fields are synthetic/proxy fields in relative units.",
    }
    write_json(MAG_DATASET / "run_metadata.json", data)


def main() -> int:
    args = parse_args()
    MAG_DATASET.mkdir(parents=True, exist_ok=True)
    all_rows = read_ep_manifest()
    selected = select_rows(all_rows, args.conditions, args.limit)
    rows = []
    for ep_row in selected:
        row = process_sample(ep_row, args)
        rows.append(row)
        print(f"{row['sample_id']}: {row['condition']} success={row['success']} {row['notes']}")
    write_csv(
        MAG_DATASET / "manifest.csv",
        rows,
        [
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
        ],
    )
    write_run_metadata(args, selected, all_rows)
    return 0 if rows and all(row["success"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
