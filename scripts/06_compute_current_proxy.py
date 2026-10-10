                      
                                                                         

from __future__ import annotations

import argparse

import numpy as np

from magnetic_common import (
    MAG_BASE,
    ROOT,
    SIM_IDS,
    compute_element_gradients,
    load_mesh,
    load_vm_tn,
    now_utc,
    sim_ep_dir,
    sim_mag_dir,
    triangle_geometry,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-id", action="append", choices=SIM_IDS, help="Simulation ID to process; repeatable.")
    parser.add_argument("--d-eff", type=float, default=1.0, help="Effective scalar D for J_eq = -D grad(Vm).")
    parser.add_argument("--coord-scale-to-m", type=float, default=1e-6, help="Scale mesh coordinates to meters.")
    parser.add_argument("--coord-units", default="um", help="Observed units of mesh.pts coordinates.")
    return parser.parse_args()


def process_sim(sim_id: str, d_eff: float, coord_scale_to_m: float, coord_units: str) -> None:
    ep_dir = sim_ep_dir(sim_id)
    out_dir = sim_mag_dir(sim_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(ep_dir, coord_scale_to_m=coord_scale_to_m, coord_units=coord_units)
    centroids, areas, grad_coeffs = triangle_geometry(mesh)
    vm_tn, times, vm_debug = load_vm_tn(ep_dir / "vm.igb", node_count=mesh.pts_m.shape[0])
    grad = compute_element_gradients(vm_tn, mesh.elements, grad_coeffs)
    j_eq = (-d_eff * grad).astype(np.float32)

    np.save(out_dir / "J_eq_elem.npy", j_eq)
    np.save(out_dir / "element_centroids.npy", centroids.astype(np.float64))
    np.save(out_dir / "element_areas.npy", areas.astype(np.float64))
    np.save(out_dir / "time_ms.npy", times.astype(np.float64))
    np.save(out_dir / "element_region_tags.npy", mesh.tags.astype(np.int64))

    ep_metadata = {}
    ep_meta_file = ep_dir / "metadata.json"
    if ep_meta_file.exists():
        import json

        ep_metadata = json.loads(ep_meta_file.read_text())

    metadata = {
        "created_utc": now_utc(),
        "sim_id": sim_id,
        "condition": ep_metadata.get("condition", ""),
        "source_ep_dir": str(ep_dir.relative_to(ROOT)),
        "mesh_format_observed": "mesh.pts first line node count; mesh.elem first line element count; all elements are Tr n0 n1 n2 tag",
        "num_nodes": int(mesh.pts_m.shape[0]),
        "num_elements": int(mesh.elements.shape[0]),
        "element_type": "Tr",
        "coordinate_units_observed": coord_units,
        "coord_scale_to_m": coord_scale_to_m,
        "mesh_bbox_raw": {
            "min": mesh.pts_raw.min(axis=0).tolist(),
            "max": mesh.pts_raw.max(axis=0).tolist(),
        },
        "mesh_bbox_m": {
            "min": mesh.pts_m.min(axis=0).tolist(),
            "max": mesh.pts_m.max(axis=0).tolist(),
        },
        "source_definition": "J_eq_elem = -D_eff * grad(Vm), element-centered",
        "D_eff": d_eff,
        "gradient_method": "triangle element plane linear reconstruction using mesh.pts and mesh.elem connectivity",
        "vm_debug": vm_debug,
        "outputs": {
            "J_eq_elem": str((out_dir / "J_eq_elem.npy").relative_to(ROOT)),
            "element_centroids": str((out_dir / "element_centroids.npy").relative_to(ROOT)),
            "element_areas": str((out_dir / "element_areas.npy").relative_to(ROOT)),
            "time_ms": str((out_dir / "time_ms.npy").relative_to(ROOT)),
            "element_region_tags": str((out_dir / "element_region_tags.npy").relative_to(ROOT)),
        },
        "limitations": "Equivalent current proxy only; not true intracellular current reconstruction.",
        "success": True,
    }
    write_json(out_dir / "current_proxy_metadata.json", metadata)
    print(f"{sim_id}: wrote current proxy to {out_dir}")


def main() -> int:
    args = parse_args()
    MAG_BASE.mkdir(parents=True, exist_ok=True)
    sim_ids = args.sim_id or SIM_IDS
    for sim_id in sim_ids:
        process_sim(sim_id, args.d_eff, args.coord_scale_to_m, args.coord_units)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

