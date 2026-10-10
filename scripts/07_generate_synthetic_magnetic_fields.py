                      
                                                                     

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from magnetic_common import (
    FIGURE_NAMES,
    ROOT,
    SIM_IDS,
    biot_savart_projection,
    generate_sensor_grid,
    load_mesh,
    load_vm_tn,
    now_utc,
    read_json,
    sim_ep_dir,
    sim_mag_dir,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-id", action="append", choices=SIM_IDS, help="Simulation ID to process; repeatable.")
    parser.add_argument("--sensor-nx", type=int, default=8)
    parser.add_argument("--sensor-ny", type=int, default=8)
    parser.add_argument("--sensor-height-m", type=float, default=0.05)
    parser.add_argument("--margin-fraction", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-18)
    return parser.parse_args()


def plot_bz_snapshot(sim_id: str, b_z: np.ndarray, sensors: np.ndarray, out_dir) -> int:
    rms = np.sqrt(np.mean(b_z * b_z, axis=1))
    tidx = int(np.argmax(rms))
    fig_dir = ROOT / "figures" / "magnetic_maps"
    fig_dir.mkdir(parents=True, exist_ok=True)
    prefix = FIGURE_NAMES[sim_id]
    fig_path = fig_dir / f"{prefix}_Bz_snapshot.png"
    plt.figure(figsize=(5.8, 5.0))
    sc = plt.scatter(sensors[:, 0] * 1000.0, sensors[:, 1] * 1000.0, c=b_z[tidx], s=95, cmap="coolwarm")
    plt.gca().set_aspect("equal")
    plt.xlabel("sensor x (mm)")
    plt.ylabel("sensor y (mm)")
    plt.title(f"{sim_id} Bz proxy at t index {tidx}")
    cbar = plt.colorbar(sc)
    cbar.set_label("Bz proxy")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()
    return tidx


def plot_mixed_pipeline_demo(tidx: int) -> None:
    sim_id = "ischemia_like_mixed_smoke"
    ep_dir = sim_ep_dir(sim_id)
    out_dir = sim_mag_dir(sim_id)
    mesh = load_mesh(ep_dir)
    vm_tn, _times, _debug = load_vm_tn(ep_dir / "vm.igb", mesh.pts_m.shape[0])
    j_eq = np.load(out_dir / "J_eq_elem.npy")
    centroids = np.load(out_dir / "element_centroids.npy")
    b_z = np.load(out_dir / "Bz.npy")
    sensors = np.load(out_dir / "sensor_positions.npy")
    tri = mtri.Triangulation(mesh.pts_m[:, 0] * 1000.0, mesh.pts_m[:, 1] * 1000.0, mesh.elements)

    fig_dir = ROOT / "figures" / "pipeline"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    pc0 = axes[0].tripcolor(tri, vm_tn[tidx], shading="gouraud", cmap="turbo")
    axes[0].set_title("Vm")
    axes[0].set_aspect("equal")
    fig.colorbar(pc0, ax=axes[0], label="mV")

    step = max(1, len(centroids) // 500)
    c = centroids[::step]
    j = j_eq[tidx, ::step]
    mag = np.linalg.norm(j[:, :2], axis=1)
    scale = 0.00018
    qx = np.divide(j[:, 0], mag, out=np.zeros_like(mag), where=mag > 0) * scale
    qy = np.divide(j[:, 1], mag, out=np.zeros_like(mag), where=mag > 0) * scale
    axes[1].tripcolor(tri, np.zeros(mesh.pts_m.shape[0]), shading="flat", alpha=0.06)
    axes[1].quiver(c[:, 0] * 1000.0, c[:, 1] * 1000.0, qx * 1000.0, qy * 1000.0, angles="xy", scale_units="xy", scale=1)
    axes[1].set_title("J_eq direction")
    axes[1].set_aspect("equal")

    pc2 = axes[2].scatter(sensors[:, 0] * 1000.0, sensors[:, 1] * 1000.0, c=b_z[tidx], s=80, cmap="coolwarm")
    axes[2].set_title("Bz sensor proxy")
    axes[2].set_aspect("equal")
    fig.colorbar(pc2, ax=axes[2], label="Bz proxy")

    for ax in axes:
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    fig.suptitle(f"Mixed EP to magnetic proxy demo, t index {tidx}")
    fig.tight_layout()
    fig.savefig(fig_dir / "mixed_ep_to_magnetic_demo.png", dpi=180)
    plt.close(fig)


def process_sim(sim_id: str, args: argparse.Namespace) -> None:
    ep_dir = sim_ep_dir(sim_id)
    out_dir = sim_mag_dir(sim_id)
    current_file = out_dir / "J_eq_elem.npy"
    if not current_file.exists():
        raise FileNotFoundError(f"Missing {current_file}; run scripts/06_compute_current_proxy.py first")
    j_eq = np.load(current_file)
    centroids = np.load(out_dir / "element_centroids.npy")
    areas = np.load(out_dir / "element_areas.npy")
    times = np.load(out_dir / "time_ms.npy") if (out_dir / "time_ms.npy").exists() else np.arange(j_eq.shape[0], dtype=float)

    current_metadata = read_json(out_dir / "current_proxy_metadata.json")
    mesh = load_mesh(ep_dir, current_metadata.get("coord_scale_to_m", 1e-6), current_metadata.get("coordinate_units_observed", "um"))
    sensors, sensor_metadata = generate_sensor_grid(
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
    write_json(out_dir / "sensor_metadata.json", sensor_metadata)
    tidx = plot_bz_snapshot(sim_id, b_z, sensors, out_dir)

    metadata = {
        "created_utc": now_utc(),
        "sim_id": sim_id,
        "condition": current_metadata.get("condition", ""),
        "source_ep_dir": str(ep_dir.relative_to(ROOT)),
        "magnetic_output_dir": str(out_dir.relative_to(ROOT)),
        "source_current_file": str(current_file.relative_to(ROOT)),
        "biot_savart": {
            "formula": "B = mu0/(4*pi) sum_e cross(J_e, s - x_e) / (|s - x_e|^3 + eps) * area_e",
            "mu0_over_4pi": 1e-7,
            "eps": args.eps,
            "weight": "triangle area; no tissue thickness calibration in v0",
        },
        "units_note": "Synthetic/proxy magnetic fields; D_eff/thickness/conductivity calibration is not finalized.",
        "B_vec_shape": list(b_vec.shape),
        "Bz_shape": list(b_z.shape),
        "Bz_snapshot_time_index": tidx,
        "time_start_ms": float(times[0]) if len(times) else None,
        "time_end_ms": float(times[-1]) if len(times) else None,
        "outputs": {
            "B_vec": str((out_dir / "B_vec.npy").relative_to(ROOT)),
            "Bz": str((out_dir / "Bz.npy").relative_to(ROOT)),
            "sensor_positions": str((out_dir / "sensor_positions.npy").relative_to(ROOT)),
            "sensor_metadata": str((out_dir / "sensor_metadata.json").relative_to(ROOT)),
        },
        "success": bool(np.all(np.isfinite(b_vec))),
    }
    write_json(out_dir / "magnetic_metadata.json", metadata)
    print(f"{sim_id}: wrote B fields to {out_dir}")


def main() -> int:
    args = parse_args()
    for sim_id in args.sim_id or SIM_IDS:
        process_sim(sim_id, args)
    if "ischemia_like_mixed_smoke" in (args.sim_id or SIM_IDS):
        meta = read_json(sim_mag_dir("ischemia_like_mixed_smoke") / "magnetic_metadata.json")
        plot_mixed_pipeline_demo(int(meta["Bz_snapshot_time_index"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

