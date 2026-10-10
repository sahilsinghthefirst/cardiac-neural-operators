                      
                                                       

from __future__ import annotations

import csv
import json

import numpy as np

from magnetic_common import MAG_BASE, SIM_IDS, biot_savart_projection, sim_mag_dir


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    errors: list[str] = []
    run_biot_savart_sanity_checks(errors)
    for sim_id in SIM_IDS:
        out_dir = sim_mag_dir(sim_id)
        require(out_dir.exists(), f"Missing magnetic output directory {out_dir}", errors)
        if not out_dir.exists():
            continue
        paths = {
            "Bz": out_dir / "Bz.npy",
            "B_vec": out_dir / "B_vec.npy",
            "sensor_positions": out_dir / "sensor_positions.npy",
            "mig_curves": out_dir / "mig_curves.csv",
            "mig_summary": out_dir / "mig_summary.json",
            "magnetic_metadata": out_dir / "magnetic_metadata.json",
        }
        for label, path in paths.items():
            require(path.exists() and path.stat().st_size > 0, f"Missing/empty {label} for {sim_id}: {path}", errors)
        if not all(path.exists() for path in paths.values()):
            continue

        b_z = np.load(paths["Bz"])
        b_vec = np.load(paths["B_vec"])
        sensors = np.load(paths["sensor_positions"])
        require(b_z.ndim == 2, f"{sim_id} Bz should be [time, sensors], got {b_z.shape}", errors)
        require(b_vec.ndim == 3 and b_vec.shape[2] == 3, f"{sim_id} B_vec should be [time, sensors, 3], got {b_vec.shape}", errors)
        require(b_vec.shape[:2] == b_z.shape, f"{sim_id} B_vec/Bz shape mismatch: {b_vec.shape} vs {b_z.shape}", errors)
        require(sensors.ndim == 2 and sensors.shape[1] == 3, f"{sim_id} sensors should be [sensors, 3], got {sensors.shape}", errors)
        require(sensors.shape[0] == b_z.shape[1], f"{sim_id} sensor count mismatch", errors)
        require(np.all(np.isfinite(b_z)), f"{sim_id} Bz contains NaN/Inf", errors)
        require(np.all(np.isfinite(b_vec)), f"{sim_id} B_vec contains NaN/Inf", errors)

        curves = np.genfromtxt(paths["mig_curves"], delimiter=",", names=True)
        for col in ["MI", "DI", "RMS"]:
            require(col in curves.dtype.names, f"{sim_id} missing {col} in mig_curves.csv", errors)
        if all(col in curves.dtype.names for col in ["MI", "DI", "RMS"]):
            require(np.all(np.isfinite(curves["MI"])), f"{sim_id} MI contains NaN/Inf", errors)
            require(np.all((curves["MI"] >= -1e-9) & (curves["MI"] <= 1.0 + 1e-9)), f"{sim_id} MI outside [0, 1]", errors)
            require(np.allclose(curves["DI"], 1.0 - curves["MI"], atol=1e-8), f"{sim_id} DI != 1 - MI", errors)
            require(np.all(curves["RMS"] >= -1e-20), f"{sim_id} RMS contains negative values", errors)
        summary = json.loads(paths["mig_summary"].read_text())
        require(summary.get("sim_id") == sim_id, f"{sim_id} mig_summary sim_id mismatch", errors)

    manifest = MAG_BASE / "manifest.csv"
    require(manifest.exists() and manifest.stat().st_size > 0, f"Missing/empty magnetic manifest {manifest}", errors)
    if manifest.exists():
        rows = list(csv.DictReader(manifest.open()))
        ids = {row["sim_id"] for row in rows}
        for sim_id in SIM_IDS:
            require(sim_id in ids, f"Magnetic manifest missing {sim_id}", errors)
    if errors:
        print("Magnetic output checks failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Magnetic output checks passed.")
    return 0


def run_biot_savart_sanity_checks(errors: list[str]) -> None:
    centroids = np.array([[0.0, 0.0, 0.0]], dtype=float)
    areas = np.array([1.0], dtype=float)
    sensors_low = np.array([[0.0, 0.0, 0.05]], dtype=float)
    sensors_high = np.array([[0.0, 0.0, 0.10]], dtype=float)
    j_zero = np.zeros((2, 1, 3), dtype=float)
    j_x = np.array([[[1.0, 0.0, 0.0]]], dtype=float)

    b_zero = biot_savart_projection(j_zero, centroids, areas, sensors_low)
    require(np.allclose(b_zero, 0.0), "Biot-Savart sanity failed: zero J did not produce zero B", errors)

    b_pos = biot_savart_projection(j_x, centroids, areas, sensors_low)
    b_neg = biot_savart_projection(-j_x, centroids, areas, sensors_low)
    require(np.allclose(b_neg, -b_pos), "Biot-Savart sanity failed: reversing J did not reverse B", errors)

    b_high = biot_savart_projection(j_x, centroids, areas, sensors_high)
    require(
        float(np.mean(np.abs(b_high))) < float(np.mean(np.abs(b_pos))),
        "Biot-Savart sanity failed: higher sensor did not reduce mean |B|",
        errors,
    )


if __name__ == "__main__":
    raise SystemExit(main())
