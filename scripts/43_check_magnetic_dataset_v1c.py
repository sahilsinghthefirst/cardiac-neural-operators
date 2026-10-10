                      
                                                              

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bocf_v1c_common import (
    FULL_MODE,
    PILOT_MODE,
    array_file,
    load_array,
    mode_paths,
    monotonic_storage_summary,
    read_csv_rows,
    rel,
    truthy,
    write_csv_rows,
    write_json,
)


def check_curves(path: Path) -> tuple[bool, list[str]]:
    problems = []
    if not path.exists() or path.stat().st_size == 0:
        return False, ["missing_mig_curves"]
    arr = np.genfromtxt(path, delimiter=",", names=True)
    for name in ["MI", "DI", "RMS"]:
        vals = np.atleast_1d(arr[name]).astype(float)
        if not np.all(np.isfinite(vals)):
            problems.append(f"{name}_nonfinite")
    mi = np.atleast_1d(arr["MI"]).astype(float)
    di = np.atleast_1d(arr["DI"]).astype(float)
    rms = np.atleast_1d(arr["RMS"]).astype(float)
    if np.any((mi < -1e-8) | (mi > 1.0 + 1e-8)):
        problems.append("MI_out_of_range")
    if not np.allclose(di, 1.0 - mi, atol=1e-6):
        problems.append("DI_not_1_minus_MI")
    if np.any(rms < -1e-20):
        problems.append("RMS_negative")
    return not problems, problems


def check_row(row: dict) -> dict:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / row["magnetic_output_dir"]
    problems = []
    bz_path = array_file(out_dir, "Bz")
    sensor_path = array_file(out_dir, "sensor_positions")
    if not truthy(row.get("success")):
        problems.append("manifest_row_unsuccessful")
    if not bz_path:
        problems.append("missing_Bz")
        bz = None
    else:
        bz = load_array(bz_path)
        if bz.ndim != 2:
            problems.append(f"Bz_bad_ndim:{bz.ndim}")
        if not np.all(np.isfinite(bz)):
            problems.append("Bz_nonfinite")
        if np.allclose(bz, 0.0):
            problems.append("Bz_all_zero")
    if not sensor_path:
        problems.append("missing_sensor_positions")
    else:
        sensors = load_array(sensor_path)
        if sensors.ndim != 2 or sensors.shape[1] != 3:
            problems.append(f"sensor_shape_bad:{sensors.shape}")
        if bz is not None and sensors.shape[0] != bz.shape[1]:
            problems.append("sensor_count_mismatch")
    ok_curves, curve_problems = check_curves(out_dir / "mig_curves.csv")
    if not ok_curves:
        problems.extend(curve_problems)
    for name in ["mig_summary.json", "magnetic_metadata.json", "current_proxy_metadata.json", "sensor_metadata.json"]:
        p = out_dir / name
        if not p.exists() or p.stat().st_size == 0:
            problems.append(f"missing:{name}")
    return {
        "sample_id": row["sample_id"],
        "condition": row["condition"],
        "success": not problems,
        "problem_count": len(problems),
        "problems": ";".join(problems),
        "magnetic_output_dir": row["magnetic_output_dir"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[PILOT_MODE, FULL_MODE], default=PILOT_MODE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = mode_paths(args.mode)
    mag_rows = read_csv_rows(paths["mag"] / "manifest.csv")
    ep_success = [r for r in read_csv_rows(paths["ep"] / "manifest.csv") if truthy(r.get("success"))]
    report_rows = [check_row(r) for r in mag_rows]
    failed = [r for r in report_rows if not r["success"]]
    problems = []
    if len(mag_rows) != len(ep_success):
        problems.append(f"magnetic rows {len(mag_rows)} != successful EP rows {len(ep_success)}")
    if failed:
        problems.append(f"{len(failed)} magnetic rows failed checks")
    debug_dirs = [p for p in paths["mag_debug"].glob("*") if p.is_dir()] if paths["mag_debug"].exists() else []
    debug_problems = []
    for d in debug_dirs:
        for name in ["J_eq_elem.npy", "B_vec.npy", "element_centroids.npy", "element_areas.npy"]:
            if not (d / name).exists() or (d / name).stat().st_size == 0:
                debug_problems.append(f"{d.name}:{name}")
    if len(ep_success) >= 40 and len(debug_dirs) < 40:
        debug_problems.append(f"debug subset has {len(debug_dirs)} dirs, expected at least 40")
    if debug_problems:
        problems.append(f"debug subset problems: {debug_problems[:10]}")
    payload = {
        "mode": args.mode,
        "manifest": rel(paths["mag"] / "manifest.csv"),
        "expected_rows": len(ep_success),
        "row_count": len(mag_rows),
        "passed_rows": len(report_rows) - len(failed),
        "failed_rows": len(failed),
        "debug_full_dir": rel(paths["mag_debug"]),
        "debug_full_count": len(debug_dirs),
        "storage": monotonic_storage_summary(paths["mag"], len(mag_rows)),
        "debug_storage": monotonic_storage_summary(paths["mag_debug"], max(len(debug_dirs), 1)),
        "passed": not problems,
        "problems": problems,
    }
    write_csv_rows(paths["mag"] / "magnetic_quality_report.csv", report_rows)
    write_json(paths["mag"] / "magnetic_quality_report.json", payload)
    print(json.dumps({k: payload[k] for k in ["passed", "row_count", "passed_rows", "problems"]}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
