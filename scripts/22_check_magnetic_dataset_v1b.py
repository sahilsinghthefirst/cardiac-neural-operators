                      
                                                                       

from __future__ import annotations

import csv
import json

import numpy as np

from magnetic_common import ROOT, write_csv, write_json


EP_DATASET = ROOT / "outputs" / "bocf_dataset_v1b_conditioned"
MAG_DATASET = ROOT / "outputs" / "magnetic_dataset_v1b"


def truthy(value) -> bool:
    return value is True or str(value).lower() == "true"


def load_csv(path):
    return list(csv.DictReader(path.open())) if path.exists() else []


def check_row(row: dict) -> dict:
    out_dir = ROOT / row["magnetic_output_dir"]
    result = {
        "sample_id": row["sample_id"],
        "condition": row["condition"],
        "success": truthy(row.get("success")),
        "passed": True,
        "notes": "",
    }
    problems: list[str] = []
    if not out_dir.exists():
        problems.append("missing magnetic output directory")

    required = [
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
    for name in required:
        path = out_dir / name
        if result["success"] and not path.exists():
            problems.append(f"missing {name}")
        elif path.exists() and path.stat().st_size == 0:
            problems.append(f"empty {name}")

    b_z = None
    if (out_dir / "Bz.npy").exists():
        b_z = np.load(out_dir / "Bz.npy")
        if b_z.ndim != 2:
            problems.append(f"Bz.npy shape {b_z.shape} is not [time, sensors]")
        if not np.all(np.isfinite(b_z)):
            problems.append("Bz.npy has NaN/Inf")
        if np.allclose(b_z, 0.0):
            problems.append("Bz.npy is all zero without documented exception")

    if (out_dir / "B_vec.npy").exists():
        b_vec = np.load(out_dir / "B_vec.npy")
        if b_vec.ndim != 3 or b_vec.shape[2] != 3:
            problems.append(f"B_vec.npy shape {b_vec.shape} is not [time, sensors, 3]")
        if b_z is not None and b_vec.ndim == 3 and b_vec.shape[:2] != b_z.shape:
            problems.append("B_vec and Bz time/sensor shapes differ")
        if not np.all(np.isfinite(b_vec)):
            problems.append("B_vec.npy has NaN/Inf")

    if (out_dir / "sensor_positions.npy").exists():
        sensors = np.load(out_dir / "sensor_positions.npy")
        if sensors.ndim != 2 or sensors.shape[1] != 3:
            problems.append(f"sensor_positions.npy shape {sensors.shape} is not [sensors, 3]")
        if b_z is not None and sensors.ndim == 2 and sensors.shape[0] != b_z.shape[1]:
            problems.append("sensor count differs from Bz sensor dimension")

    curves_file = out_dir / "mig_curves.csv"
    if curves_file.exists():
        curves = np.genfromtxt(curves_file, delimiter=",", names=True)
        if curves.size == 0:
            problems.append("mig_curves.csv is empty")
        else:
            mi = np.atleast_1d(curves["MI"])
            di = np.atleast_1d(curves["DI"])
            rms = np.atleast_1d(curves["RMS"])
            if not np.all(np.isfinite(mi)) or not np.all(np.isfinite(di)) or not np.all(np.isfinite(rms)):
                problems.append("MIG curves contain NaN/Inf")
            if np.any(mi < -1e-9) or np.any(mi > 1.0 + 1e-9):
                problems.append("MI outside [0, 1]")
            if not np.allclose(di, 1.0 - mi, atol=1e-8):
                problems.append("DI is not 1 - MI")
            if np.any(rms < -1e-18):
                problems.append("RMS is negative")
            if b_z is not None and len(mi) != b_z.shape[0]:
                problems.append("MIG curve length differs from Bz time dimension")

    if (out_dir / "mig_summary.json").exists():
        summary = json.loads((out_dir / "mig_summary.json").read_text())
        for key in ["mean_MI", "max_MI", "mean_DI", "max_DI", "mean_RMS", "max_RMS", "area_RMS"]:
            if key not in summary:
                problems.append(f"mig_summary missing {key}")

    result["passed"] = not problems
    result["notes"] = "; ".join(problems)
    return result


def main() -> int:
    manifest = MAG_DATASET / "manifest.csv"
    rows = load_csv(manifest)
    report_rows = [check_row(row) for row in rows]
    ep_rows = [r for r in load_csv(EP_DATASET / "manifest.csv") if truthy(r.get("success"))]
    manifest_success = [r for r in rows if truthy(r.get("success"))]
    ep_success_ids = {r["sample_id"] for r in ep_rows}
    mag_success_ids = {r["sample_id"] for r in manifest_success}
    missing_from_magnetic = sorted(ep_success_ids - mag_success_ids)
    coverage_complete = len(missing_from_magnetic) == 0
    passed = bool(rows) and all(r["passed"] for r in report_rows)
    summary = {
        "manifest": str(manifest.relative_to(ROOT)) if manifest.exists() else "",
        "ep_successful_samples": len(ep_rows),
        "magnetic_manifest_rows": len(rows),
        "magnetic_success_rows": len(manifest_success),
        "passed_rows": sum(1 for r in report_rows if r["passed"]),
        "coverage_complete": coverage_complete,
        "missing_successful_ep_samples": missing_from_magnetic,
        "passed": passed,
        "notes": "" if passed else "See magnetic_quality_report.csv for per-sample issues.",
    }
    write_json(MAG_DATASET / "magnetic_quality_report.json", summary)
    write_csv(MAG_DATASET / "magnetic_quality_report.csv", report_rows, ["sample_id", "condition", "success", "passed", "notes"])
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
