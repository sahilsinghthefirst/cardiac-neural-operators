                      
                                                              

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bocf_v1c_common import (
    FULL_MODE,
    PILOT_MODE,
    VARIANTS,
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
    mi = np.atleast_1d(arr["MI"]).astype(float)
    di = np.atleast_1d(arr["DI"]).astype(float)
    rms = np.atleast_1d(arr["RMS"]).astype(float)
    if not np.all(np.isfinite(mi)) or np.any((mi < -1e-8) | (mi > 1.0 + 1e-8)):
        problems.append("MI_bad")
    if not np.all(np.isfinite(di)) or not np.allclose(di, 1.0 - mi, atol=1e-6):
        problems.append("DI_bad")
    if not np.all(np.isfinite(rms)) or np.any(rms < -1e-20):
        problems.append("RMS_bad")
    return not problems, problems


def check_row(row: dict) -> dict:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / row["magnetic_output_dir"]
    problems = []
    if not truthy(row.get("success")):
        problems.append("manifest_row_unsuccessful")
    bz_path = array_file(out_dir, "Bz")
    mask_path = array_file(out_dir, "sensor_mask")
    sensor_path = array_file(out_dir, "sensor_positions")
    if not bz_path:
        problems.append("missing_Bz")
        bz = None
    else:
        bz = load_array(bz_path)
        if bz.ndim != 2:
            problems.append(f"Bz_bad_ndim:{bz.ndim}")
    if not mask_path:
        problems.append("missing_sensor_mask")
        mask = None
    else:
        mask = load_array(mask_path).astype(bool)
        if not np.any(mask):
            problems.append("no_valid_sensors")
    if not sensor_path:
        problems.append("missing_sensor_positions")
    else:
        sensors = load_array(sensor_path)
        if sensors.ndim != 2 or sensors.shape[1] != 3:
            problems.append(f"sensor_shape_bad:{sensors.shape}")
    if bz is not None and mask is not None:
        if bz.shape[1] != mask.shape[0]:
            problems.append("mask_sensor_count_mismatch")
        valid_bz = bz[:, mask]
        if not np.all(np.isfinite(valid_bz)):
            problems.append("valid_Bz_nonfinite")
        if np.allclose(valid_bz, 0.0):
            problems.append("valid_Bz_all_zero")
        if row["variant"].startswith("dropout_") or row["variant"] == "combined_medium":
            if not np.any(~mask):
                problems.append("dropout_variant_no_dropped_sensors")
    ok_curves, curve_problems = check_curves(out_dir / "mig_curves.csv")
    if not ok_curves:
        problems.extend(curve_problems)
    for name in ["mig_summary.json", "augmentation_metadata.json"]:
        if not (out_dir / name).exists() or (out_dir / name).stat().st_size == 0:
            problems.append(f"missing:{name}")
    return {
        "augmented_sample_id": row["augmented_sample_id"],
        "base_sample_id": row["base_sample_id"],
        "condition": row["condition"],
        "variant": row["variant"],
        "success": not problems,
        "problem_count": len(problems),
        "problems": ";".join(problems),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[PILOT_MODE, FULL_MODE], default=PILOT_MODE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = mode_paths(args.mode)
    rows = read_csv_rows(paths["aug"] / "manifest.csv")
    base_rows = [r for r in read_csv_rows(paths["mag"] / "manifest.csv") if truthy(r.get("success"))]
    report_rows = [check_row(r) for r in rows]
    failed = [r for r in report_rows if not r["success"]]
    expected = len(base_rows) * len(VARIANTS)
    problems = []
    if len(rows) != expected:
        problems.append(f"augmented rows {len(rows)} != expected {expected}")
    if failed:
        problems.append(f"{len(failed)} augmented rows failed checks")
    payload = {
        "mode": args.mode,
        "manifest": rel(paths["aug"] / "manifest.csv"),
        "expected_rows": expected,
        "row_count": len(rows),
        "passed_rows": len(rows) - len(failed),
        "failed_rows": len(failed),
        "storage": monotonic_storage_summary(paths["aug"], len(rows)),
        "passed": not problems,
        "problems": problems,
    }
    write_csv_rows(paths["aug"] / "augmentation_quality_report.csv", report_rows)
    write_json(paths["aug"] / "augmentation_quality_report.json", payload)
    print(json.dumps({k: payload[k] for k in ["passed", "row_count", "passed_rows", "problems"]}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
