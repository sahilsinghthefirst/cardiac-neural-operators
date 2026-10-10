                      
                                                                           

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUG_MAG = ROOT / "outputs" / "magnetic_dataset_v1b_augmented"
EXPECTED_ROWS = 70 * 11


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def truthy(value) -> bool:
    return value is True or str(value).lower() == "true"


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["augmented_sample_id", "base_sample_id", "condition", "variant", "success", "passed", "notes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def check_row(row: dict) -> dict:
    out_dir = ROOT / row["magnetic_output_dir"]
    problems: list[str] = []
    success = truthy(row.get("success"))
    required = ["Bz.npy", "B_vec.npy", "sensor_positions.npy", "sensor_mask.npy", "time_ms.npy", "mig_curves.csv", "mig_summary.json", "augmentation_metadata.json"]
    for name in required:
        path = out_dir / name
        if success and (not path.exists() or path.stat().st_size == 0):
            problems.append(f"missing or empty {name}")
    if not problems:
        b_z = np.load(out_dir / "Bz.npy")
        b_vec = np.load(out_dir / "B_vec.npy")
        sensors = np.load(out_dir / "sensor_positions.npy")
        mask = np.load(out_dir / "sensor_mask.npy").astype(bool)
        if b_z.ndim != 2:
            problems.append(f"Bz shape {b_z.shape} is not [time, sensors]")
        if b_vec.ndim != 3 or b_vec.shape[2] != 3:
            problems.append(f"B_vec shape {b_vec.shape} is not [time, sensors, 3]")
        if b_vec.shape[:2] != b_z.shape:
            problems.append("B_vec/Bz time-sensor dimensions differ")
        if sensors.shape != (b_z.shape[1], 3):
            problems.append(f"sensor_positions shape {sensors.shape} incompatible with Bz")
        if mask.shape != (b_z.shape[1],):
            problems.append(f"sensor_mask shape {mask.shape} incompatible with Bz")
        if int(np.sum(mask)) <= 0:
            problems.append("non-dropped sensor count is zero")
        valid_bz = b_z[:, mask]
        valid_bvec = b_vec[:, mask, :]
        if not np.all(np.isfinite(valid_bz)):
            problems.append("valid Bz sensors contain NaN/Inf")
        if not np.all(np.isfinite(valid_bvec)):
            problems.append("valid B_vec sensors contain NaN/Inf")
        if np.allclose(valid_bz, 0.0):
            problems.append("valid Bz is all zero")
        curves = np.genfromtxt(out_dir / "mig_curves.csv", delimiter=",", names=True)
        curves = np.atleast_1d(curves)
        mi = np.atleast_1d(curves["MI"]).astype(float)
        di = np.atleast_1d(curves["DI"]).astype(float)
        rms = np.atleast_1d(curves["RMS"]).astype(float)
        if not (np.all(np.isfinite(mi)) and np.all(np.isfinite(di)) and np.all(np.isfinite(rms))):
            problems.append("MIG curves contain NaN/Inf")
        if np.any(mi < -1e-9) or np.any(mi > 1.0 + 1e-9):
            problems.append("MI outside [0, 1]")
        if not np.allclose(di, 1.0 - mi, atol=1e-8):
            problems.append("DI != 1 - MI")
        if np.any(rms < -1e-18):
            problems.append("RMS is negative")
        summary = json.loads((out_dir / "mig_summary.json").read_text())
        if int(summary.get("num_valid_sensors", 0)) != int(np.sum(mask)):
            problems.append("mig_summary num_valid_sensors differs from sensor_mask")
    return {
        "augmented_sample_id": row["augmented_sample_id"],
        "base_sample_id": row["base_sample_id"],
        "condition": row["condition"],
        "variant": row["variant"],
        "success": success,
        "passed": not problems,
        "notes": "; ".join(problems),
    }


def main() -> int:
    manifest = AUG_MAG / "manifest.csv"
    rows = read_rows(manifest)
    report_rows = [check_row(row) for row in rows]
    passed = len(rows) == EXPECTED_ROWS and all(r["passed"] for r in report_rows)
    summary = {
        "created_utc": now_utc(),
        "expected_rows": EXPECTED_ROWS,
        "manifest_rows": len(rows),
        "successful_rows": sum(1 for r in rows if truthy(r.get("success"))),
        "passed_rows": sum(1 for r in report_rows if r["passed"]),
        "passed": passed,
        "notes": "" if passed else "See augmentation_quality_report.csv for per-row issues.",
    }
    (AUG_MAG / "augmentation_quality_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(AUG_MAG / "augmentation_quality_report.csv", report_rows)
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
