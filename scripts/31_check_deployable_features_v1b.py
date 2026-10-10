                      
                                                                 

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
READY_BASE = ROOT / "outputs" / "dataset_v1b_ready"
EXPECTED_COUNT = 70


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def numeric_keys(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    return [k for k in rows[0] if k not in {"sample_id", "condition"}]


def script_uses_forbidden_window_inputs(path: Path) -> list[str]:
    text = path.read_text().lower()
    forbidden = ["lat_file", "rt_file", "apd80", "apd90", "mask_file", "region_mask", "lat.dat", "rt_repol_proxy"]
    return [item for item in forbidden if item in text]


def main() -> int:
    problems: list[str] = []
    oracle_csv = READY_BASE / "features_mig_oracle_windowed.csv"
    oracle_json = READY_BASE / "features_mig_oracle_windowed.json"
    fixed_csv = READY_BASE / "features_mig_deployable_fixed.csv"
    fixed_json = READY_BASE / "features_mig_deployable_fixed.json"
    magnetic_csv = READY_BASE / "features_mig_deployable_magnetic_windows.csv"
    targets_csv = READY_BASE / "targets_ep.csv"
    script_path = ROOT / "scripts" / "29_compute_deployable_mig_features_v1b.py"

    for path in [oracle_csv, oracle_json, fixed_csv, fixed_json]:
        if not path.exists() or path.stat().st_size == 0:
            problems.append(f"missing or empty {path.relative_to(ROOT)}")

    fixed_rows = read_rows(fixed_csv)
    target_rows = read_rows(targets_csv)
    if len(fixed_rows) != EXPECTED_COUNT:
        problems.append(f"deployable fixed feature file has {len(fixed_rows)} rows, expected {EXPECTED_COUNT}")
    fixed_ids = {r["sample_id"] for r in fixed_rows}
    target_ids = {r["sample_id"] for r in target_rows}
    if fixed_ids != target_ids:
        problems.append("deployable fixed sample IDs do not align with targets_ep")

    for row in fixed_rows:
        for key in numeric_keys(fixed_rows):
            try:
                value = float(row[key])
            except ValueError:
                problems.append(f"{row['sample_id']}: nonnumeric deployable fixed feature {key}")
                continue
            if not np.isfinite(value):
                problems.append(f"{row['sample_id']}: nonfinite deployable fixed feature {key}")
        for prefix in ["full", "early_activation_fixed", "mid_recovery_fixed", "late_repolarization_fixed", "broad_st_t_fixed"]:
            mi = float(row[f"{prefix}_MI_mean"])
            di = float(row[f"{prefix}_DI_mean"])
            rms = float(row[f"{prefix}_RMS_mean"])
            if mi < -1e-9 or mi > 1.0 + 1e-9:
                problems.append(f"{row['sample_id']}: {prefix} MI_mean outside [0,1]")
            if not np.isclose(di, 1.0 - mi, atol=1e-8):
                problems.append(f"{row['sample_id']}: {prefix} DI_mean != 1 - MI_mean")
            if rms < -1e-18:
                problems.append(f"{row['sample_id']}: {prefix} RMS_mean negative")

    forbidden_hits = script_uses_forbidden_window_inputs(script_path)
    if forbidden_hits:
        problems.append(f"deployable feature script contains forbidden EP-ground-truth references: {forbidden_hits}")

    magnetic_rows = read_rows(magnetic_csv)
    magnetic_status = "present" if magnetic_rows else "missing"
    if magnetic_rows and len(magnetic_rows) != EXPECTED_COUNT:
        problems.append(f"magnetic-window feature file has {len(magnetic_rows)} rows, expected {EXPECTED_COUNT}")

    summary = {
        "created_utc": now_utc(),
        "oracle_csv_exists": oracle_csv.exists(),
        "oracle_json_exists": oracle_json.exists(),
        "deployable_fixed_rows": len(fixed_rows),
        "target_rows": len(target_rows),
        "sample_ids_align": fixed_ids == target_ids,
        "deployable_magnetic_windows": magnetic_status,
        "forbidden_script_references": forbidden_hits,
        "passed": not problems,
        "problem_count": len(problems),
        "problems": problems,
    }
    (READY_BASE / "deployable_features_quality_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
