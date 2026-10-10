                      
                                                                

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v1d_field_common import CONDITION_LABELS, PILOT_DIR, READY_DIR, ROOT, ensure_dirs, grouped_split_rows, no_group_leakage, write_json


SAMPLE_INDEX = READY_DIR / "sample_index.csv"
SPLIT_JSON = READY_DIR / "splits" / "grouped_train_val_test.json"
SPLIT_CSV = READY_DIR / "splits" / "grouped_train_val_test.csv"
REPORT = READY_DIR / "dataset_quality_report.json"


def main() -> int:
    ensure_dirs()
    pilot = pd.read_csv(PILOT_DIR / "manifest.csv")
    targets = pd.read_csv(PILOT_DIR / "field_target_manifest.csv")
    mag = pd.read_csv(PILOT_DIR / "magnetic_input_manifest.csv")
    rows = mag.merge(targets, on=["base_sample_id", "condition"], how="inner", validate="many_to_one")
    rows = rows.merge(pilot[["base_sample_id", "condition", "apd_scale", "conductivity_scale", "patch_radius_mm"]], on=["base_sample_id", "condition"], how="left")
    rows = rows.sort_values(["base_sample_id", "measurement_variant"]).reset_index(drop=True)
    rows["condition_label"] = rows["condition"].map(CONDITION_LABELS)
    rows.to_csv(SAMPLE_INDEX, index=False)
    split_rows, split_payload = grouped_split_rows(rows.to_dict(orient="records"))
    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(split_rows).to_csv(SPLIT_CSV, index=False)
    SPLIT_JSON.write_text(json.dumps(split_payload, indent=2) + "\n")
    merged = rows.merge(pd.DataFrame(split_rows)[["field_sample_id", "split"]], on="field_sample_id", how="left")
    merged.to_csv(SAMPLE_INDEX, index=False)

    missing_files = []
    shape_errors = []
    for _, row in merged.iterrows():
        for col in [
            "B_meas_file",
            "coordinate_channels_file",
            "sensor_mask_file",
            "LAT_grid_file",
            "APD80_grid_file",
            "APD90_grid_file",
            "patch_mask_grid_file",
            "valid_mask_grid_file",
        ]:
            if not (ROOT / row[col]).exists():
                missing_files.append(f"{row['field_sample_id']}:{col}")
        try:
            b = np.load(ROOT / row["B_meas_file"])
            lat = np.load(ROOT / row["LAT_grid_file"])
            valid = np.load(ROOT / row["valid_mask_grid_file"])
            if b.shape[1:3] != lat.shape or lat.shape != valid.shape:
                shape_errors.append(row["field_sample_id"])
        except Exception:
            shape_errors.append(row["field_sample_id"])
    leakage = no_group_leakage(split_rows)
    report = {
        "row_count": int(len(merged)),
        "base_sample_count": int(merged["base_sample_id"].nunique()),
        "condition_counts_base": pilot["condition"].value_counts().to_dict(),
        "variant_counts": merged["measurement_variant"].value_counts().to_dict(),
        "split_counts": merged["split"].value_counts().to_dict(),
        "missing_files": missing_files[:50],
        "missing_file_count": len(missing_files),
        "shape_errors": shape_errors[:50],
        "shape_error_count": len(shape_errors),
        "group_leakage": leakage,
        "passed": bool(len(merged) > 0 and not missing_files and not shape_errors and not leakage),
        "leakage_guardrail": "Target fields are labels only; B_meas and coordinate channels are deployable inputs.",
    }
    write_json(REPORT, report)
    print(json.dumps({k: report[k] for k in ["row_count", "base_sample_count", "split_counts", "passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
