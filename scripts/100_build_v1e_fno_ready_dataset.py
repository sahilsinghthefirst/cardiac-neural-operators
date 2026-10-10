                      
                                                              

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from v1e_direct_common import CONDITION_LABELS, PILOT_DIR, READY_DIR, ROOT, ensure_dirs, grouped_split_rows, no_group_leakage, write_json


SAMPLE_INDEX = READY_DIR / "sample_index.csv"
SPLIT_JSON = READY_DIR / "splits" / "grouped_train_val_test.json"
SPLIT_CSV = READY_DIR / "splits" / "grouped_train_val_test.csv"
REPORT = READY_DIR / "dataset_quality_report.json"


def main() -> int:
    ensure_dirs()
    pilot = pd.read_csv(PILOT_DIR / "manifest.csv")
    targets = pd.read_csv(PILOT_DIR / "target_manifest.csv")
    mag = pd.read_csv(PILOT_DIR / "direct_magnetic_manifest.csv")
    rows = mag.merge(targets, on=["base_sample_id", "condition"], how="inner", validate="many_to_one")
    rows = rows.merge(
        pilot[
            [
                "base_sample_id",
                "condition",
                "apd_scale",
                "conductivity_scale",
                "patch_radius_mm",
                "patch_center_x",
                "patch_center_y",
                "stimulus_location",
            ]
        ],
        on=["base_sample_id", "condition"],
        how="left",
    )
    rows = rows.sort_values(["base_sample_id", "measurement_variant"]).reset_index(drop=True)
    rows["condition_label"] = rows["condition"].map(CONDITION_LABELS)
    split_rows, split_payload = grouped_split_rows(rows.to_dict(orient="records"))
    split_df = pd.DataFrame(split_rows)
    rows = rows.merge(split_df[["row_id", "split"]], on="row_id", how="left")
    SAMPLE_INDEX.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(SAMPLE_INDEX, index=False)
    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(SPLIT_CSV, index=False)
    SPLIT_JSON.write_text(json.dumps(split_payload, indent=2) + "\n")

    missing_files = []
    shape_errors = []
    target_input_leakage = []
    for _, row in rows.iterrows():
        for col in [
            "B_meas_file",
            "coordinate_channels_file",
            "sensor_mask_file",
            "LAT_grid_file",
            "APD80_grid_file",
            "APD90_grid_file",
            "patch_mask_grid_file",
            "valid_mask_grid_file",
            "patch_param_targets_file",
        ]:
            if not (ROOT / row[col]).exists():
                missing_files.append(f"{row['row_id']}:{col}")
        try:
            b = np.load(ROOT / row["B_meas_file"])
            lat = np.load(ROOT / row["LAT_grid_file"])
            valid = np.load(ROOT / row["valid_mask_grid_file"])
            if b.shape[1:3] != lat.shape or lat.shape != valid.shape:
                shape_errors.append(row["row_id"])
        except Exception:
            shape_errors.append(row["row_id"])
    for col in rows.columns:
        if col.endswith("_grid_file") or col == "patch_param_targets_file":
            continue
        if col in {"LAT_grid", "APD90_grid", "patch_mask_grid"}:
            target_input_leakage.append(col)
    leakage = no_group_leakage(split_rows)
    report = {
        "row_count": int(len(rows)),
        "base_sample_count": int(rows["base_sample_id"].nunique()),
        "condition_counts_base": pilot["condition"].value_counts().to_dict(),
        "variant_counts": rows["measurement_variant"].value_counts().to_dict(),
        "split_counts": rows["split"].value_counts().to_dict(),
        "missing_file_count": len(missing_files),
        "missing_files": missing_files[:50],
        "shape_error_count": len(shape_errors),
        "shape_errors": shape_errors[:50],
        "group_leakage": leakage,
        "target_input_leakage_columns": target_input_leakage,
        "passed": bool(len(rows) > 0 and not missing_files and not shape_errors and not leakage and not target_input_leakage),
        "leakage_guardrail": "Model inputs are B_meas and coordinate channels only; target fields and patch parameters are labels only.",
    }
    write_json(REPORT, report)
    print(json.dumps({k: report[k] for k in ["row_count", "base_sample_count", "split_counts", "passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
