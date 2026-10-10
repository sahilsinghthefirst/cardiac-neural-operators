from __future__ import annotations

import json

import numpy as np
import pandas as pd

from v1e_500_common import (
    CONDITION_LABELS,
    PILOT_DIR,
    READY_DIR,
    ROOT,
    ensure_dirs,
    grouped_split_rows,
    no_group_leakage,
    write_json,
)


SAMPLE_INDEX = READY_DIR / "sample_index.csv"
SPLIT_JSON = READY_DIR / "splits" / "grouped_train_val_test.json"
SPLIT_CSV = READY_DIR / "splits" / "grouped_train_val_test.csv"
REPORT = READY_DIR / "dataset_quality_report.json"


def main() -> int:
    ensure_dirs()
    pilot = pd.read_csv(PILOT_DIR / "manifest.csv")
    targets = pd.read_csv(PILOT_DIR / "target_manifest.csv")
    combined = pd.read_csv(PILOT_DIR / "combined_input_manifest.csv")
    rows = combined.merge(targets, on=["base_sample_id", "condition"], how="inner", validate="many_to_one")
    meta_cols = [
        "base_sample_id",
        "condition",
        "apd_scale",
        "conductivity_scale",
        "patch_radius_mm",
        "patch_center_x",
        "patch_center_y",
        "stimulus_location",
        "output_dir",
    ]
    rows = rows.merge(pilot[[c for c in meta_cols if c in pilot.columns]], on=["base_sample_id", "condition"], how="left")
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

    missing = []
    shape_errors = []
    for _, row in rows.iterrows():
        for col in [
            "B_meas_file",
            "coordinate_channels_file",
            "sensor_mask_file",
            "LAT_grid_file",
            "APD90_grid_file",
            "patch_mask_grid_file",
            "valid_mask_grid_file",
            "patch_param_targets_file",
        ]:
            if not (ROOT / row[col]).exists():
                missing.append(f"{row['row_id']}:{col}")
        try:
            b = np.load(ROOT / row["B_meas_file"])
            coords = np.load(ROOT / row["coordinate_channels_file"])
            mask = np.load(ROOT / row["sensor_mask_file"])
            apd = np.load(ROOT / row["APD90_grid_file"])
            patch = np.load(ROOT / row["patch_mask_grid_file"])
            if b.ndim != 4 or b.shape[1:3] != apd.shape or patch.shape != apd.shape:
                shape_errors.append(f"{row['row_id']}:B/target shape")
            if coords.shape[:2] != apd.shape or coords.shape[-1] < 3:
                shape_errors.append(f"{row['row_id']}:coords shape")
            if mask.shape != b.shape[1:]:
                shape_errors.append(f"{row['row_id']}:sensor mask shape")
        except Exception as exc:
            shape_errors.append(f"{row['row_id']}:{exc!r}")
    leakage = no_group_leakage(split_rows)
    report = {
        "row_count": int(len(rows)),
        "base_sample_count": int(rows["base_sample_id"].nunique()),
        "condition_counts_base": pilot["condition"].value_counts().to_dict(),
        "variant_counts": rows["measurement_variant"].value_counts().to_dict(),
        "split_counts_rows": rows["split"].value_counts().to_dict(),
        "split_counts_base": split_payload["base_counts_by_split"],
        "missing_file_count": len(missing),
        "missing_files": missing[:50],
        "shape_error_count": len(shape_errors),
        "shape_errors": shape_errors[:50],
        "group_leakage": leakage,
        "target_input_guardrail": "Model inputs are B_meas plus coordinate channels only; patch params and target grids are supervised labels.",
        "passed": bool(len(rows) == 1500 and rows["base_sample_id"].nunique() == 500 and not missing and not shape_errors and not leakage),
    }
    write_json(REPORT, report)
    print(json.dumps({k: report[k] for k in ["row_count", "base_sample_count", "split_counts_base", "passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
