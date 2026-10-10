from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v1e_500_common import PILOT_DIR, ROOT, no_group_leakage, write_json


FIELD_READY = PILOT_DIR / "field_ready"
SAMPLE_INDEX = FIELD_READY / "sample_index.csv"
SPLIT_JSON = FIELD_READY / "splits" / "grouped_train_val_test.json"
SPLIT_CSV = FIELD_READY / "splits" / "grouped_train_val_test.csv"
REPORT = FIELD_READY / "dataset_quality_report.json"


def main() -> int:
    FIELD_READY.mkdir(parents=True, exist_ok=True)
    combined = pd.read_csv(PILOT_DIR / "combined_input_manifest.csv")
    targets = pd.read_csv(PILOT_DIR / "target_manifest.csv")
    param_ready = pd.read_csv(PILOT_DIR / "parametric_ready" / "sample_index.csv")
    split_map = param_ready[["base_sample_id", "split"]].drop_duplicates()
    rows = combined.merge(targets, on=["base_sample_id", "condition"], how="inner", validate="many_to_one")
    rows = rows.merge(split_map, on="base_sample_id", how="left", validate="many_to_one")
    rows = rows.rename(columns={"measurement_variant": "combined_variant"})
    rows["measurement_variant"] = rows["combined_variant"]
    rows = rows.sort_values(["base_sample_id", "combined_variant"]).reset_index(drop=True)
    rows.to_csv(SAMPLE_INDEX, index=False)

    split_rows = rows[["row_id", "base_sample_id", "condition", "combined_variant", "split"]].rename(columns={"combined_variant": "measurement_variant"})
    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    split_rows.to_csv(SPLIT_CSV, index=False)
    split_payload = {
        "strategy": "preserved v1e 500 parametric grouped split by base_sample_id",
        "base_sample_count": int(rows["base_sample_id"].nunique()),
        "row_count": int(len(rows)),
        "base_counts_by_split": rows.drop_duplicates("base_sample_id")["split"].value_counts().to_dict(),
        "row_counts_by_split": rows["split"].value_counts().to_dict(),
        "base_split": dict(zip(split_map["base_sample_id"], split_map["split"])),
    }
    SPLIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_JSON.write_text(json.dumps(split_payload, indent=2) + "\n")

    required = [
        "B_meas_file",
        "coordinate_channels_file",
        "sensor_mask_file",
        "LAT_grid_file",
        "APD80_grid_file",
        "APD90_grid_file",
        "patch_mask_grid_file",
        "valid_mask_grid_file",
        "patch_param_targets_file",
    ]
    missing = []
    shape_errors = []
    for row in rows.itertuples(index=False):
        for col in required:
            val = getattr(row, col, "")
            if not val or not (ROOT / val).exists():
                missing.append(f"{row.row_id}:{col}")
        try:
            b = np.load(ROOT / row.B_meas_file)
            coords = np.load(ROOT / row.coordinate_channels_file)
            mask = np.load(ROOT / row.sensor_mask_file)
            lat = np.load(ROOT / row.LAT_grid_file)
            apd80 = np.load(ROOT / row.APD80_grid_file)
            apd90 = np.load(ROOT / row.APD90_grid_file)
            patch = np.load(ROOT / row.patch_mask_grid_file)
            valid = np.load(ROOT / row.valid_mask_grid_file)
            if b.ndim != 4 or b.shape[1:3] != lat.shape:
                shape_errors.append(f"{row.row_id}:B_meas/field")
            if coords.shape[:2] != lat.shape or coords.shape[-1] < 3:
                shape_errors.append(f"{row.row_id}:coords")
            if mask.shape != b.shape[1:]:
                shape_errors.append(f"{row.row_id}:sensor_mask")
            if not (lat.shape == apd80.shape == apd90.shape == patch.shape == valid.shape):
                shape_errors.append(f"{row.row_id}:target_shapes")
        except Exception as exc:
            shape_errors.append(f"{row.row_id}:{exc!r}")
    leakage = no_group_leakage(split_rows.to_dict(orient="records"))
    report = {
        "row_count": int(len(rows)),
        "base_sample_count": int(rows["base_sample_id"].nunique()),
        "variant_counts": rows["combined_variant"].value_counts().to_dict(),
        "split_counts_base": split_payload["base_counts_by_split"],
        "split_counts_rows": split_payload["row_counts_by_split"],
        "missing_file_count": len(missing),
        "missing_files": missing[:50],
        "shape_error_count": len(shape_errors),
        "shape_errors": shape_errors[:50],
        "group_leakage": leakage,
        "leakage_guardrail": "Inputs are B_meas, coordinate channels, and optional B_meas-derived descriptors only. Target fields and patch params are labels.",
        "passed": bool(len(rows) == 1500 and rows["base_sample_id"].nunique() == 500 and not missing and not shape_errors and not leakage),
    }
    write_json(REPORT, report)
    print(json.dumps({k: report[k] for k in ["row_count", "base_sample_count", "split_counts_base", "passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
