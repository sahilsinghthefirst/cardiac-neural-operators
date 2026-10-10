                      
                                                                      

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import pandas as pd

from neural_data_common import (
    BzTensorDataset,
    check_no_group_leakage,
    load_bz_as_grid,
    load_neural_index,
    load_split_map,
    select_deployable_feature_columns,
)


READY_DIR = PROJECT_ROOT / "outputs" / "dataset_v1c_5000_augmented_ready"
OUT_DIR = PROJECT_ROOT / "outputs" / "neural_v1c"
REPORT = OUT_DIR / "neural_ready_quality_report.json"
FIXED = READY_DIR / "features_mig_deployable_fixed_augmented.csv"
MAG = READY_DIR / "features_mig_deployable_magnetic_augmented.csv"
TARGETS = READY_DIR / "targets_ep_augmented.csv"
SAMPLE_INDEX = READY_DIR / "sample_index_augmented.csv"
FORBIDDEN_INPUT_TOKENS = {"lat", "apd", "apd80", "apd90", "rt", "mask"}


def feature_check(path: Path, table_name: str) -> dict[str, object]:
    df = pd.read_csv(path)
    cols = select_deployable_feature_columns(df)
    values = df[cols].to_numpy(float)
    forbidden = []
    for col in cols:
        tokens = set(col.lower().replace("-", "_").split("_"))
        if tokens & FORBIDDEN_INPUT_TOKENS:
            forbidden.append(col)
    return {
        "table": table_name,
        "rows": int(len(df)),
        "feature_count": int(len(cols)),
        "finite": bool(np.isfinite(values).all()),
        "forbidden_leakage_like_columns": forbidden,
        "feature_columns_preview": cols[:10],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260525)
    sample_index = pd.read_csv(SAMPLE_INDEX)
    targets = pd.read_csv(TARGETS)
    split_map = load_split_map()
    index = load_neural_index()

    leakage = check_no_group_leakage(index)
    split_counts = index["split"].value_counts().to_dict()
    target_counts = targets[["y_condition", "y_lat_range", "y_apd90_patch_delta", "y_patch_area_fraction"]].notna().sum().to_dict()

    sampled_rows = []
    forced = []
    for variant in ["clean", "dropout_10pct", "dropout_25pct", "combined_medium"]:
        rows = index[index["variant"] == variant]
        if not rows.empty:
            forced.append(rows.sample(n=1, random_state=20260525).iloc[0])
    remaining = index.sample(n=28, random_state=20260525).to_dict(orient="records")
    sampled_rows = [row.to_dict() if hasattr(row, "to_dict") else row for row in forced] + remaining

    bz_checks = []
    for row in sampled_rows:
        bz_path = PROJECT_ROOT / row["Bz_file"]
        mask_path = PROJECT_ROOT / row["sensor_mask_file"]
        raw = np.load(bz_path)
        mask = np.load(mask_path).astype(bool)
        grid = load_bz_as_grid(row["Bz_file"], row["sensor_mask_file"])
        bz_checks.append(
            {
                "augmented_sample_id": row["augmented_sample_id"],
                "variant": row["variant"],
                "bz_exists": bz_path.exists(),
                "mask_exists": mask_path.exists(),
                "raw_shape": list(raw.shape),
                "grid_shape": list(grid.shape),
                "raw_nan_count": int(np.isnan(raw).sum()),
                "valid_sensor_count": int(mask.sum()),
                "grid_finite": bool(np.isfinite(grid).all()),
            }
        )

    dataset = BzTensorDataset(rows=index.sample(n=12, random_state=7))
    dataset_shapes = []
    for item in dataset:
        dataset_shapes.append(
            {
                "x_bz_shape": list(item["x_bz"].shape),
                "x_bz_finite": bool(np.isfinite(item["x_bz"].numpy()).all()),
                "target_mask": item["target_mask"].numpy().astype(bool).tolist(),
            }
        )

    fixed_check = feature_check(FIXED, "deployable_fixed")
    magnetic_check = feature_check(MAG, "deployable_magnetic")
    problems = []
    if len(sample_index) != 55000:
        problems.append(f"sample_index row count {len(sample_index)} != 55000")
    if len(targets) != 55000:
        problems.append(f"targets row count {len(targets)} != 55000")
    if leakage:
        problems.append(f"base_sample_id leakage: {leakage[:10]}")
    for check in bz_checks:
        if check["raw_shape"] != [301, 64] or check["grid_shape"] != [301, 8, 8] or not check["grid_finite"]:
            problems.append(f"Bz check failed for {check['augmented_sample_id']}")
    for check in [fixed_check, magnetic_check]:
        if not check["finite"]:
            problems.append(f"{check['table']} contains non-finite feature values")
        if check["forbidden_leakage_like_columns"]:
            problems.append(f"{check['table']} has leakage-like input columns: {check['forbidden_leakage_like_columns']}")

    report = {
        "passed": not problems,
        "problems": problems,
        "sample_index_rows": int(len(sample_index)),
        "target_rows": int(len(targets)),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "base_sample_count": int(index["base_sample_id"].nunique()),
        "target_nonmissing_counts": {k: int(v) for k, v in target_counts.items()},
        "bz_sample_checks": bz_checks,
        "dataset_shapes": dataset_shapes,
        "feature_checks": [fixed_check, magnetic_check],
        "leakage_policy": "grouped by base_sample_id; no oracle LAT/RT/APD-windowed features used",
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({"report": str(REPORT.relative_to(PROJECT_ROOT)), "passed": report["passed"], "problems": problems}, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
