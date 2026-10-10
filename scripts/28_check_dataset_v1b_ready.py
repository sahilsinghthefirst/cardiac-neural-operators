                      
                                                                      

from __future__ import annotations

import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
READY_BASE = ROOT / "outputs" / "dataset_v1b_ready"
EXPECTED_COUNT = 70


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def truthy(value) -> bool:
    return value is True or str(value).lower() == "true"


def fval(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return math.nan


def is_missing(value) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return value in ("", None)


def mask_nonempty(path: Path) -> bool:
    mask = np.genfromtxt(path, delimiter=",", names=True)
    vals = np.atleast_1d(mask["abnormal"])
    return bool(np.sum(vals) > 0)


def main() -> int:
    problems: list[str] = []
    index_rows = read_csv(READY_BASE / "sample_index.csv")
    target_rows = read_csv(READY_BASE / "targets_ep.csv")
    feature_rows = read_csv(READY_BASE / "features_mig_windowed.csv")
    if len(index_rows) != EXPECTED_COUNT:
        problems.append(f"sample_index has {len(index_rows)} rows, expected {EXPECTED_COUNT}")
    if len(target_rows) != EXPECTED_COUNT:
        problems.append(f"targets_ep has {len(target_rows)} rows, expected {EXPECTED_COUNT}")
    if len(feature_rows) != EXPECTED_COUNT:
        problems.append(f"features_mig_windowed has {len(feature_rows)} rows, expected {EXPECTED_COUNT}")

    ids_index = {r["sample_id"] for r in index_rows}
    ids_targets = {r["sample_id"] for r in target_rows}
    ids_features = {r["sample_id"] for r in feature_rows}
    if not (ids_index == ids_targets == ids_features):
        problems.append("sample IDs do not align across index, targets, and features")

    target_by_id = {r["sample_id"]: r for r in target_rows}
    feature_by_id = {r["sample_id"]: r for r in feature_rows}

    feature_numeric = [k for k in (feature_rows[0].keys() if feature_rows else []) if k not in {"sample_id", "condition"}]
    for row in index_rows:
        sid = row["sample_id"]
        if truthy(row.get("success")):
            for key in ["Bz_file", "mig_curves_file", "mig_summary_file"]:
                path = ROOT / row[key]
                if not path.exists() or path.stat().st_size == 0:
                    problems.append(f"{sid}: missing or empty {key}")
        condition = row["condition"]
        target = target_by_id.get(sid, {})
        if condition == "homogeneous":
            for key in ["APD80_patch_delta", "APD90_patch_delta", "LAT_patch_delta", "patch_area_fraction"]:
                if not is_missing(target.get(key)):
                    problems.append(f"{sid}: homogeneous {key} should be NaN/missing")
        else:
            if not mask_nonempty(ROOT / row["mask_file"]):
                problems.append(f"{sid}: non-homogeneous patch mask is empty")
            for key in ["APD90_patch_delta", "LAT_patch_delta", "patch_area_fraction"]:
                if not np.isfinite(fval(target, key)):
                    problems.append(f"{sid}: expected finite {key}")

        feature = feature_by_id.get(sid, {})
        for key in feature_numeric:
            value = fval(feature, key)
            if not np.isfinite(value):
                problems.append(f"{sid}: non-finite feature {key}")
        for window in ["full", "activation_like", "post_activation_like", "repolarization_like", "st_t_like"]:
            mi_mean = fval(feature, f"{window}_MI_mean")
            di_mean = fval(feature, f"{window}_DI_mean")
            rms_mean = fval(feature, f"{window}_RMS_mean")
            if mi_mean < -1e-9 or mi_mean > 1 + 1e-9:
                problems.append(f"{sid}: {window} MI_mean outside [0,1]")
            if not np.isclose(di_mean, 1.0 - mi_mean, atol=1e-8):
                problems.append(f"{sid}: {window} DI_mean != 1 - MI_mean")
            if rms_mean < -1e-18:
                problems.append(f"{sid}: {window} RMS_mean negative")

    summary = {
        "created_utc": now_utc(),
        "sample_index_rows": len(index_rows),
        "targets_ep_rows": len(target_rows),
        "features_mig_windowed_rows": len(feature_rows),
        "sample_ids_align": ids_index == ids_targets == ids_features,
        "passed": not problems,
        "problem_count": len(problems),
        "problems": problems[:200],
    }
    READY_BASE.mkdir(parents=True, exist_ok=True)
    (READY_BASE / "dataset_ready_quality_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
