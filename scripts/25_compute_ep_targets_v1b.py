                      
                                                                        

from __future__ import annotations

import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
READY_BASE = ROOT / "outputs" / "dataset_v1b_ready"
CONDITION_LABELS = {
    "homogeneous": 0,
    "apd_repolarization_perturbation": 1,
    "conduction_slowing_perturbation": 2,
    "mixed_apd_conduction_perturbation": 3,
}

FIELDS = [
    "sample_id",
    "condition",
    "condition_label",
    "APD80_mean",
    "APD80_std",
    "APD80_min",
    "APD80_max",
    "APD90_mean",
    "APD90_std",
    "APD90_min",
    "APD90_max",
    "LAT_min",
    "LAT_max",
    "LAT_range",
    "APD80_inside_patch",
    "APD80_outside_patch",
    "APD80_patch_delta",
    "APD90_inside_patch",
    "APD90_outside_patch",
    "APD90_patch_delta",
    "LAT_inside_patch",
    "LAT_outside_patch",
    "LAT_patch_delta",
    "patch_area_fraction",
    "y_condition",
    "y_apd90_patch_delta",
    "y_lat_range",
    "y_patch_area_fraction",
    "notes",
]


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def valid(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr) & (arr >= 0)]


def stats(arr: np.ndarray, prefix: str) -> dict:
    vals = valid(arr)
    if vals.size == 0:
        return {f"{prefix}_{k}": math.nan for k in ["mean", "std", "min", "max"]}
    return {
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_std": float(np.std(vals)),
        f"{prefix}_min": float(np.min(vals)),
        f"{prefix}_max": float(np.max(vals)),
    }


def mean_or_nan(arr: np.ndarray) -> float:
    vals = valid(arr)
    return float(np.mean(vals)) if vals.size else math.nan


def load_mask(path: Path) -> np.ndarray:
    mask = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(mask["abnormal"]).astype(bool)


def json_clean(value):
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return json_clean(value.item())
    return value


def main() -> int:
    index_rows = read_csv(READY_BASE / "sample_index.csv")
    rows: list[dict] = []
    for row in index_rows:
        condition = row["condition"]
        apd80 = np.loadtxt(ROOT / row["apd80_file"])
        apd90 = np.loadtxt(ROOT / row["apd90_file"])
        lat = np.loadtxt(ROOT / row["lat_file"])
        mask = load_mask(ROOT / row["mask_file"])
        patch_nonempty = bool(np.any(mask))
        out = {
            "sample_id": row["sample_id"],
            "condition": condition,
            "condition_label": CONDITION_LABELS[condition],
            **stats(apd80, "APD80"),
            **stats(apd90, "APD90"),
            "LAT_min": math.nan,
            "LAT_max": math.nan,
            "LAT_range": math.nan,
            "APD80_inside_patch": math.nan,
            "APD80_outside_patch": math.nan,
            "APD80_patch_delta": math.nan,
            "APD90_inside_patch": math.nan,
            "APD90_outside_patch": math.nan,
            "APD90_patch_delta": math.nan,
            "LAT_inside_patch": math.nan,
            "LAT_outside_patch": math.nan,
            "LAT_patch_delta": math.nan,
            "patch_area_fraction": math.nan,
            "notes": "",
        }
        lat_valid = valid(lat)
        if lat_valid.size:
            out["LAT_min"] = float(np.min(lat_valid))
            out["LAT_max"] = float(np.max(lat_valid))
            out["LAT_range"] = float(np.max(lat_valid) - np.min(lat_valid))
        if patch_nonempty:
            patch_fraction = float(np.mean(mask))
            apd80_in = mean_or_nan(apd80[mask])
            apd80_out = mean_or_nan(apd80[~mask])
            apd90_in = mean_or_nan(apd90[mask])
            apd90_out = mean_or_nan(apd90[~mask])
            lat_in = mean_or_nan(lat[mask])
            lat_out = mean_or_nan(lat[~mask])
            out.update(
                {
                    "APD80_inside_patch": apd80_in,
                    "APD80_outside_patch": apd80_out,
                    "APD80_patch_delta": apd80_in - apd80_out,
                    "APD90_inside_patch": apd90_in,
                    "APD90_outside_patch": apd90_out,
                    "APD90_patch_delta": apd90_in - apd90_out,
                    "LAT_inside_patch": lat_in,
                    "LAT_outside_patch": lat_out,
                    "LAT_patch_delta": lat_in - lat_out,
                    "patch_area_fraction": patch_fraction,
                }
            )
        elif condition != "homogeneous":
            out["notes"] = "non-homogeneous sample has empty patch mask"
        out["y_condition"] = out["condition_label"]
        out["y_apd90_patch_delta"] = out["APD90_patch_delta"]
        out["y_lat_range"] = out["LAT_range"]
        out["y_patch_area_fraction"] = out["patch_area_fraction"]
        rows.append(out)

    READY_BASE.mkdir(parents=True, exist_ok=True)
    with (READY_BASE / "targets_ep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "created_utc": now_utc(),
        "row_count": len(rows),
        "condition_labels": CONDITION_LABELS,
        "recommended_target_variables": ["y_condition", "y_apd90_patch_delta", "y_lat_range", "y_patch_area_fraction"],
        "rows": [json_clean(r) for r in rows],
    }
    (READY_BASE / "targets_ep.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {READY_BASE / 'targets_ep.csv'} with {len(rows)} rows")
    return 0 if len(rows) == 70 else 1


if __name__ == "__main__":
    raise SystemExit(main())
