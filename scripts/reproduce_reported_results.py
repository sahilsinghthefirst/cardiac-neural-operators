                      
                                                                     

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


FIELDS = (
    "scope",
    "model",
    "selection_role",
    "LAT_global_R2",
    "APD90_global_R2",
    "patch_Dice_nonhomogeneous_micro",
    "APD90_patch_delta_R2",
    "degenerate_region_count",
    "n_base_cases",
    "n_nonhomogeneous_cases",
    "threshold",
    "resampling_unit",
    "measurement_variant",
)


def r2(true: np.ndarray, pred: np.ndarray) -> float:
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    finite = np.isfinite(true) & np.isfinite(pred)
    true = true[finite]
    pred = pred[finite]
    denominator = np.sum((true - np.mean(true)) ** 2)
    if true.size < 2 or denominator <= 0:
        return math.nan
    return float(1.0 - np.sum((true - pred) ** 2) / denominator)


def read_cohort(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pooled_field_r2(
    maps: dict[str, np.ndarray], indices: np.ndarray, model: str, field: str
) -> float:
    valid = maps["valid_mask"].astype(bool)
    true = maps[f"true_{field}"]
    pred = maps[f"{model}_field_pred_{field}"]
    true_values = []
    pred_values = []
    for index in indices:
        active = valid[index] & np.isfinite(true[index]) & np.isfinite(pred[index])
        true_values.append(true[index][active])
        pred_values.append(pred[index][active])
    return r2(np.concatenate(true_values), np.concatenate(pred_values))


def localization_metrics(
    maps: dict[str, np.ndarray],
    indices: np.ndarray,
    model: str,
    threshold: float,
    cohort_true_contrast: np.ndarray,
) -> tuple[float, float, int]:
    valid = maps["valid_mask"].astype(bool)
    truth = maps["true_patch"] > 0.5
    predicted = maps[f"{model}_field_pred_patch_prob"] > threshold
    predicted_apd90 = maps[f"{model}_field_pred_apd90"]

    tp = fp = fn = 0
    true_contrast: list[float] = []
    predicted_contrast: list[float] = []
    degenerate = 0

    for index in indices:
        active = valid[index]
        true_region = truth[index] & active
        predicted_region = predicted[index] & active

        tp += int(np.sum(predicted_region & true_region))
        fp += int(np.sum(predicted_region & ~true_region & active))
        fn += int(np.sum(~predicted_region & true_region & active))

        true_delta = float(cohort_true_contrast[index])
        predicted_size = int(np.sum(predicted_region))
        if predicted_size == 0 or predicted_size == int(np.sum(active)):
            degenerate += 1
            predicted_delta = 0.0
        else:
            predicted_delta = float(
                np.mean(predicted_apd90[index][predicted_region])
                - np.mean(predicted_apd90[index][active & ~predicted_region])
            )
        true_contrast.append(true_delta)
        predicted_contrast.append(predicted_delta)

    denominator = 2 * tp + fp + fn
    dice = float(2 * tp / denominator) if denominator else math.nan
    contrast_r2 = r2(np.asarray(true_contrast), np.asarray(predicted_contrast))
    return dice, contrast_r2, degenerate


def recompute(results_dir: Path) -> list[dict[str, object]]:
    analysis = results_dir / "01_locked_analysis" / "icbinb_required_analysis"
    cohort = read_cohort(analysis / "paired_test_cohort.csv")
    with np.load(analysis / "paired_test_cohort_maps.npz") as archive:
        maps = {name: archive[name] for name in archive.files}

    conditions = np.asarray([row["condition"] for row in cohort], dtype=object)
    nonhomogeneous = np.asarray(
        [float(row["true_patch_present"]) >= 0.5 for row in cohort], dtype=bool
    )
    cohort_true_contrast = np.asarray(
        [
            float(row["true_APD90_patch_delta"])
            if row["true_APD90_patch_delta"].strip()
            else math.nan
            for row in cohort
        ],
        dtype=np.float64,
    )
    scopes = {
        "overall": np.arange(len(cohort), dtype=np.int64),
        "conduction_slowing_perturbation": np.flatnonzero(
            conditions == "conduction_slowing_perturbation"
        ),
    }

    rows: list[dict[str, object]] = []
    for scope, indices in scopes.items():
        localization_indices = indices[nonhomogeneous[indices]]
        for model, display_name in (("fno", "FNO"), ("deeponet", "DeepONet")):
            dice, contrast_r2, degenerate = localization_metrics(
                maps, localization_indices, model, 0.5, cohort_true_contrast
            )
            rows.append(
                {
                    "scope": scope,
                    "model": display_name,
                    "selection_role": "field_selected",
                    "LAT_global_R2": pooled_field_r2(maps, indices, model, "lat"),
                    "APD90_global_R2": pooled_field_r2(maps, indices, model, "apd90"),
                    "patch_Dice_nonhomogeneous_micro": dice,
                    "APD90_patch_delta_R2": contrast_r2,
                    "degenerate_region_count": degenerate,
                    "n_base_cases": int(indices.size),
                    "n_nonhomogeneous_cases": int(localization_indices.size),
                    "threshold": 0.5,
                    "resampling_unit": "base_sample_id",
                    "measurement_variant": "combined_bz_axial",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_reference(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["scope"], row["model"]): row for row in csv.DictReader(handle)
        }


def validate(
    rows: list[dict[str, object]], reference_path: Path, tolerance: float
) -> dict[str, object]:
    reference = read_reference(reference_path)
    checks = []
    for row in rows:
        key = (str(row["scope"]), str(row["model"]))
        expected = reference[key]
        for metric in (
            "LAT_global_R2",
            "APD90_global_R2",
            "patch_Dice_nonhomogeneous_micro",
            "APD90_patch_delta_R2",
        ):
            actual_value = float(row[metric])
            expected_value = float(expected[metric])
            difference = abs(actual_value - expected_value)
            checks.append(
                {
                    "scope": key[0],
                    "model": key[1],
                    "metric": metric,
                    "expected": expected_value,
                    "recomputed": actual_value,
                    "absolute_difference": difference,
                    "pass": difference <= tolerance,
                }
            )
        actual_count = int(row["degenerate_region_count"])
        expected_count = int(float(expected["degenerate_region_count"]))
        checks.append(
            {
                "scope": key[0],
                "model": key[1],
                "metric": "degenerate_region_count",
                "expected": expected_count,
                "recomputed": actual_count,
                "absolute_difference": abs(actual_count - expected_count),
                "pass": actual_count == expected_count,
            }
        )
    return {
        "status": "pass" if all(item["pass"] for item in checks) else "fail",
        "tolerance": tolerance,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reproduced"))
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = results_dir / "01_locked_analysis" / "icbinb_required_analysis"

    rows = recompute(results_dir)
    write_csv(output_dir / "same_checkpoint_recomputed.csv", rows)
    report = validate(rows, analysis / "same_checkpoint_audit.csv", args.tolerance)
    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "output_dir": str(output_dir)}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
