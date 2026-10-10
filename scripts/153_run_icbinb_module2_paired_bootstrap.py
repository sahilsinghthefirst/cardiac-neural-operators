                      
                                                                   
 
                                                                           
                                                                           
                                                                       
   

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "outputs" / "icbinb_required_analysis"
CONTRACT_PATH = ANALYSIS_DIR / "evaluation_contract.json"
INVENTORY_PATH = ANALYSIS_DIR / "artifact_inventory.json"
COHORT_PATH = ANALYSIS_DIR / "paired_test_cohort.csv"
MAPS_PATH = ANALYSIS_DIR / "paired_test_cohort_maps.npz"
COHORT_VALIDATION_PATH = ANALYSIS_DIR / "cohort_validation.json"

FINAL_FILES = {
    "case_statistics": ANALYSIS_DIR / "module2_case_statistics.csv",
    "point_estimates": ANALYSIS_DIR / "module2_point_estimates.csv",
    "condition_specific": ANALYSIS_DIR / "module2_condition_specific.csv",
    "bootstrap_intervals": ANALYSIS_DIR / "module2_bootstrap_intervals.csv",
    "paired_differences": ANALYSIS_DIR / "module2_paired_differences.csv",
    "support_diagnostics": ANALYSIS_DIR / "module2_support_diagnostics.csv",
    "bootstrap_replicates": ANALYSIS_DIR / "module2_bootstrap_replicates.npz",
    "summary": ANALYSIS_DIR / "module2_summary.json",
    "validation": ANALYSIS_DIR / "module2_validation.json",
}

EXPECTED_CONDITIONS = {
    "apd_repolarization_perturbation": 18,
    "conduction_slowing_perturbation": 18,
    "homogeneous": 12,
    "mixed_apd_conduction_perturbation": 27,
}
MODELS = ("fno", "deeponet")
METRICS = (
    "LAT_global_R2",
    "APD90_global_R2",
    "patch_Dice_nonhomogeneous_micro",
    "patch_center_error_mean",
    "homogeneous_false_positive_area_mean",
    "APD90_patch_delta_deployed_R2",
)
DIRECTIONS = {
    "LAT_global_R2": "higher_is_better",
    "APD90_global_R2": "higher_is_better",
    "patch_Dice_nonhomogeneous_micro": "higher_is_better",
    "patch_center_error_mean": "lower_is_better",
    "homogeneous_false_positive_area_mean": "lower_is_better",
    "APD90_patch_delta_deployed_R2": "higher_is_better",
}
MODEL_ROLE = {
    "LAT_global_R2": "global_field_comparison",
    "APD90_global_R2": "global_field_comparison",
    "patch_Dice_nonhomogeneous_micro": "localization_comparison",
    "patch_center_error_mean": "localization_comparison",
    "homogeneous_false_positive_area_mean": "localization_comparison",
    "APD90_patch_delta_deployed_R2": "localization_comparison",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inventory_artifact(
    inventory_by_id: dict[str, dict[str, Any]], artifact_id: str
) -> dict[str, Any]:
    if artifact_id not in inventory_by_id:
        raise RuntimeError(f"Inventory lacks required artifact {artifact_id!r}.")
    record = inventory_by_id[artifact_id]
    path = Path(record["path"])
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file():
        raise RuntimeError(f"Required artifact is missing: {path}")
    observed = sha256(path)
    if observed != record.get("sha256"):
        raise RuntimeError(
            f"Artifact hash changed for {artifact_id}: expected {record.get('sha256')}, "
            f"observed {observed}"
        )
    return {
        "path": path,
        "sha256": observed,
        "bytes": path.stat().st_size,
    }


def r2_from_sufficient(
    n: np.ndarray | float,
    true_sum: np.ndarray | float,
    true_sumsq: np.ndarray | float,
    sse: np.ndarray | float,
) -> np.ndarray:
    n_array = np.asarray(n, dtype=np.float64)
    true_sum_array = np.asarray(true_sum, dtype=np.float64)
    true_sumsq_array = np.asarray(true_sumsq, dtype=np.float64)
    sse_array = np.asarray(sse, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        sst = true_sumsq_array - (true_sum_array * true_sum_array) / n_array
        result = 1.0 - sse_array / sst
    return np.where((n_array >= 2) & np.isfinite(sst) & (sst > 0.0), result, np.nan)


def scalar_r2(true: np.ndarray, pred: np.ndarray) -> float:
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    finite = np.isfinite(true) & np.isfinite(pred)
    if int(np.sum(finite)) < 2:
        return math.nan
    true = true[finite]
    pred = pred[finite]
    sst = float(np.sum((true - np.mean(true)) ** 2, dtype=np.float64))
    if not math.isfinite(sst) or sst <= 0.0:
        return math.nan
    return float(1.0 - np.sum((true - pred) ** 2, dtype=np.float64) / sst)


def metrics_for_scope(scope: str) -> tuple[str, ...]:
    if scope == "overall":
        return METRICS
    if scope == "homogeneous":
        return (
            "LAT_global_R2",
            "APD90_global_R2",
            "homogeneous_false_positive_area_mean",
        )
    return (
        "LAT_global_R2",
        "APD90_global_R2",
        "patch_Dice_nonhomogeneous_micro",
        "patch_center_error_mean",
        "APD90_patch_delta_deployed_R2",
    )


def population_mask(case_stats: pd.DataFrame, metric: str) -> np.ndarray:
    if metric in {
        "patch_Dice_nonhomogeneous_micro",
        "patch_center_error_mean",
        "APD90_patch_delta_deployed_R2",
    }:
        return case_stats["is_nonhomogeneous"].to_numpy(dtype=bool)
    if metric == "homogeneous_false_positive_area_mean":
        return case_stats["is_homogeneous"].to_numpy(dtype=bool)
    return np.ones(len(case_stats), dtype=bool)


def metric_estimate(
    case_stats: pd.DataFrame,
    indices: np.ndarray,
    metric: str,
    model: str,
) -> tuple[float, int, int]:
    applicable = population_mask(case_stats, metric)
    selected = np.asarray(indices, dtype=np.int64)
    selected = selected[applicable[selected]]
    n_cases = int(selected.size)
    if n_cases == 0:
        return math.nan, 0, 0
    if metric == "LAT_global_R2":
        n = float(case_stats[f"{model}_lat_n_pixels"].to_numpy()[selected].sum())
        estimate = r2_from_sufficient(
            n,
            case_stats["lat_true_sum"].to_numpy()[selected].sum(),
            case_stats["lat_true_sumsq"].to_numpy()[selected].sum(),
            case_stats[f"{model}_lat_sse"].to_numpy()[selected].sum(),
        )
        return float(estimate), n_cases, int(n)
    if metric == "APD90_global_R2":
        n = float(case_stats[f"{model}_apd90_n_pixels"].to_numpy()[selected].sum())
        estimate = r2_from_sufficient(
            n,
            case_stats["apd90_true_sum"].to_numpy()[selected].sum(),
            case_stats["apd90_true_sumsq"].to_numpy()[selected].sum(),
            case_stats[f"{model}_apd90_sse"].to_numpy()[selected].sum(),
        )
        return float(estimate), n_cases, int(n)
    if metric == "patch_Dice_nonhomogeneous_micro":
        tp = float(case_stats[f"{model}_patch_tp"].to_numpy()[selected].sum())
        fp = float(case_stats[f"{model}_patch_fp"].to_numpy()[selected].sum())
        fn = float(case_stats[f"{model}_patch_fn"].to_numpy()[selected].sum())
        denominator = 2.0 * tp + fp + fn
        estimate = 2.0 * tp / denominator if denominator > 0.0 else math.nan
        n_pixels = int(case_stats["valid_pixel_count"].to_numpy()[selected].sum())
        return float(estimate), n_cases, n_pixels
    if metric == "patch_center_error_mean":
        values = case_stats[f"{model}_center_error"].to_numpy(dtype=np.float64)[selected]
        return float(np.mean(values)), n_cases, n_cases
    if metric == "homogeneous_false_positive_area_mean":
        values = case_stats[f"{model}_homogeneous_false_positive_area"].to_numpy(
            dtype=np.float64
        )[selected]
        return float(np.mean(values)), n_cases, n_cases
    if metric == "APD90_patch_delta_deployed_R2":
        true = case_stats["true_APD90_patch_delta"].to_numpy(dtype=np.float64)[selected]
        pred = case_stats[f"{model}_deployed_APD90_patch_delta"].to_numpy(
            dtype=np.float64
        )[selected]
        return scalar_r2(true, pred), n_cases, n_cases
    raise KeyError(metric)


def bootstrap_metric(
    case_stats: pd.DataFrame,
    global_draws: np.ndarray,
    metric: str,
    model: str,
) -> np.ndarray:
    applicable = population_mask(case_stats, metric)
    weights = applicable.astype(np.float64)
    draws = np.asarray(global_draws, dtype=np.int64)
    if metric in {"LAT_global_R2", "APD90_global_R2"}:
        prefix = "lat" if metric.startswith("LAT") else "apd90"
        n_case = case_stats[f"{model}_{prefix}_n_pixels"].to_numpy(dtype=np.float64) * weights
        true_sum_case = case_stats[f"{prefix}_true_sum"].to_numpy(dtype=np.float64) * weights
        true_sumsq_case = case_stats[f"{prefix}_true_sumsq"].to_numpy(dtype=np.float64) * weights
        sse_case = case_stats[f"{model}_{prefix}_sse"].to_numpy(dtype=np.float64) * weights
        return r2_from_sufficient(
            n_case[draws].sum(axis=1),
            true_sum_case[draws].sum(axis=1),
            true_sumsq_case[draws].sum(axis=1),
            sse_case[draws].sum(axis=1),
        )
    if metric == "patch_Dice_nonhomogeneous_micro":
        tp = (case_stats[f"{model}_patch_tp"].to_numpy(dtype=np.float64) * weights)[draws].sum(axis=1)
        fp = (case_stats[f"{model}_patch_fp"].to_numpy(dtype=np.float64) * weights)[draws].sum(axis=1)
        fn = (case_stats[f"{model}_patch_fn"].to_numpy(dtype=np.float64) * weights)[draws].sum(axis=1)
        denominator = 2.0 * tp + fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            values = 2.0 * tp / denominator
        return np.where(denominator > 0.0, values, np.nan)
    if metric in {
        "patch_center_error_mean",
        "homogeneous_false_positive_area_mean",
    }:
        column = (
            f"{model}_center_error"
            if metric == "patch_center_error_mean"
            else f"{model}_homogeneous_false_positive_area"
        )
        values = case_stats[column].to_numpy(dtype=np.float64)
        finite_weights = weights * np.isfinite(values)
        safe_values = np.where(np.isfinite(values), values, 0.0) * weights
        counts = finite_weights[draws].sum(axis=1)
        totals = safe_values[draws].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            means = totals / counts
        return np.where(counts > 0.0, means, np.nan)
    if metric == "APD90_patch_delta_deployed_R2":
        true = case_stats["true_APD90_patch_delta"].to_numpy(dtype=np.float64)
        pred = case_stats[f"{model}_deployed_APD90_patch_delta"].to_numpy(dtype=np.float64)
        finite_weights = weights * np.isfinite(true) * np.isfinite(pred)
        true_safe = np.where(np.isfinite(true), true, 0.0)
        pred_safe = np.where(np.isfinite(pred), pred, 0.0)
        n = finite_weights[draws].sum(axis=1)
        true_sum = (true_safe * finite_weights)[draws].sum(axis=1)
        true_sumsq = (true_safe * true_safe * finite_weights)[draws].sum(axis=1)
        sse = ((true_safe - pred_safe) ** 2 * finite_weights)[draws].sum(axis=1)
        return r2_from_sufficient(n, true_sum, true_sumsq, sse)
    raise KeyError(metric)


def percentile_interval(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan, math.nan, 0
    lower, upper = np.percentile(finite, [2.5, 97.5])
    return float(lower), float(upper), int(finite.size)


def direct_metric(
    table: pd.DataFrame,
    maps: dict[str, np.ndarray],
    indices: np.ndarray,
    metric: str,
    model: str,
) -> float:
    indices = np.asarray(indices, dtype=np.int64)
    nonhom = table["true_patch_present"].to_numpy(dtype=float) >= 0.5
    hom = ~nonhom
    valid = maps["valid_mask"].astype(bool)
    if metric in {"LAT_global_R2", "APD90_global_R2"}:
        field = "lat" if metric.startswith("LAT") else "apd90"
        true = maps[f"true_{field}"]
        pred = maps[f"{model}_field_pred_{field}"]
        selected_true: list[np.ndarray] = []
        selected_pred: list[np.ndarray] = []
        for index in indices:
            mask = valid[index] & np.isfinite(true[index]) & np.isfinite(pred[index])
            selected_true.append(true[index][mask].astype(np.float64))
            selected_pred.append(pred[index][mask].astype(np.float64))
        return scalar_r2(np.concatenate(selected_true), np.concatenate(selected_pred))
    if metric == "patch_Dice_nonhomogeneous_micro":
        selected = indices[nonhom[indices]]
        truth = maps["true_patch"] > 0.5
        pred = maps[f"{model}_localization_pred_patch_mask"].astype(bool)
        tp = fp = fn = 0
        for index in selected:
            active = valid[index]
            tp += int(np.sum(pred[index] & truth[index] & active))
            fp += int(np.sum(pred[index] & ~truth[index] & active))
            fn += int(np.sum(~pred[index] & truth[index] & active))
        return float(2 * tp / (2 * tp + fp + fn))
    if metric == "patch_center_error_mean":
        selected = indices[nonhom[indices]]
        dx = (
            table[f"{model}_localization__pred_center_x_norm"].to_numpy(dtype=float)[selected]
            - table["true_center_x_norm"].to_numpy(dtype=float)[selected]
        )
        dy = (
            table[f"{model}_localization__pred_center_y_norm"].to_numpy(dtype=float)[selected]
            - table["true_center_y_norm"].to_numpy(dtype=float)[selected]
        )
        return float(np.mean(np.sqrt(dx * dx + dy * dy)))
    if metric == "homogeneous_false_positive_area_mean":
        selected = indices[hom[indices]]
        pred = maps[f"{model}_localization_pred_patch_mask"].astype(bool)
        areas = [float(np.mean(pred[index][valid[index]])) for index in selected]
        return float(np.mean(areas))
    if metric == "APD90_patch_delta_deployed_R2":
        selected = indices[nonhom[indices]]
        true = table["true_APD90_patch_delta"].to_numpy(dtype=float)[selected]
        raw = table[f"{model}_localization__map_APD90_patch_delta"].to_numpy(dtype=float)[selected]
        deployed = np.where(np.isfinite(raw), raw, 0.0)
        return scalar_r2(true, deployed)
    raise KeyError(metric)


def build_case_statistics(
    table: pd.DataFrame, maps: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid_all = maps["valid_mask"].astype(bool)
    true_patch_all = maps["true_patch"] > 0.5
    threshold_checks: dict[str, bool] = {}
    for model in MODELS:
        recomputed = (
            maps[f"{model}_localization_pred_patch_prob"] > 0.5
        ) & valid_all
        saved = maps[f"{model}_localization_pred_patch_mask"].astype(bool)
        threshold_checks[model] = bool(np.array_equal(recomputed, saved))

    for index, source in table.reset_index(drop=True).iterrows():
        valid = valid_all[index]
        true_patch = true_patch_all[index]
        present = bool(float(source["true_patch_present"]) >= 0.5)
        row: dict[str, Any] = {
            "cohort_index": int(source["cohort_index"]),
            "row_id": str(source["row_id"]),
            "base_sample_id": str(source["base_sample_id"]),
            "split": str(source["split"]),
            "condition": str(source["condition"]),
            "combined_variant": str(source["combined_variant"]),
            "is_nonhomogeneous": present,
            "is_homogeneous": not present,
            "valid_pixel_count": int(np.sum(valid)),
            "true_patch_pixel_count": int(np.sum(true_patch & valid)),
            "true_APD90_patch_delta": float(source["true_APD90_patch_delta"])
            if np.isfinite(source["true_APD90_patch_delta"])
            else math.nan,
        }
        for field in ("lat", "apd90"):
            true = maps[f"true_{field}"][index].astype(np.float64)
            fno_pred = maps[f"fno_field_pred_{field}"][index].astype(np.float64)
            deep_pred = maps[f"deeponet_field_pred_{field}"][index].astype(np.float64)
            common = valid & np.isfinite(true) & np.isfinite(fno_pred) & np.isfinite(deep_pred)
            true_values = true[common]
            row[f"{field}_true_sum"] = float(np.sum(true_values, dtype=np.float64))
            row[f"{field}_true_sumsq"] = float(
                np.sum(true_values * true_values, dtype=np.float64)
            )
            for model, pred in (("fno", fno_pred), ("deeponet", deep_pred)):
                residual = true_values - pred[common]
                row[f"{model}_{field}_n_pixels"] = int(true_values.size)
                row[f"{model}_{field}_sse"] = float(
                    np.sum(residual * residual, dtype=np.float64)
                )
        for model in MODELS:
            pred_mask = maps[f"{model}_localization_pred_patch_mask"][index].astype(bool) & valid
            row[f"{model}_patch_tp"] = int(np.sum(pred_mask & true_patch & valid))
            row[f"{model}_patch_fp"] = int(np.sum(pred_mask & ~true_patch & valid))
            row[f"{model}_patch_fn"] = int(np.sum(~pred_mask & true_patch & valid))
            pred_area = float(np.mean(pred_mask[valid]))
            row[f"{model}_predicted_patch_area"] = pred_area
            row[f"{model}_homogeneous_false_positive_area"] = pred_area
            dx = float(source[f"{model}_localization__pred_center_x_norm"]) - float(
                source["true_center_x_norm"]
            )
            dy = float(source[f"{model}_localization__pred_center_y_norm"]) - float(
                source["true_center_y_norm"]
            )
            row[f"{model}_center_error"] = (
                float(math.hypot(dx, dy)) if present and np.isfinite(dx) and np.isfinite(dy) else math.nan
            )
            raw_delta = float(source[f"{model}_localization__map_APD90_patch_delta"])
            support_valid = bool(np.isfinite(raw_delta))
            row[f"{model}_predicted_support_nondegenerate"] = support_valid
            row[f"{model}_raw_APD90_patch_delta"] = raw_delta if support_valid else math.nan
            row[f"{model}_deployed_APD90_patch_delta"] = raw_delta if support_valid else 0.0
            row[f"{model}_degenerate_support_zero_fallback"] = bool(present and not support_valid)
        rows.append(row)

    case_stats = pd.DataFrame(rows)
    presence_from_maps = case_stats["true_patch_pixel_count"].to_numpy(dtype=int) > 0
    presence_from_table = case_stats["is_nonhomogeneous"].to_numpy(dtype=bool)
    diagnostics = {
        "threshold_mask_matches_saved": threshold_checks,
        "truth_presence_matches_maps": bool(
            np.array_equal(presence_from_maps, presence_from_table)
        ),
    }
    return case_stats, diagnostics


def support_diagnostics(case_stats: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    scopes = [("overall", np.arange(len(case_stats), dtype=int))]
    scopes.extend(
        (condition, np.flatnonzero(case_stats["condition"].eq(condition).to_numpy()))
        for condition in EXPECTED_CONDITIONS
        if condition != "homogeneous"
    )
    nonhom = case_stats["is_nonhomogeneous"].to_numpy(dtype=bool)
    for scope, scope_indices in scopes:
        selected = scope_indices[nonhom[scope_indices]]
        shared = np.ones(selected.size, dtype=bool)
        for model in MODELS:
            shared &= case_stats[f"{model}_predicted_support_nondegenerate"].to_numpy(dtype=bool)[selected]
        shared_true = case_stats["true_APD90_patch_delta"].to_numpy(dtype=float)[selected][shared]
        for model in MODELS:
            support_ok = case_stats[f"{model}_predicted_support_nondegenerate"].to_numpy(dtype=bool)[selected]
            raw = case_stats[f"{model}_raw_APD90_patch_delta"].to_numpy(dtype=float)[selected]
            true = case_stats["true_APD90_patch_delta"].to_numpy(dtype=float)[selected]
            deployed = case_stats[f"{model}_deployed_APD90_patch_delta"].to_numpy(dtype=float)[selected]
            records.append(
                {
                    "scope": scope,
                    "model_family": model,
                    "eligible_nonhomogeneous_cases": int(selected.size),
                    "nondegenerate_support_cases": int(np.sum(support_ok)),
                    "degenerate_support_cases": int(np.sum(~support_ok)),
                    "degenerate_support_fraction": float(np.mean(~support_ok)),
                    "primary_zero_fallback_R2": scalar_r2(true, deployed),
                    "model_specific_complete_case_R2": scalar_r2(true[support_ok], raw[support_ok]),
                    "shared_complete_case_count": int(np.sum(shared)),
                    "shared_complete_case_R2": scalar_r2(
                        shared_true,
                        raw[shared],
                    ),
                }
            )
    return pd.DataFrame(records)


def main() -> int:
    existing = [str(path) for path in FINAL_FILES.values() if path.exists()]
    if existing:
        raise SystemExit(
            "Refusing to overwrite existing Module 2 artifacts: " + ", ".join(existing)
        )

    contract = read_json(CONTRACT_PATH)
    inventory = read_json(INVENTORY_PATH)
    cohort_validation = read_json(COHORT_VALIDATION_PATH)
    if not contract.get("validation", {}).get("passed"):
        raise SystemExit("Evaluation contract does not pass.")
    if not cohort_validation.get("passed"):
        raise SystemExit("Canonical cohort validation does not pass.")
    if contract["measurement_policy"]["primary_variant"] != "combined_bz_axial":
        raise SystemExit("Primary measurement variant changed.")
    bootstrap_config = contract["bootstrap"]
    if bootstrap_config.get("independent_unit") != "base_sample_id":
        raise SystemExit("Bootstrap independent unit is not base_sample_id.")
    n_bootstrap = int(bootstrap_config["replicates"])
    seed = int(bootstrap_config["random_seed"])
    if n_bootstrap != 5000 or seed != 20260901:
        raise SystemExit(
            f"Locked bootstrap configuration changed: replicates={n_bootstrap}, seed={seed}"
        )

    inventory_by_id = {record["id"]: record for record in inventory["artifacts"]}
    source_artifacts = {
        artifact_id: verify_inventory_artifact(inventory_by_id, artifact_id)
        for artifact_id in (
            "paired_test_cohort_table",
            "paired_test_cohort_maps",
            "paired_test_cohort_validation",
        )
    }

    table = pd.read_csv(COHORT_PATH).sort_values("cohort_index").reset_index(drop=True)
    with np.load(MAPS_PATH) as archive:
        maps = {name: np.asarray(archive[name]) for name in archive.files}
    errors: list[str] = []
    if len(table) != 75 or table["base_sample_id"].nunique() != 75:
        errors.append("Canonical cohort is not exactly 75 unique base cases.")
    if set(table["split"]) != {"test"}:
        errors.append("Canonical cohort contains non-test rows.")
    if set(table["combined_variant"]) != {"combined_bz_axial"}:
        errors.append("Canonical cohort contains a non-primary measurement variant.")
    condition_counts = {
        str(key): int(value)
        for key, value in table["condition"].value_counts().sort_index().items()
    }
    if condition_counts != EXPECTED_CONDITIONS:
        errors.append(
            f"Condition counts changed: expected {EXPECTED_CONDITIONS}, observed {condition_counts}"
        )
    required_map_arrays = {
        "true_lat",
        "true_apd90",
        "true_patch",
        "valid_mask",
        "fno_field_pred_lat",
        "fno_field_pred_apd90",
        "deeponet_field_pred_lat",
        "deeponet_field_pred_apd90",
        "fno_localization_pred_patch_prob",
        "deeponet_localization_pred_patch_prob",
        "fno_localization_pred_patch_mask",
        "deeponet_localization_pred_patch_mask",
    }
    missing_maps = sorted(required_map_arrays - set(maps))
    if missing_maps:
        errors.append(f"Cohort map archive lacks required arrays: {missing_maps}")
    for name in required_map_arrays & set(maps):
        if maps[name].shape != (75, 32, 32):
            errors.append(f"Unexpected map shape for {name}: {maps[name].shape}")
    if errors:
        raise SystemExit("Module 2 input validation failed: " + " | ".join(errors))

    case_stats, case_diagnostics = build_case_statistics(table, maps)
    if not all(case_diagnostics["threshold_mask_matches_saved"].values()):
        errors.append(
            f"Saved localization masks do not match threshold 0.5: {case_diagnostics}"
        )
    if not case_diagnostics["truth_presence_matches_maps"]:
        errors.append("Scalar patch-presence labels do not match truth maps.")
    if case_stats["base_sample_id"].duplicated().any():
        errors.append("Case-statistics table contains duplicate base IDs.")

    model_ids = {
        "fno": {
            "global_field_comparison": contract["model_roles"]["global_field_comparison"]["fno"]["model_id"],
            "localization_comparison": contract["model_roles"]["localization_comparison"]["fno"]["model_id"],
        },
        "deeponet": {
            "global_field_comparison": contract["model_roles"]["global_field_comparison"]["deeponet"]["model_id"],
            "localization_comparison": contract["model_roles"]["localization_comparison"]["deeponet"]["model_id"],
        },
    }

    scopes: list[tuple[str, np.ndarray]] = [
        ("overall", np.arange(len(case_stats), dtype=np.int64))
    ]
    scopes.extend(
        (
            condition,
            np.flatnonzero(case_stats["condition"].eq(condition).to_numpy()).astype(np.int64),
        )
        for condition in EXPECTED_CONDITIONS
    )

    point_records: list[dict[str, Any]] = []
    condition_records: list[dict[str, Any]] = []
    point_lookup: dict[tuple[str, str, str], float] = {}
    direct_parity: dict[str, Any] = {}
    for scope, indices in scopes:
        for metric in metrics_for_scope(scope):
            for model in MODELS:
                estimate, n_cases, n_observations = metric_estimate(
                    case_stats, indices, metric, model
                )
                direct = direct_metric(table, maps, indices, metric, model)
                difference = abs(estimate - direct)
                parity_key = f"{scope}__{metric}__{model}"
                direct_parity[parity_key] = {
                    "sufficient_statistic_estimate": estimate,
                    "direct_estimate": direct,
                    "absolute_difference": difference,
                    "tolerance": 1e-10,
                    "passed": bool(difference <= 1e-10),
                }
                if not direct_parity[parity_key]["passed"]:
                    errors.append(f"Direct metric parity failed for {parity_key}: {difference}")
                role = MODEL_ROLE[metric]
                record = {
                    "scope": scope,
                    "metric": metric,
                    "direction": DIRECTIONS[metric],
                    "model_family": model,
                    "model_id": model_ids[model][role],
                    "model_role": role,
                    "estimate": estimate,
                    "n_base_cases": n_cases,
                    "n_observations": n_observations,
                    "measurement_variant": "combined_bz_axial",
                }
                point_lookup[(scope, metric, model)] = estimate
                if scope == "overall":
                    point_records.append(record)
                else:
                    condition_records.append(record)

    point_df = pd.DataFrame(point_records)
    condition_df = pd.DataFrame(condition_records)
    if not np.all(np.isfinite(point_df["estimate"].to_numpy(dtype=float))):
        errors.append("At least one overall point estimate is nonfinite.")
    if not np.all(np.isfinite(condition_df["estimate"].to_numpy(dtype=float))):
        errors.append("At least one condition-specific point estimate is nonfinite.")

    rng = np.random.default_rng(seed)
    bootstrap_arrays: dict[str, np.ndarray] = {}
    interval_records: list[dict[str, Any]] = []
    paired_records: list[dict[str, Any]] = []
    for scope, scope_indices in scopes:
        local_draws = rng.integers(
            0,
            len(scope_indices),
            size=(n_bootstrap, len(scope_indices)),
            dtype=np.int16,
        )
        global_draws = scope_indices[local_draws]
        bootstrap_arrays[f"scope_indices__{scope}"] = scope_indices.astype(np.int16)
        bootstrap_arrays[f"draws__{scope}"] = global_draws.astype(np.int16)
        for metric in metrics_for_scope(scope):
            by_model: dict[str, np.ndarray] = {}
            for model in MODELS:
                values = bootstrap_metric(case_stats, global_draws, metric, model)
                by_model[model] = values
                bootstrap_arrays[f"metric__{scope}__{metric}__{model}"] = values.astype(
                    np.float64
                )
                lower, upper, valid_replicates = percentile_interval(values)
                _, n_cases, n_observations = metric_estimate(
                    case_stats, scope_indices, metric, model
                )
                interval_records.append(
                    {
                        "scope": scope,
                        "metric": metric,
                        "direction": DIRECTIONS[metric],
                        "model_family": model,
                        "model_id": model_ids[model][MODEL_ROLE[metric]],
                        "estimate": point_lookup[(scope, metric, model)],
                        "ci_lower_2p5": lower,
                        "ci_upper_97p5": upper,
                        "confidence_level": 0.95,
                        "bootstrap_replicates_requested": n_bootstrap,
                        "bootstrap_replicates_valid": valid_replicates,
                        "n_base_cases": n_cases,
                        "n_observations": n_observations,
                        "resampling_unit": "base_sample_id",
                    }
                )
                if valid_replicates < int(0.99 * n_bootstrap):
                    errors.append(
                        f"Too few valid bootstrap replicates for {scope}/{metric}/{model}: "
                        f"{valid_replicates}/{n_bootstrap}"
                    )
            difference = by_model["fno"] - by_model["deeponet"]
            bootstrap_arrays[f"paired_difference__{scope}__{metric}"] = difference.astype(
                np.float64
            )
            lower, upper, valid_replicates = percentile_interval(difference)
            point_difference = (
                point_lookup[(scope, metric, "fno")]
                - point_lookup[(scope, metric, "deeponet")]
            )
            finite = difference[np.isfinite(difference)]
            if DIRECTIONS[metric] == "higher_is_better":
                probability_fno_better = float(np.mean(finite > 0.0))
            else:
                probability_fno_better = float(np.mean(finite < 0.0))
            paired_records.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "direction": DIRECTIONS[metric],
                    "difference_definition": "FNO_minus_DeepONet",
                    "fno_estimate": point_lookup[(scope, metric, "fno")],
                    "deeponet_estimate": point_lookup[(scope, metric, "deeponet")],
                    "paired_difference": point_difference,
                    "ci_lower_2p5": lower,
                    "ci_upper_97p5": upper,
                    "confidence_level": 0.95,
                    "bootstrap_replicates_requested": n_bootstrap,
                    "bootstrap_replicates_valid": valid_replicates,
                    "bootstrap_probability_fno_better": probability_fno_better,
                    "resampling_unit": "base_sample_id",
                }
            )
            if valid_replicates < int(0.99 * n_bootstrap):
                errors.append(
                    f"Too few paired bootstrap replicates for {scope}/{metric}: "
                    f"{valid_replicates}/{n_bootstrap}"
                )

    interval_df = pd.DataFrame(interval_records)
    paired_df = pd.DataFrame(paired_records)
    support_df = support_diagnostics(case_stats)
    for frame_name, frame in (
        ("bootstrap intervals", interval_df),
        ("paired differences", paired_df),
        ("support diagnostics", support_df),
    ):
        numeric_columns = frame.select_dtypes(include=[np.number]).columns
        if frame[numeric_columns].isna().any().any():
            errors.append(f"{frame_name} contains nonfinite numeric output values.")
    if len(point_df) != 12:
        errors.append(f"Expected 12 overall model-metric rows, observed {len(point_df)}.")
    if len(condition_df) != 36:
        errors.append(
            f"Expected 36 condition-specific model-metric rows, observed {len(condition_df)}."
        )
    if len(interval_df) != 48:
        errors.append(f"Expected 48 interval rows, observed {len(interval_df)}.")
    if len(paired_df) != 24:
        errors.append(f"Expected 24 paired-difference rows, observed {len(paired_df)}.")
    for _, row in paired_df.iterrows():
        expected_difference = float(row["fno_estimate"] - row["deeponet_estimate"])
        if not math.isclose(
            float(row["paired_difference"]), expected_difference, rel_tol=0.0, abs_tol=1e-15
        ):
            errors.append(
                f"Paired point difference is inconsistent for {row['scope']}/{row['metric']}."
            )

    validation_core = {
        "passed": not errors,
        "errors": errors,
        "independent_unit": "base_sample_id",
        "paired_models": True,
        "primary_variant": "combined_bz_axial",
        "cohort_rows": int(len(case_stats)),
        "unique_base_sample_ids": int(case_stats["base_sample_id"].nunique()),
        "condition_counts": condition_counts,
        "bootstrap": {
            "replicates": n_bootstrap,
            "random_seed": seed,
            "confidence_level": 0.95,
            "interval_method": "percentile",
            "difference_direction": "FNO_minus_DeepONet",
        },
        "endpoint_policy": {
            "APD90_patch_delta_deployed_R2": (
                "Degenerate empty/full deployed support is assigned 0 ms recovered contrast "
                "on the complete non-homogeneous population. Complete-case results and "
                "degenerate-support counts are exported as sensitivity diagnostics."
            ),
            "center_error": "Euclidean error of the saved parametric center head in normalized coordinates.",
            "field_R2": "Recomputed over all shared finite valid-grid pixels within each replicate.",
            "patch_Dice": "Micro-Dice over concatenated valid-grid pixels from non-homogeneous cases.",
        },
        "case_diagnostics": case_diagnostics,
        "direct_metric_parity": direct_parity,
        "output_row_counts": {
            "overall_point_estimates": int(len(point_df)),
            "condition_specific_point_estimates": int(len(condition_df)),
            "bootstrap_intervals": int(len(interval_df)),
            "paired_differences": int(len(paired_df)),
            "support_diagnostics": int(len(support_df)),
        },
        "safety": {
            "training_run": False,
            "model_inference_run": False,
            "test_set_used_for_selection_or_calibration": False,
            "augmented_rows_treated_as_independent_cases": False,
            "legacy_outputs_modified": False,
        },
    }
    if errors:
        print(json.dumps(sanitize(validation_core), indent=2), flush=True)
        raise SystemExit("Module 2 validation failed; no outputs were published.")

    interval_lookup = {
        (row["scope"], row["metric"], row["model_family"]): row
        for row in interval_records
    }
    summary_payload = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "purpose": "Locked paired 75-base-case Module 2 estimates and uncertainty.",
        "models": model_ids,
        "overall": [
            {
                **record,
                "ci_lower_2p5": interval_lookup[("overall", record["metric"], record["model_family"])]["ci_lower_2p5"],
                "ci_upper_97p5": interval_lookup[("overall", record["metric"], record["model_family"])]["ci_upper_97p5"],
            }
            for record in point_records
        ],
        "overall_paired_differences": [
            record for record in paired_records if record["scope"] == "overall"
        ],
        "condition_specific": condition_records,
        "support_diagnostics": support_df.to_dict(orient="records"),
        "interpretation_boundary": (
            "These are grouped-test estimates for a controlled synthetic benchmark. They do "
            "not establish patient-specific, clinical, real-SQUID, or physics-informed validation."
        ),
    }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".module2_", dir=ANALYSIS_DIR) as temp_name:
        temp_dir = Path(temp_name)
        temp_paths = {key: temp_dir / path.name for key, path in FINAL_FILES.items()}
        case_stats.to_csv(temp_paths["case_statistics"], index=False)
        point_df.to_csv(temp_paths["point_estimates"], index=False)
        condition_df.to_csv(temp_paths["condition_specific"], index=False)
        interval_df.to_csv(temp_paths["bootstrap_intervals"], index=False)
        paired_df.to_csv(temp_paths["paired_differences"], index=False)
        support_df.to_csv(temp_paths["support_diagnostics"], index=False)
        np.savez_compressed(temp_paths["bootstrap_replicates"], **bootstrap_arrays)
        write_json(temp_paths["summary"], summary_payload)
        output_hashes = {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for key, path in temp_paths.items()
            if key != "validation"
        }
        validation_payload = {
            "schema_version": "1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            **validation_core,
            "source_contract": {
                "path": str(CONTRACT_PATH.relative_to(ROOT)),
                "sha256": sha256(CONTRACT_PATH),
            },
            "source_inventory": {
                "path": str(INVENTORY_PATH.relative_to(ROOT)),
                "sha256": sha256(INVENTORY_PATH),
            },
            "source_artifacts": {
                artifact_id: {
                    "path": str(record["path"].relative_to(ROOT)),
                    "sha256": record["sha256"],
                    "bytes": record["bytes"],
                }
                for artifact_id, record in source_artifacts.items()
            },
            "outputs": output_hashes,
        }
        write_json(temp_paths["validation"], validation_payload)
        for key, final_path in FINAL_FILES.items():
            temp_paths[key].replace(final_path)

    print(
        json.dumps(
            {
                "passed": True,
                "status": "complete",
                "bootstrap_replicates": n_bootstrap,
                "overall_point_rows": int(len(point_df)),
                "condition_specific_rows": int(len(condition_df)),
                "paired_difference_rows": int(len(paired_df)),
                "output_directory": str(ANALYSIS_DIR),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
