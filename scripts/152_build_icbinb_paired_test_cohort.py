                      
                                                                 

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

FINAL_TABLE = ANALYSIS_DIR / "paired_test_cohort.csv"
FINAL_MAPS = ANALYSIS_DIR / "paired_test_cohort_maps.npz"
FINAL_VALIDATION = ANALYSIS_DIR / "cohort_validation.json"

PRIMARY_VARIANT = "combined_bz_axial"
KEY_COLUMNS = ["row_id", "base_sample_id", "split", "condition", "combined_variant"]
SOURCE_TRUTH_COLUMNS = {
    "true_patch_present": "patch_present",
    "true_center_x_norm": "center_x_norm",
    "true_center_y_norm": "center_y_norm",
    "true_patch_area_param": "patch_area_fraction",
    "true_radius_param": "equivalent_radius_norm",
    "true_log_area_fraction": "log_area_fraction",
    "true_log_radius": "log_radius",
    "true_area_bin": "area_bin",
    "true_LAT_range": "LAT_range",
    "true_APD90_patch_delta": "APD90_patch_delta",
    "true_patch_area_fraction": "patch_area_fraction",
}
TRUTH_COLUMNS = [
    "true_patch_present",
    "true_center_x_norm",
    "true_center_y_norm",
    "true_patch_area_param",
    "true_radius_param",
    "true_log_area_fraction",
    "true_log_radius",
    "true_area_bin",
    "true_LAT_range",
    "true_APD90_patch_delta",
    "true_patch_area_fraction",
]
LOG_INVERSE_TOLERANCE = 2e-6
LAT_RANGE_SCALAR_TOLERANCE_MS = 3e-6
TRUE_APD90_DELTA_TOLERANCE_MS = 2e-5
MAP_SCALAR_APD90_TOLERANCE_MS = 2e-4


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def verify_source_artifacts(
    inventory: dict[str, Any], artifact_ids: list[str]
) -> dict[str, dict[str, Any]]:
    records = {record["id"]: record for record in inventory["artifacts"]}
    verified: dict[str, dict[str, Any]] = {}
    for artifact_id in artifact_ids:
        if artifact_id not in records:
            raise RuntimeError(f"Artifact {artifact_id!r} is missing from the inventory.")
        record = records[artifact_id]
        path = resolve(record["path"])
        if not path.is_file():
            raise RuntimeError(f"Artifact file is missing: {path}")
        observed_hash = sha256(path)
        if observed_hash != record.get("sha256"):
            raise RuntimeError(
                f"Artifact hash changed for {artifact_id}: "
                f"expected {record.get('sha256')}, observed {observed_hash}"
            )
        verified[artifact_id] = {
            "path": path,
            "sha256": observed_hash,
            "bytes": path.stat().st_size,
        }
    return verified


def compare_columns(
    reference: pd.Series,
    candidate: pd.Series,
    *,
    tolerance: float,
) -> dict[str, Any]:
    ref_numeric = pd.to_numeric(reference, errors="coerce").to_numpy(dtype=float)
    cand_numeric = pd.to_numeric(candidate, errors="coerce").to_numpy(dtype=float)
    ref_nan = ~np.isfinite(ref_numeric)
    cand_nan = ~np.isfinite(cand_numeric)
    nan_pattern_equal = bool(np.array_equal(ref_nan, cand_nan))
    finite = ~ref_nan & ~cand_nan
    max_difference = (
        float(np.max(np.abs(ref_numeric[finite] - cand_numeric[finite])))
        if np.any(finite)
        else 0.0
    )
    return {
        "passed": nan_pattern_equal and max_difference <= tolerance,
        "nan_pattern_equal": nan_pattern_equal,
        "max_absolute_difference": max_difference,
        "tolerance": tolerance,
    }


def compare_scalar_truth(
    reference: pd.Series,
    candidate: pd.Series,
    *,
    column: str,
    patch_present: pd.Series,
) -> dict[str, Any]:
    if column not in {"true_patch_area_param", "true_radius_param"}:
        tolerance = {
            "true_LAT_range": LAT_RANGE_SCALAR_TOLERANCE_MS,
            "true_APD90_patch_delta": TRUE_APD90_DELTA_TOLERANCE_MS,
        }.get(column, 1e-6)
        result = compare_columns(reference, candidate, tolerance=tolerance)
        result["comparison_scope"] = "all_rows"
        return result

    present = pd.to_numeric(patch_present, errors="coerce").to_numpy(dtype=float) > 0.5
    result = compare_columns(
        reference[present],
        candidate[present],
        tolerance=LOG_INVERSE_TOLERANCE,
    )
    candidate_absent = pd.to_numeric(
        candidate[~present], errors="coerce"
    ).to_numpy(dtype=float)
    if column == "true_patch_area_param":
        absent_encoding_valid = bool(
            np.all(~np.isfinite(candidate_absent) | (np.abs(candidate_absent) <= LOG_INVERSE_TOLERANCE))
        )
        accepted_absent_encoding = "zero_or_nan"
    else:
        absent_encoding_valid = bool(np.all(~np.isfinite(candidate_absent)))
        accepted_absent_encoding = "nan"
    result.update(
        {
            "passed": bool(result["passed"] and absent_encoding_valid),
            "comparison_scope": "patch_present_rows",
            "patch_present_rows": int(np.sum(present)),
            "patch_absent_rows": int(np.sum(~present)),
            "patch_absent_encoding_valid": absent_encoding_valid,
            "accepted_patch_absent_encoding": accepted_absent_encoding,
            "rationale": (
                "The canonical raw target is compared on patch-present rows; model exports "
                "may encode absent parameter targets as zero or NaN. FNO inverse-log values "
                "include the target-construction epsilon of 1e-6."
            ),
        }
    )
    return result


def selected_scalar_rows(
    frame: pd.DataFrame,
    model: str,
    canonical_row_ids: list[str],
) -> pd.DataFrame:
    selected = frame[
        frame["model"].eq(model)
        & frame["split"].eq("test")
        & frame["combined_variant"].eq(PRIMARY_VARIANT)
    ].copy()
    if len(selected) != 75 or selected["row_id"].nunique() != 75:
        raise RuntimeError(
            f"Scalar coverage failure for {model}: rows={len(selected)}, "
            f"unique_row_ids={selected['row_id'].nunique()}"
        )
    missing = sorted(set(canonical_row_ids) - set(selected["row_id"]))
    extra = sorted(set(selected["row_id"]) - set(canonical_row_ids))
    if missing or extra:
        raise RuntimeError(
            f"Scalar row mismatch for {model}: missing={missing[:5]}, extra={extra[:5]}"
        )
    return selected.set_index("row_id", drop=False).loc[canonical_row_ids].reset_index(drop=True)


def selected_map_rows(
    index: pd.DataFrame,
    maps_path: Path,
    model: str,
    canonical_row_ids: list[str],
) -> dict[str, np.ndarray]:
    source_positions = (
        pd.to_numeric(index["array_index"], errors="raise").to_numpy(dtype=int)
        if "array_index" in index.columns
        else np.arange(len(index), dtype=int)
    )
    indexed = index.copy()
    indexed["_source_position"] = source_positions
    selected = indexed[
        indexed["model"].eq(model)
        & indexed["split"].eq("test")
        & indexed["combined_variant"].eq(PRIMARY_VARIANT)
    ].copy()
    if len(selected) != 75 or selected["row_id"].nunique() != 75:
        raise RuntimeError(
            f"Map coverage failure for {model}: rows={len(selected)}, "
            f"unique_row_ids={selected['row_id'].nunique()}"
        )
    missing = sorted(set(canonical_row_ids) - set(selected["row_id"]))
    extra = sorted(set(selected["row_id"]) - set(canonical_row_ids))
    if missing or extra:
        raise RuntimeError(
            f"Map row mismatch for {model}: missing={missing[:5]}, extra={extra[:5]}"
        )
    selected = selected.set_index("row_id", drop=False).loc[canonical_row_ids]
    positions = selected["_source_position"].to_numpy(dtype=int)
    required_arrays = [
        "true_lat",
        "true_apd90",
        "true_patch",
        "valid_mask",
        "pred_lat",
        "pred_apd90",
        "pred_patch_prob",
    ]
    with np.load(maps_path) as archive:
        missing_arrays = sorted(set(required_arrays) - set(archive.files))
        if missing_arrays:
            raise RuntimeError(f"Map archive {maps_path} is missing {missing_arrays}")
        arrays = {name: np.asarray(archive[name][positions]) for name in required_arrays}
    arrays["row_id"] = selected["row_id"].to_numpy(dtype=str)
    arrays["base_sample_id"] = selected["base_sample_id"].to_numpy(dtype=str)
    arrays["condition"] = selected["condition"].to_numpy(dtype=str)
    return arrays


def compare_map_truth(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    role: str,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    report: dict[str, Any] = {}
    if not np.array_equal(reference["row_id"], candidate["row_id"]):
        errors.append(f"Row order differs for map role {role}.")
    if not np.array_equal(reference["base_sample_id"], candidate["base_sample_id"]):
        errors.append(f"Base IDs differ for map role {role}.")
    if not np.array_equal(reference["condition"], candidate["condition"]):
        errors.append(f"Conditions differ for map role {role}.")
    for name, tolerance in (
        ("true_lat", 1e-5),
        ("true_apd90", 1e-5),
        ("true_patch", 0.0),
        ("valid_mask", 0.0),
    ):
        if tolerance == 0.0:
            passed = bool(np.array_equal(reference[name], candidate[name], equal_nan=True))
            maximum = 0.0 if passed else math.inf
        else:
            ref = np.asarray(reference[name], dtype=float)
            cand = np.asarray(candidate[name], dtype=float)
            nan_equal = np.array_equal(np.isnan(ref), np.isnan(cand))
            finite = np.isfinite(ref) & np.isfinite(cand)
            maximum = (
                float(np.max(np.abs(ref[finite] - cand[finite]))) if np.any(finite) else 0.0
            )
            passed = bool(nan_equal and maximum <= tolerance)
        report[name] = {
            "passed": passed,
            "max_absolute_difference": maximum,
            "tolerance": tolerance,
        }
        if not passed:
            errors.append(f"Truth map mismatch for {role}/{name}: {report[name]}")
    return errors, report


def per_row_area(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.full(mask.shape[0], np.nan, dtype=np.float64)
    for index in range(mask.shape[0]):
        active = valid[index].astype(bool)
        if np.any(active):
            values[index] = float(np.mean(mask[index][active]))
    return values


def per_row_apd90_delta(
    apd90: np.ndarray,
    mask: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    values = np.full(mask.shape[0], np.nan, dtype=np.float64)
    for index in range(mask.shape[0]):
        active = valid[index].astype(bool) & np.isfinite(apd90[index])
        inside = mask[index].astype(bool) & active
        outside = ~mask[index].astype(bool) & active
        if np.any(inside) and np.any(outside):
            values[index] = float(
                np.mean(apd90[index][inside]) - np.mean(apd90[index][outside])
            )
    return values


def compare_vectors(
    observed: np.ndarray,
    expected: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    observed_finite = np.isfinite(observed)
    expected_finite = np.isfinite(expected)
    finite_pattern_equal = bool(np.array_equal(observed_finite, expected_finite))
    common = observed_finite & expected_finite
    maximum = (
        float(np.max(np.abs(observed[common] - expected[common]))) if np.any(common) else 0.0
    )
    return {
        "passed": finite_pattern_equal and maximum <= tolerance,
        "finite_pattern_equal": finite_pattern_equal,
        "max_absolute_difference": maximum,
        "tolerance": tolerance,
    }


def main() -> int:
    for path in (FINAL_TABLE, FINAL_MAPS, FINAL_VALIDATION):
        if path.exists():
            raise SystemExit(
                f"Refusing to overwrite existing canonical cohort artifact: {path}"
            )
    contract = read_json(CONTRACT_PATH)
    inventory = read_json(INVENTORY_PATH)
    if not contract.get("validation", {}).get("passed"):
        raise SystemExit("Evaluation contract is not valid.")
    if contract["measurement_policy"]["primary_variant"] != PRIMARY_VARIANT:
        raise SystemExit("Contract primary measurement variant changed.")

    source_ids = [
        "grouped_split_json",
        "deeponet_sample_index",
        "fno_predictions_scalar",
        "deeponet_predictions_scalar",
        "fno_map_cache_index",
        "fno_map_cache_arrays",
        "deeponet_map_cache_index",
        "deeponet_map_cache_arrays",
    ]
    sources = verify_source_artifacts(inventory, source_ids)

    roles = {
        "fno_field": contract["model_roles"]["global_field_comparison"]["fno"]["model_id"],
        "deeponet_field": contract["model_roles"]["global_field_comparison"]["deeponet"]["model_id"],
        "fno_localization": contract["model_roles"]["localization_comparison"]["fno"]["model_id"],
        "deeponet_localization": contract["model_roles"]["localization_comparison"]["deeponet"]["model_id"],
    }

    split_payload = read_json(sources["grouped_split_json"]["path"])
    expected_test_ids = sorted(
        base_id
        for base_id, split in split_payload["base_split"].items()
        if split == "test"
    )
    if len(expected_test_ids) != 75:
        raise SystemExit(f"Grouped split contains {len(expected_test_ids)} test bases, not 75.")

    sample_index = pd.read_csv(sources["deeponet_sample_index"]["path"])
    required_sample_columns = KEY_COLUMNS + sorted(set(SOURCE_TRUTH_COLUMNS.values()))
    missing_sample_columns = sorted(set(required_sample_columns) - set(sample_index.columns))
    if missing_sample_columns:
        raise SystemExit(
            f"Primary sample index lacks canonical truth columns: {missing_sample_columns}"
        )
    canonical_source = sample_index[
        sample_index["split"].eq("test")
        & sample_index["combined_variant"].eq(PRIMARY_VARIANT)
    ][required_sample_columns].copy()
    canonical_source = canonical_source.sort_values("base_sample_id").reset_index(drop=True)
    canonical = canonical_source[KEY_COLUMNS].copy()
    if len(canonical) != 75 or canonical["base_sample_id"].nunique() != 75:
        raise SystemExit("Primary sample index does not contain exactly 75 unique test bases.")
    if canonical["base_sample_id"].tolist() != expected_test_ids:
        raise SystemExit("Primary sample-index IDs differ from the grouped split test IDs.")
    canonical_row_ids = canonical["row_id"].tolist()

    fno_scalars = pd.read_csv(sources["fno_predictions_scalar"]["path"])
    deep_scalars = pd.read_csv(sources["deeponet_predictions_scalar"]["path"])
    scalar_sources = {
        "fno_field": selected_scalar_rows(
            fno_scalars, roles["fno_field"], canonical_row_ids
        ),
        "fno_localization": selected_scalar_rows(
            fno_scalars, roles["fno_localization"], canonical_row_ids
        ),
        "deeponet_field": selected_scalar_rows(
            deep_scalars, roles["deeponet_field"], canonical_row_ids
        ),
        "deeponet_localization": selected_scalar_rows(
            deep_scalars, roles["deeponet_localization"], canonical_row_ids
        ),
    }

    errors: list[str] = []
    scalar_truth_report: dict[str, Any] = {}
    cohort = canonical.copy()
    cohort.insert(0, "cohort_index", np.arange(len(cohort), dtype=int))
    for column in TRUTH_COLUMNS:
        cohort[column] = canonical_source[SOURCE_TRUTH_COLUMNS[column]].to_numpy()
    for role, frame in scalar_sources.items():
        if frame[KEY_COLUMNS].astype(str).to_numpy().tolist() != canonical[KEY_COLUMNS].astype(str).to_numpy().tolist():
            errors.append(f"Scalar key/order mismatch for {role}.")
        role_report: dict[str, Any] = {}
        for column in TRUTH_COLUMNS:
            if column not in frame.columns:
                errors.append(f"Scalar source {role} lacks truth column {column}.")
                continue
            result = compare_scalar_truth(
                cohort[column],
                frame[column],
                column=column,
                patch_present=cohort["true_patch_present"],
            )
            role_report[column] = result
            if not result["passed"]:
                errors.append(f"Scalar truth mismatch for {role}/{column}: {result}")
        scalar_truth_report[role] = role_report
        prediction_columns = [
            column
            for column in frame.columns
            if column.startswith("pred_") or column.startswith("homogeneous_false_positive_")
        ]
        for column in prediction_columns:
            cohort[f"{role}__{column}"] = frame[column].to_numpy()

    fno_map_index = pd.read_csv(sources["fno_map_cache_index"]["path"])
    deep_map_index = pd.read_csv(sources["deeponet_map_cache_index"]["path"])
    map_sources = {
        "fno_field": selected_map_rows(
            fno_map_index,
            sources["fno_map_cache_arrays"]["path"],
            roles["fno_field"],
            canonical_row_ids,
        ),
        "fno_localization": selected_map_rows(
            fno_map_index,
            sources["fno_map_cache_arrays"]["path"],
            roles["fno_localization"],
            canonical_row_ids,
        ),
        "deeponet_field": selected_map_rows(
            deep_map_index,
            sources["deeponet_map_cache_arrays"]["path"],
            roles["deeponet_field"],
            canonical_row_ids,
        ),
        "deeponet_localization": selected_map_rows(
            deep_map_index,
            sources["deeponet_map_cache_arrays"]["path"],
            roles["deeponet_localization"],
            canonical_row_ids,
        ),
    }

    map_truth_report: dict[str, Any] = {}
    map_reference = map_sources["fno_field"]
    for role, arrays in map_sources.items():
        role_errors, role_report = compare_map_truth(map_reference, arrays, role)
        errors.extend(role_errors)
        map_truth_report[role] = role_report
        valid = arrays["valid_mask"].astype(bool)
        for name in ("pred_lat", "pred_apd90", "pred_patch_prob"):
            nonfinite = int(np.sum(~np.isfinite(arrays[name][valid])))
            if nonfinite:
                errors.append(f"Found {nonfinite} nonfinite valid pixels in {role}/{name}.")

    output_arrays: dict[str, np.ndarray] = {
        "true_lat": map_reference["true_lat"].astype(np.float32),
        "true_apd90": map_reference["true_apd90"].astype(np.float32),
        "true_patch": map_reference["true_patch"].astype(np.float32),
        "valid_mask": map_reference["valid_mask"].astype(bool),
    }
    for role, arrays in map_sources.items():
        output_arrays[f"{role}_pred_lat"] = arrays["pred_lat"].astype(np.float32)
        output_arrays[f"{role}_pred_apd90"] = arrays["pred_apd90"].astype(np.float32)
        output_arrays[f"{role}_pred_patch_prob"] = arrays["pred_patch_prob"].astype(np.float32)
    for role in ("fno_localization", "deeponet_localization"):
        output_arrays[f"{role}_pred_patch_mask"] = (
            output_arrays[f"{role}_pred_patch_prob"] > 0.5
        ) & output_arrays["valid_mask"]

    scalar_map_alignment: dict[str, Any] = {}
    true_delta_from_maps = per_row_apd90_delta(
        output_arrays["true_apd90"],
        output_arrays["true_patch"] > 0.5,
        output_arrays["valid_mask"],
    )
    if "true_APD90_patch_delta" in cohort.columns:
        comparison = compare_vectors(
            true_delta_from_maps,
            pd.to_numeric(cohort["true_APD90_patch_delta"], errors="coerce").to_numpy(),
            TRUE_APD90_DELTA_TOLERANCE_MS,
        )
        scalar_map_alignment["true_APD90_patch_delta"] = comparison
        if not comparison["passed"]:
            errors.append(f"True APD90 delta scalar/map mismatch: {comparison}")
    for role in ("fno_localization", "deeponet_localization"):
        predicted_mask = output_arrays[f"{role}_pred_patch_mask"]
        predicted_area = per_row_area(predicted_mask, output_arrays["valid_mask"])
        predicted_delta = per_row_apd90_delta(
            output_arrays[f"{role}_pred_apd90"],
            predicted_mask,
            output_arrays["valid_mask"],
        )
        cohort[f"{role}__map_patch_area_fraction"] = predicted_area
        cohort[f"{role}__map_APD90_patch_delta"] = predicted_delta
        area_column = f"{role}__pred_patch_area_fraction_pixel_mask"
        delta_column = f"{role}__pred_APD90_patch_delta_pixel_mask_pred_apd"
        role_report: dict[str, Any] = {}
        if area_column in cohort.columns:
            role_report["patch_area_fraction"] = compare_vectors(
                predicted_area,
                pd.to_numeric(cohort[area_column], errors="coerce").to_numpy(),
                1e-7,
            )
        else:
            errors.append(f"Missing scalar area column {area_column}.")
        if delta_column in cohort.columns:
            role_report["APD90_patch_delta"] = compare_vectors(
                predicted_delta,
                pd.to_numeric(cohort[delta_column], errors="coerce").to_numpy(),
                MAP_SCALAR_APD90_TOLERANCE_MS,
            )
        else:
            errors.append(f"Missing scalar delta column {delta_column}.")
        scalar_map_alignment[role] = role_report
        for metric, result in role_report.items():
            if not result["passed"]:
                errors.append(f"Scalar/map mismatch for {role}/{metric}: {result}")

    condition_counts = {
        str(key): int(value)
        for key, value in cohort["condition"].value_counts().sort_index().items()
    }
    expected_conditions = {
        "apd_repolarization_perturbation": 18,
        "conduction_slowing_perturbation": 18,
        "homogeneous": 12,
        "mixed_apd_conduction_perturbation": 27,
    }
    if condition_counts != expected_conditions:
        errors.append(
            f"Condition counts changed: expected {expected_conditions}, observed {condition_counts}"
        )
    if set(cohort["split"]) != {"test"}:
        errors.append(f"Non-test rows entered the cohort: {sorted(set(cohort['split']))}")
    if cohort["base_sample_id"].duplicated().any():
        errors.append("Duplicate base_sample_id values entered the canonical cohort.")

    validation_core = {
        "passed": not errors,
        "errors": errors,
        "independent_unit": "base_sample_id",
        "primary_variant": PRIMARY_VARIANT,
        "row_count": int(len(cohort)),
        "unique_base_sample_ids": int(cohort["base_sample_id"].nunique()),
        "condition_counts": condition_counts,
        "model_roles": roles,
        "pairing_keys": ["base_sample_id", "combined_variant"],
        "no_train_or_validation_rows": set(cohort["split"]) == {"test"},
        "scalar_truth_alignment": scalar_truth_report,
        "map_truth_alignment": map_truth_report,
        "scalar_map_alignment": scalar_map_alignment,
        "numeric_tolerance_policy": {
            "raw_vs_inverse_log_area_and_radius": LOG_INVERSE_TOLERANCE,
            "LAT_range_scalar_reduction_ms": LAT_RANGE_SCALAR_TOLERANCE_MS,
            "true_APD90_patch_delta_scalar_reduction_ms": TRUE_APD90_DELTA_TOLERANCE_MS,
            "fno_cached_map_vs_legacy_scalar_APD90_patch_delta_ms": MAP_SCALAR_APD90_TOLERANCE_MS,
            "notes": [
                "Canonical scalar truths come from the model-ready sample index, not a model export.",
                "FNO area/radius truths were legacy-exported by exponentiating log(value + 1e-6).",
                "LAT range and true APD90 delta tolerances cover observed float32 field-reduction serialization drift.",
                "The APD90 tolerance bounds observed float32 inference/reduction drift and is negligible relative to millisecond-scale outcomes.",
            ],
        },
    }
    if errors:
        print(json.dumps(validation_core, indent=2), flush=True)
        raise SystemExit("Canonical paired cohort validation failed; no outputs were published.")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".paired_cohort_", dir=ANALYSIS_DIR) as temp_name:
        temp_dir = Path(temp_name)
        temp_table = temp_dir / FINAL_TABLE.name
        temp_maps = temp_dir / FINAL_MAPS.name
        temp_validation = temp_dir / FINAL_VALIDATION.name
        cohort.to_csv(temp_table, index=False)
        np.savez_compressed(temp_maps, **output_arrays)
        output_hashes = {
            "paired_test_cohort.csv": {
                "sha256": sha256(temp_table),
                "bytes": temp_table.stat().st_size,
            },
            "paired_test_cohort_maps.npz": {
                "sha256": sha256(temp_maps),
                "bytes": temp_maps.stat().st_size,
            },
        }
        validation_payload = {
            "schema_version": "1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            **validation_core,
            "map_arrays": {
                name: {"shape": list(array.shape), "dtype": str(array.dtype)}
                for name, array in output_arrays.items()
            },
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
                for artifact_id, record in sources.items()
            },
            "outputs": output_hashes,
            "safety": {
                "source_files_modified": False,
                "test_rows_used_for_model_or_threshold_selection": False,
                "augmented_rows_treated_as_independent_biological_cases": False,
            },
        }
        write_json(temp_validation, validation_payload)
        temp_table.replace(FINAL_TABLE)
        temp_maps.replace(FINAL_MAPS)
        temp_validation.replace(FINAL_VALIDATION)

    print(
        json.dumps(
            {
                "passed": True,
                "table": str(FINAL_TABLE),
                "maps": str(FINAL_MAPS),
                "validation": str(FINAL_VALIDATION),
                "rows": int(len(cohort)),
                "condition_counts": condition_counts,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
