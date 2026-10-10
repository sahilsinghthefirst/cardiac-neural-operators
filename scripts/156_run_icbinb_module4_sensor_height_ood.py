from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from v1e_deeponet_data_common import DeepONetStats, make_loader as make_deep_loader, torch
from v1e_field_data_common import make_loader as make_fno_loader


MODULE4_DIR = PROJECT_ROOT / "outputs" / "icbinb_sensor_height_ood"
MANIFEST_PATH = MODULE4_DIR / "sensor_height_manifest.csv"
GENERATION_VALIDATION_PATH = MODULE4_DIR / "generation_validation.json"
ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "icbinb_required_analysis"
CONTRACT_PATH = ANALYSIS_DIR / "evaluation_contract.json"
INVENTORY_PATH = ANALYSIS_DIR / "artifact_inventory.json"
COHORT_PATH = ANALYSIS_DIR / "paired_test_cohort.csv"
CANONICAL_MAPS_PATH = ANALYSIS_DIR / "paired_test_cohort_maps.npz"
FIELD_INDEX_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_v1e_500_combined_pilot"
    / "field_ready"
    / "sample_index.csv"
)

FINAL_FILES = {
    "prediction_index": MODULE4_DIR / "module4_prediction_index.csv",
    "prediction_maps": MODULE4_DIR / "module4_prediction_maps.npz",
    "case_statistics": MODULE4_DIR / "module4_case_statistics.csv",
    "point_estimates": MODULE4_DIR / "module4_point_estimates.csv",
    "bootstrap_intervals": MODULE4_DIR / "module4_bootstrap_intervals.csv",
    "height_deltas": MODULE4_DIR / "module4_height_deltas.csv",
    "architecture_differences": MODULE4_DIR / "module4_architecture_differences.csv",
    "condition_specific": MODULE4_DIR / "module4_condition_specific.csv",
    "support_diagnostics": MODULE4_DIR / "module4_support_diagnostics.csv",
    "bootstrap_replicates": MODULE4_DIR / "module4_bootstrap_replicates.npz",
    "figure_png": MODULE4_DIR / "module4_sensor_height_ood.png",
    "figure_pdf": MODULE4_DIR / "module4_sensor_height_ood.pdf",
    "summary": MODULE4_DIR / "module4_summary.json",
    "validation": MODULE4_DIR / "module4_validation.json",
}

HEIGHTS_MM = (35, 50, 65)
MODELS = ("fno", "deeponet")
MODEL_IDS = {
    "fno": "fno_field_only",
    "deeponet": "deeponet_cnn_branch_fourier_trunk",
}
METRICS = (
    "LAT_global_R2",
    "APD90_global_R2",
    "patch_Dice_nonhomogeneous_micro",
    "patch_center_error_deployed_mean",
    "homogeneous_false_positive_area_mean",
    "APD90_patch_delta_deployed_R2",
)
PRIMARY_METRICS = {"LAT_global_R2", "APD90_global_R2"}
LOWER_IS_BETTER = {
    "patch_center_error_deployed_mean",
    "homogeneous_false_positive_area_mean",
}
EXPECTED_CONDITIONS = (
    "apd_repolarization_perturbation",
    "conduction_slowing_perturbation",
    "homogeneous",
    "mixed_apd_conduction_perturbation",
)


def load_numeric_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FNO_EXPORT = load_numeric_module(
    "_icbinb_fno_export", PROJECT_ROOT / "scripts" / "151_export_icbinb_fno_full_grid_cache.py"
)
DEEP_EVAL = load_numeric_module(
    "_icbinb_deeponet_eval", PROJECT_ROOT / "scripts" / "129_evaluate_v1e_500_deeponet.py"
)


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
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_deep_stats(payload: dict[str, Any]) -> DeepONetStats:
    return DeepONetStats(
        input_mean=float(payload["input_mean"]),
        input_std=float(payload["input_std"]),
        field_mean=np.asarray(payload["field_mean"], dtype=np.float32),
        field_std=np.asarray(payload["field_std"], dtype=np.float32),
        param_mean=np.asarray(payload["param_mean"], dtype=np.float32),
        param_std=np.asarray(payload["param_std"], dtype=np.float32),
        descriptor_mean=np.asarray(payload["descriptor_mean"], dtype=np.float32),
        descriptor_std=np.asarray(payload["descriptor_std"], dtype=np.float32),
        descriptor_columns=list(payload["descriptor_columns"]),
    )


def prerequisite_state() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    required = (
        MANIFEST_PATH,
        GENERATION_VALIDATION_PATH,
        CONTRACT_PATH,
        INVENTORY_PATH,
        COHORT_PATH,
        CANONICAL_MAPS_PATH,
        FIELD_INDEX_PATH,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing Module 4 prerequisites: {missing}")
    existing = [str(path) for path in FINAL_FILES.values() if path.exists()]
    if existing:
        raise SystemExit(f"Refusing to overwrite existing Module 4 analysis outputs: {existing}")
    generation = read_json(GENERATION_VALIDATION_PATH)
    contract = read_json(CONTRACT_PATH)
    inventory = read_json(INVENTORY_PATH)
    if not generation.get("passed"):
        raise SystemExit("The Module 4 input-generation validation did not pass.")
    if not contract.get("validation", {}).get("passed"):
        raise SystemExit("The locked ICBINB evaluation contract is invalid.")
    artifacts = {record["id"]: record for record in inventory["artifacts"]}
    for artifact_id in ("fno_checkpoint_field", "deeponet_checkpoint_field", "fno_training_statistics"):
        if artifact_id not in artifacts:
            raise SystemExit(f"Locked artifact {artifact_id!r} is missing from the inventory.")
        path = resolve(artifacts[artifact_id]["path"])
        if not path.is_file() or sha256(path) != artifacts[artifact_id]["sha256"]:
            raise SystemExit(f"Locked artifact failed hash validation: {artifact_id}")
    return generation, contract, artifacts


def build_height_rows() -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    cohort = pd.read_csv(COHORT_PATH).sort_values("cohort_index").reset_index(drop=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    source = pd.read_csv(FIELD_INDEX_PATH)
    source = source[
        source["split"].eq("test") & source["combined_variant"].eq("combined_bz_axial")
    ].copy()
    source = cohort[["cohort_index", "base_sample_id"]].merge(
        source, on="base_sample_id", how="left", validate="one_to_one"
    ).sort_values("cohort_index").reset_index(drop=True)
    if len(source) != 75 or source["B_meas_file"].isna().any():
        raise SystemExit("Could not rebuild the canonical 75-case source index.")
    if set(manifest["height_mm"].astype(int)) != set(HEIGHTS_MM):
        raise SystemExit("Unexpected sensor heights in the Module 4 manifest.")

    by_height: dict[int, pd.DataFrame] = {}
    path_columns = (
        "B_meas_file",
        "sensor_positions_file",
        "sensor_mask_file",
        "coordinate_channels_file",
        "time_ms_file",
        "measurement_metadata_file",
    )
    for height in HEIGHTS_MM:
        height_manifest = manifest[manifest["height_mm"].astype(int).eq(height)][
            ["cohort_index", "base_sample_id", "ood_row_id", *path_columns]
        ].copy()
        rows = source.drop(columns=list(path_columns)).merge(
            height_manifest,
            on=["cohort_index", "base_sample_id"],
            how="left",
            validate="one_to_one",
        ).sort_values("cohort_index").reset_index(drop=True)
        rows["source_row_id"] = rows["row_id"]
        rows["row_id"] = rows["ood_row_id"]
        rows["height_mm"] = height
        if len(rows) != 75 or rows[list(path_columns)].isna().any().any():
            raise SystemExit(f"Incomplete inference rows at {height} mm.")
        for column in path_columns:
            missing = [str(resolve(value)) for value in rows[column] if not resolve(value).is_file()]
            if missing:
                raise SystemExit(f"Missing files at {height} mm in {column}: {missing[:3]}")
        by_height[height] = rows
    return cohort, by_height


def run_fno(rows: pd.DataFrame, checkpoint_path: Path, stats, device, batch_size: int):
    model, replay = FNO_EXPORT.build_model(checkpoint_path, stats, device)
    loader = make_fno_loader(
        rows,
        stats,
        batch_size=batch_size,
        shuffle=False,
        input_norm=replay["input_norm"],
        descriptors=None,
    )
    truth = {key: [] for key in ("lat", "apd90", "patch", "valid")}
    pred = {key: [] for key in ("lat", "apd90", "patch_prob")}
    with torch.inference_mode():
        for batch in loader:
            field, _params, _area, _condition = model(
                batch["x"].to(device).float(), batch["desc"].to(device).float()
            )
            field_np = field.detach().cpu().numpy()
            target = batch["y_fields"].numpy()
            truth["lat"].append(target[:, 0] * stats.field_std[0] + stats.field_mean[0])
            truth["apd90"].append(target[:, 2] * stats.field_std[2] + stats.field_mean[2])
            truth["patch"].append(target[:, 3])
            truth["valid"].append(batch["valid_mask"].numpy().astype(bool))
            pred["lat"].append(field_np[:, 0] * stats.field_std[0] + stats.field_mean[0])
            pred["apd90"].append(field_np[:, 2] * stats.field_std[2] + stats.field_mean[2])
            pred["patch_prob"].append(torch.sigmoid(field[:, 3]).detach().cpu().numpy())
    del loader, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    truth_arrays = {
        key: np.concatenate(value).astype(bool if key == "valid" else np.float32)
        for key, value in truth.items()
    }
    return (
        truth_arrays,
        {key: np.concatenate(value).astype(np.float32) for key, value in pred.items()},
        replay,
    )


def run_deeponet(rows: pd.DataFrame, checkpoint_path: Path, device, batch_size: int):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stats = load_deep_stats(checkpoint["stats"])
    model = DEEP_EVAL.build_model_from_checkpoint(checkpoint, len(stats.descriptor_columns))
    model.to(device)
    model.eval()
    input_norm = str(checkpoint.get("args", {}).get("input_norm", "per_sample"))
    loader = make_deep_loader(
        rows,
        stats,
        batch_size=batch_size,
        shuffle=False,
        mode="full",
        k_queries=1024,
        input_norm=input_norm,
        descriptors=None,
        seed=20260901,
    )
    truth = {key: [] for key in ("lat", "apd90", "patch", "valid")}
    pred = {key: [] for key in ("lat", "apd90", "patch_prob")}
    with torch.inference_mode():
        for batch in loader:
            query, _params, _area, _condition = model(
                batch["branch_x"].to(device).float(),
                batch["trunk_xy"].to(device).float(),
                batch["descriptor_x"].to(device).float(),
            )
            query_np = query.detach().cpu().numpy()
            target = batch["y_query"].numpy()
            batch_count = query_np.shape[0]
            truth["lat"].append(
                (target[:, :, 0] * stats.field_std[0] + stats.field_mean[0]).reshape(batch_count, 32, 32)
            )
            truth["apd90"].append(
                (target[:, :, 2] * stats.field_std[2] + stats.field_mean[2]).reshape(batch_count, 32, 32)
            )
            truth["patch"].append(target[:, :, 3].reshape(batch_count, 32, 32))
            truth["valid"].append(
                batch["valid_query_mask"].numpy().astype(bool).reshape(batch_count, 32, 32)
            )
            pred["lat"].append(
                (query_np[:, :, 0] * stats.field_std[0] + stats.field_mean[0]).reshape(batch_count, 32, 32)
            )
            pred["apd90"].append(
                (query_np[:, :, 2] * stats.field_std[2] + stats.field_mean[2]).reshape(batch_count, 32, 32)
            )
            pred["patch_prob"].append(
                torch.sigmoid(query[:, :, 3]).detach().cpu().numpy().reshape(batch_count, 32, 32)
            )
    replay = {
        "input_norm": input_norm,
        "branch_in_channels": int(checkpoint["branch_in_channels"]),
        "use_descriptors": bool(checkpoint.get("config", {}).get("descriptors", False)),
        "fourier_trunk": bool(checkpoint.get("config", {}).get("fourier", False)),
        "sensor_coordinate_or_height_metadata_consumed": False,
    }
    del loader, model, checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    truth_arrays = {
        key: np.concatenate(value).astype(bool if key == "valid" else np.float32)
        for key, value in truth.items()
    }
    return (
        truth_arrays,
        {key: np.concatenate(value).astype(np.float32) for key, value in pred.items()},
        replay,
    )


def scalar_r2(true: np.ndarray, pred: np.ndarray) -> float:
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    finite = np.isfinite(true) & np.isfinite(pred)
    if finite.sum() < 2 or np.allclose(true[finite], true[finite][0]):
        return math.nan
    residual = float(np.sum((true[finite] - pred[finite]) ** 2))
    total = float(np.sum((true[finite] - np.mean(true[finite])) ** 2))
    return 1.0 - residual / total if total > 0 else math.nan


def r2_sufficient(n, true_sum, true_sumsq, sse):
    n = np.asarray(n, dtype=np.float64)
    true_sum = np.asarray(true_sum, dtype=np.float64)
    true_sumsq = np.asarray(true_sumsq, dtype=np.float64)
    sse = np.asarray(sse, dtype=np.float64)
    total = true_sumsq - true_sum * true_sum / np.maximum(n, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = 1.0 - sse / total
    result = np.where((n >= 2) & (total > 0), result, np.nan)
    return float(result) if result.ndim == 0 else result


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    y, x = np.nonzero(mask)
    if len(x) == 0:
        return None
    return float(np.mean(x) / max(mask.shape[1] - 1, 1)), float(np.mean(y) / max(mask.shape[0] - 1, 1))


def patch_delta(field: np.ndarray, support: np.ndarray, valid: np.ndarray) -> float:
    inside = support & valid & np.isfinite(field)
    outside = (~support) & valid & np.isfinite(field)
    if not inside.any() or not outside.any():
        return math.nan
    return float(np.mean(field[inside]) - np.mean(field[outside]))


def build_case_statistics(cohort, truth, predictions) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    valid_all = truth["valid"].astype(bool)
    true_patch_all = truth["patch"] > 0.5
    for model in MODELS:
        for height in HEIGHTS_MM:
            pred = predictions[(model, height)]
            for index, source in cohort.iterrows():
                valid = valid_all[index]
                true_patch = true_patch_all[index] & valid
                pred_mask = (pred["patch_prob"][index] > 0.5) & valid
                present = bool(true_patch.any())
                true_center = centroid(true_patch)
                pred_center = centroid(pred_mask)
                if present and true_center is not None and pred_center is not None:
                    center_error = float(math.hypot(pred_center[0] - true_center[0], pred_center[1] - true_center[1]))
                elif present:
                    center_error = math.sqrt(2.0)
                else:
                    center_error = math.nan
                record: dict[str, Any] = {
                    "cohort_index": int(source["cohort_index"]),
                    "base_sample_id": str(source["base_sample_id"]),
                    "condition": str(source["condition"]),
                    "model_family": model,
                    "model_id": MODEL_IDS[model],
                    "height_mm": height,
                    "is_nonhomogeneous": present,
                    "is_homogeneous": not present,
                    "valid_pixel_count": int(valid.sum()),
                    "true_patch_pixel_count": int(true_patch.sum()),
                    "predicted_patch_pixel_count": int(pred_mask.sum()),
                    "predicted_support_nondegenerate": bool(pred_mask.any() and ((~pred_mask) & valid).any()),
                    "patch_tp": int(np.sum(pred_mask & true_patch)),
                    "patch_fp": int(np.sum(pred_mask & ~true_patch & valid)),
                    "patch_fn": int(np.sum(~pred_mask & true_patch)),
                    "predicted_patch_area": float(np.mean(pred_mask[valid])),
                    "center_error_deployed": center_error,
                    "true_APD90_patch_delta": patch_delta(truth["apd90"][index], true_patch, valid),
                }
                raw_delta = patch_delta(pred["apd90"][index], pred_mask, valid)
                record["raw_APD90_patch_delta"] = raw_delta
                record["deployed_APD90_patch_delta"] = raw_delta if np.isfinite(raw_delta) else 0.0
                for field in ("lat", "apd90"):
                    true_field = truth[field][index].astype(np.float64)
                    pred_field = pred[field][index].astype(np.float64)
                    active = valid & np.isfinite(true_field) & np.isfinite(pred_field)
                    true_values = true_field[active]
                    residual = true_values - pred_field[active]
                    record[f"{field}_n_pixels"] = int(true_values.size)
                    record[f"{field}_true_sum"] = float(np.sum(true_values, dtype=np.float64))
                    record[f"{field}_true_sumsq"] = float(np.sum(true_values * true_values, dtype=np.float64))
                    record[f"{field}_sse"] = float(np.sum(residual * residual, dtype=np.float64))
                records.append(record)
    return pd.DataFrame(records).sort_values(["model_family", "height_mm", "cohort_index"]).reset_index(drop=True)


def applicable(table: pd.DataFrame, metric: str) -> np.ndarray:
    if metric in {
        "patch_Dice_nonhomogeneous_micro",
        "patch_center_error_deployed_mean",
        "APD90_patch_delta_deployed_R2",
    }:
        return table["is_nonhomogeneous"].to_numpy(dtype=bool)
    if metric == "homogeneous_false_positive_area_mean":
        return table["is_homogeneous"].to_numpy(dtype=bool)
    return np.ones(len(table), dtype=bool)


def estimate(table: pd.DataFrame, metric: str, selected: np.ndarray | None = None) -> tuple[float, int, int]:
    if selected is None:
        selected = np.arange(len(table), dtype=np.int64)
    selected = np.asarray(selected, dtype=np.int64)
    selected = selected[applicable(table, metric)[selected]]
    if selected.size == 0:
        return math.nan, 0, 0
    if metric in {"LAT_global_R2", "APD90_global_R2"}:
        prefix = "lat" if metric.startswith("LAT") else "apd90"
        n = int(table[f"{prefix}_n_pixels"].to_numpy()[selected].sum())
        value = r2_sufficient(
            n,
            table[f"{prefix}_true_sum"].to_numpy()[selected].sum(),
            table[f"{prefix}_true_sumsq"].to_numpy()[selected].sum(),
            table[f"{prefix}_sse"].to_numpy()[selected].sum(),
        )
        return float(value), int(selected.size), n
    if metric == "patch_Dice_nonhomogeneous_micro":
        tp = float(table["patch_tp"].to_numpy()[selected].sum())
        fp = float(table["patch_fp"].to_numpy()[selected].sum())
        fn = float(table["patch_fn"].to_numpy()[selected].sum())
        denominator = 2.0 * tp + fp + fn
        value = 2.0 * tp / denominator if denominator > 0 else math.nan
        return value, int(selected.size), int(table["valid_pixel_count"].to_numpy()[selected].sum())
    if metric == "patch_center_error_deployed_mean":
        values = table["center_error_deployed"].to_numpy(dtype=np.float64)[selected]
        return float(np.mean(values)), int(selected.size), int(selected.size)
    if metric == "homogeneous_false_positive_area_mean":
        values = table["predicted_patch_area"].to_numpy(dtype=np.float64)[selected]
        return float(np.mean(values)), int(selected.size), int(selected.size)
    if metric == "APD90_patch_delta_deployed_R2":
        true = table["true_APD90_patch_delta"].to_numpy(dtype=np.float64)[selected]
        pred = table["deployed_APD90_patch_delta"].to_numpy(dtype=np.float64)[selected]
        return scalar_r2(true, pred), int(selected.size), int(selected.size)
    raise KeyError(metric)


def bootstrap_values(table: pd.DataFrame, metric: str, draws: np.ndarray) -> np.ndarray:
    weights = applicable(table, metric).astype(np.float64)
    if metric in {"LAT_global_R2", "APD90_global_R2"}:
        prefix = "lat" if metric.startswith("LAT") else "apd90"
        n = (table[f"{prefix}_n_pixels"].to_numpy(dtype=np.float64) * weights)[draws].sum(axis=1)
        true_sum = (table[f"{prefix}_true_sum"].to_numpy(dtype=np.float64) * weights)[draws].sum(axis=1)
        true_sumsq = (table[f"{prefix}_true_sumsq"].to_numpy(dtype=np.float64) * weights)[draws].sum(axis=1)
        sse = (table[f"{prefix}_sse"].to_numpy(dtype=np.float64) * weights)[draws].sum(axis=1)
        return r2_sufficient(n, true_sum, true_sumsq, sse)
    if metric == "patch_Dice_nonhomogeneous_micro":
        tp = (table["patch_tp"].to_numpy(dtype=np.float64) * weights)[draws].sum(axis=1)
        fp = (table["patch_fp"].to_numpy(dtype=np.float64) * weights)[draws].sum(axis=1)
        fn = (table["patch_fn"].to_numpy(dtype=np.float64) * weights)[draws].sum(axis=1)
        denominator = 2.0 * tp + fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            values = 2.0 * tp / denominator
        return np.where(denominator > 0, values, np.nan)
    if metric in {"patch_center_error_deployed_mean", "homogeneous_false_positive_area_mean"}:
        column = "center_error_deployed" if metric.startswith("patch_center") else "predicted_patch_area"
        values = table[column].to_numpy(dtype=np.float64)
        finite = np.isfinite(values).astype(np.float64) * weights
        safe = np.where(np.isfinite(values), values, 0.0) * weights
        counts = finite[draws].sum(axis=1)
        totals = safe[draws].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = totals / counts
        return np.where(counts > 0, result, np.nan)
    if metric == "APD90_patch_delta_deployed_R2":
        true = table["true_APD90_patch_delta"].to_numpy(dtype=np.float64)
        pred = table["deployed_APD90_patch_delta"].to_numpy(dtype=np.float64)
        finite = np.isfinite(true) & np.isfinite(pred)
        case_weights = weights * finite
        true_safe = np.where(finite, true, 0.0)
        pred_safe = np.where(finite, pred, 0.0)
        n = case_weights[draws].sum(axis=1)
        true_sum = (true_safe * case_weights)[draws].sum(axis=1)
        true_sumsq = (true_safe * true_safe * case_weights)[draws].sum(axis=1)
        sse = ((true_safe - pred_safe) ** 2 * case_weights)[draws].sum(axis=1)
        return r2_sufficient(n, true_sum, true_sumsq, sse)
    raise KeyError(metric)


def interval(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return math.nan, math.nan, 0
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high), int(finite.size)


def analyze(case_stats: pd.DataFrame, replicates: int, seed: int):
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, 75, size=(replicates, 75), dtype=np.int16)
    point_records: list[dict[str, Any]] = []
    interval_records: list[dict[str, Any]] = []
    condition_records: list[dict[str, Any]] = []
    replicate_arrays: dict[tuple[str, int, str], np.ndarray] = {}
    packed_replicates: dict[str, np.ndarray] = {}

    for model in MODELS:
        for height in HEIGHTS_MM:
            table = case_stats[
                case_stats["model_family"].eq(model) & case_stats["height_mm"].eq(height)
            ].sort_values("cohort_index").reset_index(drop=True)
            if len(table) != 75:
                raise RuntimeError(f"Expected 75 cases for {model}/{height} mm.")
            for metric in METRICS:
                value, n_cases, n_obs = estimate(table, metric)
                reps = bootstrap_values(table, metric, draws)
                low, high, finite_reps = interval(reps)
                replicate_arrays[(model, height, metric)] = reps
                packed_replicates[f"{model}__h{height:03d}__{metric}"] = reps.astype(np.float32)
                common = {
                    "scope": "overall",
                    "model_family": model,
                    "model_id": MODEL_IDS[model],
                    "height_mm": height,
                    "metric": metric,
                    "endpoint_role": "primary" if metric in PRIMARY_METRICS else "secondary_diagnostic",
                    "estimate": value,
                    "eligible_base_cases": n_cases,
                    "observations": n_obs,
                }
                point_records.append(common)
                interval_records.append(
                    {
                        **common,
                        "ci_lower": low,
                        "ci_upper": high,
                        "confidence_level": 0.95,
                        "bootstrap_replicates": replicates,
                        "finite_bootstrap_replicates": finite_reps,
                    }
                )
                for condition in EXPECTED_CONDITIONS:
                    selected = np.flatnonzero(table["condition"].eq(condition).to_numpy())
                    condition_value, condition_n, condition_obs = estimate(table, metric, selected)
                    if condition_n:
                        condition_records.append(
                            {
                                "condition": condition,
                                "model_family": model,
                                "model_id": MODEL_IDS[model],
                                "height_mm": height,
                                "metric": metric,
                                "estimate": condition_value,
                                "eligible_base_cases": condition_n,
                                "observations": condition_obs,
                            }
                        )

    delta_records: list[dict[str, Any]] = []
    architecture_records: list[dict[str, Any]] = []
    for model in MODELS:
        for height in (35, 65):
            for metric in METRICS:
                current = next(
                    row["estimate"]
                    for row in point_records
                    if row["model_family"] == model and row["height_mm"] == height and row["metric"] == metric
                )
                nominal = next(
                    row["estimate"]
                    for row in point_records
                    if row["model_family"] == model and row["height_mm"] == 50 and row["metric"] == metric
                )
                raw_delta = current - nominal
                reps = replicate_arrays[(model, height, metric)] - replicate_arrays[(model, 50, metric)]
                low, high, finite = interval(reps)
                oriented = -raw_delta if metric in LOWER_IS_BETTER else raw_delta
                oriented_reps = -reps if metric in LOWER_IS_BETTER else reps
                oriented_low, oriented_high, _ = interval(oriented_reps)
                delta_records.append(
                    {
                        "model_family": model,
                        "model_id": MODEL_IDS[model],
                        "height_mm": height,
                        "reference_height_mm": 50,
                        "metric": metric,
                        "ood_estimate": current,
                        "nominal_estimate": nominal,
                        "raw_ood_minus_nominal": raw_delta,
                        "raw_ci_lower": low,
                        "raw_ci_upper": high,
                        "performance_change": oriented,
                        "performance_change_ci_lower": oriented_low,
                        "performance_change_ci_upper": oriented_high,
                        "performance_change_interpretation": "negative means worse than nominal",
                        "finite_bootstrap_replicates": finite,
                    }
                )
    for height in HEIGHTS_MM:
        for metric in METRICS:
            fno = next(
                row["estimate"]
                for row in point_records
                if row["model_family"] == "fno" and row["height_mm"] == height and row["metric"] == metric
            )
            deep = next(
                row["estimate"]
                for row in point_records
                if row["model_family"] == "deeponet" and row["height_mm"] == height and row["metric"] == metric
            )
            reps = replicate_arrays[("fno", height, metric)] - replicate_arrays[("deeponet", height, metric)]
            low, high, finite = interval(reps)
            architecture_records.append(
                {
                    "height_mm": height,
                    "metric": metric,
                    "fno_estimate": fno,
                    "deeponet_estimate": deep,
                    "fno_minus_deeponet": fno - deep,
                    "ci_lower": low,
                    "ci_upper": high,
                    "finite_bootstrap_replicates": finite,
                    "difference_direction": "FNO_minus_DeepONet; sign is raw and metric-dependent",
                }
            )
    return (
        pd.DataFrame(point_records),
        pd.DataFrame(interval_records),
        pd.DataFrame(delta_records),
        pd.DataFrame(architecture_records),
        pd.DataFrame(condition_records),
        packed_replicates,
    )


def support_diagnostics(case_stats: pd.DataFrame) -> pd.DataFrame:
    records = []
    for model in MODELS:
        for height in HEIGHTS_MM:
            table = case_stats[
                case_stats["model_family"].eq(model) & case_stats["height_mm"].eq(height)
            ]
            for scope, selected in [("overall_nonhomogeneous", table[table["is_nonhomogeneous"]])]:
                records.append(
                    {
                        "scope": scope,
                        "model_family": model,
                        "height_mm": height,
                        "eligible_cases": int(len(selected)),
                        "nondegenerate_support_cases": int(selected["predicted_support_nondegenerate"].sum()),
                        "degenerate_support_cases": int((~selected["predicted_support_nondegenerate"]).sum()),
                        "zero_fallback_cases": int((~np.isfinite(selected["raw_APD90_patch_delta"])).sum()),
                    }
                )
    return pd.DataFrame(records)


def make_figure(intervals: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    panels = (
        ("LAT_global_R2", "LAT global $R^2$"),
        ("APD90_global_R2", "APD90 global $R^2$"),
        ("patch_Dice_nonhomogeneous_micro", "Non-homogeneous patch Dice"),
        ("homogeneous_false_positive_area_mean", "Homogeneous false-positive area"),
    )
    colors = {"fno": "#1464A5", "deeponet": "#D55E00"}
    labels = {"fno": "FNO", "deeponet": "DeepONet"}
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), constrained_layout=True)
    for ax, (metric, title) in zip(axes.ravel(), panels):
        for model in MODELS:
            data = intervals[
                intervals["metric"].eq(metric) & intervals["model_family"].eq(model)
            ].sort_values("height_mm")
            y = data["estimate"].to_numpy(dtype=float)
            lower = y - data["ci_lower"].to_numpy(dtype=float)
            upper = data["ci_upper"].to_numpy(dtype=float) - y
            ax.errorbar(
                data["height_mm"], y, yerr=np.vstack([lower, upper]), marker="o",
                linewidth=1.8, capsize=3, color=colors[model], label=labels[model],
            )
        ax.axvline(50, color="#777777", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Sensor-tissue standoff (mm)")
        ax.set_xticks(HEIGHTS_MM)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Zero-shot sensitivity to sensor-tissue standoff", fontsize=12)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--parity-tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 6:
        raise SystemExit("Use batch size 1 through 6 to keep inference memory bounded.")
    if args.bootstrap_replicates < 1000:
        raise SystemExit("At least 1000 grouped bootstrap replicates are required.")
    generation, contract, artifacts = prerequisite_state()
    cohort, rows_by_height = build_height_rows()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")

    fno_checkpoint = resolve(artifacts["fno_checkpoint_field"]["path"])
    deep_checkpoint = resolve(artifacts["deeponet_checkpoint_field"]["path"])
    fno_stats_path = resolve(artifacts["fno_training_statistics"]["path"])
    fno_stats = FNO_EXPORT.load_stats(fno_stats_path)
    predictions: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    truth_reference: dict[str, np.ndarray] | None = None
    replay: dict[str, Any] = {}
    for height in HEIGHTS_MM:
        print(f"FNO frozen inference at {height} mm...", flush=True)
        truth, pred, model_replay = run_fno(
            rows_by_height[height], fno_checkpoint, fno_stats, device, args.batch_size
        )
        if truth_reference is None:
            truth_reference = truth
        else:
            for key in truth:
                if not np.array_equal(truth_reference[key], truth[key], equal_nan=True):
                    raise RuntimeError(f"Truth changed across heights for FNO: {key}")
        predictions[("fno", height)] = pred
        replay[f"fno_h{height:03d}"] = model_replay
    for height in HEIGHTS_MM:
        print(f"DeepONet frozen inference at {height} mm...", flush=True)
        truth, pred, model_replay = run_deeponet(
            rows_by_height[height], deep_checkpoint, device, args.batch_size
        )
        assert truth_reference is not None
        for key in truth:
            if not np.array_equal(truth_reference[key], truth[key], equal_nan=True):
                raise RuntimeError(f"Truth changed between model loaders: {key}")
        predictions[("deeponet", height)] = pred
        replay[f"deeponet_h{height:03d}"] = model_replay
    assert truth_reference is not None

    with np.load(CANONICAL_MAPS_PATH) as canonical:
        pixel_replay = {}
        for model in MODELS:
            for field, canonical_name in (
                ("lat", f"{model}_field_pred_lat"),
                ("apd90", f"{model}_field_pred_apd90"),
                ("patch_prob", f"{model}_field_pred_patch_prob"),
            ):
                difference = np.abs(predictions[(model, 50)][field] - canonical[canonical_name])
                max_abs = float(np.max(difference))
                pixel_replay[f"{model}_{field}"] = {
                    "max_absolute_difference": max_abs,
                    "mean_absolute_difference": float(np.mean(difference)),
                    "bit_exact": bool(max_abs == 0.0),
                }
        canonical_truth_names = {
            "lat": "true_lat",
            "apd90": "true_apd90",
            "patch": "true_patch",
            "valid": "valid_mask",
        }
        truth_parity = {
            key: bool(
                np.array_equal(
                    truth_reference[key], canonical[canonical_name], equal_nan=True
                )
            )
            for key, canonical_name in canonical_truth_names.items()
        }
        metric_replay = {}
        valid = truth_reference["valid"].astype(bool)
        nonhom = np.any((truth_reference["patch"] > 0.5) & valid, axis=(1, 2))
        for model in MODELS:
            for field, canonical_name in (
                ("lat", f"{model}_field_pred_lat"),
                ("apd90", f"{model}_field_pred_apd90"),
            ):
                observed = scalar_r2(
                    truth_reference[field][valid], predictions[(model, 50)][field][valid]
                )
                expected = scalar_r2(
                    truth_reference[field][valid], canonical[canonical_name][valid]
                )
                delta = abs(observed - expected)
                metric_replay[f"{model}_{field}_global_R2"] = {
                    "observed": observed,
                    "canonical": expected,
                    "absolute_difference": delta,
                    "tolerance": args.parity_tolerance,
                    "passed": delta <= args.parity_tolerance,
                }
            truth_mask = (truth_reference["patch"] > 0.5) & valid
            observed_mask = (predictions[(model, 50)]["patch_prob"] > 0.5) & valid
            canonical_mask = (canonical[f"{model}_field_pred_patch_prob"] > 0.5) & valid
            selected = nonhom[:, None, None]
            def dice_value(predicted):
                tp = float(np.sum(predicted & truth_mask & selected))
                fp = float(np.sum(predicted & ~truth_mask & valid & selected))
                fn = float(np.sum(~predicted & truth_mask & selected))
                return 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
            observed = dice_value(observed_mask)
            expected = dice_value(canonical_mask)
            delta = abs(observed - expected)
            metric_replay[f"{model}_patch_Dice_nonhomogeneous_micro"] = {
                "observed": observed,
                "canonical": expected,
                "absolute_difference": delta,
                "tolerance": args.parity_tolerance,
                "passed": delta <= args.parity_tolerance,
            }
    parity_passed = all(record["passed"] for record in metric_replay.values()) and all(truth_parity.values())
    if not parity_passed:
        raise RuntimeError(
            f"Nominal prediction replay failed: metrics={metric_replay}, truth={truth_parity}, "
            f"pixel_diagnostics={pixel_replay}"
        )

    case_stats = build_case_statistics(cohort, truth_reference, predictions)
    point, intervals, deltas, architectures, conditions, replicate_arrays = analyze(
        case_stats, args.bootstrap_replicates, args.seed
    )
    supports = support_diagnostics(case_stats)

    prediction_index = pd.concat(
        [
            cohort[["cohort_index", "base_sample_id", "condition"]].assign(
                model_family=model, model_id=MODEL_IDS[model], height_mm=height
            )
            for model in MODELS
            for height in HEIGHTS_MM
        ],
        ignore_index=True,
    )
    packed_maps: dict[str, np.ndarray] = {
        "true_lat": truth_reference["lat"],
        "true_apd90": truth_reference["apd90"],
        "true_patch": truth_reference["patch"],
        "valid_mask": truth_reference["valid"],
    }
    for model in MODELS:
        for height in HEIGHTS_MM:
            for field, array in predictions[(model, height)].items():
                packed_maps[f"{model}_h{height:03d}_pred_{field}"] = array

    height_channel_values = {}
    for height, rows in rows_by_height.items():
        coords = np.load(resolve(rows.iloc[0]["coordinate_channels_file"]), mmap_mode="r")
        unique = np.unique(coords[..., 2])
        height_channel_values[str(height)] = unique.astype(float).tolist()
    errors: list[str] = []
    if len(case_stats) != 450:
        errors.append(f"Expected 450 model-height-case rows, found {len(case_stats)}.")
    if case_stats.duplicated(["model_family", "height_mm", "base_sample_id"]).any():
        errors.append("Duplicate model-height-base rows in case statistics.")
    for model in MODELS:
        for height in HEIGHTS_MM:
            pred = predictions[(model, height)]
            valid = truth_reference["valid"]
            for field in ("lat", "apd90", "patch_prob"):
                if not np.all(np.isfinite(pred[field][valid])):
                    errors.append(f"Nonfinite {model}/{height}/{field} predictions.")
    expected_height_channels = {"35": [0.699999988079071], "50": [1.0], "65": [1.2999999523162842]}
    if height_channel_values != expected_height_channels:
        errors.append(f"Unexpected FNO coordinate height channels: {height_channel_values}")

    temp_name = tempfile.mkdtemp(prefix=".module4_analysis_", dir=MODULE4_DIR)
    temp_dir = Path(temp_name)
    published = False
    try:
        temp_paths = {key: temp_dir / path.name for key, path in FINAL_FILES.items()}
        prediction_index.to_csv(temp_paths["prediction_index"], index=False)
        case_stats.to_csv(temp_paths["case_statistics"], index=False)
        point.to_csv(temp_paths["point_estimates"], index=False)
        intervals.to_csv(temp_paths["bootstrap_intervals"], index=False)
        deltas.to_csv(temp_paths["height_deltas"], index=False)
        architectures.to_csv(temp_paths["architecture_differences"], index=False)
        conditions.to_csv(temp_paths["condition_specific"], index=False)
        supports.to_csv(temp_paths["support_diagnostics"], index=False)
        np.savez_compressed(temp_paths["prediction_maps"], **packed_maps)
        np.savez_compressed(temp_paths["bootstrap_replicates"], **replicate_arrays)
        make_figure(intervals, temp_paths["figure_png"], temp_paths["figure_pdf"])

        headline = {}
        for model in MODELS:
            headline[model] = {}
            for height in HEIGHTS_MM:
                selected = point[
                    point["model_family"].eq(model)
                    & point["height_mm"].eq(height)
                    & point["metric"].isin(PRIMARY_METRICS | {"patch_Dice_nonhomogeneous_micro"})
                ]
                headline[model][str(height)] = {
                    row.metric: float(row.estimate) for row in selected.itertuples(index=False)
                }
        summary = {
            "schema_version": "1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "module": 4,
            "status": "complete" if not errors else "failed_validation",
            "passed": not errors,
            "title": "Zero-shot sensor-tissue standoff sensitivity on 75 held-out synthetic cases",
            "independent_unit": "base_sample_id",
            "base_cases": 75,
            "heights_mm": list(HEIGHTS_MM),
            "models": MODEL_IDS,
            "training_performed": False,
            "adaptation_performed": False,
            "ep_simulations_rerun": False,
            "bootstrap": {
                "replicates": args.bootstrap_replicates,
                "seed": args.seed,
                "interval": "95% percentile",
                "resampling_unit": "75 held-out base_sample_id values",
            },
            "endpoint_policy": {
                "primary": sorted(PRIMARY_METRICS),
                "secondary_diagnostic": sorted(set(METRICS) - PRIMARY_METRICS),
                "patch_threshold": 0.5,
                "degenerate_patch_delta_fallback": 0.0,
                "empty_support_center_error_fallback": math.sqrt(2.0),
            },
            "headline_point_estimates": headline,
            "nominal_prediction_metric_replay": metric_replay,
            "nominal_prediction_pixel_diagnostics": pixel_replay,
            "truth_replay_exact": truth_parity,
            "replay_configuration": replay,
            "input_normalization_caveat": (
                "Both frozen checkpoints use per-sample measurement normalization. Absolute field-amplitude attenuation "
                "with standoff is therefore largely normalized away; the stress test primarily probes spatial/temporal "
                "pattern shift, plus the FNO's explicit but training-constant height coordinate."
            ),
            "geometry_conditioning_caveat": (
                "FNO receives a height coordinate (0.7, 1.0, 1.3) that was constant at 1.0 during training. "
                "DeepONet receives only measurement tensors and sensor masks, not sensor coordinates or height metadata."
            ),
            "scientific_interpretation": "Case-specific synthetic acquisition-geometry sensitivity; zero-shot OOD stress test.",
            "claim_boundary": "Not patient-specific validation, clinical validation, real-SQUID validation, or learned geometry conditioning.",
            "errors": errors,
        }
        write_json(temp_paths["summary"], summary)
        validation = {
            "passed": not errors,
            "errors": errors,
            "generation_validation_passed": bool(generation["passed"]),
            "nominal_projection_bit_exact_cases": int(generation["nominal_bit_exact_cases"]),
            "nominal_prediction_replay_passed": parity_passed,
            "nominal_prediction_metric_replay": metric_replay,
            "nominal_prediction_pixel_diagnostics": pixel_replay,
            "truth_replay_exact": truth_parity,
            "height_channel_values": height_channel_values,
            "case_rows": int(len(case_stats)),
            "grouped_base_cases_per_model_height": 75,
            "checkpoint_hashes": {
                "fno": sha256(fno_checkpoint),
                "deeponet": sha256(deep_checkpoint),
            },
            "source_hashes": {
                "contract": sha256(CONTRACT_PATH),
                "inventory": sha256(INVENTORY_PATH),
                "cohort": sha256(COHORT_PATH),
                "sensor_height_manifest": sha256(MANIFEST_PATH),
            },
            "safety": {
                "legacy_outputs_overwritten": False,
                "one_model_loaded_at_a_time": True,
                "batch_size_at_most_legacy": args.batch_size <= 6,
                "training_or_finetuning": False,
            },
        }
        write_json(temp_paths["validation"], validation)
        if errors:
            print(json.dumps(validation, indent=2), flush=True)
            raise RuntimeError("Module 4 analysis validation failed; no final analysis files published.")
        for key, final_path in FINAL_FILES.items():
            temp_paths[key].replace(final_path)
        published = True
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    print(
        json.dumps(
            {
                "passed": published,
                "module": 4,
                "base_cases": 75,
                "heights_mm": list(HEIGHTS_MM),
                "bootstrap_replicates": args.bootstrap_replicates,
                "headline": headline,
                "summary": str(FINAL_FILES["summary"]),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
