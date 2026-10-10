                      
                                                                        
 
                                                                           
                                                                         
                                                                                 
                                                         
   

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SKLEARN = PROJECT_ROOT / ".local_deps" / "sklearn"
if LOCAL_SKLEARN.exists():
    sys.path.insert(0, str(LOCAL_SKLEARN))

import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:                                                          
    raise SystemExit(
        "scikit-learn is required for these baselines. Install it into the "
        "project venv or run `venv_carputils/bin/python -m pip install "
        "--target .local_deps/sklearn scikit-learn`."
    ) from exc


READY_DIR = PROJECT_ROOT / "outputs" / "dataset_v1c_5000_augmented_ready"
OUT_DIR = PROJECT_ROOT / "outputs" / "baselines_v1c"

FIXED_FEATURES = READY_DIR / "features_mig_deployable_fixed_augmented.csv"
MAG_FEATURES = READY_DIR / "features_mig_deployable_magnetic_augmented.csv"
TARGETS = READY_DIR / "targets_ep_augmented.csv"
SPLIT_JSON = READY_DIR / "splits" / "grouped_train_val_test.json"

RESULTS_CSV = OUT_DIR / "classical_baseline_results.csv"
RESULTS_JSON = OUT_DIR / "classical_baseline_results.json"
PREDICTIONS_CSV = OUT_DIR / "classical_predictions.csv"
IMPORTANCE_CSV = OUT_DIR / "random_forest_feature_importance.csv"

RANDOM_STATE = 20260525
ID_COLUMNS = {
    "augmented_sample_id",
    "base_sample_id",
    "variant",
    "sample_id",
    "condition",
    "notes",
}
TARGET_COLUMNS = {
    "condition_label",
    "y_condition",
    "y_apd90_patch_delta",
    "y_lat_range",
    "y_patch_area_fraction",
}
REGRESSION_TARGETS = ["y_lat_range", "y_apd90_patch_delta", "y_patch_area_fraction"]


@dataclass(frozen=True)
class FeatureSetting:
    name: str
    feature_table: str
    scaling: str
    path: Path
    standardize: bool


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_split_map() -> dict[str, str]:
    data = json.loads(SPLIT_JSON.read_text())
    base_split = data.get("base_split")
    if not isinstance(base_split, dict):
        raise ValueError(f"{SPLIT_JSON} does not contain a base_split mapping")
    allowed = {"train", "val", "test"}
    bad = {k: v for k, v in base_split.items() if v not in allowed}
    if bad:
        raise ValueError(f"Unexpected split labels in {SPLIT_JSON}: {list(bad.items())[:5]}")
    return base_split


def load_targets(split_map: dict[str, str]) -> pd.DataFrame:
    targets = pd.read_csv(TARGETS)
    required = {"augmented_sample_id", "base_sample_id", "variant", "condition", *TARGET_COLUMNS}
    missing = required - set(targets.columns)
    if missing:
        raise ValueError(f"Missing target columns: {sorted(missing)}")

    targets["split"] = targets["base_sample_id"].map(split_map)
    if targets["split"].isna().any():
        missing_base = targets.loc[targets["split"].isna(), "base_sample_id"].unique()[:10]
        raise ValueError(f"Targets contain base samples missing from grouped split: {missing_base}")

    split_counts = targets.groupby("base_sample_id")["split"].nunique()
    leaked = split_counts[split_counts > 1]
    if not leaked.empty:
        raise ValueError(f"Grouped split leakage detected for base samples: {leaked.index[:10].tolist()}")

    return targets


def select_feature_columns(df: pd.DataFrame, table_name: str) -> list[str]:
    excluded = set(ID_COLUMNS) | set(TARGET_COLUMNS)
    cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if col.endswith("_start_ms") or col.endswith("_end_ms") or col.endswith("_n_timepoints"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    if not cols:
        raise ValueError(f"No numeric deployable feature columns found for {table_name}")
    return cols


def load_joined_feature_table(path: Path, targets: pd.DataFrame, table_name: str) -> tuple[pd.DataFrame, list[str]]:
    features = pd.read_csv(path)
    required = {"augmented_sample_id", "base_sample_id", "variant", "condition"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    feature_cols = select_feature_columns(features, table_name)

    target_cols = [
        "augmented_sample_id",
        "base_sample_id",
        "variant",
        "condition",
        "split",
        "y_condition",
        "y_lat_range",
        "y_apd90_patch_delta",
        "y_patch_area_fraction",
    ]
    joined = features.merge(
        targets[target_cols],
        on=["augmented_sample_id", "base_sample_id", "variant", "condition"],
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(features) or len(joined) != len(targets):
        raise ValueError(
            f"Join mismatch for {path}: features={len(features)} targets={len(targets)} joined={len(joined)}"
        )
    if joined[feature_cols].replace([np.inf, -np.inf], np.nan).isna().any().any():
        bad = joined[feature_cols].columns[
            joined[feature_cols].replace([np.inf, -np.inf], np.nan).isna().any()
        ].tolist()
        raise ValueError(f"Non-finite deployable feature values in {table_name}: {bad[:10]}")
    return joined, feature_cols


def make_settings() -> list[FeatureSetting]:
    return [
        FeatureSetting("fixed_raw", "deployable_fixed", "raw", FIXED_FEATURES, False),
        FeatureSetting("fixed_standardized", "deployable_fixed", "standardized", FIXED_FEATURES, True),
        FeatureSetting("magnetic_raw", "deployable_magnetic", "raw", MAG_FEATURES, False),
        FeatureSetting("magnetic_standardized", "deployable_magnetic", "standardized", MAG_FEATURES, True),
    ]


def wrap_estimator(estimator, standardize: bool):
    if not standardize:
        return estimator
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def classification_models(standardize: bool) -> dict[str, object]:
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": wrap_estimator(
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
            standardize,
        ),
        "random_forest_classifier": RandomForestClassifier(
            n_estimators=180,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def regression_models(standardize: bool) -> dict[str, object]:
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "ridge_regression": wrap_estimator(Ridge(alpha=1.0, random_state=RANDOM_STATE), standardize),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=180,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def finite_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return math.nan
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return math.nan
    return float(pearsonr(y_true, y_pred).statistic)


def regression_metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else math.nan,
        "pearson_r": finite_pearson(y_true, y_pred),
    }


def save_feature_importance(
    rows: list[dict[str, object]],
    model_name: str,
    target: str,
    setting: FeatureSetting,
    model,
    feature_cols: list[str],
) -> None:
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif isinstance(model, Pipeline) and hasattr(model[-1], "feature_importances_"):
        importances = model[-1].feature_importances_
    if importances is None:
        return
    for feature, value in zip(feature_cols, importances):
        rows.append(
            {
                "task": "classification" if target == "y_condition" else "regression",
                "target": target,
                "feature_setting": setting.name,
                "feature_table": setting.feature_table,
                "scaling": setting.scaling,
                "model": model_name,
                "feature": feature,
                "importance": float(value),
            }
        )


def add_prediction_rows(
    rows: list[dict[str, object]],
    df: pd.DataFrame,
    indices: Iterable[int],
    task: str,
    target: str,
    setting: FeatureSetting,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    meta = df.loc[list(indices)][
        ["augmented_sample_id", "base_sample_id", "variant", "condition", "split"]
    ].reset_index(drop=True)
    for i, row in meta.iterrows():
        rows.append(
            {
                "task": task,
                "target": target,
                "feature_setting": setting.name,
                "feature_table": setting.feature_table,
                "scaling": setting.scaling,
                "model": model_name,
                "split": row["split"],
                "augmented_sample_id": row["augmented_sample_id"],
                "base_sample_id": row["base_sample_id"],
                "variant": row["variant"],
                "condition": row["condition"],
                "y_true": y_true[i],
                "y_pred": y_pred[i],
                "residual": float(y_pred[i] - y_true[i]) if task == "regression" else math.nan,
            }
        )


def run_classification(
    df: pd.DataFrame,
    feature_cols: list[str],
    setting: FeatureSetting,
    result_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    importance_rows: list[dict[str, object]],
) -> None:
    labels = sorted(df["y_condition"].dropna().astype(int).unique().tolist())
    split_indices = {split: df.index[df["split"] == split].to_numpy() for split in ["train", "val", "test"]}
    x_train = df.loc[split_indices["train"], feature_cols].to_numpy(dtype=float)
    y_train = df.loc[split_indices["train"], "y_condition"].to_numpy(dtype=int)

    for model_name, model in classification_models(setting.standardize).items():
        model.fit(x_train, y_train)
        save_feature_importance(importance_rows, model_name, "y_condition", setting, model, feature_cols)

        for split, indices in split_indices.items():
            x_eval = df.loc[indices, feature_cols].to_numpy(dtype=float)
            y_true = df.loc[indices, "y_condition"].to_numpy(dtype=int)
            y_pred = model.predict(x_eval).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            result_rows.append(
                {
                    "task": "classification",
                    "target": "y_condition",
                    "feature_setting": setting.name,
                    "feature_table": setting.feature_table,
                    "scaling": setting.scaling,
                    "model": model_name,
                    "split": split,
                    "n_train": int(len(y_train)),
                    "n_eval": int(len(y_true)),
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                    "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                    "mae": math.nan,
                    "rmse": math.nan,
                    "r2": math.nan,
                    "pearson_r": math.nan,
                    "confusion_matrix": json.dumps(cm.tolist()),
                    "labels": json.dumps(labels),
                }
            )
            add_prediction_rows(
                prediction_rows, df, indices, "classification", "y_condition", setting, model_name, y_true, y_pred
            )


def run_regression(
    df: pd.DataFrame,
    feature_cols: list[str],
    setting: FeatureSetting,
    result_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    importance_rows: list[dict[str, object]],
) -> None:
    for target in REGRESSION_TARGETS:
        finite_df = df[np.isfinite(pd.to_numeric(df[target], errors="coerce"))].copy()
        if finite_df.empty:
            continue
        split_indices = {
            split: finite_df.index[finite_df["split"] == split].to_numpy() for split in ["train", "val", "test"]
        }
        if len(split_indices["train"]) == 0 or len(split_indices["val"]) == 0 or len(split_indices["test"]) == 0:
            continue

        x_train = finite_df.loc[split_indices["train"], feature_cols].to_numpy(dtype=float)
        y_train = finite_df.loc[split_indices["train"], target].to_numpy(dtype=float)

        for model_name, model in regression_models(setting.standardize).items():
            model.fit(x_train, y_train)
            save_feature_importance(importance_rows, model_name, target, setting, model, feature_cols)
            for split, indices in split_indices.items():
                x_eval = finite_df.loc[indices, feature_cols].to_numpy(dtype=float)
                y_true = finite_df.loc[indices, target].to_numpy(dtype=float)
                y_pred = model.predict(x_eval).astype(float)
                metrics = regression_metric_row(y_true, y_pred)
                result_rows.append(
                    {
                        "task": "regression",
                        "target": target,
                        "feature_setting": setting.name,
                        "feature_table": setting.feature_table,
                        "scaling": setting.scaling,
                        "model": model_name,
                        "split": split,
                        "n_train": int(len(y_train)),
                        "n_eval": int(len(y_true)),
                        "accuracy": math.nan,
                        "balanced_accuracy": math.nan,
                        "macro_f1": math.nan,
                        **metrics,
                        "confusion_matrix": "",
                        "labels": "",
                    }
                )
                add_prediction_rows(
                    prediction_rows, finite_df, indices, "regression", target, setting, model_name, y_true, y_pred
                )


def write_json_summary(results: pd.DataFrame, metadata: dict[str, object]) -> None:
    records = []
    for row in results.to_dict(orient="records"):
        clean = {}
        for key, value in row.items():
            if isinstance(value, float) and math.isnan(value):
                clean[key] = None
            else:
                clean[key] = value
        records.append(clean)
    RESULTS_JSON.write_text(json.dumps({"metadata": metadata, "results": records}, indent=2))


def main() -> None:
    ensure_dirs()
    split_map = load_split_map()
    targets = load_targets(split_map)

    result_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    metadata = {
        "feature_policy": "deployable fixed-time and magnetic-derived MIG features only; oracle LAT/RT/APD windows excluded",
        "split_policy": "grouped by base_sample_id; augmented variants stay in the same split",
        "targets_path": str(TARGETS.relative_to(PROJECT_ROOT)),
        "split_json": str(SPLIT_JSON.relative_to(PROJECT_ROOT)),
        "n_augmented_rows": int(len(targets)),
        "n_base_samples": int(targets["base_sample_id"].nunique()),
        "random_state": RANDOM_STATE,
    }

    for setting in make_settings():
        print(f"Running setting: {setting.name}")
        df, feature_cols = load_joined_feature_table(setting.path, targets, setting.feature_table)
        metadata[f"{setting.name}_n_features"] = len(feature_cols)
        metadata[f"{setting.name}_feature_columns"] = feature_cols
        run_classification(df, feature_cols, setting, result_rows, prediction_rows, importance_rows)
        run_regression(df, feature_cols, setting, result_rows, prediction_rows, importance_rows)

    results = pd.DataFrame(result_rows)
    predictions = pd.DataFrame(prediction_rows)
    importances = pd.DataFrame(importance_rows)
    results.to_csv(RESULTS_CSV, index=False)
    predictions.to_csv(PREDICTIONS_CSV, index=False)
    importances.to_csv(IMPORTANCE_CSV, index=False)
    write_json_summary(results, metadata)

    test_summary = results[results["split"] == "test"].copy()
    best_class = test_summary[test_summary["task"] == "classification"].sort_values("macro_f1", ascending=False).head(5)
    best_reg = test_summary[test_summary["task"] == "regression"].sort_values("r2", ascending=False).head(10)

    print(json.dumps({
        "results_csv": str(RESULTS_CSV.relative_to(PROJECT_ROOT)),
        "results_json": str(RESULTS_JSON.relative_to(PROJECT_ROOT)),
        "predictions_csv": str(PREDICTIONS_CSV.relative_to(PROJECT_ROOT)),
        "feature_importance_csv": str(IMPORTANCE_CSV.relative_to(PROJECT_ROOT)),
        "n_result_rows": int(len(results)),
        "n_prediction_rows": int(len(predictions)),
        "best_test_classification": best_class[
            ["feature_setting", "model", "accuracy", "balanced_accuracy", "macro_f1"]
        ].to_dict(orient="records"),
        "best_test_regression": best_reg[
            ["target", "feature_setting", "model", "mae", "rmse", "r2", "pearson_r"]
        ].to_dict(orient="records"),
    }, indent=2))


if __name__ == "__main__":
    main()
