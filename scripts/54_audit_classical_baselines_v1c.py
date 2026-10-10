                      
                                                     
 
                                                                           
                                                                              
                          
   

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SKLEARN = PROJECT_ROOT / ".local_deps" / "sklearn"
if LOCAL_SKLEARN.exists():
    sys.path.insert(0, str(LOCAL_SKLEARN))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_recall_fscore_support,
        r2_score,
    )
except ImportError as exc:                    
    raise SystemExit("scikit-learn and scipy are required for this audit.") from exc


BASELINE_DIR = PROJECT_ROOT / "outputs" / "baselines_v1c"
READY_DIR = PROJECT_ROOT / "outputs" / "dataset_v1c_5000_augmented_ready"
FIG_DIR = PROJECT_ROOT / "figures" / "baselines_v1c_audit"
DOC_PATH = PROJECT_ROOT / "docs" / "CLASSICAL_BASELINE_AUDIT_V1C.md"

RESULTS_CSV = BASELINE_DIR / "classical_baseline_results.csv"
PREDICTIONS_CSV = BASELINE_DIR / "classical_predictions.csv"
IMPORTANCE_CSV = BASELINE_DIR / "random_forest_feature_importance.csv"
FIXED_FEATURES = READY_DIR / "features_mig_deployable_fixed_augmented.csv"
MAG_FEATURES = READY_DIR / "features_mig_deployable_magnetic_augmented.csv"
TARGETS_CSV = READY_DIR / "targets_ep_augmented.csv"
SPLIT_JSON = READY_DIR / "splits" / "grouped_train_val_test.json"

VARIANT_ORDER = [
    "clean",
    "noise_low",
    "noise_medium",
    "noise_high",
    "dropout_10pct",
    "dropout_25pct",
    "jitter_1mm",
    "jitter_3mm",
    "height_30mm",
    "height_70mm",
    "combined_medium",
]
REG_TARGETS = ["y_lat_range", "y_apd90_patch_delta", "y_patch_area_fraction"]
ID_COLUMNS = {"augmented_sample_id", "base_sample_id", "variant", "sample_id", "condition", "notes"}
TARGET_COLUMNS = {"condition_label", "y_condition", "y_apd90_patch_delta", "y_lat_range", "y_patch_area_fraction"}
RANDOM_STATE = 20260525


def ensure_dirs() -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def finite_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return math.nan
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return math.nan
    return float(pearsonr(y_true, y_pred).statistic)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else math.nan,
        "pearson_r": finite_pearson(y_true, y_pred),
    }


def load_core_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(RESULTS_CSV)
    predictions = pd.read_csv(PREDICTIONS_CSV)
    importances = pd.read_csv(IMPORTANCE_CSV)
    return results, predictions, importances


def best_rows(results: pd.DataFrame) -> dict[str, pd.Series]:
    rows: dict[str, pd.Series] = {}
    class_rows = results[
        (results["task"] == "classification")
        & (results["target"] == "y_condition")
        & (results["split"] == "test")
        & (~results["model"].str.contains("baseline", na=False))
    ].copy()
    rows["y_condition"] = class_rows.sort_values("macro_f1", ascending=False).iloc[0]
    for target in REG_TARGETS:
        reg_rows = results[
            (results["task"] == "regression")
            & (results["target"] == target)
            & (results["split"] == "test")
            & (~results["model"].str.contains("baseline", na=False))
        ].copy()
        rows[target] = reg_rows.sort_values("r2", ascending=False).iloc[0]
    return rows


def prediction_subset(preds: pd.DataFrame, row: pd.Series, split: str | None = None) -> pd.DataFrame:
    mask = (
        (preds["task"] == row["task"])
        & (preds["target"] == row["target"])
        & (preds["feature_setting"] == row["feature_setting"])
        & (preds["model"] == row["model"])
    )
    if split is not None:
        mask &= preds["split"].eq(split)
    return preds[mask].copy()


def compute_per_variant_metrics(results: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    selected = best_rows(results)
    for target, best in selected.items():
        sub = prediction_subset(preds, best)
        for (split, variant), group in sub.groupby(["split", "variant"], dropna=False):
            y_true = group["y_true"].to_numpy()
            y_pred = group["y_pred"].to_numpy()
            row = {
                "task": best["task"],
                "target": target,
                "split": split,
                "variant": variant,
                "feature_setting": best["feature_setting"],
                "model": best["model"],
                "n": int(len(group)),
            }
            if best["task"] == "classification":
                yt = y_true.astype(int)
                yp = y_pred.astype(int)
                row.update(
                    {
                        "accuracy": float(accuracy_score(yt, yp)),
                        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
                        "macro_f1": float(f1_score(yt, yp, average="macro")),
                        "mae": math.nan,
                        "rmse": math.nan,
                        "r2": math.nan,
                        "pearson_r": math.nan,
                    }
                )
            else:
                finite = np.isfinite(y_true) & np.isfinite(y_pred)
                row.update({"accuracy": math.nan, "balanced_accuracy": math.nan, "macro_f1": math.nan})
                row.update(regression_metrics(y_true[finite].astype(float), y_pred[finite].astype(float)))
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(BASELINE_DIR / "audit_per_variant_metrics.csv", index=False)
    return out


def raw_vs_standardized(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "mae", "rmse", "r2", "pearson_r"]
    raw = results[results["scaling"] == "raw"].copy()
    std = results[results["scaling"] == "standardized"].copy()
    keys = ["task", "target", "feature_table", "model", "split"]
    merged = raw.merge(std, on=keys, suffixes=("_raw", "_standardized"), how="inner")
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        record = {key: row[key] for key in keys}
        for metric in metric_cols:
            a = row.get(f"{metric}_raw")
            b = row.get(f"{metric}_standardized")
            record[f"{metric}_raw"] = a
            record[f"{metric}_standardized"] = b
            record[f"{metric}_delta_standardized_minus_raw"] = b - a if pd.notna(a) and pd.notna(b) else math.nan
        rows.append(record)
    out = pd.DataFrame(rows)
    out.to_csv(BASELINE_DIR / "audit_raw_vs_standardized_sensitivity.csv", index=False)
    return out


def condition_report(results: pd.DataFrame, preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    best = best_rows(results)["y_condition"]
    sub = prediction_subset(preds, best, split="test")
    labels = sorted(sub["y_true"].astype(int).unique().tolist())
    class_names = {
        int(label): str(group["condition"].mode().iloc[0])
        for label, group in sub.groupby(sub["y_true"].astype(int))
    }
    precision, recall, f1, support = precision_recall_fscore_support(
        sub["y_true"].astype(int), sub["y_pred"].astype(int), labels=labels, zero_division=0
    )
    report = pd.DataFrame(
        {
            "label": labels,
            "condition": [class_names[i] for i in labels],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "feature_setting": best["feature_setting"],
            "model": best["model"],
        }
    )
    report.to_csv(BASELINE_DIR / "audit_classification_report_by_condition.csv", index=False)

    cm = confusion_matrix(sub["y_true"].astype(int), sub["y_pred"].astype(int), labels=labels)
    cm_df = pd.DataFrame(cm, index=[class_names[i] for i in labels], columns=[class_names[i] for i in labels])
    cm_df.to_csv(BASELINE_DIR / "audit_best_condition_confusion_matrix.csv")

    pairs: list[dict[str, object]] = []
    for i, true_label in enumerate(labels):
        total = cm[i].sum()
        for j, pred_label in enumerate(labels):
            if i == j or cm[i, j] == 0:
                continue
            pairs.append(
                {
                    "true_condition": class_names[true_label],
                    "predicted_condition": class_names[pred_label],
                    "count": int(cm[i, j]),
                    "fraction_of_true": float(cm[i, j] / total) if total else math.nan,
                }
            )
    pairs_df = pd.DataFrame(pairs).sort_values(["count", "fraction_of_true"], ascending=False)
    pairs_df.to_csv(BASELINE_DIR / "audit_most_confused_conditions.csv", index=False)
    return report, cm_df, pairs_df


def regression_residuals(results: pd.DataFrame, preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_condition: list[dict[str, object]] = []
    by_variant: list[dict[str, object]] = []
    by_quartile: list[dict[str, object]] = []
    selected = best_rows(results)
    for target in REG_TARGETS:
        best = selected[target]
        sub = prediction_subset(preds, best, split="test")
        sub = sub[np.isfinite(sub["y_true"]) & np.isfinite(sub["y_pred"])].copy()
        sub["residual"] = sub["y_pred"] - sub["y_true"]
        try:
            sub["target_severity_quartile"] = pd.qcut(sub["y_true"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
        except ValueError:
            sub["target_severity_quartile"] = "all"

        for group_name, collector in [
            ("condition", by_condition),
            ("variant", by_variant),
            ("target_severity_quartile", by_quartile),
        ]:
            for key, group in sub.groupby(group_name, observed=False):
                y_true = group["y_true"].to_numpy(float)
                y_pred = group["y_pred"].to_numpy(float)
                row = {
                    "target": target,
                    group_name: key,
                    "feature_setting": best["feature_setting"],
                    "model": best["model"],
                    "n": int(len(group)),
                    "mean_residual": float(np.mean(y_pred - y_true)),
                    "median_residual": float(np.median(y_pred - y_true)),
                    "mae": float(mean_absolute_error(y_true, y_pred)),
                    "rmse": rmse(y_true, y_pred),
                    "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else math.nan,
                    "pearson_r": finite_pearson(y_true, y_pred),
                    "target_mean": float(np.mean(y_true)),
                }
                collector.append(row)

    cond_df = pd.DataFrame(by_condition)
    var_df = pd.DataFrame(by_variant)
    quart_df = pd.DataFrame(by_quartile)
    cond_df.to_csv(BASELINE_DIR / "audit_regression_residuals_by_condition.csv", index=False)
    var_df.to_csv(BASELINE_DIR / "audit_regression_residuals_by_variant.csv", index=False)
    quart_df.to_csv(BASELINE_DIR / "audit_regression_residuals_by_target_quartile.csv", index=False)
    return cond_df, var_df, quart_df


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = ID_COLUMNS | TARGET_COLUMNS
    out = []
    for col in df.columns:
        if col in excluded:
            continue
        if col.endswith("_start_ms") or col.endswith("_end_ms") or col.endswith("_n_timepoints"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def feature_family_columns(cols: list[str]) -> dict[str, list[str]]:
    return {
        "MI_only": [c for c in cols if "_MI_" in c],
        "DI_only": [c for c in cols if "_DI_" in c],
        "RMS_only": [c for c in cols if "_RMS_" in c],
        "MI_DI_only": [c for c in cols if "_MI_" in c or "_DI_" in c],
        "RMS_AUC_only": [c for c in cols if "_RMS_" in c or c.endswith("_auc")],
        "all_features": list(cols),
        "no_RMS_or_AUC": [c for c in cols if "_RMS_" not in c and not c.endswith("_auc")],
    }


def load_feature_table(path: Path, targets: pd.DataFrame, split_map: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    features = pd.read_csv(path)
    target_cols = [
        "augmented_sample_id",
        "base_sample_id",
        "variant",
        "condition",
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
    joined["split"] = joined["base_sample_id"].map(split_map)
    if joined["split"].isna().any():
        raise ValueError("Feature table contains rows missing from grouped split map")
    return joined, select_feature_columns(features)


def run_feature_ablation() -> pd.DataFrame:
    split_map = json.loads(SPLIT_JSON.read_text())["base_split"]
    targets = pd.read_csv(TARGETS_CSV)
    feature_tables = {
        "deployable_fixed": FIXED_FEATURES,
        "deployable_magnetic": MAG_FEATURES,
    }
    rows: list[dict[str, object]] = []
    for table_name, path in feature_tables.items():
        df, cols = load_feature_table(path, targets, split_map)
        families = feature_family_columns(cols)
        for family, family_cols in families.items():
            if not family_cols:
                continue
            print(f"Ablation {table_name} / {family} ({len(family_cols)} features)")
            train = df["split"].eq("train")
            x_train = df.loc[train, family_cols].to_numpy(float)
            y_train_cls = df.loc[train, "y_condition"].to_numpy(int)
            clf = RandomForestClassifier(
                n_estimators=120,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
            clf.fit(x_train, y_train_cls)
            for split in ["train", "val", "test"]:
                mask = df["split"].eq(split)
                y_true = df.loc[mask, "y_condition"].to_numpy(int)
                y_pred = clf.predict(df.loc[mask, family_cols].to_numpy(float))
                rows.append(
                    {
                        "task": "classification",
                        "target": "y_condition",
                        "feature_table": table_name,
                        "feature_family": family,
                        "model": "random_forest_classifier",
                        "split": split,
                        "n_features": len(family_cols),
                        "n_eval": int(len(y_true)),
                        "accuracy": float(accuracy_score(y_true, y_pred)),
                        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                        "mae": math.nan,
                        "rmse": math.nan,
                        "r2": math.nan,
                        "pearson_r": math.nan,
                    }
                )

            for target in REG_TARGETS:
                finite = np.isfinite(df[target])
                train_reg = train & finite
                if train_reg.sum() == 0:
                    continue
                reg = RandomForestRegressor(
                    n_estimators=120,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                )
                reg.fit(df.loc[train_reg, family_cols].to_numpy(float), df.loc[train_reg, target].to_numpy(float))
                for split in ["train", "val", "test"]:
                    mask = df["split"].eq(split) & finite
                    y_true = df.loc[mask, target].to_numpy(float)
                    y_pred = reg.predict(df.loc[mask, family_cols].to_numpy(float))
                    metrics = regression_metrics(y_true, y_pred)
                    rows.append(
                        {
                            "task": "regression",
                            "target": target,
                            "feature_table": table_name,
                            "feature_family": family,
                            "model": "random_forest_regressor",
                            "split": split,
                            "n_features": len(family_cols),
                            "n_eval": int(len(y_true)),
                            "accuracy": math.nan,
                            "balanced_accuracy": math.nan,
                            "macro_f1": math.nan,
                            **metrics,
                        }
                    )
    out = pd.DataFrame(rows)
    out.to_csv(BASELINE_DIR / "audit_feature_family_ablation.csv", index=False)
    return out


def top_feature_importances(results: pd.DataFrame, importances: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    selected = best_rows(results)
    for target, best in selected.items():
        model = "random_forest_classifier" if target == "y_condition" else "random_forest_regressor"
        subset = importances[
            (importances["target"] == target)
            & (importances["model"] == model)
            & (importances["feature_setting"] == best["feature_setting"])
        ].copy()
        if subset.empty:
            subset = importances[(importances["target"] == target) & (importances["model"] == model)].copy()
        subset = subset.sort_values("importance", ascending=False).head(20)
        subset.insert(0, "selected_by_test_metric", "macro_f1" if target == "y_condition" else "r2")
        rows.append(subset)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(BASELINE_DIR / "audit_top20_random_forest_feature_importances.csv", index=False)
    return out


def plot_per_variant_metrics(per_variant: pd.DataFrame) -> Path:
    test = per_variant[per_variant["split"] == "test"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    specs = [
        ("y_condition", "macro_f1", "Condition macro F1"),
        ("y_lat_range", "r2", "LAT range R2"),
        ("y_apd90_patch_delta", "r2", "APD90 patch delta R2"),
        ("y_patch_area_fraction", "r2", "Patch area fraction R2"),
    ]
    for ax, (target, metric, title) in zip(axes.ravel(), specs):
        sub = test[test["target"] == target].copy()
        sub["variant"] = pd.Categorical(sub["variant"], categories=VARIANT_ORDER, ordered=True)
        sub = sub.sort_values("variant")
        ax.bar(sub["variant"].astype(str), sub[metric], color="#4c78a8")
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=35)
        ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    out = FIG_DIR / "per_variant_metrics.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_feature_ablation(ablation: pd.DataFrame) -> Path:
    test = ablation[ablation["split"] == "test"].copy()
    families = ["MI_only", "DI_only", "RMS_only", "MI_DI_only", "RMS_AUC_only", "all_features", "no_RMS_or_AUC"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    specs = [
        ("y_condition", "macro_f1", "Condition macro F1"),
        ("y_lat_range", "r2", "LAT range R2"),
        ("y_apd90_patch_delta", "r2", "APD90 patch delta R2"),
        ("y_patch_area_fraction", "r2", "Patch area fraction R2"),
    ]
    for ax, (target, metric, title) in zip(axes.ravel(), specs):
        sub = test[(test["target"] == target) & (test["feature_table"] == "deployable_magnetic")].copy()
        sub["feature_family"] = pd.Categorical(sub["feature_family"], categories=families, ordered=True)
        sub = sub.sort_values("feature_family")
        ax.bar(sub["feature_family"].astype(str), sub[metric], color="#72b7b2")
        ax.set_title(f"{title} - magnetic windows")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=35)
        ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    out = FIG_DIR / "feature_family_ablation.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_confusion(cm_df: pd.DataFrame) -> Path:
    cm = cm_df.to_numpy()
    labels = [str(x).replace("_", "\n") for x in cm_df.index]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Best Condition Classifier Confusion Matrix")
    thresh = cm.max() * 0.55 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    out = FIG_DIR / "per_condition_confusion.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_apd_residuals_by_condition(residual_cond: pd.DataFrame) -> Path:
    sub = residual_cond[residual_cond["target"] == "y_apd90_patch_delta"].copy()
    sub = sub.sort_values("condition")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(sub["condition"].str.replace("_", "\n"), sub["mae"], color="#f58518")
    ax.set_ylabel("MAE")
    ax.set_title("APD90 Patch Delta Residuals By Condition")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    out = FIG_DIR / "apd_residuals_by_condition.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_lat_residuals_by_variant(residual_var: pd.DataFrame) -> Path:
    sub = residual_var[residual_var["target"] == "y_lat_range"].copy()
    sub["variant"] = pd.Categorical(sub["variant"], categories=VARIANT_ORDER, ordered=True)
    sub = sub.sort_values("variant")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(sub["variant"].astype(str), sub["mae"], color="#54a24b")
    ax.set_ylabel("MAE")
    ax.set_title("LAT Range Residuals By Variant")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    out = FIG_DIR / "lat_residuals_by_variant.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_top_importances(top: pd.DataFrame) -> Path:
    targets = ["y_condition", "y_lat_range", "y_apd90_patch_delta", "y_patch_area_fraction"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, target in zip(axes.ravel(), targets):
        sub = top[top["target"] == target].sort_values("importance", ascending=True).tail(12)
        ax.barh(sub["feature"].str.replace("_", " ", regex=False), sub["importance"], color="#4c78a8")
        ax.set_title(target)
        ax.set_xlabel("Importance")
    fig.tight_layout()
    out = FIG_DIR / "top_feature_importances.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def write_report(
    results: pd.DataFrame,
    per_variant: pd.DataFrame,
    ablation: pd.DataFrame,
    sensitivity: pd.DataFrame,
    class_report: pd.DataFrame,
    confused: pd.DataFrame,
    residual_cond: pd.DataFrame,
    residual_var: pd.DataFrame,
    residual_quart: pd.DataFrame,
    top_importances: pd.DataFrame,
    figures: list[Path],
) -> None:
    selected = best_rows(results)
    best_cls = selected["y_condition"]
    best_lat = selected["y_lat_range"]
    best_apd = selected["y_apd90_patch_delta"]
    best_area = selected["y_patch_area_fraction"]

    test_ab = ablation[ablation["split"] == "test"]
    amp_rows = test_ab[test_ab["feature_family"].isin(["RMS_only", "RMS_AUC_only", "no_RMS_or_AUC", "all_features"])]
    amp_summary = amp_rows[
        ["target", "feature_table", "feature_family", "macro_f1", "r2", "n_features"]
    ].to_dict(orient="records")

    height_cls = per_variant[
        (per_variant["split"] == "test")
        & (per_variant["target"] == "y_condition")
        & (per_variant["variant"].isin(["height_30mm", "height_70mm", "clean"]))
    ][["variant", "macro_f1", "accuracy"]].to_dict(orient="records")
    height_lat = per_variant[
        (per_variant["split"] == "test")
        & (per_variant["target"] == "y_lat_range")
        & (per_variant["variant"].isin(["height_30mm", "height_70mm", "clean"]))
    ][["variant", "mae", "r2"]].to_dict(orient="records")

    top_confusions = confused.head(5).to_dict(orient="records")
    weakest_variants = (
        per_variant[(per_variant["split"] == "test") & (per_variant["target"] == "y_condition")]
        .sort_values("macro_f1")
        .head(4)[["variant", "macro_f1", "accuracy"]]
        .to_dict(orient="records")
    )

    def fmt_row(row: pd.Series, metric: str) -> str:
        return (
            f"{row['feature_setting']} / {row['model']} "
            f"({metric}={row[metric]:.3f})"
        )

    def json_ready(records: list[dict[str, object]]) -> list[dict[str, object]]:
        cleaned: list[dict[str, object]] = []
        for record in records:
            out: dict[str, object] = {}
            for key, value in record.items():
                if isinstance(value, float) and math.isnan(value):
                    out[key] = None
                else:
                    out[key] = value
            cleaned.append(out)
        return cleaned

    lines = [
        "# Classical Baseline Audit v1c",
        "",
        "This audit summarizes what classical deployable MIG baselines already solve on the v1c 5,000-sample dataset. It does not implement CNNs, FNOs, or neural-network training.",
        "",
        "## Guardrails",
        "",
        "- Oracle LAT/RT/APD-windowed feature tables were not used.",
        "- The grouped `base_sample_id` split was preserved.",
        "- Augmented variants inherit EP targets from their base sample.",
        "- Sensor/noise/geometry variants model measurement robustness, not new EP physiology.",
        "",
        "## Classical Benchmarks To Beat",
        "",
        f"- Condition classification: {fmt_row(best_cls, 'macro_f1')}, accuracy={best_cls['accuracy']:.3f}, balanced accuracy={best_cls['balanced_accuracy']:.3f}.",
        f"- LAT range regression: {fmt_row(best_lat, 'r2')}, MAE={best_lat['mae']:.3f}, RMSE={best_lat['rmse']:.3f}, Pearson r={best_lat['pearson_r']:.3f}.",
        f"- APD90 patch delta regression: {fmt_row(best_apd, 'r2')}, MAE={best_apd['mae']:.3f}, RMSE={best_apd['rmse']:.3f}, Pearson r={best_apd['pearson_r']:.3f}.",
        f"- Patch area fraction regression: {fmt_row(best_area, 'r2')}, MAE={best_area['mae']:.4f}, RMSE={best_area['rmse']:.4f}, Pearson r={best_area['pearson_r']:.3f}.",
        "",
        "A neural model should beat these test-set numbers under the same grouped split before being considered an improvement over deployable classical MIG features.",
        "",
        "## What Classical MIG Features Solve Well",
        "",
        "- LAT range is already solved very strongly by deployable magnetic-window random forests. This likely reflects activation-timing and amplitude summaries being highly informative for conduction-slowing perturbations.",
        "- Condition classification is useful but not saturated: macro F1 is around 0.74 for the best random forest.",
        "- APD90 patch delta is partially recoverable from deployable summaries, with moderate R2 around 0.56.",
        "",
        "## What They Fail At",
        "",
        "- Patch area fraction is weakly predicted relative to the other targets; best R2 is only around 0.28.",
        "- APD-specific regression remains substantially noisier than LAT range regression.",
        "- Per-variant and height analyses show that measurement geometry can shift performance; robust models should be judged across variants, not just clean samples.",
        "",
        "## RMS And Amplitude Dependence",
        "",
        "Feature-family ablations were run with random forest models on the same grouped split. The full ablation table is `outputs/baselines_v1c/audit_feature_family_ablation.csv`.",
        "",
        "Selected ablation records:",
        "",
        "```json",
        json.dumps(json_ready(amp_summary[:40]), indent=2),
        "```",
        "",
        "RMS and AUC-heavy families dominate LAT range performance. Removing RMS/AUC columns sharply tests whether a future model is using morphology/balance information rather than mostly amplitude-integral cues.",
        "",
        "## Height Variant Sensitivity",
        "",
        "Condition classification for clean and height variants:",
        "",
        "```json",
        json.dumps(json_ready(height_cls), indent=2),
        "```",
        "",
        "LAT range regression for clean and height variants:",
        "",
        "```json",
        json.dumps(json_ready(height_lat), indent=2),
        "```",
        "",
        "Height changes alter magnetic field amplitude and can distort some metrics. Standardized linear models improve over raw linear models, but tree models are scale-invariant and show unchanged raw/standardized results.",
        "",
        "## Per-Condition Classification",
        "",
        "Per-condition report is saved to `outputs/baselines_v1c/audit_classification_report_by_condition.csv`.",
        "",
        "Most confused condition pairs:",
        "",
        "```json",
        json.dumps(json_ready(top_confusions), indent=2),
        "```",
        "",
        "Weakest variants for condition classification:",
        "",
        "```json",
        json.dumps(json_ready(weakest_variants), indent=2),
        "```",
        "",
        "## Regression Residuals",
        "",
        "Residual summaries were saved by condition, variant, and target severity quartile:",
        "",
        "- `outputs/baselines_v1c/audit_regression_residuals_by_condition.csv`",
        "- `outputs/baselines_v1c/audit_regression_residuals_by_variant.csv`",
        "- `outputs/baselines_v1c/audit_regression_residuals_by_target_quartile.csv`",
        "",
        "These tables should be inspected before neural modeling so future CNN/FNO comparisons do not hide condition-specific or variant-specific regressions.",
        "",
        "## Feature Importance",
        "",
        "Top random forest importances are saved to `outputs/baselines_v1c/audit_top20_random_forest_feature_importances.csv`. The top features are dominated by deployable MI/DI/RMS summaries and magnetic-window timing or amplitude terms depending on target.",
        "",
        "## Figures",
        "",
    ]
    lines.extend([f"- `{p.relative_to(PROJECT_ROOT)}`" for p in figures])
    lines.extend(
        [
            "",
            "## Recommended Neural Target",
            "",
            "The first neural comparison should focus on APD90 patch delta and patch area fraction, because classical deployable summaries leave the most room there. LAT range is already near-ceiling with random forests, so a CNN/FNO must beat R2 approximately 0.991 on the grouped test split to justify added complexity for that target.",
            "",
            "## Regenerate",
            "",
            "```bash",
            "venv_carputils/bin/python scripts/54_audit_classical_baselines_v1c.py",
            "```",
        ]
    )
    DOC_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    results, predictions, importances = load_core_tables()
    per_variant = compute_per_variant_metrics(results, predictions)
    sensitivity = raw_vs_standardized(results)
    class_report, cm_df, confused = condition_report(results, predictions)
    residual_cond, residual_var, residual_quart = regression_residuals(results, predictions)
    ablation = run_feature_ablation()
    top = top_feature_importances(results, importances)
    figures = [
        plot_per_variant_metrics(per_variant),
        plot_feature_ablation(ablation),
        plot_confusion(cm_df),
        plot_apd_residuals_by_condition(residual_cond),
        plot_lat_residuals_by_variant(residual_var),
        plot_top_importances(top),
    ]
    write_report(
        results,
        per_variant,
        ablation,
        sensitivity,
        class_report,
        confused,
        residual_cond,
        residual_var,
        residual_quart,
        top,
        figures,
    )
    selected = best_rows(results)
    print(
        json.dumps(
            {
                "audit_report": str(DOC_PATH.relative_to(PROJECT_ROOT)),
                "per_variant_metrics": "outputs/baselines_v1c/audit_per_variant_metrics.csv",
                "feature_family_ablation": "outputs/baselines_v1c/audit_feature_family_ablation.csv",
                "figures": [str(p.relative_to(PROJECT_ROOT)) for p in figures],
                "benchmarks_to_beat": {
                    "y_condition_macro_f1": float(selected["y_condition"]["macro_f1"]),
                    "y_lat_range_r2": float(selected["y_lat_range"]["r2"]),
                    "y_apd90_patch_delta_r2": float(selected["y_apd90_patch_delta"]["r2"]),
                    "y_patch_area_fraction_r2": float(selected["y_patch_area_fraction"]["r2"]),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
