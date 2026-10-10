                      
                                                                   

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = PROJECT_ROOT / "outputs" / "baselines_v1c"
FIG_DIR = PROJECT_ROOT / "figures" / "baselines_v1c"

RESULTS_CSV = BASELINE_DIR / "classical_baseline_results.csv"
PREDICTIONS_CSV = BASELINE_DIR / "classical_predictions.csv"
IMPORTANCE_CSV = BASELINE_DIR / "random_forest_feature_importance.csv"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [p for p in [RESULTS_CSV, PREDICTIONS_CSV, IMPORTANCE_CSV] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Run scripts/52_run_classical_mig_baselines_v1c.py first; missing {missing}")
    return pd.read_csv(RESULTS_CSV), pd.read_csv(PREDICTIONS_CSV), pd.read_csv(IMPORTANCE_CSV)


def condition_label_map(preds: pd.DataFrame) -> dict[int, str]:
    class_preds = preds[(preds["task"] == "classification") & (preds["target"] == "y_condition")]
    label_map: dict[int, str] = {}
    for label, group in class_preds.groupby("y_true"):
        label_map[int(label)] = str(group["condition"].mode().iloc[0])
    return label_map


def best_test_row(results: pd.DataFrame, task: str, target: str, metric: str, exclude_baseline: bool = True) -> pd.Series:
    rows = results[(results["split"] == "test") & (results["task"] == task) & (results["target"] == target)].copy()
    if exclude_baseline:
        rows = rows[~rows["model"].str.contains("baseline", na=False)]
    if rows.empty:
        raise ValueError(f"No result rows for {task}/{target}")
    return rows.sort_values(metric, ascending=False).iloc[0]


def matching_predictions(preds: pd.DataFrame, row: pd.Series, split: str = "test") -> pd.DataFrame:
    mask = (
        (preds["task"] == row["task"])
        & (preds["target"] == row["target"])
        & (preds["feature_setting"] == row["feature_setting"])
        & (preds["model"] == row["model"])
        & (preds["split"] == split)
    )
    out = preds[mask].copy()
    if out.empty:
        raise ValueError(f"No predictions found for {row.to_dict()}")
    return out


def plot_confusion_matrix(results: pd.DataFrame, preds: pd.DataFrame) -> Path:
    row = best_test_row(results, "classification", "y_condition", "macro_f1")
    selected = matching_predictions(preds, row)
    labels = sorted(selected["y_true"].dropna().astype(int).unique().tolist())
    label_map = condition_label_map(preds)
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    pos = {label: i for i, label in enumerate(labels)}
    for y_true, y_pred in zip(selected["y_true"].astype(int), selected["y_pred"].astype(int)):
        if y_true in pos and y_pred in pos:
            cm[pos[y_true], pos[y_pred]] += 1

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    tick_labels = [label_map.get(label, str(label)).replace("_", "\n") for label in labels]
    ax.set_xticks(range(len(labels)), tick_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), tick_labels)
    ax.set_xlabel("Predicted condition")
    ax.set_ylabel("True condition")
    ax.set_title(
        f"Condition Confusion Matrix\n{row['feature_setting']} / {row['model']} / macro F1={row['macro_f1']:.3f}"
    )
    threshold = cm.max() * 0.55 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    out = FIG_DIR / "confusion_matrix_condition.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_regression_scatter(results: pd.DataFrame, preds: pd.DataFrame, target: str, filename: str) -> Path:
    row = best_test_row(results, "regression", target, "r2")
    selected = matching_predictions(preds, row)
    y_true = selected["y_true"].astype(float).to_numpy()
    y_pred = selected["y_pred"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true, y_pred, s=8, alpha=0.35, linewidths=0)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.any():
        lo = float(min(y_true[finite].min(), y_pred[finite].min()))
        hi = float(max(y_true[finite].max(), y_pred[finite].max()))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1, linestyle="--")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"True {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(
        f"{target} Test Predictions\n{row['feature_setting']} / {row['model']} / R2={row['r2']:.3f}"
    )
    fig.tight_layout()
    out = FIG_DIR / filename
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_feature_importance(importances: pd.DataFrame) -> Path:
    rows = importances[
        (importances["task"] == "classification")
        & (importances["target"] == "y_condition")
        & (importances["feature_setting"] == "fixed_standardized")
        & (importances["model"] == "random_forest_classifier")
    ].copy()
    if rows.empty:
        rows = importances[importances["model"].str.contains("random_forest", na=False)].copy()
    rows = rows.sort_values("importance", ascending=False).head(25).sort_values("importance")

    fig, ax = plt.subplots(figsize=(8, 8))
    labels = rows["feature"].str.replace("_", " ", regex=False)
    ax.barh(labels, rows["importance"], color="#4c78a8")
    ax.set_xlabel("Random forest importance")
    ax.set_title("Top Deployable MIG Feature Importances")
    fig.tight_layout()
    out = FIG_DIR / "feature_importance_random_forest.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_raw_vs_standardized(results: pd.DataFrame) -> Path:
    test = results[results["split"] == "test"].copy()
    clf = test[
        (test["task"] == "classification")
        & (~test["model"].str.contains("baseline", na=False))
    ].copy()
    reg = test[
        (test["task"] == "regression")
        & (~test["model"].str.contains("baseline", na=False))
    ].copy()

    clf_summary = (
        clf.groupby(["feature_table", "scaling", "model"], as_index=False)["macro_f1"]
        .mean()
        .sort_values(["feature_table", "model", "scaling"])
    )
    reg_summary = (
        reg.groupby(["feature_table", "scaling", "model"], as_index=False)["r2"]
        .mean()
        .sort_values(["feature_table", "model", "scaling"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=False)
    for ax, summary, metric, title in [
        (axes[0], clf_summary, "macro_f1", "Classification Macro F1"),
        (axes[1], reg_summary, "r2", "Regression Mean R2"),
    ]:
        labels = [f"{r.feature_table}\n{r.model}".replace("deployable_", "") for r in summary.itertuples()]
        colors = ["#72b7b2" if s == "standardized" else "#f58518" for s in summary["scaling"]]
        ax.bar(np.arange(len(summary)), summary[metric], color=colors)
        ax.set_xticks(np.arange(len(summary)), labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.axhline(0, color="black", linewidth=0.8)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#f58518", label="raw"),
        plt.Rectangle((0, 0), 1, 1, color="#72b7b2", label="standardized"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIG_DIR / "raw_vs_standardized_results.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    ensure_dirs()
    results, preds, importances = load_outputs()
    outputs = [
        plot_confusion_matrix(results, preds),
        plot_regression_scatter(results, preds, "y_lat_range", "regression_lat_range.png"),
        plot_regression_scatter(results, preds, "y_apd90_patch_delta", "regression_apd90_delta.png"),
        plot_feature_importance(importances),
        plot_raw_vs_standardized(results),
    ]
    print(json.dumps({"figures": [str(p.relative_to(PROJECT_ROOT)) for p in outputs]}, indent=2))


if __name__ == "__main__":
    main()
