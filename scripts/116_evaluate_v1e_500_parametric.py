                      
                                                                  

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if (PROJECT_ROOT / ".local_deps" / "sklearn").exists():
    sys.path.append(str(PROJECT_ROOT / ".local_deps" / "sklearn"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from v1e_500_common import NEURAL_DIR, NEURAL_FIG_DIR, READY_DIR, ROOT, write_json


RAW_PRED = NEURAL_DIR / "v1e_500_parametric_predictions_raw.csv"
PRED_CSV = NEURAL_DIR / "v1e_500_parametric_predictions.csv"
RESULTS_CSV = NEURAL_DIR / "v1e_500_parametric_results.csv"
RESULTS_JSON = NEURAL_DIR / "v1e_500_parametric_results.json"


def safe_r2(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 2 or np.allclose(y[m], y[m][0]):
        return math.nan
    return float(r2_score(y[m], p[m]))


def safe_mae(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    return float(mean_absolute_error(y[m], p[m])) if m.sum() else math.nan


def safe_rmse(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(mean_squared_error(y[m], p[m]))) if m.sum() else math.nan


def param_mask(cx, cy, radius, present_prob, shape=(32, 32), threshold=0.5):
    if present_prob < threshold or not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(radius):
        return np.zeros(shape, dtype=bool)
    cx = float(np.clip(cx, 0.0, 1.0))
    cy = float(np.clip(cy, 0.0, 1.0))
    radius = float(np.clip(radius, 1e-4, 0.75))
    yy, xx = np.indices(shape)
    x = xx / max(shape[1] - 1, 1)
    y = yy / max(shape[0] - 1, 1)
    return ((x - cx) ** 2 + (y - cy) ** 2) <= radius**2


def dice_iou(true: np.ndarray, pred: np.ndarray):
    true = true.astype(bool)
    pred = pred.astype(bool)
    tp = float(np.sum(true & pred))
    fp = float(np.sum(~true & pred))
    fn = float(np.sum(true & ~pred))
    dice = (2 * tp) / max(2 * tp + fp + fn, 1.0)
    iou = tp / max(tp + fp + fn, 1.0)
    return dice, iou


def patch_delta(apd90: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> float:
    inside = mask.astype(bool) & valid.astype(bool)
    outside = (~mask.astype(bool)) & valid.astype(bool)
    if inside.sum() == 0 or outside.sum() == 0:
        return math.nan
    return float(np.mean(apd90[inside]) - np.mean(apd90[outside]))


def enrich_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    ready = pd.read_csv(READY_DIR / "sample_index.csv")
    file_cols = [
        "row_id",
        "APD90_grid_file",
        "patch_mask_grid_file",
        "valid_mask_grid_file",
    ]
    preds = preds.merge(ready[file_cols].drop_duplicates("row_id"), on="row_id", how="left")
    pred_area = np.exp(preds["pred_log_area_fraction"].astype(float)).clip(0.0, 1.0)
    pred_radius = np.exp(preds["pred_log_radius"].astype(float)).clip(0.0, 1.0)
    preds["pred_patch_area_fraction"] = pred_area
    preds["pred_equivalent_radius_norm"] = pred_radius
    rows = []
    for row in preds.itertuples(index=False):
        true_mask = np.load(ROOT / row.patch_mask_grid_file).astype(bool)
        valid = np.load(ROOT / row.valid_mask_grid_file).astype(bool)
        apd90 = np.load(ROOT / row.APD90_grid_file).astype(float)
        pred_mask = param_mask(row.pred_center_x_norm, row.pred_center_y_norm, row.pred_equivalent_radius_norm, row.pred_patch_present_prob, true_mask.shape)
        dice, iou = dice_iou(true_mask & valid, pred_mask & valid)
        true_delta = patch_delta(apd90, true_mask, valid)
        pred_delta = patch_delta(apd90, pred_mask, valid)
        rows.append(
            {
                "row_id": row.row_id,
                "model": row.model,
                "parametric_mask_dice_row": dice,
                "parametric_mask_iou_row": iou,
                "predicted_mask_area_fraction": float(np.mean(pred_mask[valid])),
                "true_apd90_patch_delta": true_delta,
                "parametric_mask_true_apd90_patch_delta": pred_delta,
                "homogeneous_false_positive_area_row": float(np.mean(pred_mask[valid])) if row.patch_present == 0 else math.nan,
            }
        )
    return preds.merge(pd.DataFrame(rows), on=["row_id", "model"], how="left")


def summarize(preds: pd.DataFrame) -> pd.DataFrame:
    out = []
    test = preds[preds["split"].eq("test")].copy()
    for model, sub in test.groupby("model"):
        y = sub["patch_present"].astype(int)
        prob = sub["pred_patch_present_prob"].astype(float)
        pred = (prob >= 0.5).astype(int)
        rec = {
            "model": model,
            "split": "test",
            "target": "patch_present",
            "n": int(len(sub)),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
        }
        try:
            rec["auroc"] = float(roc_auc_score(y, prob)) if y.nunique() > 1 else math.nan
        except Exception:
            rec["auroc"] = math.nan
        out.append(rec)
        if "pred_condition_label" in sub:
            out.append(
                {
                    "model": model,
                    "split": "test",
                    "target": "condition",
                    "n": int(len(sub)),
                    "accuracy": float(accuracy_score(sub["condition_label"], sub["pred_condition_label"])),
                    "balanced_accuracy": float(balanced_accuracy_score(sub["condition_label"], sub["pred_condition_label"])),
                    "macro_f1": float(f1_score(sub["condition_label"], sub["pred_condition_label"], average="macro")),
                }
            )
        non = sub[sub["patch_present"].eq(1)].copy()
        if len(non):
            center_err = np.sqrt((non["center_x_norm"] - non["pred_center_x_norm"]) ** 2 + (non["center_y_norm"] - non["pred_center_y_norm"]) ** 2)
            out.append(
                {
                    "model": model,
                    "split": "test",
                    "target": "patch_center_error",
                    "n": int(len(non)),
                    "mae": float(np.mean(center_err)),
                    "rmse": float(np.sqrt(np.mean(center_err * center_err))),
                }
            )
            for target, pred_col in [
                ("patch_area_fraction", "pred_patch_area_fraction"),
                ("log_area_fraction", "pred_log_area_fraction"),
                ("equivalent_radius_norm", "pred_equivalent_radius_norm"),
                ("log_radius", "pred_log_radius"),
                ("parametric_mask_area_fraction", "predicted_mask_area_fraction"),
                ("APD90_patch_delta_parametric_mask_true_apd90", "parametric_mask_true_apd90_patch_delta"),
            ]:
                true_col = "patch_area_fraction" if target == "parametric_mask_area_fraction" else target
                if target == "APD90_patch_delta_parametric_mask_true_apd90":
                    true_col = "true_apd90_patch_delta"
                out.append(
                    {
                        "model": model,
                        "split": "test",
                        "target": target,
                        "n": int(np.isfinite(non[true_col]).sum()),
                        "mae": safe_mae(non[true_col], non[pred_col]),
                        "rmse": safe_rmse(non[true_col], non[pred_col]),
                        "r2": safe_r2(non[true_col], non[pred_col]),
                    }
                )
            out.append(
                {
                    "model": model,
                    "split": "test",
                    "target": "area_bin",
                    "n": int(len(non)),
                    "accuracy": float(accuracy_score(non["area_bin"], non["pred_area_bin"])),
                    "macro_f1": float(f1_score(non["area_bin"], non["pred_area_bin"], average="macro")),
                }
            )
            out.append(
                {
                    "model": model,
                    "split": "test",
                    "target": "parametric_mask_dice",
                    "n": int(len(non)),
                    "dice": float(non["parametric_mask_dice_row"].mean()),
                    "iou": float(non["parametric_mask_iou_row"].mean()),
                }
            )
        hom = sub[sub["patch_present"].eq(0)].copy()
        if len(hom):
            out.append(
                {
                    "model": model,
                    "split": "test",
                    "target": "homogeneous_false_positive_area",
                    "n": int(len(hom)),
                    "area": float(hom["homogeneous_false_positive_area_row"].mean()),
                }
            )
    return pd.DataFrame(out)


def mean_center_baseline(preds: pd.DataFrame) -> float:
    train = preds[preds["split"].eq("train") & preds["patch_present"].eq(1)]
    test = preds[preds["split"].eq("test") & preds["patch_present"].eq(1)]
    if train.empty or test.empty:
        return math.nan
    cx = float(train["center_x_norm"].mean())
    cy = float(train["center_y_norm"].mean())
    err = np.sqrt((test["center_x_norm"] - cx) ** 2 + (test["center_y_norm"] - cy) ** 2)
    return float(np.mean(err))


def make_figures(results: pd.DataFrame, preds: pd.DataFrame) -> None:
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = results[(results["target"].eq("patch_present")) & (results["split"].eq("test"))]
    ax.bar(vals["model"], vals["macro_f1"])
    ax.set_ylabel("macro F1")
    ax.set_title("Patch Present")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "v1e_500_patch_present.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    vals = results[(results["target"].eq("patch_center_error")) & (results["split"].eq("test"))]
    ax.bar(vals["model"], vals["mae"])
    ax.axhline(mean_center_baseline(preds), color="black", linestyle="--", linewidth=1, label="mean center baseline")
    ax.set_ylabel("normalized center MAE")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "v1e_500_center_error.png", dpi=180)
    plt.close(fig)

    test_non = preds[preds["split"].eq("test") & preds["patch_present"].eq(1)]
    fig, ax = plt.subplots(figsize=(5, 5))
    for model, sub in test_non.groupby("model"):
        ax.scatter(sub["patch_area_fraction"], sub["pred_patch_area_fraction"], s=10, alpha=0.6, label=model)
    ax.set_xlabel("true area")
    ax.set_ylabel("predicted area from log-area")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "v1e_500_area_predictions.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    metric = results[results["target"].isin(["parametric_mask_dice", "patch_area_fraction", "log_area_fraction", "area_bin"])]
    labels = metric["model"] + "\n" + metric["target"]
    vals = metric["dice"].fillna(metric["r2"]).fillna(metric["macro_f1"]).fillna(0)
    ax.bar(range(len(metric)), vals)
    ax.set_xticks(range(len(metric)))
    ax.set_xticklabels(labels, rotation=75, fontsize=7)
    ax.set_title("v1e 500 Parametric Localization")
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "v1e_500_fno_vs_cnn.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for model, sub in test_non.groupby("model"):
        ax.scatter(sub["true_apd90_patch_delta"], sub["parametric_mask_true_apd90_patch_delta"], s=10, alpha=0.6, label=model)
    ax.set_xlabel("true APD90 patch delta")
    ax.set_ylabel("parametric-mask APD90 delta")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "v1e_500_apd90_delta_parametric.png", dpi=180)
    plt.close(fig)


def main() -> int:
    preds = pd.read_csv(RAW_PRED)
    preds = enrich_predictions(preds)
    results = summarize(preds)
    baseline = mean_center_baseline(preds[preds["model"].eq("tabular_rf_signal_descriptors")])
    results.loc[len(results)] = {"model": "mean_center_baseline", "split": "test", "target": "patch_center_error", "n": int(preds[preds["split"].eq("test") & preds["patch_present"].eq(1)]["base_sample_id"].nunique()), "mae": baseline}
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(PRED_CSV, index=False)
    results.to_csv(RESULTS_CSV, index=False)
    make_figures(results, preds)
    payload = {
        "results": results.replace({np.nan: None}).to_dict(orient="records"),
        "mean_center_baseline_mae": baseline,
        "comparison_targets": {"previous_100_sample_best_dice": 0.431, "v1d_patch_dice_target": 0.449},
        "prediction_file": str(PRED_CSV),
    }
    write_json(RESULTS_JSON, payload)
    print(json.dumps({"results": str(RESULTS_CSV), "predictions": str(PRED_CSV), "mean_center_baseline_mae": baseline}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
