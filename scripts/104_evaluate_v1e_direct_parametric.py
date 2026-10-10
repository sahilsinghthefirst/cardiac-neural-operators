                      
                                                                  

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
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from v1e_direct_common import NEURAL_DIR, NEURAL_FIG_DIR, ROOT, write_json


OUT_CSV = NEURAL_DIR / "v1e_parametric_evaluation.csv"
OUT_JSON = NEURAL_DIR / "v1e_parametric_evaluation.json"


def reg_metrics(df: pd.DataFrame, true_col: str, pred_col: str) -> dict:
    mask = np.isfinite(df[true_col]) & np.isfinite(df[pred_col])
    if not mask.any():
        return {"n": 0, "mae": math.nan, "rmse": math.nan, "r2": math.nan}
    y = df.loc[mask, true_col].to_numpy(float)
    p = df.loc[mask, pred_col].to_numpy(float)
    return {
        "n": int(mask.sum()),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(y) > 1 else math.nan,
    }


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    preds = []
    for model_name, rpath, ppath in [
        ("v1e_fno", NEURAL_DIR / "v1e_fno_field_parametric_results.csv", NEURAL_DIR / "v1e_fno_field_parametric_predictions.csv"),
        ("v1e_cnn", NEURAL_DIR / "v1e_cnn_field_parametric_results.csv", NEURAL_DIR / "v1e_cnn_field_parametric_predictions.csv"),
    ]:
        if rpath.exists():
            r = pd.read_csv(rpath)
            r["source_table"] = model_name
            results.append(r)
        if ppath.exists():
            p = pd.read_csv(ppath)
            p["source_table"] = model_name
            preds.append(p)
    if not results:
        raise SystemExit("No v1e model result tables found.")
    return pd.concat(results, ignore_index=True), pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def summarize_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if preds.empty:
        return pd.DataFrame(rows)
    test = preds[preds["split"].eq("test")].copy()
    for model, sub in test.groupby("model"):
        present_true = sub["true_patch_present"].to_numpy(float)
        present_pred = (sub["pred_patch_present_prob"].to_numpy(float) >= 0.5).astype(int)
        rows.append(
            {
                "model": model,
                "split": "test",
                "target": "patch_present",
                "metric_scope": "parametric",
                "n": int(len(sub)),
                "accuracy": float(accuracy_score(present_true.astype(int), present_pred)),
                "macro_f1": float(f1_score(present_true.astype(int), present_pred, average="macro")),
            }
        )
        for true_col, pred_col, target in [
            ("true_patch_center_x_norm", "pred_patch_center_x_norm", "patch_center_x"),
            ("true_patch_center_y_norm", "pred_patch_center_y_norm", "patch_center_y"),
            ("true_patch_radius_norm", "pred_patch_radius_norm", "patch_radius"),
            ("true_patch_area_param", "pred_patch_area_param", "patch_area_param"),
            ("true_patch_area_fraction", "pred_patch_area_fraction_param_mask", "patch_area_fraction_param_mask"),
            ("true_patch_area_fraction", "pred_patch_area_fraction_pixel_mask", "patch_area_fraction_pixel_mask"),
            ("true_APD90_patch_delta", "pred_APD90_patch_delta_param_mask", "APD90_patch_delta_param_mask"),
            ("true_APD90_patch_delta", "pred_APD90_patch_delta_pixel_mask", "APD90_patch_delta_pixel_mask"),
            ("true_LAT_range", "pred_LAT_range", "LAT_range"),
        ]:
            rows.append({"model": model, "split": "test", "target": target, "metric_scope": "summary_from_predictions", **reg_metrics(sub, true_col, pred_col)})
        mask = np.isfinite(sub["true_patch_center_x_norm"]) & np.isfinite(sub["pred_patch_center_x_norm"]) & np.isfinite(sub["true_patch_center_y_norm"]) & np.isfinite(sub["pred_patch_center_y_norm"])
        if mask.any():
            err = np.sqrt(
                (sub.loc[mask, "true_patch_center_x_norm"] - sub.loc[mask, "pred_patch_center_x_norm"]) ** 2
                + (sub.loc[mask, "true_patch_center_y_norm"] - sub.loc[mask, "pred_patch_center_y_norm"]) ** 2
            )
            rows.append({"model": model, "split": "test", "target": "patch_center_error", "metric_scope": "parametric", "n": int(mask.sum()), "mae": float(np.mean(err)), "rmse": float(np.sqrt(np.mean(err * err))), "r2": math.nan})
    return pd.DataFrame(rows)


def make_figures(results: pd.DataFrame, preds: pd.DataFrame, summary: pd.DataFrame) -> None:
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    test = results[results["split"].eq("test")].copy()
    keep = test[test["target"].isin(["LAT", "APD90", "APD80", "patch_mask", "APD90_patch_delta_pixel_mask", "APD90_patch_delta_param_mask", "patch_area_fraction_pixel_mask", "patch_area_fraction_param_mask"])]
    if len(keep):
        fig, ax = plt.subplots(figsize=(10, 4))
        labels = keep["model"] + "\n" + keep["target"]
        vals = keep["r2"].where(keep["r2"].notna(), keep.get("dice", np.nan))
        ax.bar(range(len(keep)), vals)
        ax.set_xticks(range(len(keep)))
        ax.set_xticklabels(labels, rotation=80, fontsize=7)
        ax.set_title("v1e Direct FNO/CNN Summary")
        fig.tight_layout()
        fig.savefig(NEURAL_FIG_DIR / "fno_vs_cnn_direct_summary.png", dpi=180)
        plt.close(fig)
    if not preds.empty:
        testp = preds[preds["split"].eq("test")]
        for pred_col, fname, title in [
            ("pred_APD90_patch_delta_param_mask", "apd90_delta_recovery.png", "APD90 Delta Recovery"),
            ("pred_patch_area_param", "fno_patch_param_predictions.png", "Patch Area Parametric Prediction"),
        ]:
            fig, ax = plt.subplots(figsize=(5, 5))
            for model, sub in testp.groupby("model"):
                true_col = "true_APD90_patch_delta" if "APD90" in pred_col else "true_patch_area_param"
                mask = np.isfinite(sub[true_col]) & np.isfinite(sub[pred_col])
                if mask.any():
                    ax.scatter(sub.loc[mask, true_col], sub.loc[mask, pred_col], s=12, alpha=0.6, label=model)
            ax.set_title(title)
            ax.set_xlabel("true")
            ax.set_ylabel("predicted")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(NEURAL_FIG_DIR / fname, dpi=180)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 5))
        sub = testp[np.isfinite(testp["pred_patch_area_fraction_param_mask"]) & np.isfinite(testp["pred_patch_area_fraction_pixel_mask"])]
        ax.scatter(sub["pred_patch_area_fraction_pixel_mask"], sub["pred_patch_area_fraction_param_mask"], s=10, alpha=0.5)
        ax.set_xlabel("pixel-mask area")
        ax.set_ylabel("parametric-mask area")
        ax.set_title("Parametric Mask vs Pixel Mask")
        fig.tight_layout()
        fig.savefig(NEURAL_FIG_DIR / "parametric_mask_vs_pixel_mask.png", dpi=180)
        plt.close(fig)
    src_fig = PROJECT_ROOT / "figures" / "v1e_direct_magnetic_pilot" / "direct_bz_examples.png"
    if src_fig.exists():
                                                                             
                                                                               
        pass


def main() -> int:
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    results, preds = load_tables()
    summary = summarize_predictions(preds)
    out = pd.concat([results, summary], ignore_index=True, sort=False)
    out.to_csv(OUT_CSV, index=False)
    make_figures(results, preds, summary)
    payload = {
        "results_sources": ["v1e_fno_field_parametric_results.csv", "v1e_cnn_field_parametric_results.csv"],
        "prediction_sources": ["v1e_fno_field_parametric_predictions.csv", "v1e_cnn_field_parametric_predictions.csv"],
        "records": out.replace({np.nan: None}).to_dict(orient="records"),
        "figures": {
            "direct_input_examples": "figures/v1e_direct_magnetic_pilot/direct_bz_examples.png",
            "fno_vs_cnn_direct_summary": "figures/neural_v1e/fno_vs_cnn_direct_summary.png",
            "fno_patch_param_predictions": "figures/neural_v1e/fno_patch_param_predictions.png",
            "parametric_mask_vs_pixel_mask": "figures/neural_v1e/parametric_mask_vs_pixel_mask.png",
            "apd90_delta_recovery": "figures/neural_v1e/apd90_delta_recovery.png",
        },
    }
    write_json(OUT_JSON, payload)
    print(json.dumps({"evaluation": str(OUT_CSV), "rows": int(len(out))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
