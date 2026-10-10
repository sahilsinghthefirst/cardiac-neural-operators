                      
                                                                  

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v1e_500_common import NEURAL_DIR, NEURAL_FIG_DIR, ROOT, write_json


OUT_RESULTS = NEURAL_DIR / "field_parametric_results.csv"
OUT_JSON = NEURAL_DIR / "field_parametric_results.json"
OUT_PREDS = NEURAL_DIR / "field_parametric_predictions.csv"


def load_optional(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def metric(results: pd.DataFrame, model: str, target: str, col: str):
    sub = results[(results["split"].eq("test")) & (results["model"].eq(model)) & (results["target"].eq(target))]
    if sub.empty or col not in sub.columns:
        return math.nan
    vals = sub[col].dropna()
    return float(vals.iloc[0]) if len(vals) else math.nan


def make_figures(results: pd.DataFrame, preds: pd.DataFrame) -> None:
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    test = results[results["split"].eq("test")].copy()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, target, title, col in zip(axes, ["LAT", "APD90", "patch_mask"], ["LAT R2", "APD90 R2", "Patch Dice"], ["r2", "r2", "dice"]):
        vals = test[test["target"].eq(target)]
        ax.bar(vals["model"], vals[col])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=75, labelsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "field_parametric_summary.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    vals = test[test["target"].isin(["APD90_patch_delta_pixel_mask_pred_apd", "APD90_patch_delta_param_mask_pred_apd", "APD90_patch_delta_param_mask_true_apd", "APD90_patch_delta_true_mask_pred_apd"])]
    ax.bar(vals["model"] + "\n" + vals["target"], vals["r2"])
    ax.tick_params(axis="x", rotation=80, labelsize=6)
    ax.set_ylabel("R2")
    ax.set_title("APD90 Delta Recovery")
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "apd90_delta_recovery.png", dpi=180)
    plt.close(fig)

    test_pred = preds[preds["split"].eq("test")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, true_col, pred_col, title in [
        (axes[0], "true_patch_area_param", "pred_patch_area_param", "Area"),
        (axes[1], "true_radius_param", "pred_radius_param", "Radius"),
    ]:
        for model, sub in test_pred.groupby("model"):
            m = np.isfinite(sub[true_col]) & np.isfinite(sub[pred_col])
            ax.scatter(sub.loc[m, true_col], sub.loc[m, pred_col], s=8, alpha=0.55, label=model)
        ax.set_xlabel(f"true {title.lower()}")
        ax.set_ylabel(f"pred {title.lower()}")
        ax.set_title(title)
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "area_radius_predictions.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    patch = test[test["target"].eq("patch_mask")][["model", "dice"]]
    ax.bar(patch["model"], patch["dice"])
    ax.axhline(0.438, color="black", linestyle="--", linewidth=1, label="v1e 500 parametric Dice")
    ax.set_ylabel("Dice")
    ax.set_title("Patch Mask And Parametric Comparison")
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "patch_mask_and_parametric_examples.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    desc_pairs = []
    for base in ["fno_field_plus_param_heads", "cnn_field_plus_param_heads"]:
        desc_model = base + "_plus_descriptors" if base.startswith("cnn") else base + "_plus_descriptors"
        for target, col in [("patch_mask", "dice"), ("APD90_patch_delta_param_mask_true_apd", "r2"), ("patch_area_param_head", "r2")]:
            base_val = metric(test, base, target, col)
            desc_val = metric(test, desc_model, target, col)
            if np.isfinite(base_val) and np.isfinite(desc_val):
                desc_pairs.append({"label": f"{base}\n{target}", "delta": desc_val - base_val})
    if desc_pairs:
        ddf = pd.DataFrame(desc_pairs)
        ax.bar(ddf["label"], ddf["delta"])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Descriptor Branch Effect")
    ax.set_ylabel("descriptor - no descriptor")
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "descriptor_branch_effect.png", dpi=180)
    plt.close(fig)

                                                                               
                                                                       
    fig, ax = plt.subplots(figsize=(6, 5))
    vals = test[test["target"].isin(["LAT", "APD90"])].copy()
    ax.scatter(vals["target"], vals["r2"], c=vals["model"].astype("category").cat.codes, s=35)
    ax.set_title("FNO vs CNN Field Metrics")
    ax.set_ylabel("pixel R2")
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "fno_vs_cnn_field_maps.png", dpi=180)
    plt.close(fig)


def main() -> int:
    fno_res = load_optional(NEURAL_DIR / "fno_field_parametric_results.csv")
    cnn_res = load_optional(NEURAL_DIR / "cnn_field_parametric_results.csv")
    fno_pred = load_optional(NEURAL_DIR / "fno_field_parametric_predictions.csv")
    cnn_pred = load_optional(NEURAL_DIR / "cnn_field_parametric_predictions.csv")
    results = pd.concat([fno_res, cnn_res], ignore_index=True)
    preds = pd.concat([fno_pred, cnn_pred], ignore_index=True)
    if results.empty or preds.empty:
        raise SystemExit("Missing FNO/CNN field-parametric outputs")
    results.to_csv(OUT_RESULTS, index=False)
    preds.to_csv(OUT_PREDS, index=False)
    make_figures(results, preds)
    test = results[results["split"].eq("test")].copy()
    summary = {
        "result_file": str(OUT_RESULTS),
        "prediction_file": str(OUT_PREDS),
        "models": sorted(results["model"].unique().tolist()),
        "test_metrics": test.replace({np.nan: None}).to_dict(orient="records"),
        "comparisons": {
            "v1d_fno_field_baseline": {"LAT_R2": 0.976, "APD90_R2": 0.981, "patch_Dice": 0.449, "APD90_delta_R2": 0.078, "patch_area_R2": -0.120},
            "v1e_500_parametric_pilot": {"parametric_mask_Dice": 0.438, "APD90_delta_param_mask_true_APD90_R2": 0.704, "area_R2": 0.262, "radius_R2": 0.246},
        },
        "leakage_guardrail": "Evaluation consumes predictions and labels only; model inputs remain magnetic tensors, coordinates, and optional B_meas-derived descriptors.",
    }
    write_json(OUT_JSON, summary)
    print(json.dumps({"results": str(OUT_RESULTS), "predictions": str(OUT_PREDS), "models": summary["models"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
