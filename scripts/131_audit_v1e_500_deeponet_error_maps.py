                      
                                                                    

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v1e_500_common import NEURAL_DIR, NEURAL_FIG_DIR, require_cuda, write_json
from v1e_deeponet_calibration_common import (
    area_fraction,
    centroid_error,
    load_or_build_full_grid_cache,
    patch_metrics,
    regression_metrics,
    scalar_targets,
    threshold_mask,
    torch,
)


AUDIT_CSV = NEURAL_DIR / "deeponet_error_map_audit.csv"
AUDIT_JSON = NEURAL_DIR / "deeponet_error_map_audit.json"
EXAMPLES_CSV = NEURAL_DIR / "deeponet_good_bad_examples.csv"


def row_metrics(index: pd.DataFrame, maps: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for i, row in index.iterrows():
        true_patch = maps["true_patch"][i]
        valid = maps["valid_mask"][i]
        pred_prob = maps["pred_patch_prob"][i]
        pred_mask = threshold_mask(pred_prob, valid, 0.5)
        true_apd90 = maps["true_apd90"][i]
        pred_apd90 = maps["pred_apd90"][i]
        inside = (true_patch > 0.5) & valid
        outside = (true_patch <= 0.5) & valid
        apd_err = pred_apd90 - true_apd90
        pm = patch_metrics(true_patch, pred_mask, valid)
        true_area = area_fraction(true_patch, valid)
        pred_area = area_fraction(pred_mask, valid)
        true_delta = scalar_targets(true_apd90, true_patch, valid)
        pred_delta = scalar_targets(pred_apd90, pred_mask, valid)
        pred_delta_true_apd = scalar_targets(true_apd90, pred_mask, valid)
        true_mask_pred_delta = scalar_targets(pred_apd90, true_patch, valid)
        prob_total = float(np.sum(pred_prob[valid])) if np.any(valid) else 0.0
        prob_inside = float(np.sum(pred_prob[inside])) if inside.any() else math.nan
        prob_outside = float(np.sum(pred_prob[outside])) if outside.any() else math.nan
        rows.append(
            {
                "record_type": "row",
                "model": row["model"],
                "split": row["split"],
                "row_id": row["row_id"],
                "base_sample_id": row["base_sample_id"],
                "condition": row["condition"],
                "combined_variant": row["combined_variant"],
                "patch_present": float(row["true_patch_present"]),
                "dice": pm["dice"],
                "iou": pm["iou"],
                "precision": pm["precision"],
                "recall": pm["recall"],
                "apd90_abs_error_inside_true_patch": float(np.mean(np.abs(apd_err[inside]))) if inside.any() else math.nan,
                "apd90_abs_error_outside_true_patch": float(np.mean(np.abs(apd_err[outside]))) if outside.any() else math.nan,
                "apd90_signed_error_inside_true_patch": float(np.mean(apd_err[inside])) if inside.any() else math.nan,
                "apd90_signed_error_outside_true_patch": float(np.mean(apd_err[outside])) if outside.any() else math.nan,
                "probability_mass_inside_true_patch": prob_inside,
                "probability_mass_outside_true_patch": prob_outside,
                "probability_mass_inside_fraction": prob_inside / prob_total if prob_total > 0 and np.isfinite(prob_inside) else math.nan,
                "mean_probability_inside_true_patch": float(np.mean(pred_prob[inside])) if inside.any() else math.nan,
                "mean_probability_outside_true_patch": float(np.mean(pred_prob[outside])) if outside.any() else math.nan,
                "centroid_error_probability": centroid_error(pred_prob, true_patch, valid),
                "true_area_fraction": true_area,
                "predicted_area_fraction": pred_area,
                "area_bias": pred_area - true_area if np.isfinite(pred_area) and np.isfinite(true_area) else math.nan,
                "homogeneous_false_positive_area": pred_area if float(row["true_patch_present"]) < 0.5 else math.nan,
                "true_apd90_patch_delta": true_delta,
                "pred_apd90_patch_delta_pixel_mask_pred_apd": pred_delta,
                "pred_apd90_patch_delta_pixel_mask_true_apd": pred_delta_true_apd,
                "pred_apd90_patch_delta_true_mask_pred_apd": true_mask_pred_delta,
                "abs_delta_error_pixel_mask_pred_apd": abs(pred_delta - true_delta) if np.isfinite(pred_delta) and np.isfinite(true_delta) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def group_summaries(rows: pd.DataFrame) -> pd.DataFrame:
    summary = []
    for group_cols, label in [
        (["model"], "model"),
        (["model", "condition"], "condition"),
        (["model", "combined_variant"], "combined_variant"),
    ]:
        for keys, sub in rows.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            rec = {"record_type": label, "split": "test", "n": int(len(sub))}
            for col, val in zip(group_cols, keys):
                rec[col] = val
            for metric in [
                "dice",
                "iou",
                "precision",
                "recall",
                "apd90_abs_error_inside_true_patch",
                "apd90_abs_error_outside_true_patch",
                "probability_mass_inside_fraction",
                "mean_probability_inside_true_patch",
                "mean_probability_outside_true_patch",
                "centroid_error_probability",
                "area_bias",
                "homogeneous_false_positive_area",
            ]:
                rec[f"mean_{metric}"] = float(sub[metric].mean(skipna=True)) if metric in sub else math.nan
            rec["APD90_patch_delta_R2_pixel_mask_pred_apd"] = regression_metrics(
                sub["true_apd90_patch_delta"], sub["pred_apd90_patch_delta_pixel_mask_pred_apd"]
            )["r2"]
            rec["APD90_patch_delta_R2_pixel_mask_true_apd"] = regression_metrics(
                sub["true_apd90_patch_delta"], sub["pred_apd90_patch_delta_pixel_mask_true_apd"]
            )["r2"]
            rec["APD90_patch_delta_R2_true_mask_pred_apd"] = regression_metrics(
                sub["true_apd90_patch_delta"], sub["pred_apd90_patch_delta_true_mask_pred_apd"]
            )["r2"]
            summary.append(rec)
    return pd.DataFrame(summary)


def good_bad_examples(rows: pd.DataFrame) -> pd.DataFrame:
    non = rows[(rows["record_type"].eq("row")) & (rows["patch_present"].eq(1))].copy()
    examples = []
    for model, sub in non.groupby("model"):
        good = sub.sort_values(["dice", "abs_delta_error_pixel_mask_pred_apd"], ascending=[False, True]).head(5)
        bad = sub.sort_values(["dice", "abs_delta_error_pixel_mask_pred_apd"], ascending=[True, False]).head(5)
        delta_bad = sub.sort_values("abs_delta_error_pixel_mask_pred_apd", ascending=False).head(5)
        for tag, part in [("good_dice", good), ("bad_dice", bad), ("bad_delta", delta_bad)]:
            for rank, (_, row) in enumerate(part.iterrows(), start=1):
                examples.append(
                    {
                        "model": model,
                        "example_type": tag,
                        "rank": rank,
                        "row_id": row["row_id"],
                        "base_sample_id": row["base_sample_id"],
                        "condition": row["condition"],
                        "combined_variant": row["combined_variant"],
                        "dice": row["dice"],
                        "centroid_error_probability": row["centroid_error_probability"],
                        "area_bias": row["area_bias"],
                        "true_apd90_patch_delta": row["true_apd90_patch_delta"],
                        "pred_apd90_patch_delta_pixel_mask_pred_apd": row["pred_apd90_patch_delta_pixel_mask_pred_apd"],
                        "abs_delta_error_pixel_mask_pred_apd": row["abs_delta_error_pixel_mask_pred_apd"],
                    }
                )
    return pd.DataFrame(examples)


def make_figures(rows: pd.DataFrame, examples: pd.DataFrame, index: pd.DataFrame, maps: dict[str, np.ndarray]) -> None:
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    row_df = rows[rows["record_type"].eq("row")].copy()

    best_model = row_df.groupby("model")["dice"].mean().sort_values(ascending=False).index[0]
    ex_rows = examples[(examples["model"].eq(best_model)) & (examples["example_type"].isin(["good_dice", "bad_dice"]))].groupby("example_type").head(2)
    selected = []
    for _, ex in ex_rows.iterrows():
        matches = np.flatnonzero((index["model"].eq(ex["model"]) & index["row_id"].eq(ex["row_id"])).to_numpy())
        if len(matches):
            selected.append((ex, int(matches[0])))
    if selected:
        fig, axes = plt.subplots(len(selected), 4, figsize=(12, 3 * len(selected)))
        axes = np.atleast_2d(axes)
        for axrow, (ex, idx) in zip(axes, selected):
            valid = maps["valid_mask"][idx]
            pred_mask = threshold_mask(maps["pred_patch_prob"][idx], valid, 0.5)
            apd_err = np.where(valid, maps["pred_apd90"][idx] - maps["true_apd90"][idx], np.nan)
            items = [
                (maps["true_patch"][idx], f"{ex['example_type']} true", "magma", 0, 1),
                (maps["pred_patch_prob"][idx], "probability", "magma", 0, 1),
                (pred_mask, "threshold 0.5", "magma", 0, 1),
                (apd_err, "APD90 error", "coolwarm", None, None),
            ]
            for ax, (arr, title, cmap, vmin, vmax) in zip(axrow, items):
                im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f"{title}\nDice {ex['dice']:.3f}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.tight_layout()
        fig.savefig(NEURAL_FIG_DIR / "deeponet_error_maps_good_bad.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    model_order = row_df.groupby("model")["probability_mass_inside_fraction"].mean().sort_values(ascending=False).index.tolist()
    data = [row_df[row_df["model"].eq(m)]["probability_mass_inside_fraction"].dropna().to_numpy() for m in model_order]
    ax.boxplot(data, labels=model_order, showfliers=False)
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    ax.set_ylabel("probability mass inside true patch / total mass")
    ax.set_title("DeepONet Patch Probability Mass")
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "deeponet_patch_probability_mass.png", dpi=180)
    plt.close(fig)

    cond = row_df.groupby(["model", "condition"], as_index=False)["dice"].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    for model, sub in cond.groupby("model"):
        ax.plot(sub["condition"], sub["dice"], marker="o", label=model)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.set_ylabel("mean Dice")
    ax.set_title("DeepONet Dice By Condition")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "deeponet_dice_by_condition.png", dpi=180)
    plt.close(fig)

    var = row_df.groupby(["model", "combined_variant"], as_index=False)["dice"].mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    for model, sub in var.groupby("model"):
        ax.plot(sub["combined_variant"], sub["dice"], marker="o", label=model)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.set_ylabel("mean Dice")
    ax.set_title("DeepONet Dice By Combined Variant")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "deeponet_dice_by_variant.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    err = row_df.groupby("model", as_index=False)[["apd90_abs_error_inside_true_patch", "apd90_abs_error_outside_true_patch"]].mean()
    x = np.arange(len(err))
    ax.bar(x - 0.18, err["apd90_abs_error_inside_true_patch"], width=0.36, label="inside patch")
    ax.bar(x + 0.18, err["apd90_abs_error_outside_true_patch"], width=0.36, label="outside patch")
    ax.set_xticks(x)
    ax.set_xticklabels(err["model"], rotation=75, fontsize=7)
    ax.set_ylabel("APD90 MAE")
    ax.set_title("APD90 Error Inside vs Outside True Patch")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "deeponet_apd90_error_inside_outside_patch.png", dpi=180)
    plt.close(fig)


def main() -> int:
    require_cuda(torch)
    index, maps = load_or_build_full_grid_cache(splits=("val", "test"), batch_size=4)
    test_mask = index["split"].eq("test")
    test_index = index[test_mask].reset_index(drop=True)
    test_maps = {k: v[np.flatnonzero(test_mask.to_numpy())] for k, v in maps.items()}
    row_df = row_metrics(test_index, test_maps)
    summary_df = group_summaries(row_df)
    audit = pd.concat([row_df, summary_df], ignore_index=True, sort=False)
    examples = good_bad_examples(row_df)
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_CSV, index=False)
    examples.to_csv(EXAMPLES_CSV, index=False)
    make_figures(row_df, examples, test_index, test_maps)

    model_summary = summary_df[summary_df["record_type"].eq("model")].copy()
    payload = {
        "audit_csv": str(AUDIT_CSV),
        "examples_csv": str(EXAMPLES_CSV),
        "models": sorted(row_df["model"].unique().tolist()),
        "model_summary": model_summary.replace({np.nan: None}).to_dict(orient="records"),
        "figures": [
            "figures/neural_v1e_500/deeponet_error_maps_good_bad.png",
            "figures/neural_v1e_500/deeponet_patch_probability_mass.png",
            "figures/neural_v1e_500/deeponet_dice_by_condition.png",
            "figures/neural_v1e_500/deeponet_dice_by_variant.png",
            "figures/neural_v1e_500/deeponet_apd90_error_inside_outside_patch.png",
        ],
        "leakage_guardrail": "Audit regenerates predictions from checkpoints for analysis only. Target fields/masks are labels for evaluation, not model inputs.",
    }
    write_json(AUDIT_JSON, payload)
    best = model_summary.sort_values("mean_dice", ascending=False).head(1)
    print(json.dumps({"audit": str(AUDIT_CSV), "examples": str(EXAMPLES_CSV), "best_model": best[["model", "mean_dice"]].to_dict(orient="records")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
