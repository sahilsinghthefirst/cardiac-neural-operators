                      
                                                                    

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
    centroid,
    centroid_error,
    disk_mask,
    load_or_build_full_grid_cache,
    patch_metrics,
    regression_metrics,
    scalar_targets,
    soft_delta,
    threshold_mask,
    topk_mask,
    torch,
)


VAL_CSV = NEURAL_DIR / "deeponet_mask_calibration_validation.csv"
TEST_CSV = NEURAL_DIR / "deeponet_mask_calibration_test.csv"
OUT_JSON = NEURAL_DIR / "deeponet_mask_calibration.json"


THRESHOLDS = np.round(np.linspace(0.01, 0.99, 99), 2)
BASELINE_DELTA_R2 = 0.3346258400089735


def mask_for_row(row, true_patch, valid, prob, method: str, threshold: float | None = None) -> np.ndarray:
    present = float(row.get("pred_patch_present_prob", 1.0))
    if method.endswith("_threshold"):
        return threshold_mask(prob, valid, float(threshold if threshold is not None else 0.5))
    if method == "validation_topk_area_prior":
        area = float(row.get("calibration_area_prior", math.nan))
        return topk_mask(prob, valid, area) if present >= 0.5 else np.zeros_like(true_patch, dtype=np.float32)
    if method == "parametric_area_topk":
        return topk_mask(prob, valid, float(row.get("pred_patch_area_param", math.nan))) if present >= 0.5 else np.zeros_like(true_patch, dtype=np.float32)
    if method == "hybrid_prob_centroid_param_size_disk":
        area = float(row.get("pred_patch_area_param", math.nan))
        radius_area = math.sqrt(max(area, 0.0) / math.pi) if np.isfinite(area) else math.nan
        radius_pred = float(row.get("pred_radius_param", math.nan))
        if np.isfinite(radius_area) and np.isfinite(radius_pred):
            radius = 0.5 * (radius_area + radius_pred)
        elif np.isfinite(radius_area):
            radius = radius_area
        else:
            radius = radius_pred
        cx, cy = centroid(prob, valid, weighted=True)
        return disk_mask(cx, cy, radius, present=present, shape=true_patch.shape)
    raise ValueError(f"Unknown binary mask method: {method}")


def evaluate_method(
    index: pd.DataFrame,
    maps: dict[str, np.ndarray],
    *,
    model: str,
    method: str,
    settings: dict | None = None,
    split: str,
    threshold: float | None = None,
    calibration_area_prior: float | None = None,
) -> dict:
    mask = index["model"].eq(model) & index["split"].eq(split)
    sub = index[mask].reset_index(drop=True).copy()
    idxs = np.flatnonzero(mask.to_numpy())
    if calibration_area_prior is not None:
        sub["calibration_area_prior"] = calibration_area_prior
    rows = []
    all_true = []
    all_pred = []
    all_valid = []
    for local, global_idx in enumerate(idxs):
        row = sub.iloc[local]
        valid = maps["valid_mask"][global_idx]
        true_patch = maps["true_patch"][global_idx]
        prob = maps["pred_patch_prob"][global_idx]
        if method == "soft_mask":
            pred_area = float(np.mean(prob[valid])) if np.any(valid) else math.nan
            pred_delta = soft_delta(maps["pred_apd90"][global_idx], prob, valid)
            pred_delta_true = soft_delta(maps["true_apd90"][global_idx], prob, valid)
            bin_mask = threshold_mask(prob, valid, 0.5)
        else:
            thr = threshold
            if settings:
                if method == "condition_threshold":
                    thr = settings.get((model, row["condition"]), settings.get((model, "__global__"), 0.5))
                elif method == "variant_threshold":
                    thr = settings.get((model, row["combined_variant"]), settings.get((model, "__global__"), 0.5))
                elif method == "condition_variant_threshold":
                    thr = settings.get((model, row["condition"], row["combined_variant"]), settings.get((model, "__global__"), 0.5))
                elif method == "global_threshold":
                    thr = settings.get((model, "__global__"), 0.5)
            bin_mask = mask_for_row(row, true_patch, valid, prob, method if method not in {"global_threshold", "condition_threshold", "variant_threshold", "condition_variant_threshold"} else "global_threshold", thr)
            pred_area = area_fraction(bin_mask, valid)
            pred_delta = scalar_targets(maps["pred_apd90"][global_idx], bin_mask, valid)
            pred_delta_true = scalar_targets(maps["true_apd90"][global_idx], bin_mask, valid)
        true_delta = scalar_targets(maps["true_apd90"][global_idx], true_patch, valid)
        pm = patch_metrics(true_patch, bin_mask, valid)
        true_area = area_fraction(true_patch, valid)
        all_true.append(true_patch)
        all_pred.append(bin_mask)
        all_valid.append(valid)
        rows.append(
            {
                "true_patch_present": float(row["true_patch_present"]),
                "condition": row["condition"],
                "combined_variant": row["combined_variant"],
                "dice": pm["dice"],
                "iou": pm["iou"],
                "precision": pm["precision"],
                "recall": pm["recall"],
                "true_area": true_area,
                "pred_area": pred_area,
                "area_bias": pred_area - true_area if np.isfinite(pred_area) and np.isfinite(true_area) else math.nan,
                "true_delta": true_delta,
                "pred_delta_pred_apd": pred_delta,
                "pred_delta_true_apd": pred_delta_true,
                "centroid_error": centroid_error(prob if method == "soft_mask" else bin_mask, true_patch, valid),
                "homogeneous_false_positive_area": pred_area if float(row["true_patch_present"]) < 0.5 else math.nan,
            }
        )
    df = pd.DataFrame(rows)
    all_true_cat = np.stack(all_true) if all_true else np.zeros((0, 32, 32))
    all_pred_cat = np.stack(all_pred) if all_pred else np.zeros((0, 32, 32))
    all_valid_cat = np.stack(all_valid).astype(bool) if all_valid else np.zeros((0, 32, 32), dtype=bool)
    micro = patch_metrics(all_true_cat, all_pred_cat, all_valid_cat) if len(df) else {"dice": math.nan, "iou": math.nan, "precision": math.nan, "recall": math.nan}
    non = df[df["true_patch_present"].eq(1)]
    rec = {
        "split": split,
        "model": model,
        "method": method,
        "threshold": threshold,
        "diagnostic_non_deployable": method in {"condition_threshold", "variant_threshold", "condition_variant_threshold"},
        "n": int(len(df)),
        "dice": float(df["dice"].mean(skipna=True)),
        "micro_dice": micro["dice"],
        "iou": float(df["iou"].mean(skipna=True)),
        "precision": float(df["precision"].mean(skipna=True)),
        "recall": float(df["recall"].mean(skipna=True)),
        "homogeneous_false_positive_area": float(df["homogeneous_false_positive_area"].mean(skipna=True)),
        "patch_area_r2": regression_metrics(df["true_area"], df["pred_area"])["r2"],
        "patch_area_mae": regression_metrics(df["true_area"], df["pred_area"])["mae"],
        "area_bias": float(df["area_bias"].mean(skipna=True)),
        "APD90_patch_delta_R2_pred_apd": regression_metrics(non["true_delta"], non["pred_delta_pred_apd"])["r2"] if len(non) else math.nan,
        "APD90_patch_delta_R2_true_apd": regression_metrics(non["true_delta"], non["pred_delta_true_apd"])["r2"] if len(non) else math.nan,
        "APD90_patch_delta_MAE_pred_apd": regression_metrics(non["true_delta"], non["pred_delta_pred_apd"])["mae"] if len(non) else math.nan,
        "centroid_error": float(non["centroid_error"].mean(skipna=True)) if len(non) else math.nan,
    }
    if split == "val":
        delta = rec["APD90_patch_delta_R2_pred_apd"]
        rec["selection_score"] = (delta if np.isfinite(delta) else -1.0) + 0.25 * rec["dice"] - 0.5 * rec["homogeneous_false_positive_area"]
    return rec


def select_thresholds(index: pd.DataFrame, maps: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, dict]]:
    val_records = []
    settings: dict[str, dict] = {"global_threshold": {}, "condition_threshold": {}, "variant_threshold": {}, "condition_variant_threshold": {}}
    models = sorted(index["model"].unique().tolist())
    for model in models:
        best_global = None
        for thr in THRESHOLDS:
            rec = evaluate_method(index, maps, model=model, method="global_threshold", split="val", threshold=float(thr))
            rec["candidate_scope"] = "global_sweep"
            val_records.append(rec)
            if best_global is None or rec["selection_score"] > best_global["selection_score"]:
                best_global = rec
        settings["global_threshold"][(model, "__global__")] = float(best_global["threshold"])
        val_records.append({**best_global, "candidate_scope": "selected_global"})

        val_model = index[(index["model"].eq(model)) & (index["split"].eq("val"))]
        for group_col, method, scope in [
            ("condition", "condition_threshold", "selected_condition"),
            ("combined_variant", "variant_threshold", "selected_variant"),
        ]:
            for group_value in sorted(val_model[group_col].unique().tolist()):
                best = None
                for thr in THRESHOLDS:
                    group_idxs = np.flatnonzero(((index["model"].eq(model)) & (index["split"].eq("val")) & (index[group_col].eq(group_value))).to_numpy())
                    tmp_rows = []
                    for gi in group_idxs:
                        row = index.iloc[gi]
                        valid = maps["valid_mask"][gi]
                        true_patch = maps["true_patch"][gi]
                        pred_mask = threshold_mask(maps["pred_patch_prob"][gi], valid, float(thr))
                        true_delta = scalar_targets(maps["true_apd90"][gi], true_patch, valid)
                        pred_delta = scalar_targets(maps["pred_apd90"][gi], pred_mask, valid)
                        pm = patch_metrics(true_patch, pred_mask, valid)
                        tmp_rows.append(
                            {
                                "true_present": float(row["true_patch_present"]),
                                "dice": pm["dice"],
                                "hom_fp": area_fraction(pred_mask, valid) if float(row["true_patch_present"]) < 0.5 else math.nan,
                                "true_delta": true_delta,
                                "pred_delta": pred_delta,
                            }
                        )
                    tmp = pd.DataFrame(tmp_rows)
                    non = tmp[tmp["true_present"].eq(1)]
                    delta_r2 = regression_metrics(non["true_delta"], non["pred_delta"])["r2"] if len(non) else math.nan
                    score = (delta_r2 if np.isfinite(delta_r2) else -1.0) + 0.25 * float(tmp["dice"].mean()) - 0.5 * float(tmp["hom_fp"].mean(skipna=True) if tmp["hom_fp"].notna().any() else 0.0)
                    rec = {
                        "split": "val",
                        "model": model,
                        "method": method,
                        group_col: group_value,
                        "threshold": float(thr),
                        "selection_score": score,
                        "APD90_patch_delta_R2_pred_apd": delta_r2,
                        "dice": float(tmp["dice"].mean()),
                        "homogeneous_false_positive_area": float(tmp["hom_fp"].mean(skipna=True) if tmp["hom_fp"].notna().any() else 0.0),
                        "candidate_scope": scope,
                        "diagnostic_non_deployable": True,
                    }
                    if best is None or rec["selection_score"] > best["selection_score"]:
                        best = rec
                key = (model, group_value)
                settings[method][key] = float(best["threshold"])
                val_records.append(best)

        for condition in sorted(val_model["condition"].unique().tolist()):
            for variant in sorted(val_model["combined_variant"].unique().tolist()):
                group_idxs = np.flatnonzero(((index["model"].eq(model)) & (index["split"].eq("val")) & (index["condition"].eq(condition)) & (index["combined_variant"].eq(variant))).to_numpy())
                if len(group_idxs) == 0:
                    continue
                best = None
                for thr in THRESHOLDS:
                    tmp_rows = []
                    for gi in group_idxs:
                        row = index.iloc[gi]
                        valid = maps["valid_mask"][gi]
                        true_patch = maps["true_patch"][gi]
                        pred_mask = threshold_mask(maps["pred_patch_prob"][gi], valid, float(thr))
                        true_delta = scalar_targets(maps["true_apd90"][gi], true_patch, valid)
                        pred_delta = scalar_targets(maps["pred_apd90"][gi], pred_mask, valid)
                        pm = patch_metrics(true_patch, pred_mask, valid)
                        tmp_rows.append({"true_present": float(row["true_patch_present"]), "dice": pm["dice"], "hom_fp": area_fraction(pred_mask, valid) if float(row["true_patch_present"]) < 0.5 else math.nan, "true_delta": true_delta, "pred_delta": pred_delta})
                    tmp = pd.DataFrame(tmp_rows)
                    non = tmp[tmp["true_present"].eq(1)]
                    delta_r2 = regression_metrics(non["true_delta"], non["pred_delta"])["r2"] if len(non) else math.nan
                    score = (delta_r2 if np.isfinite(delta_r2) else -1.0) + 0.25 * float(tmp["dice"].mean()) - 0.5 * float(tmp["hom_fp"].mean(skipna=True) if tmp["hom_fp"].notna().any() else 0.0)
                    rec = {"split": "val", "model": model, "method": "condition_variant_threshold", "condition": condition, "combined_variant": variant, "threshold": float(thr), "selection_score": score, "APD90_patch_delta_R2_pred_apd": delta_r2, "dice": float(tmp["dice"].mean()), "homogeneous_false_positive_area": float(tmp["hom_fp"].mean(skipna=True) if tmp["hom_fp"].notna().any() else 0.0), "candidate_scope": "selected_condition_variant", "diagnostic_non_deployable": True}
                    if best is None or rec["selection_score"] > best["selection_score"]:
                        best = rec
                settings["condition_variant_threshold"][(model, condition, variant)] = float(best["threshold"])
                val_records.append(best)
    return pd.DataFrame(val_records), settings


def validation_area_priors(index: pd.DataFrame, maps: dict[str, np.ndarray]) -> dict[str, float]:
    priors = {}
    for model, sub in index[index["split"].eq("val")].groupby("model"):
        areas = []
        for idx in sub.index:
            if float(index.loc[idx, "true_patch_present"]) >= 0.5:
                areas.append(area_fraction(maps["true_patch"][idx], maps["valid_mask"][idx]))
        priors[model] = float(np.mean(areas)) if areas else 0.08
    return priors


def selected_method_metrics(index, maps, settings, area_priors, split: str) -> pd.DataFrame:
    records = []
    models = sorted(index["model"].unique().tolist())
    for model in models:
        methods = [
            "global_threshold",
            "condition_threshold",
            "variant_threshold",
            "condition_variant_threshold",
            "soft_mask",
            "validation_topk_area_prior",
            "parametric_area_topk",
            "hybrid_prob_centroid_param_size_disk",
        ]
        for method in methods:
            rec = evaluate_method(
                index,
                maps,
                model=model,
                method=method,
                settings=settings.get(method),
                split=split,
                calibration_area_prior=area_priors.get(model),
            )
            if method == "global_threshold":
                rec["threshold"] = settings["global_threshold"].get((model, "__global__"))
            rec["calibration_area_prior"] = area_priors.get(model)
            rec["selected_on"] = "validation" if method in {"global_threshold", "condition_threshold", "variant_threshold", "condition_variant_threshold", "validation_topk_area_prior"} else "model_prediction_no_test_labels"
            records.append(rec)
    return pd.DataFrame(records)


def make_figures(val_df: pd.DataFrame, test_df: pd.DataFrame, index: pd.DataFrame, maps: dict[str, np.ndarray], settings: dict, area_priors: dict) -> None:
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    sweep = val_df[val_df.get("candidate_scope", "").eq("global_sweep")].copy()
    if len(sweep):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for model, sub in sweep.groupby("model"):
            axes[0].plot(sub["threshold"], sub["selection_score"], label=model)
            axes[1].plot(sub["threshold"], sub["APD90_patch_delta_R2_pred_apd"], label=model)
        axes[0].set_title("Validation Selection Score")
        axes[1].set_title("Validation APD90 Delta R2")
        for ax in axes:
            ax.set_xlabel("threshold")
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(NEURAL_FIG_DIR / "deeponet_threshold_sweeps.png", dpi=180)
        plt.close(fig)

    best_row = test_df.sort_values("APD90_patch_delta_R2_pred_apd", ascending=False).head(1)
    if len(best_row):
        model = str(best_row.iloc[0]["model"])
        method = str(best_row.iloc[0]["method"])
        candidates = np.flatnonzero(((index["model"].eq(model)) & (index["split"].eq("test")) & (index["true_patch_present"].ge(1))).to_numpy())[:3]
        if len(candidates):
            fig, axes = plt.subplots(len(candidates), 4, figsize=(12, 3 * len(candidates)))
            axes = np.atleast_2d(axes)
            for axrow, idx in zip(axes, candidates):
                valid = maps["valid_mask"][idx]
                prob = maps["pred_patch_prob"][idx]
                row = index.iloc[idx].copy()
                row["calibration_area_prior"] = area_priors.get(model)
                if method == "global_threshold":
                    mask = threshold_mask(prob, valid, settings["global_threshold"].get((model, "__global__"), 0.5))
                elif method == "soft_mask":
                    mask = prob
                elif method == "validation_topk_area_prior":
                    mask = topk_mask(prob, valid, area_priors.get(model, 0.08))
                elif method == "parametric_area_topk":
                    mask = topk_mask(prob, valid, row["pred_patch_area_param"])
                elif method == "hybrid_prob_centroid_param_size_disk":
                    mask = mask_for_row(row, maps["true_patch"][idx], valid, prob, method)
                else:
                    mask = threshold_mask(prob, valid, 0.5)
                items = [(maps["true_patch"][idx], "true patch"), (prob, "probability"), (mask, method), (maps["pred_apd90"][idx] - maps["true_apd90"][idx], "APD90 error")]
                for ax, (arr, title) in zip(axrow, items):
                    im = ax.imshow(arr, origin="lower", cmap="magma" if title != "APD90 error" else "coolwarm")
                    ax.set_title(title, fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            fig.tight_layout()
            fig.savefig(NEURAL_FIG_DIR / "deeponet_calibrated_mask_examples.png", dpi=180)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    comp = test_df[test_df["method"].isin(["global_threshold", "soft_mask"])]
    for method, sub in comp.groupby("method"):
        ax.bar(sub["model"] + "\n" + method, sub["APD90_patch_delta_R2_pred_apd"], label=method)
    ax.axhline(BASELINE_DELTA_R2, color="black", linestyle="--", linewidth=1, label="original best 0.335")
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    ax.set_ylabel("APD90 delta R2, predicted APD90")
    ax.set_title("Soft vs Binary Delta Extraction")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "deeponet_soft_vs_binary_delta.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    comp = test_df[test_df["method"].isin(["global_threshold", "parametric_area_topk", "hybrid_prob_centroid_param_size_disk"])]
    labels = comp["model"] + "\n" + comp["method"]
    ax.bar(range(len(comp)), comp["patch_area_r2"])
    ax.set_xticks(range(len(comp)))
    ax.set_xticklabels(labels, rotation=80, fontsize=6)
    ax.set_ylabel("patch area R2")
    ax.set_title("Parametric Area Mask Effect")
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "deeponet_parametric_area_mask_effect.png", dpi=180)
    plt.close(fig)


def main() -> int:
    require_cuda(torch)
    index, maps = load_or_build_full_grid_cache(splits=("val", "test"), batch_size=4)
    val_sweep, settings = select_thresholds(index, maps)
    area_priors = validation_area_priors(index, maps)
    val_selected = selected_method_metrics(index, maps, settings, area_priors, "val")
    test_selected = selected_method_metrics(index, maps, settings, area_priors, "test")
    val_out = pd.concat([val_sweep, val_selected], ignore_index=True, sort=False)
    val_out.to_csv(VAL_CSV, index=False)
    test_selected.to_csv(TEST_CSV, index=False)
    make_figures(val_out, test_selected, index, maps, settings, area_priors)
    best = test_selected.sort_values("APD90_patch_delta_R2_pred_apd", ascending=False).head(1)
    payload = {
        "validation_csv": str(VAL_CSV),
        "test_csv": str(TEST_CSV),
        "selected_settings": {
            method: {"|".join(map(str, key)): value for key, value in vals.items()} for method, vals in settings.items()
        },
        "validation_area_priors": area_priors,
        "best_test_predicted_apd_delta": best.replace({np.nan: None}).to_dict(orient="records"),
        "baseline_predicted_pixel_mask_apd_delta_r2": BASELINE_DELTA_R2,
        "no_test_label_calibration": True,
        "nondeployable_diagnostic_methods": ["condition_threshold", "variant_threshold", "condition_variant_threshold"],
        "leakage_guardrail": "Validation labels select thresholds/area priors. Test labels are used once for evaluation only. Calibration settings are not model inputs.",
    }
    write_json(OUT_JSON, payload)
    print(json.dumps({"validation": str(VAL_CSV), "test": str(TEST_CSV), "best": payload["best_test_predicted_apd_delta"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
