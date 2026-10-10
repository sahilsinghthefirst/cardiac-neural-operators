                      
                                                                                

from __future__ import annotations

import argparse
import importlib.util
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

from v1e_500_common import CHECKPOINT_DIR, NEURAL_DIR, NEURAL_FIG_DIR, ROOT, SEED, require_cuda, write_json
from v1e_deeponet_calibration_common import area_fraction, centroid, disk_mask, patch_metrics, regression_metrics, scalar_targets
from v1e_deeponet_data_common import compute_stats, load_descriptors, torch
from v1e_deeponet_mask_refine_data_common import make_mask_refine_loader, merge_rows_with_refine_targets


TRAIN_SPEC = importlib.util.spec_from_file_location("_mask_refine_train", PROJECT_ROOT / "scripts" / "137_train_v1e_500_deeponet_mask_refinement.py")
if TRAIN_SPEC is None or TRAIN_SPEC.loader is None:
    raise ImportError("Could not load mask-refinement training module")
MASK_TRAIN = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(MASK_TRAIN)


RESULTS_CSV = NEURAL_DIR / "mask_refine_results.csv"
RESULTS_JSON = NEURAL_DIR / "mask_refine_results.json"
PREDICTIONS_CSV = NEURAL_DIR / "mask_refine_predictions.csv"
CALIBRATION_CSV = NEURAL_DIR / "mask_refine_calibration.csv"

THRESHOLDS = np.round(np.linspace(0.01, 0.99, 99), 2)
BASELINE_DICE = 0.5846666146361768
BASELINE_BALANCED_DELTA_R2 = 0.5996209808256749
BASELINE_SPARSE_DELTA_R2 = 0.7313469085858489
BASELINE_CENTER_MAE = 0.1198441061594165


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def checkpoint_paths() -> list[Path]:
    return sorted(CHECKPOINT_DIR.glob("deeponet_mask_refine_*.pt"))


def build_model(ckpt: dict):
    args = ckpt.get("args", {})
    config = ckpt.get("config", {})
    model = MASK_TRAIN.MaskRefineDeepONet(
        int(ckpt["branch_in_channels"]),
        int(ckpt["descriptor_dim"]),
        basis_dim=int(ckpt.get("basis_dim", args.get("basis_dim", 64))),
        branch_latent=int(ckpt.get("branch_latent", args.get("branch_latent", 96))),
        trunk_hidden=int(ckpt.get("trunk_hidden", args.get("trunk_hidden", 96))),
        dropout=float(ckpt.get("dropout", args.get("dropout", 0.08))),
        use_descriptors=bool(config.get("descriptors", False)),
        fourier_trunk=bool(config.get("fourier", True)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def peak_center(center_prob: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    v = valid.astype(bool)
    if not np.any(v):
        return math.nan, math.nan
    score = np.where(v, center_prob, -np.inf)
    idx = int(np.nanargmax(score))
    y, x = np.unravel_index(idx, center_prob.shape)
    return float(x / max(center_prob.shape[1] - 1, 1)), float(y / max(center_prob.shape[0] - 1, 1))


def center_error(cx: float, cy: float, tx: float, ty: float) -> float:
    if not all(np.isfinite(v) for v in [cx, cy, tx, ty]):
        return math.nan
    return float(math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2))


def unscale_params(params_np: np.ndarray, stats) -> np.ndarray:
    out = params_np.copy()
    out[:, 1:] = out[:, 1:] * stats.param_std[1:] + stats.param_mean[1:]
    return out


def collect_predictions(rows: pd.DataFrame, desc: pd.DataFrame, stats, args, device):
    records = []
    arrays = {
        "true_patch": [],
        "valid": [],
        "patch_prob": [],
        "center_prob": [],
        "pred_apd90": [],
        "true_apd90": [],
    }
    for path in checkpoint_paths():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model_name = str(ckpt.get("model_name", path.stem.replace("deeponet_mask_refine_", "")))
        model = build_model(ckpt).to(device)
        for split in ["val", "test"]:
            sub = rows[rows["split"].eq(split)].reset_index(drop=True)
            loader = make_mask_refine_loader(sub, stats, batch_size=args.batch_size, shuffle=False, mode="full", k_queries=1024, input_norm=args.input_norm, descriptors=desc, seed=SEED + 505)
            offset = 0
            with torch.no_grad():
                for batch in loader:
                    query, params, area_logits = model(
                        batch["branch_x"].to(device).float(),
                        batch["trunk_xy"].to(device).float(),
                        batch["descriptor_x"].to(device).float(),
                    )
                    query_np = query.detach().cpu().numpy()
                    params_np = params.detach().cpu().numpy()
                    area_np = area_logits.detach().cpu().numpy()
                    p_unscaled = unscale_params(params_np, stats)
                    y_param = batch["y_param"].numpy()
                    p_mask = batch["param_mask"].numpy().astype(bool)
                    raw_area_radius = batch["raw_area_radius"].numpy()
                    valid_np = batch["valid_query_mask"].numpy().astype(bool)
                    true_patch_np = batch["patch_query"].numpy()
                    true_apd90_np = batch["y_field"].numpy()[:, :, 1] * stats.field_std[2] + stats.field_mean[2]
                    pred_apd90_np = query_np[:, :, 1] * stats.field_std[2] + stats.field_mean[2]
                    patch_prob_np = sigmoid_np(query_np[:, :, 2])
                    center_prob_np = sigmoid_np(query_np[:, :, 3])
                    present_prob = sigmoid_np(params_np[:, 0])
                    for b in range(query_np.shape[0]):
                        src = sub.iloc[offset + b]
                        valid = valid_np[b].reshape(32, 32)
                        true_patch = true_patch_np[b].reshape(32, 32)
                        patch_prob = patch_prob_np[b].reshape(32, 32)
                        center_prob = center_prob_np[b].reshape(32, 32)
                        prob_cx, prob_cy = centroid(patch_prob, valid, weighted=True)
                        heat_cx, heat_cy = peak_center(center_prob, valid)
                        true_cx = y_param[b, 1] * stats.param_std[1] + stats.param_mean[1] if p_mask[b, 1] else math.nan
                        true_cy = y_param[b, 2] * stats.param_std[2] + stats.param_mean[2] if p_mask[b, 2] else math.nan
                        rec = {
                            "model": model_name,
                            "split": split,
                            "row_id": src["row_id"],
                            "base_sample_id": src["base_sample_id"],
                            "condition": src["condition"],
                            "combined_variant": src["combined_variant"],
                            "true_patch_present": float(y_param[b, 0]),
                            "pred_patch_present_prob": float(present_prob[b]),
                            "true_center_x_norm": true_cx,
                            "pred_center_x_norm": float(p_unscaled[b, 1]),
                            "true_center_y_norm": true_cy,
                            "pred_center_y_norm": float(p_unscaled[b, 2]),
                            "center_heatmap_peak_x_norm": heat_cx,
                            "center_heatmap_peak_y_norm": heat_cy,
                            "probability_centroid_x_norm": prob_cx,
                            "probability_centroid_y_norm": prob_cy,
                            "param_head_center_error": center_error(float(p_unscaled[b, 1]), float(p_unscaled[b, 2]), true_cx, true_cy),
                            "center_heatmap_peak_error": center_error(heat_cx, heat_cy, true_cx, true_cy),
                            "probability_centroid_error": center_error(prob_cx, prob_cy, true_cx, true_cy),
                            "true_patch_area_fraction": float(raw_area_radius[b, 0]) if np.isfinite(raw_area_radius[b, 0]) else math.nan,
                            "pred_patch_area_param": float(math.exp(p_unscaled[b, 3])) if np.isfinite(p_unscaled[b, 3]) else math.nan,
                            "true_radius_param": float(raw_area_radius[b, 1]) if np.isfinite(raw_area_radius[b, 1]) else math.nan,
                            "pred_radius_param": float(math.exp(p_unscaled[b, 4])) if np.isfinite(p_unscaled[b, 4]) else math.nan,
                            "true_log_area_fraction": float(y_param[b, 3] * stats.param_std[3] + stats.param_mean[3]) if p_mask[b, 3] else math.nan,
                            "pred_log_area_fraction": float(p_unscaled[b, 3]),
                            "true_log_radius": float(y_param[b, 4] * stats.param_std[4] + stats.param_mean[4]) if p_mask[b, 4] else math.nan,
                            "pred_log_radius": float(p_unscaled[b, 4]),
                            "true_area_bin": int(batch["area_bin"].numpy()[b]) if bool(batch["area_bin_mask"].numpy()[b]) else -1,
                            "pred_area_bin": int(np.argmax(area_np[b])),
                            "true_APD90_patch_delta": float(batch["apd90_patch_delta"].numpy()[b]),
                            "array_index": len(arrays["true_patch"]),
                        }
                        records.append(rec)
                        arrays["true_patch"].append(true_patch.astype(np.float32))
                        arrays["valid"].append(valid.astype(bool))
                        arrays["patch_prob"].append(patch_prob.astype(np.float32))
                        arrays["center_prob"].append(center_prob.astype(np.float32))
                        arrays["pred_apd90"].append(pred_apd90_np[b].reshape(32, 32).astype(np.float32))
                        arrays["true_apd90"].append(true_apd90_np[b].reshape(32, 32).astype(np.float32))
                    offset += query_np.shape[0]
        print(f"evaluated full-grid refinement predictions for {model_name}", flush=True)
    index = pd.DataFrame(records)
    maps = {k: np.stack(v) for k, v in arrays.items()}
    return index, maps


def mask_for_method(row, maps, method: str, threshold: float | None = None):
    idx = int(row["array_index"])
    valid = maps["valid"][idx]
    prob = maps["patch_prob"][idx]
    present = float(row["pred_patch_present_prob"])
    if method in {"uncalibrated_threshold_0.5", "calibrated_sparse_threshold", "calibrated_balanced_threshold"}:
        thr = 0.5 if threshold is None else float(threshold)
        return ((prob >= thr) & valid).astype(np.float32)
    if method == "center_heatmap_peak_radius_disk":
        return disk_mask(float(row["center_heatmap_peak_x_norm"]), float(row["center_heatmap_peak_y_norm"]), float(row["pred_radius_param"]), present, prob.shape)
    if method == "probability_centroid_radius_disk":
        return disk_mask(float(row["probability_centroid_x_norm"]), float(row["probability_centroid_y_norm"]), float(row["pred_radius_param"]), present, prob.shape)
    if method == "true_mask_upper_bound":
        return maps["true_patch"][idx].astype(np.float32)
    raise ValueError(f"unknown mask method: {method}")


def evaluate_mask_method(index: pd.DataFrame, maps: dict[str, np.ndarray], model: str, split: str, method: str, threshold: float | None = None) -> tuple[dict, pd.DataFrame]:
    sub = index[index["model"].eq(model) & index["split"].eq(split)].copy()
    rows = []
    all_true = []
    all_pred = []
    all_valid = []
    for _, row in sub.iterrows():
        idx = int(row["array_index"])
        valid = maps["valid"][idx]
        true_patch = maps["true_patch"][idx]
        pred_mask = mask_for_method(row, maps, method, threshold)
        true_delta = scalar_targets(maps["true_apd90"][idx], true_patch, valid)
        pred_delta = scalar_targets(maps["pred_apd90"][idx], pred_mask, valid)
        true_mask_delta = scalar_targets(maps["pred_apd90"][idx], true_patch, valid)
        pm = patch_metrics(true_patch, pred_mask, valid)
        pred_area = area_fraction(pred_mask, valid)
        rows.append(
            {
                "model": model,
                "split": split,
                "method": method,
                "threshold": threshold,
                "row_id": row["row_id"],
                "condition": row["condition"],
                "combined_variant": row["combined_variant"],
                "true_patch_present": row["true_patch_present"],
                "dice": pm["dice"],
                "iou": pm["iou"],
                "precision": pm["precision"],
                "recall": pm["recall"],
                "true_area": area_fraction(true_patch, valid),
                "pred_area": pred_area,
                "true_delta": true_delta,
                "pred_delta": pred_delta,
                "true_mask_delta": true_mask_delta,
                "homogeneous_false_positive_area": pred_area if float(row["true_patch_present"]) < 0.5 else math.nan,
            }
        )
        all_true.append(true_patch)
        all_pred.append(pred_mask)
        all_valid.append(valid)
    df = pd.DataFrame(rows)
    micro = patch_metrics(np.stack(all_true), np.stack(all_pred), np.stack(all_valid).astype(bool)) if rows else {"dice": math.nan, "iou": math.nan, "precision": math.nan, "recall": math.nan}
    non = df[df["true_patch_present"].eq(1)]
    rec = {
        "model": model,
        "split": split,
        "target": "patch_mask",
        "method": method,
        "metric_scope": "valid_grid_pixels",
        "threshold": threshold,
        "n": int(len(df)),
        "dice": micro["dice"],
        "row_mean_dice": float(df["dice"].mean(skipna=True)),
        "iou": micro["iou"],
        "precision": micro["precision"],
        "recall": micro["recall"],
        "homogeneous_false_positive_area": float(df["homogeneous_false_positive_area"].mean(skipna=True)),
        "patch_area_r2": regression_metrics(non["true_area"], non["pred_area"])["r2"] if len(non) else math.nan,
        "APD90_patch_delta_R2_pred_mask_pred_apd": regression_metrics(non["true_delta"], non["pred_delta"])["r2"] if len(non) else math.nan,
        "APD90_patch_delta_MAE_pred_mask_pred_apd": regression_metrics(non["true_delta"], non["pred_delta"])["mae"] if len(non) else math.nan,
        "APD90_patch_delta_R2_true_mask_pred_apd": regression_metrics(non["true_delta"], non["true_mask_delta"])["r2"] if len(non) else math.nan,
    }
    return rec, df


def select_thresholds(index, maps) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    records = []
    settings: dict[str, dict[str, float]] = {"sparse": {}, "balanced": {}}
    for model in sorted(index["model"].unique()):
        candidates = []
        for thr in THRESHOLDS:
            rec, _ = evaluate_mask_method(index, maps, model, "val", "calibrated_sparse_threshold", float(thr))
            rec["candidate_scope"] = "threshold_sweep"
            rec["selection_score_sparse"] = (rec["APD90_patch_delta_R2_pred_mask_pred_apd"] if np.isfinite(rec["APD90_patch_delta_R2_pred_mask_pred_apd"]) else -1.0) - 0.6 * rec["homogeneous_false_positive_area"]
            rec["selection_score_balanced"] = (rec["APD90_patch_delta_R2_pred_mask_pred_apd"] if np.isfinite(rec["APD90_patch_delta_R2_pred_mask_pred_apd"]) else -1.0) + 0.35 * rec["dice"] - 0.8 * rec["homogeneous_false_positive_area"]
            candidates.append(rec)
            records.append(rec)
        cdf = pd.DataFrame(candidates)
        sparse = cdf.sort_values("selection_score_sparse", ascending=False).iloc[0]
        balanced_pool = cdf[(cdf["dice"].ge(0.25)) & (cdf["homogeneous_false_positive_area"].le(0.05))]
        if balanced_pool.empty:
            balanced_pool = cdf
        balanced = balanced_pool.sort_values("selection_score_balanced", ascending=False).iloc[0]
        settings["sparse"][model] = float(sparse["threshold"])
        settings["balanced"][model] = float(balanced["threshold"])
        records.append({**sparse.to_dict(), "candidate_scope": "selected_sparse"})
        records.append({**balanced.to_dict(), "candidate_scope": "selected_balanced"})
    return pd.DataFrame(records), settings


def summarize_parametrics(index: pd.DataFrame, split: str) -> list[dict]:
    records = []
    sub_all = index[index["split"].eq(split)].copy()
    for model, sub in sub_all.groupby("model"):
        non = sub[sub["true_patch_present"].eq(1)]
        if len(non):
            center = regression_metrics(non["true_center_x_norm"], non["pred_center_x_norm"])
            cy = regression_metrics(non["true_center_y_norm"], non["pred_center_y_norm"])
            center_err = non["param_head_center_error"].to_numpy(float)
            heat_err = non["center_heatmap_peak_error"].to_numpy(float)
            prob_err = non["probability_centroid_error"].to_numpy(float)
            records.extend(
                [
                    {"model": model, "split": split, "target": "patch_center_error", "method": "param_head", "metric_scope": "parametric", "n": int(len(non)), "mae": float(np.nanmean(center_err)), "rmse": float(np.sqrt(np.nanmean(center_err * center_err)))},
                    {"model": model, "split": split, "target": "patch_center_error", "method": "center_heatmap_peak", "metric_scope": "query_center", "n": int(len(non)), "mae": float(np.nanmean(heat_err)), "rmse": float(np.sqrt(np.nanmean(heat_err * heat_err)))},
                    {"model": model, "split": split, "target": "patch_center_error", "method": "probability_centroid", "metric_scope": "query_center", "n": int(len(non)), "mae": float(np.nanmean(prob_err)), "rmse": float(np.sqrt(np.nanmean(prob_err * prob_err)))},
                    {"model": model, "split": split, "target": "patch_area_param_head", "method": "param_head", "metric_scope": "parametric", "n": int(len(non)), **regression_metrics(non["true_patch_area_fraction"], non["pred_patch_area_param"])},
                    {"model": model, "split": split, "target": "radius_param_head", "method": "param_head", "metric_scope": "parametric", "n": int(len(non)), **regression_metrics(non["true_radius_param"], non["pred_radius_param"])},
                    {"model": model, "split": split, "target": "log_area_param_head", "method": "param_head", "metric_scope": "parametric", "n": int(len(non)), **regression_metrics(non["true_log_area_fraction"], non["pred_log_area_fraction"])},
                    {"model": model, "split": split, "target": "log_radius_param_head", "method": "param_head", "metric_scope": "parametric", "n": int(len(non)), **regression_metrics(non["true_log_radius"], non["pred_log_radius"])},
                ]
            )
    return records


def condition_variant_rows(row_metrics: pd.DataFrame) -> list[dict]:
    records = []
    for group_col, scope in [("condition", "condition"), ("combined_variant", "combined_variant")]:
        for (model, method, group), sub in row_metrics.groupby(["model", "method", group_col]):
            non = sub[sub["true_patch_present"].eq(1)]
            records.append(
                {
                    "model": model,
                    "split": "test",
                    "target": "patch_mask",
                    "method": method,
                    "metric_scope": scope,
                    group_col: group,
                    "n": int(len(sub)),
                    "row_mean_dice": float(sub["dice"].mean(skipna=True)),
                    "row_mean_precision": float(sub["precision"].mean(skipna=True)),
                    "row_mean_recall": float(sub["recall"].mean(skipna=True)),
                    "homogeneous_false_positive_area": float(sub["homogeneous_false_positive_area"].mean(skipna=True)),
                    "APD90_patch_delta_R2_pred_mask_pred_apd": regression_metrics(non["true_delta"], non["pred_delta"])["r2"] if len(non) else math.nan,
                }
            )
    return records


def make_figures(results: pd.DataFrame, predictions: pd.DataFrame, row_metrics: pd.DataFrame, index: pd.DataFrame, maps: dict, settings: dict) -> None:
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    test_main = results[(results["split"].eq("test")) & (results["metric_scope"].eq("valid_grid_pixels"))]
    best_model = str(test_main[test_main["method"].eq("calibrated_balanced_threshold")].sort_values("dice", ascending=False).iloc[0]["model"])

    non = row_metrics[(row_metrics["model"].eq(best_model)) & (row_metrics["method"].eq("calibrated_balanced_threshold")) & (row_metrics["true_patch_present"].eq(1))].copy()
    selected = pd.concat([non.sort_values("dice", ascending=False).head(2), non.sort_values("dice", ascending=True).head(2)], ignore_index=True)
    if len(selected):
        fig, axes = plt.subplots(len(selected), 5, figsize=(14, 3 * len(selected)))
        axes = np.atleast_2d(axes)
        for axrow, row in zip(axes, selected.itertuples(index=False)):
            pred_row = index[(index["model"].eq(row.model)) & (index["split"].eq("test")) & (index["row_id"].eq(row.row_id))].iloc[0]
            idx = int(pred_row["array_index"])
            valid = maps["valid"][idx]
            thr = settings["balanced"][best_model]
            mask = mask_for_method(pred_row, maps, "calibrated_balanced_threshold", thr)
            apd_err = np.where(valid, maps["pred_apd90"][idx] - maps["true_apd90"][idx], np.nan)
            items = [
                (maps["true_patch"][idx], "true patch", "magma", 0, 1),
                (maps["patch_prob"][idx], "patch prob", "magma", 0, 1),
                (maps["center_prob"][idx], "center heat", "magma", 0, 1),
                (mask, "balanced mask", "magma", 0, 1),
                (apd_err, "APD90 error", "coolwarm", None, None),
            ]
            for ax, (arr, title, cmap, vmin, vmax) in zip(axrow, items):
                im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f"{title}\nDice {row.dice:.3f}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.tight_layout()
        fig.savefig(NEURAL_FIG_DIR / "mask_refine_good_bad_examples.png", dpi=180)
        plt.close(fig)

    center = results[(results["split"].eq("test")) & (results["target"].eq("patch_center_error"))].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = center["model"] + "\n" + center["method"]
    ax.bar(range(len(center)), center["mae"])
    ax.axhline(BASELINE_CENTER_MAE, color="black", linestyle="--", linewidth=1, label="prior best 0.120")
    ax.set_xticks(range(len(center)))
    ax.set_xticklabels(labels, rotation=80, fontsize=6)
    ax.set_ylabel("center MAE")
    ax.set_title("Mask Refinement Center Error")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "mask_refine_center_error.png", dpi=180)
    plt.close(fig)

    cond = results[(results["split"].eq("test")) & (results["metric_scope"].eq("condition")) & (results["method"].eq("calibrated_balanced_threshold"))].copy()
    fig, ax = plt.subplots(figsize=(10, 4))
    for model, sub in cond.groupby("model"):
        ax.plot(sub["condition"], sub["row_mean_dice"], marker="o", label=model)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.set_ylabel("row mean Dice")
    ax.set_title("Mask Refinement Condition Breakdown")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "mask_refine_condition_breakdown.png", dpi=180)
    plt.close(fig)

    delta = test_main[test_main["method"].isin(["uncalibrated_threshold_0.5", "calibrated_sparse_threshold", "calibrated_balanced_threshold", "center_heatmap_peak_radius_disk", "probability_centroid_radius_disk", "true_mask_upper_bound"])]
    fig, ax = plt.subplots(figsize=(max(9, len(delta) * 0.35), 4.5))
    labels = delta["model"] + "\n" + delta["method"]
    ax.bar(range(len(delta)), delta["APD90_patch_delta_R2_pred_mask_pred_apd"])
    ax.axhline(0.335, color="gray", linestyle="--", linewidth=1, label="prior uncalibrated 0.335")
    ax.axhline(BASELINE_BALANCED_DELTA_R2, color="black", linestyle="--", linewidth=1, label="prior balanced 0.600")
    ax.set_xticks(range(len(delta)))
    ax.set_xticklabels(labels, rotation=82, fontsize=6)
    ax.set_ylabel("APD90 delta R2, predicted APD90")
    ax.set_title("Mask Refinement Delta Recovery")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "mask_refine_delta_recovery.png", dpi=180)
    plt.close(fig)

    comp = test_main[test_main["method"].isin(["calibrated_sparse_threshold", "calibrated_balanced_threshold"])]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(comp))
    ax.bar(x - 0.18, comp["dice"], width=0.36, label="Dice")
    ax.bar(x + 0.18, comp["APD90_patch_delta_R2_pred_mask_pred_apd"], width=0.36, label="delta R2")
    ax.set_xticks(x)
    ax.set_xticklabels(comp["model"] + "\n" + comp["method"], rotation=80, fontsize=6)
    ax.set_title("Sparse vs Balanced Mask Calibration")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "mask_refine_sparse_vs_balanced_masks.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-norm", choices=["per_sample", "train_global"], default="per_sample")
    args = parser.parse_args()
    require_cuda(torch)
    rows = merge_rows_with_refine_targets()
    desc = load_descriptors()
    stats = compute_stats(rows, desc)
    device = torch.device("cuda")
    if not checkpoint_paths():
        raise SystemExit("No mask-refinement checkpoints found")

    index, maps = collect_predictions(rows, desc, stats, args, device)
    calibration_sweep, settings = select_thresholds(index, maps)
    results = []
    row_metric_frames = []
    for model in sorted(index["model"].unique()):
        for split in ["val", "test"]:
            methods = [
                ("uncalibrated_threshold_0.5", None),
                ("calibrated_sparse_threshold", settings["sparse"][model]),
                ("calibrated_balanced_threshold", settings["balanced"][model]),
                ("center_heatmap_peak_radius_disk", None),
                ("probability_centroid_radius_disk", None),
                ("true_mask_upper_bound", None),
            ]
            for method, threshold in methods:
                rec, row_df = evaluate_mask_method(index, maps, model, split, method, threshold)
                results.append(rec)
                if split == "test":
                    row_metric_frames.append(row_df)
    results.extend(summarize_parametrics(index, "test"))
    test_rows_for_groups = pd.concat(row_metric_frames, ignore_index=True)
    results.extend(condition_variant_rows(test_rows_for_groups))
    results_df = pd.DataFrame(results)

    predictions = index[index["split"].eq("test")].copy()
    for method in [
        "uncalibrated_threshold_0.5",
        "calibrated_sparse_threshold",
        "calibrated_balanced_threshold",
        "center_heatmap_peak_radius_disk",
        "probability_centroid_radius_disk",
        "true_mask_upper_bound",
    ]:
        method_rows = []
        for model in sorted(predictions["model"].unique()):
            threshold = None
            if method == "calibrated_sparse_threshold":
                threshold = settings["sparse"][model]
            elif method == "calibrated_balanced_threshold":
                threshold = settings["balanced"][model]
            _rec, row_df = evaluate_mask_method(index, maps, model, "test", method, threshold)
            slim = row_df[["row_id", "model", "pred_delta", "pred_area", "dice"]].rename(
                columns={
                    "pred_delta": f"{method}_APD90_patch_delta_pred_apd",
                    "pred_area": f"{method}_patch_area_fraction",
                    "dice": f"{method}_row_dice",
                }
            )
            method_rows.append(slim)
        predictions = predictions.merge(pd.concat(method_rows, ignore_index=True), on=["model", "row_id"], how="left")

    calibration_test_rows = []
    for model in sorted(index["model"].unique()):
        for label, method, threshold in [
            ("selected_sparse_test", "calibrated_sparse_threshold", settings["sparse"][model]),
            ("selected_balanced_test", "calibrated_balanced_threshold", settings["balanced"][model]),
        ]:
            rec, _ = evaluate_mask_method(index, maps, model, "test", method, threshold)
            rec["candidate_scope"] = label
            calibration_test_rows.append(rec)
    calibration_df = pd.concat([calibration_sweep, pd.DataFrame(calibration_test_rows)], ignore_index=True, sort=False)

    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_CSV, index=False)
    predictions.drop(columns=["array_index"]).to_csv(PREDICTIONS_CSV, index=False)
    calibration_df.to_csv(CALIBRATION_CSV, index=False)
    make_figures(results_df, predictions, test_rows_for_groups, index, maps, settings)

    test = results_df[results_df["split"].eq("test")].copy()
    test_global = test[test["metric_scope"].eq("valid_grid_pixels")].copy()
    bests = {
        "best_balanced_dice": test_global[test_global["method"].eq("calibrated_balanced_threshold")].sort_values("dice", ascending=False).head(1).replace({np.nan: None}).to_dict(orient="records"),
        "best_sparse_delta_r2": test_global[test_global["method"].eq("calibrated_sparse_threshold")].sort_values("APD90_patch_delta_R2_pred_mask_pred_apd", ascending=False).head(1).replace({np.nan: None}).to_dict(orient="records"),
        "best_balanced_delta_r2": test_global[test_global["method"].eq("calibrated_balanced_threshold")].sort_values("APD90_patch_delta_R2_pred_mask_pred_apd", ascending=False).head(1).replace({np.nan: None}).to_dict(orient="records"),
        "best_center_mae": test[test["target"].eq("patch_center_error")].sort_values("mae", ascending=True).head(1).replace({np.nan: None}).to_dict(orient="records"),
    }
    payload = {
        "results_csv": str(RESULTS_CSV),
        "predictions_csv": str(PREDICTIONS_CSV),
        "calibration_csv": str(CALIBRATION_CSV),
        "threshold_settings": settings,
        "bests": bests,
        "baselines": {
            "plain_deeponet_patch_dice": BASELINE_DICE,
            "prior_balanced_delta_r2": BASELINE_BALANCED_DELTA_R2,
            "prior_sparse_delta_r2": BASELINE_SPARSE_DELTA_R2,
            "plain_deeponet_center_mae": BASELINE_CENTER_MAE,
        },
        "figures": [
            "figures/neural_v1e_500/mask_refine_good_bad_examples.png",
            "figures/neural_v1e_500/mask_refine_center_error.png",
            "figures/neural_v1e_500/mask_refine_condition_breakdown.png",
            "figures/neural_v1e_500/mask_refine_delta_recovery.png",
            "figures/neural_v1e_500/mask_refine_sparse_vs_balanced_masks.png",
        ],
        "calibration_guardrail": "Thresholds are selected on validation rows only, then applied once to grouped-test rows.",
        "leakage_guardrail": "Evaluation uses target maps and patch parameters only as labels. Model inputs remain B_meas, optional B_meas-derived descriptors, and query coordinates.",
    }
    write_json(RESULTS_JSON, payload)
    print(json.dumps({"results": str(RESULTS_CSV), "predictions": str(PREDICTIONS_CSV), "bests": bests}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
