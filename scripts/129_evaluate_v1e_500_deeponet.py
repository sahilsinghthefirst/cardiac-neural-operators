                      
                                                                 

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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from v1e_500_common import CHECKPOINT_DIR, NEURAL_DIR, NEURAL_FIG_DIR, SEED, require_cuda, write_json
from v1e_deeponet_data_common import compute_stats, load_descriptors, load_index, make_loader, torch


TRAIN_SPEC = importlib.util.spec_from_file_location("_deeponet_train", PROJECT_ROOT / "scripts" / "127_train_v1e_500_deeponet_field_parametric.py")
if TRAIN_SPEC is None or TRAIN_SPEC.loader is None:
    raise ImportError("Could not load DeepONet training module")
DEEPO_TRAIN = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(DEEPO_TRAIN)

DESC_SPEC = importlib.util.spec_from_file_location("_descriptor_deeponet_train", PROJECT_ROOT / "scripts" / "128_train_v1e_500_mlp_descriptor_deeponet_baseline.py")
if DESC_SPEC is None or DESC_SPEC.loader is None:
    raise ImportError("Could not load descriptor DeepONet module")
DESC_TRAIN = importlib.util.module_from_spec(DESC_SPEC)
DESC_SPEC.loader.exec_module(DESC_TRAIN)


OUT_RESULTS = NEURAL_DIR / "deeponet_results.csv"
OUT_JSON = NEURAL_DIR / "deeponet_results.json"
OUT_PREDS = NEURAL_DIR / "deeponet_predictions.csv"

BASELINES = {
    "v1e_500_parametric_cnn_encoder": {
        "center_mae": 0.1376507316959553,
        "parametric_mask_dice": 0.4376892189670998,
        "APD90_delta_param_mask_true_apd_R2": 0.7037444153702479,
        "log_area_R2": 0.2292131839370053,
        "log_radius_R2": 0.2225541430695877,
    },
    "v1e_500_parametric_fno_encoder": {
        "patch_present_macro_f1": 0.8931319464234825,
        "center_mae": 0.1432190519645683,
    },
    "v1e_500_tabular_rf_descriptors": {
        "patch_present_macro_f1": 0.976,
        "patch_area_R2": 0.2622975120286949,
        "radius_R2": 0.2464200928100517,
        "area_bin_macro_f1": 0.4731462572170536,
    },
    "v1e_500_field_parametric_fno": {
        "LAT_R2": 0.9520664369301024,
        "APD90_R2": 0.9589286421515965,
        "patch_mask_dice": 0.5812682985723341,
        "patch_area_R2": 0.2892978631637591,
        "radius_R2": 0.2740862334116253,
    },
    "v1e_500_field_parametric_cnn": {
        "LAT_R2": 0.9402894230797328,
        "APD90_R2": 0.9651141087892924,
        "APD90_delta_true_mask_pred_apd_R2": 0.784,
        "APD90_delta_param_mask_true_apd_R2": 0.6099276920228602,
    },
    "v1d_fno_field_baseline": {
        "LAT_R2": 0.976,
        "APD90_R2": 0.981,
        "patch_mask_dice": 0.449,
        "APD90_delta_pixel_mask_pred_apd_R2": 0.078,
        "patch_area_pixel_mask_R2": -0.120,
    },
}


def regression_metrics(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not np.any(mask):
        return {"mae": math.nan, "rmse": math.nan, "r2": math.nan}
    out = {
        "mae": float(mean_absolute_error(y[mask], p[mask])),
        "rmse": float(np.sqrt(mean_squared_error(y[mask], p[mask]))),
    }
    if mask.sum() >= 2 and not np.allclose(y[mask], y[mask][0]):
        out["r2"] = float(r2_score(y[mask], p[mask]))
    else:
        out["r2"] = math.nan
    return out


def sigmoid_np(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(arr, dtype=np.float64)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    exp_x = np.exp(arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    if np.isscalar(x):
        return float(out)
    return out


def patch_metrics(true, pred, valid) -> dict:
    t = (true > 0.5) & valid
    p = (pred > 0.5) & valid
    tp = float(np.sum(t & p))
    fp = float(np.sum(~t & p & valid))
    fn = float(np.sum(t & ~p))
    return {
        "dice": (2.0 * tp) / max(2.0 * tp + fp + fn, 1.0),
        "iou": tp / max(tp + fp + fn, 1.0),
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
    }


def param_mask(cx, cy, radius, present, shape=(32, 32)) -> np.ndarray:
    if present < 0.5 or not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(radius):
        return np.zeros(shape, dtype=np.float32)
    yy, xx = np.indices(shape)
    x = xx / max(shape[1] - 1, 1)
    y = yy / max(shape[0] - 1, 1)
    radius = float(np.clip(radius, 1e-4, 0.75))
    return (((x - np.clip(cx, 0, 1)) ** 2 + (y - np.clip(cy, 0, 1)) ** 2) <= radius**2).astype(np.float32)


def scalar_targets(lat: np.ndarray, apd90: np.ndarray, patch: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    v = valid.astype(bool)
    lat_vals = lat[v & np.isfinite(lat)]
    out = {
        "LAT_range": float(np.nanmax(lat_vals) - np.nanmin(lat_vals)) if lat_vals.size else math.nan,
        "patch_area_fraction": float(np.mean((patch > 0.5)[v])) if np.any(v) else math.nan,
    }
    inside = (patch > 0.5) & v & np.isfinite(apd90)
    outside = (patch <= 0.5) & v & np.isfinite(apd90)
    out["APD90_patch_delta"] = float(np.mean(apd90[inside]) - np.mean(apd90[outside])) if inside.any() and outside.any() else math.nan
    return out


def checkpoint_paths() -> list[Path]:
    paths = sorted(CHECKPOINT_DIR.glob("deeponet_*.pt"))
    paths.extend(sorted(CHECKPOINT_DIR.glob("descriptor_deeponet_*.pt")))
    return paths


def build_model_from_checkpoint(ckpt: dict, desc_dim: int):
    config = ckpt.get("config", {})
    if config.get("descriptor_only"):
        model = DESC_TRAIN.DescriptorDeepONetFieldParametric(
            int(ckpt.get("descriptor_dim", desc_dim)),
            basis_dim=int(ckpt["basis_dim"]),
            branch_latent=int(ckpt["branch_latent"]),
            trunk_hidden=int(ckpt["trunk_hidden"]),
            dropout=float(ckpt["dropout"]),
            fourier_trunk=bool(config.get("fourier", True)),
        )
    else:
        model = DEEPO_TRAIN.DeepONetFieldParametric(
            int(ckpt["branch_in_channels"]),
            int(ckpt.get("descriptor_dim", desc_dim)),
            basis_dim=int(ckpt["basis_dim"]),
            branch_latent=int(ckpt["branch_latent"]),
            trunk_hidden=int(ckpt["trunk_hidden"]),
            dropout=float(ckpt["dropout"]),
            use_descriptors=bool(config.get("descriptors", False)),
            fourier_trunk=bool(config.get("fourier", False)),
        )
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def evaluate_model(model, model_name: str, rows, desc, stats, batch_size: int, device):
    test_rows = rows[rows["split"].eq("test")].reset_index(drop=True)
    loader = make_loader(
        test_rows,
        stats,
        batch_size=batch_size,
        shuffle=False,
        mode="full",
        k_queries=1024,
        input_norm="per_sample",
        descriptors=desc,
        seed=SEED + 707,
    )
    model.eval()
    field_true = {k: [] for k in ["LAT", "APD80", "APD90"]}
    field_pred = {k: [] for k in ["LAT", "APD80", "APD90"]}
    patch_true_all = []
    patch_prob_all = []
    valid_all = []
    pred_rows = []
    examples = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            query, params, area_logits, cond_logits = model(
                batch["branch_x"].to(device).float(),
                batch["trunk_xy"].to(device).float(),
                batch["descriptor_x"].to(device).float(),
            )
            query_np = query.detach().cpu().numpy()
            params_np = params.detach().cpu().numpy()
            area_np = area_logits.detach().cpu().numpy()
            cond_np = cond_logits.detach().cpu().numpy()
            y_np = batch["y_query"].numpy()
            valid_np = batch["valid_query_mask"].numpy().astype(bool)
            patch_prob = sigmoid_np(query_np[:, :, 3])
            for i, key in enumerate(["LAT", "APD80", "APD90"]):
                field_true[key].append(y_np[:, :, i] * stats.field_std[i] + stats.field_mean[i])
                field_pred[key].append(query_np[:, :, i] * stats.field_std[i] + stats.field_mean[i])
            patch_true_all.append(y_np[:, :, 3])
            patch_prob_all.append(patch_prob)
            valid_all.append(valid_np)
            for b in range(query_np.shape[0]):
                src = test_rows.iloc[offset + b]
                true_lat = y_np[b, :, 0].reshape(32, 32) * stats.field_std[0] + stats.field_mean[0]
                true_apd90 = y_np[b, :, 2].reshape(32, 32) * stats.field_std[2] + stats.field_mean[2]
                true_patch = y_np[b, :, 3].reshape(32, 32)
                valid = valid_np[b].reshape(32, 32)
                pred_lat = query_np[b, :, 0].reshape(32, 32) * stats.field_std[0] + stats.field_mean[0]
                pred_apd90 = query_np[b, :, 2].reshape(32, 32) * stats.field_std[2] + stats.field_mean[2]
                pred_patch_prob = patch_prob[b].reshape(32, 32)
                pred_patch_bin = (pred_patch_prob > 0.5).astype(np.float32)
                param_unscaled = params_np[b].copy()
                param_unscaled[1:] = param_unscaled[1:] * stats.param_std[1:] + stats.param_mean[1:]
                present_prob = float(sigmoid_np(params_np[b, 0]))
                pred_log_area = float(param_unscaled[3])
                pred_log_radius = float(param_unscaled[4])
                pred_area = float(np.exp(pred_log_area)) if np.isfinite(pred_log_area) else math.nan
                pred_radius = float(np.exp(pred_log_radius)) if np.isfinite(pred_log_radius) else math.nan
                p_mask = param_mask(float(param_unscaled[1]), float(param_unscaled[2]), pred_radius, present_prob)
                y_param = batch["y_param"].numpy()[b]
                p_mask_tensor = batch["param_mask"].numpy()[b].astype(bool)
                true_center_x = y_param[1] * stats.param_std[1] + stats.param_mean[1] if p_mask_tensor[1] else math.nan
                true_center_y = y_param[2] * stats.param_std[2] + stats.param_mean[2] if p_mask_tensor[2] else math.nan
                true_log_area = y_param[3] * stats.param_std[3] + stats.param_mean[3] if p_mask_tensor[3] else math.nan
                true_log_radius = y_param[4] * stats.param_std[4] + stats.param_mean[4] if p_mask_tensor[4] else math.nan
                raw_area_radius = batch["raw_area_radius"].numpy()[b]
                true_scalars = scalar_targets(true_lat, true_apd90, true_patch, valid)
                pixel_pred_scalars = scalar_targets(pred_lat, pred_apd90, pred_patch_bin, valid)
                param_pred_scalars = scalar_targets(pred_lat, pred_apd90, p_mask, valid)
                param_true_apd = scalar_targets(true_lat, true_apd90, p_mask, valid)
                true_mask_pred_apd = scalar_targets(pred_lat, pred_apd90, true_patch, valid)
                row = {
                    "model": model_name,
                    "row_id": src["row_id"],
                    "base_sample_id": src["base_sample_id"],
                    "split": "test",
                    "condition": src["condition"],
                    "combined_variant": src["combined_variant"],
                    "true_patch_present": float(y_param[0]),
                    "pred_patch_present_prob": present_prob,
                    "true_center_x_norm": true_center_x,
                    "pred_center_x_norm": float(param_unscaled[1]),
                    "true_center_y_norm": true_center_y,
                    "pred_center_y_norm": float(param_unscaled[2]),
                    "true_patch_area_param": float(raw_area_radius[0]) if np.isfinite(raw_area_radius[0]) else math.nan,
                    "pred_patch_area_param": pred_area,
                    "true_radius_param": float(raw_area_radius[1]) if np.isfinite(raw_area_radius[1]) else math.nan,
                    "pred_radius_param": pred_radius,
                    "true_log_area_fraction": true_log_area,
                    "pred_log_area_fraction": pred_log_area,
                    "true_log_radius": true_log_radius,
                    "pred_log_radius": pred_log_radius,
                    "true_area_bin": int(batch["area_bin"].numpy()[b]) if bool(batch["area_bin_mask"].numpy()[b]) else -1,
                    "pred_area_bin": int(np.argmax(area_np[b])),
                    "pred_condition_label": int(np.argmax(cond_np[b])),
                    "true_LAT_range": true_scalars["LAT_range"],
                    "pred_LAT_range": pixel_pred_scalars["LAT_range"],
                    "true_APD90_patch_delta": true_scalars["APD90_patch_delta"],
                    "pred_APD90_patch_delta_pixel_mask_pred_apd": pixel_pred_scalars["APD90_patch_delta"],
                    "pred_APD90_patch_delta_param_mask_pred_apd": param_pred_scalars["APD90_patch_delta"],
                    "pred_APD90_patch_delta_param_mask_true_apd": param_true_apd["APD90_patch_delta"],
                    "pred_APD90_patch_delta_true_mask_pred_apd": true_mask_pred_apd["APD90_patch_delta"],
                    "true_patch_area_fraction": true_scalars["patch_area_fraction"],
                    "pred_patch_area_fraction_pixel_mask": pixel_pred_scalars["patch_area_fraction"],
                    "pred_patch_area_fraction_param_mask": param_pred_scalars["patch_area_fraction"],
                    "homogeneous_false_positive_area_pixel_mask": pixel_pred_scalars["patch_area_fraction"] if float(y_param[0]) < 0.5 else math.nan,
                    "homogeneous_false_positive_area_param_mask": param_pred_scalars["patch_area_fraction"] if float(y_param[0]) < 0.5 else math.nan,
                }
                pred_rows.append(row)
                if len(examples) < 4 and model_name:
                    examples.append(
                        {
                            "row": row,
                            "true_lat": true_lat,
                            "pred_lat": pred_lat,
                            "true_apd90": true_apd90,
                            "pred_apd90": pred_apd90,
                            "true_patch": true_patch,
                            "pred_patch_prob": pred_patch_prob,
                            "pred_patch_bin": pred_patch_bin,
                            "param_mask": p_mask,
                            "valid": valid,
                        }
                    )
            offset += query_np.shape[0]
    results = []
    valid_cat = np.concatenate(valid_all, axis=0).astype(bool)
    patch_true_cat = np.concatenate(patch_true_all, axis=0)
    patch_prob_cat = np.concatenate(patch_prob_all, axis=0)
    results.append({"model": model_name, "split": "test", "target": "patch_mask", "metric_scope": "valid_grid_pixels", "n": int(np.sum(valid_cat)), **patch_metrics(patch_true_cat, patch_prob_cat, valid_cat)})
    for key in ["LAT", "APD80", "APD90"]:
        true = np.concatenate(field_true[key], axis=0)[valid_cat]
        pred = np.concatenate(field_pred[key], axis=0)[valid_cat]
        results.append({"model": model_name, "split": "test", "target": key, "metric_scope": "valid_grid_pixels", "n": int(true.size), **regression_metrics(true, pred)})
    pdf = pd.DataFrame(pred_rows)
    y_present = pdf["true_patch_present"].astype(int)
    p_present = pdf["pred_patch_present_prob"].astype(float)
    pred_present = (p_present >= 0.5).astype(int)
    present_row = {
        "model": model_name,
        "split": "test",
        "target": "patch_present",
        "metric_scope": "classification",
        "n": int(len(pdf)),
        "accuracy": float(accuracy_score(y_present, pred_present)),
        "balanced_accuracy": float(balanced_accuracy_score(y_present, pred_present)),
        "macro_f1": float(f1_score(y_present, pred_present, average="macro")),
    }
    try:
        present_row["auroc"] = float(roc_auc_score(y_present, p_present)) if y_present.nunique() > 1 else math.nan
    except Exception:
        present_row["auroc"] = math.nan
    results.append(present_row)
    non = pdf[pdf["true_patch_present"].eq(1)].copy()
    if len(non):
        center_err = np.sqrt((non["true_center_x_norm"] - non["pred_center_x_norm"]) ** 2 + (non["true_center_y_norm"] - non["pred_center_y_norm"]) ** 2)
        results.append({"model": model_name, "split": "test", "target": "patch_center_error", "metric_scope": "parametric", "n": int(len(non)), "mae": float(np.mean(center_err)), "rmse": float(np.sqrt(np.mean(center_err * center_err)))})
        for true_col, pred_col, target in [
            ("true_patch_area_param", "pred_patch_area_param", "patch_area_param_head"),
            ("true_log_area_fraction", "pred_log_area_fraction", "log_area_param_head"),
            ("true_radius_param", "pred_radius_param", "radius_param_head"),
            ("true_log_radius", "pred_log_radius", "log_radius_param_head"),
            ("true_LAT_range", "pred_LAT_range", "LAT_range"),
            ("true_APD90_patch_delta", "pred_APD90_patch_delta_pixel_mask_pred_apd", "APD90_patch_delta_pixel_mask_pred_apd"),
            ("true_APD90_patch_delta", "pred_APD90_patch_delta_param_mask_pred_apd", "APD90_patch_delta_param_mask_pred_apd"),
            ("true_APD90_patch_delta", "pred_APD90_patch_delta_param_mask_true_apd", "APD90_patch_delta_param_mask_true_apd"),
            ("true_APD90_patch_delta", "pred_APD90_patch_delta_true_mask_pred_apd", "APD90_patch_delta_true_mask_pred_apd"),
            ("true_patch_area_fraction", "pred_patch_area_fraction_pixel_mask", "patch_area_fraction_pixel_mask"),
            ("true_patch_area_fraction", "pred_patch_area_fraction_param_mask", "patch_area_fraction_param_mask"),
            ("true_patch_area_fraction", "pred_patch_area_param", "patch_area_fraction_param_head"),
        ]:
            results.append({"model": model_name, "split": "test", "target": target, "metric_scope": "derived_or_param_scalar", "n": int(np.isfinite(non[true_col]).sum()), **regression_metrics(non[true_col], non[pred_col])})
        area_non = non[non["true_area_bin"].ge(0)]
        if len(area_non):
            results.append({"model": model_name, "split": "test", "target": "area_bin", "metric_scope": "classification", "n": int(len(area_non)), "accuracy": float(accuracy_score(area_non["true_area_bin"], area_non["pred_area_bin"])), "macro_f1": float(f1_score(area_non["true_area_bin"], area_non["pred_area_bin"], average="macro"))})
    hom = pdf[pdf["true_patch_present"].eq(0)].copy()
    if len(hom):
        results.append({"model": model_name, "split": "test", "target": "homogeneous_false_positive_area_pixel_mask", "metric_scope": "patch", "n": int(len(hom)), "area": float(hom["homogeneous_false_positive_area_pixel_mask"].mean())})
        results.append({"model": model_name, "split": "test", "target": "homogeneous_false_positive_area_param_mask", "metric_scope": "patch", "n": int(len(hom)), "area": float(hom["homogeneous_false_positive_area_param_mask"].mean())})
    return pd.DataFrame(results), pdf, examples


def metric(results: pd.DataFrame, model: str, target: str, col: str):
    sub = results[(results["model"].eq(model)) & (results["target"].eq(target))]
    if sub.empty or col not in sub.columns:
        return math.nan
    vals = sub[col].dropna()
    return float(vals.iloc[0]) if len(vals) else math.nan


def best_model(results: pd.DataFrame, target: str, col: str) -> str:
    sub = results[results["target"].eq(target)]
    if sub.empty or col not in sub.columns:
        return str(results["model"].iloc[0])
    sub = sub[np.isfinite(sub[col])]
    if sub.empty:
        return str(results["model"].iloc[0])
    return str(sub.loc[sub[col].idxmax(), "model"])


def make_figures(results: pd.DataFrame, preds: pd.DataFrame, examples_by_model: dict[str, list[dict]]) -> None:
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    best = best_model(results, "patch_mask", "dice")
    examples = examples_by_model.get(best, [])
    if examples:
        count = min(3, len(examples))
        fig, axes = plt.subplots(count, 4, figsize=(12, 3 * count))
        axes = np.atleast_2d(axes)
        for axrow, ex in zip(axes, examples[:count]):
            items = [
                (ex["true_lat"], "true LAT", "viridis"),
                (ex["pred_lat"], "pred LAT", "viridis"),
                (ex["true_apd90"], "true APD90", "viridis"),
                (ex["pred_apd90"], "pred APD90", "viridis"),
            ]
            for ax, (arr, title, cmap) in zip(axrow, items):
                im = ax.imshow(arr, origin="lower", cmap=cmap)
                ax.set_title(title, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle(f"DeepONet Field Examples: {best}", fontsize=10)
        fig.tight_layout()
        fig.savefig(NEURAL_FIG_DIR / "deeponet_field_examples.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(count, 4, figsize=(12, 3 * count))
        axes = np.atleast_2d(axes)
        for axrow, ex in zip(axes, examples[:count]):
            items = [
                (ex["true_patch"], "true patch"),
                (ex["pred_patch_prob"], "pixel prob"),
                (ex["pred_patch_bin"], "pixel mask"),
                (ex["param_mask"], "param mask"),
            ]
            for ax, (arr, title) in zip(axrow, items):
                im = ax.imshow(arr, origin="lower", cmap="magma", vmin=0, vmax=1)
                ax.set_title(title, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle(f"DeepONet Patch Examples: {best}", fontsize=10)
        fig.tight_layout()
        fig.savefig(NEURAL_FIG_DIR / "deeponet_patch_examples.png", dpi=180)
        plt.close(fig)

    test_non = preds[preds["true_patch_present"].eq(1)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, true_col, pred_col, title in [
        (axes[0], "true_patch_area_param", "pred_patch_area_param", "Area"),
        (axes[1], "true_radius_param", "pred_radius_param", "Radius"),
    ]:
        for model, sub in test_non.groupby("model"):
            m = np.isfinite(sub[true_col]) & np.isfinite(sub[pred_col])
            ax.scatter(sub.loc[m, true_col], sub.loc[m, pred_col], s=8, alpha=0.5, label=model)
        ax.set_xlabel(f"true {title.lower()}")
        ax.set_ylabel(f"pred {title.lower()}")
        ax.set_title(title)
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "deeponet_parametric_predictions.png", dpi=180)
    plt.close(fig)

    summary_rows = []
    for model in sorted(results["model"].unique()):
        summary_rows.extend(
            [
                {"label": f"{model}\npatch Dice", "value": metric(results, model, "patch_mask", "dice")},
                {"label": f"{model}\nAPD90 R2", "value": metric(results, model, "APD90", "r2")},
                {"label": f"{model}\narea R2", "value": metric(results, model, "patch_area_param_head", "r2")},
            ]
        )
    summary_rows.extend(
        [
            {"label": "FNO field\npatch Dice", "value": BASELINES["v1e_500_field_parametric_fno"]["patch_mask_dice"]},
            {"label": "CNN field\nAPD90 R2", "value": BASELINES["v1e_500_field_parametric_cnn"]["APD90_R2"]},
            {"label": "RF desc\narea R2", "value": BASELINES["v1e_500_tabular_rf_descriptors"]["patch_area_R2"]},
        ]
    )
    s = pd.DataFrame(summary_rows)
    fig, ax = plt.subplots(figsize=(max(9, len(s) * 0.45), 4.5))
    ax.bar(range(len(s)), s["value"])
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s["label"], rotation=75, fontsize=7)
    ax.set_title("DeepONet vs FNO/CNN/RF Summary")
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "deeponet_vs_fno_cnn_summary.png", dpi=180)
    plt.close(fig)

    delta = results[results["target"].isin(["APD90_patch_delta_pixel_mask_pred_apd", "APD90_patch_delta_param_mask_pred_apd", "APD90_patch_delta_param_mask_true_apd", "APD90_patch_delta_true_mask_pred_apd"])]
    fig, ax = plt.subplots(figsize=(max(9, len(delta) * 0.32), 4))
    labels = delta["model"] + "\n" + delta["target"]
    ax.bar(range(len(delta)), delta["r2"])
    ax.axhline(BASELINES["v1e_500_parametric_cnn_encoder"]["APD90_delta_param_mask_true_apd_R2"], color="black", linestyle="--", linewidth=1, label="parametric CNN true-APD90")
    ax.set_xticks(range(len(delta)))
    ax.set_xticklabels(labels, rotation=80, fontsize=6)
    ax.set_ylabel("R2")
    ax.set_title("DeepONet APD90 Delta Recovery")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "deeponet_apd90_delta_recovery.png", dpi=180)
    plt.close(fig)

    pairs = []
    for base in ["deeponet_cnn_branch", "deeponet_cnn_branch_fourier_trunk"]:
        with_desc = base.replace("deeponet_cnn_branch", "deeponet_cnn_branch_plus_descriptors")
        if with_desc not in set(results["model"]):
            continue
        for target, col in [("patch_mask", "dice"), ("APD90", "r2"), ("patch_area_param_head", "r2"), ("APD90_patch_delta_param_mask_true_apd", "r2")]:
            b = metric(results, base, target, col)
            d = metric(results, with_desc, target, col)
            if np.isfinite(b) and np.isfinite(d):
                pairs.append({"label": f"{base}\n{target}", "delta": d - b})
    fig, ax = plt.subplots(figsize=(8, 4))
    if pairs:
        ddf = pd.DataFrame(pairs)
        ax.bar(ddf["label"], ddf["delta"])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("DeepONet Descriptor Branch Effect")
    ax.set_ylabel("descriptor - no descriptor")
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "deeponet_descriptor_branch_effect.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    require_cuda(torch)
    rows = load_index()
    desc = load_descriptors()
    stats = compute_stats(rows, desc)
    first = next(iter(make_loader(rows.head(1), stats, batch_size=1, shuffle=False, mode="full", descriptors=desc)))
    desc_dim = int(first["descriptor_x"].shape[1])
    device = torch.device("cuda")
    paths = checkpoint_paths()
    if not paths:
        raise SystemExit("No DeepONet checkpoints found in outputs/neural_v1e_500/checkpoints")
    all_results = []
    all_preds = []
    examples_by_model = {}
    for path in paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model_name = str(ckpt.get("model_name", path.stem.replace("deeponet_", "")))
        model = build_model_from_checkpoint(ckpt, desc_dim).to(device)
        res, pred, examples = evaluate_model(model, model_name, rows, desc, stats, args.batch_size, device)
        all_results.append(res)
        all_preds.append(pred)
        examples_by_model[model_name] = examples
        print(f"evaluated {model_name}", flush=True)
    results = pd.concat(all_results, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_RESULTS, index=False)
    preds.to_csv(OUT_PREDS, index=False)
    make_figures(results, preds, examples_by_model)
    test = results[results["split"].eq("test")].copy()
    bests = {
        "best_patch_dice": {
            "value": float(test[test["target"].eq("patch_mask")]["dice"].max()),
            "model": best_model(test, "patch_mask", "dice"),
        },
        "best_center_mae": {
            "value": float(test[test["target"].eq("patch_center_error")]["mae"].min()),
            "model": str(test[test["target"].eq("patch_center_error")].sort_values("mae").iloc[0]["model"]),
        },
        "best_apd90_r2": {
            "value": float(test[test["target"].eq("APD90")]["r2"].max()),
            "model": best_model(test, "APD90", "r2"),
        },
        "best_apd90_delta_param_mask_true_apd_r2": {
            "value": float(test[test["target"].eq("APD90_patch_delta_param_mask_true_apd")]["r2"].max()),
            "model": best_model(test, "APD90_patch_delta_param_mask_true_apd", "r2"),
        },
        "best_area_r2": {
            "value": float(test[test["target"].eq("patch_area_param_head")]["r2"].max()),
            "model": best_model(test, "patch_area_param_head", "r2"),
        },
        "best_radius_r2": {
            "value": float(test[test["target"].eq("radius_param_head")]["r2"].max()),
            "model": best_model(test, "radius_param_head", "r2"),
        },
    }
    payload = {
        "result_file": str(OUT_RESULTS),
        "prediction_file": str(OUT_PREDS),
        "models": sorted(results["model"].unique().tolist()),
        "test_metrics": test.replace({np.nan: None}).to_dict(orient="records"),
        "bests": bests,
        "baselines": BASELINES,
        "figures": [
            "figures/neural_v1e_500/deeponet_field_examples.png",
            "figures/neural_v1e_500/deeponet_patch_examples.png",
            "figures/neural_v1e_500/deeponet_parametric_predictions.png",
            "figures/neural_v1e_500/deeponet_vs_fno_cnn_summary.png",
            "figures/neural_v1e_500/deeponet_apd90_delta_recovery.png",
            "figures/neural_v1e_500/deeponet_descriptor_branch_effect.png",
        ],
        "leakage_guardrail": "Evaluation uses predictions and labels only; model inputs remain B_meas, optional B_meas-derived descriptors, and query coordinates.",
    }
    write_json(OUT_JSON, payload)
    print(json.dumps({"results": str(OUT_RESULTS), "predictions": str(OUT_PREDS), "bests": bests}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
