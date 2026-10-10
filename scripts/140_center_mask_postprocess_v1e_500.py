                      
                                                                                        

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
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

from v1e_500_common import CHECKPOINT_DIR, NEURAL_DIR, NEURAL_FIG_DIR, SEED, require_cuda, write_json
from v1e_deeponet_data_common import compute_stats, load_descriptors, torch
from v1e_deeponet_mask_refine_data_common import make_mask_refine_loader, merge_rows_with_refine_targets


TRAIN_SPEC = importlib.util.spec_from_file_location("_mask_refine_train", PROJECT_ROOT / "scripts" / "137_train_v1e_500_deeponet_mask_refinement.py")
if TRAIN_SPEC is None or TRAIN_SPEC.loader is None:
    raise ImportError("Could not load mask-refinement training module")
MASK_TRAIN = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(MASK_TRAIN)


RESULTS_CSV = NEURAL_DIR / "center_mask_postprocess_results.csv"
CALIBRATION_CSV = NEURAL_DIR / "center_mask_postprocess_calibration.csv"
PREDICTIONS_CSV = NEURAL_DIR / "center_mask_postprocess_predictions.csv"
RESULTS_JSON = NEURAL_DIR / "center_mask_postprocess_results.json"
GATE_JSON = NEURAL_DIR / "center_mask_postprocess_gate.json"

BASELINE_BALANCED_DELTA_R2 = 0.6069896937501429
TARGET_BALANCED_DELTA_R2 = 0.65
TARGET_BALANCED_DICE = 0.55
TARGET_HFP = 0.005
TARGET_CONDUCTION_DICE = 0.25
PRIOR_SPARSE_R2 = 0.7313469085858489


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    pos = x >= 0
    out = np.empty_like(x, dtype=np.float64)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out.astype(np.float32)


def finite_r2(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 2 or np.allclose(y[m], y[m][0]):
        return math.nan
    ss_res = float(np.sum((y[m] - p[m]) ** 2))
    ss_tot = float(np.sum((y[m] - np.mean(y[m])) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan


def regression_metrics(y, p) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if not np.any(m):
        return {"mae": math.nan, "rmse": math.nan, "r2": math.nan}
    d = y[m] - p[m]
    return {"mae": float(np.mean(np.abs(d))), "rmse": float(np.sqrt(np.mean(d * d))), "r2": finite_r2(y[m], p[m])}


def patch_metrics(true_mask: np.ndarray, pred_mask: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    v = valid.astype(bool)
    t = (true_mask > 0.5) & v
    p = (pred_mask > 0.5) & v
    tp = float(np.sum(t & p))
    fp = float(np.sum(~t & p & v))
    fn = float(np.sum(t & ~p))
    return {
        "dice": (2.0 * tp) / max(2.0 * tp + fp + fn, 1.0),
        "iou": tp / max(tp + fp + fn, 1.0),
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
    }


def area_fraction(mask: np.ndarray, valid: np.ndarray) -> float:
    v = valid.astype(bool)
    return float(np.mean((mask > 0.5)[v])) if np.any(v) else math.nan


def apd90_delta(apd90: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> float:
    v = valid.astype(bool) & np.isfinite(apd90)
    inside = (mask > 0.5) & v
    outside = (mask <= 0.5) & v
    return float(np.mean(apd90[inside]) - np.mean(apd90[outside])) if inside.any() and outside.any() else math.nan


def center_error(cx: float, cy: float, tx: float, ty: float) -> float:
    if not all(np.isfinite(x) for x in [cx, cy, tx, ty]):
        return math.nan
    return float(math.sqrt((cx - tx) ** 2 + (cy - ty) ** 2))


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


def peak_xy(score: np.ndarray) -> tuple[float, float]:
    idx = int(np.nanargmax(score))
    y, x = np.unravel_index(idx, score.shape)
    return float(x / max(score.shape[1] - 1, 1)), float(y / max(score.shape[0] - 1, 1))


def weighted_centroid(score: np.ndarray) -> tuple[float, float]:
    weights = np.clip(score, 0.0, None).astype(float)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return math.nan, math.nan
    yy, xx = np.indices(score.shape)
    return float(np.sum(xx * weights) / total / max(score.shape[1] - 1, 1)), float(np.sum(yy * weights) / total / max(score.shape[0] - 1, 1))


def unscale_params(params_np: np.ndarray, stats) -> np.ndarray:
    out = params_np.copy()
    out[:, 1:] = out[:, 1:] * stats.param_std[1:] + stats.param_mean[1:]
    return out


def collect_outputs(rows: pd.DataFrame, desc: pd.DataFrame, stats, args, device):
    records = []
    maps = defaultdict(list)
    for path in checkpoint_paths():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model_name = str(ckpt.get("model_name", path.stem.replace("deeponet_mask_refine_", "")))
        model = build_model(ckpt).to(device)
        for split in ["val", "test"]:
            sub = rows[rows["split"].eq(split)].reset_index(drop=True)
            loader = make_mask_refine_loader(sub, stats, batch_size=args.batch_size, shuffle=False, mode="full", k_queries=1024, input_norm=args.input_norm, descriptors=desc, seed=SEED + 810)
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
                    true_patch = batch["patch_query"].numpy()
                    valid = batch["valid_query_mask"].numpy().astype(bool)
                    true_apd90 = batch["y_field"].numpy()[:, :, 1] * stats.field_std[2] + stats.field_mean[2]
                    pred_apd90 = query_np[:, :, 1] * stats.field_std[2] + stats.field_mean[2]
                    patch_prob = sigmoid_np(query_np[:, :, 2])
                    center_prob = sigmoid_np(query_np[:, :, 3])
                    signed_dist = query_np[:, :, 4]
                    present_prob = sigmoid_np(params_np[:, 0])
                    for b in range(query_np.shape[0]):
                        src = sub.iloc[offset + b]
                        prob_map = patch_prob[b].reshape(32, 32)
                        heat_map = center_prob[b].reshape(32, 32)
                        heat_x, heat_y = peak_xy(heat_map)
                        prob_x, prob_y = weighted_centroid(prob_map)
                        true_cx = y_param[b, 1] * stats.param_std[1] + stats.param_mean[1] if p_mask[b, 1] else math.nan
                        true_cy = y_param[b, 2] * stats.param_std[2] + stats.param_mean[2] if p_mask[b, 2] else math.nan
                        pred_area = float(math.exp(p_unscaled[b, 3])) if np.isfinite(p_unscaled[b, 3]) else math.nan
                        pred_radius = float(math.exp(p_unscaled[b, 4])) if np.isfinite(p_unscaled[b, 4]) else math.nan
                        records.append(
                            {
                                "model": model_name,
                                "split": split,
                                "row_id": src["row_id"],
                                "base_sample_id": src["base_sample_id"],
                                "condition": src["condition"],
                                "combined_variant": src["combined_variant"],
                                "true_patch_present": float(y_param[b, 0]),
                                "pred_patch_present_prob": float(present_prob[b]),
                                "true_center_x_norm": true_cx,
                                "true_center_y_norm": true_cy,
                                "pred_center_x_norm": float(p_unscaled[b, 1]),
                                "pred_center_y_norm": float(p_unscaled[b, 2]),
                                "center_heatmap_peak_x_norm": heat_x,
                                "center_heatmap_peak_y_norm": heat_y,
                                "probability_centroid_x_norm": prob_x,
                                "probability_centroid_y_norm": prob_y,
                                "pred_patch_area_param": pred_area,
                                "pred_radius_param": pred_radius,
                                "pred_area_bin": int(np.argmax(area_np[b])),
                                "true_patch_area_fraction": float(raw_area_radius[b, 0]) if np.isfinite(raw_area_radius[b, 0]) else math.nan,
                                "true_radius_param": float(raw_area_radius[b, 1]) if np.isfinite(raw_area_radius[b, 1]) else math.nan,
                                "true_APD90_patch_delta": float(batch["apd90_patch_delta"].numpy()[b]),
                                "global_center_error": center_error(float(p_unscaled[b, 1]), float(p_unscaled[b, 2]), true_cx, true_cy),
                                "heatmap_peak_error": center_error(heat_x, heat_y, true_cx, true_cy),
                                "probability_centroid_error": center_error(prob_x, prob_y, true_cx, true_cy),
                                "array_index": len(maps["patch_prob"]),
                            }
                        )
                        maps["patch_prob"].append(prob_map.astype(np.float32))
                        maps["center_prob"].append(heat_map.astype(np.float32))
                        maps["signed_distance"].append(signed_dist[b].reshape(32, 32).astype(np.float32))
                        maps["pred_apd90"].append(pred_apd90[b].reshape(32, 32).astype(np.float32))
                        maps["true_apd90"].append(true_apd90[b].reshape(32, 32).astype(np.float32))
                        maps["true_patch"].append(true_patch[b].reshape(32, 32).astype(np.float32))
                        maps["valid"].append(valid[b].reshape(32, 32).astype(bool))
                    offset += query_np.shape[0]
        print(f"collected deployable full-grid outputs for {model_name}", flush=True)
    return pd.DataFrame(records), {k: np.stack(v) for k, v in maps.items()}


def radius_from_row(row, source: str, multiplier: float = 1.0) -> float:
    pred_radius = float(row.get("pred_radius_param", math.nan))
    pred_area = float(row.get("pred_patch_area_param", math.nan))
    area_radius = math.sqrt(max(pred_area, 0.0) / math.pi) if np.isfinite(pred_area) else math.nan
    if source == "radius":
        radius = pred_radius
    elif source == "area":
        radius = area_radius
    elif source == "mean":
        if np.isfinite(pred_radius) and np.isfinite(area_radius):
            radius = 0.5 * (pred_radius + area_radius)
        elif np.isfinite(pred_radius):
            radius = pred_radius
        else:
            radius = area_radius
    else:
        raise ValueError(f"unknown radius source {source}")
    if not np.isfinite(radius):
        return math.nan
    return float(np.clip(radius * multiplier, 1e-4, 0.75))


def disk_mask(cx: float, cy: float, radius: float, shape=(32, 32)) -> np.ndarray:
    if not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(radius):
        return np.zeros(shape, dtype=np.float32)
    yy, xx = np.indices(shape)
    x = xx / max(shape[1] - 1, 1)
    y = yy / max(shape[0] - 1, 1)
    return (((x - np.clip(cx, 0.0, 1.0)) ** 2 + (y - np.clip(cy, 0.0, 1.0)) ** 2) <= radius**2).astype(np.float32)


def center_distance_mask(cx: float, cy: float, max_radius: float, shape=(32, 32)) -> np.ndarray:
    return disk_mask(cx, cy, max_radius, shape)


def connected_component_mask(binary: np.ndarray, seed_xy: tuple[float, float], rule: str) -> np.ndarray:
    binary = binary.astype(bool)
    if not np.any(binary):
        return np.zeros_like(binary, dtype=np.float32)
    try:
        from scipy import ndimage

        labels, nlab = ndimage.label(binary, structure=np.ones((3, 3), dtype=int))
        if nlab == 0:
            return np.zeros_like(binary, dtype=np.float32)
        sizes = np.bincount(labels.reshape(-1))
        sizes[0] = 0
        if rule == "largest":
            keep = int(np.argmax(sizes))
        else:
            sx = int(round(np.clip(seed_xy[0], 0.0, 1.0) * (binary.shape[1] - 1)))
            sy = int(round(np.clip(seed_xy[1], 0.0, 1.0) * (binary.shape[0] - 1)))
            keep = int(labels[sy, sx])
            if keep == 0:
                yy, xx = np.indices(binary.shape)
                best = None
                for lab in range(1, nlab + 1):
                    m = labels == lab
                    cy = float(np.mean(yy[m]))
                    cx = float(np.mean(xx[m]))
                    dist = (cx - sx) ** 2 + (cy - sy) ** 2
                    if best is None or dist < best[0]:
                        best = (dist, lab)
                keep = int(best[1]) if best is not None else 0
        return (labels == keep).astype(np.float32)
    except Exception:
        return binary.astype(np.float32)


def candidate_mask(row: pd.Series, maps: dict[str, np.ndarray], cand: dict) -> np.ndarray:
    idx = int(row["array_index"])
    prob = maps["patch_prob"][idx]
    heat = maps["center_prob"][idx]
    dist = maps["signed_distance"][idx]
    shape = prob.shape
    family = cand["family"]
    present_ok = float(row["pred_patch_present_prob"]) >= float(cand.get("present_threshold", 0.5))
    if cand.get("present_gate", False) and not present_ok:
        return np.zeros(shape, dtype=np.float32)
    cx = float(row["pred_center_x_norm"])
    cy = float(row["pred_center_y_norm"])
    radius = radius_from_row(row, cand.get("radius_source", "radius"), float(cand.get("radius_multiplier", 1.0)))
    if family in {"global_probability_threshold", "sparse_high_threshold"}:
        return (prob >= float(cand["prob_threshold"])).astype(np.float32)
    if family == "predicted_center_disk":
        return disk_mask(cx, cy, radius, shape)
    if family == "predicted_center_disk_radius_multiplier":
        return disk_mask(cx, cy, radius, shape)
    if family == "center_constrained_probability":
        center_gate = center_distance_mask(cx, cy, radius, shape)
        return ((prob >= float(cand["prob_threshold"])) & (center_gate > 0.5)).astype(np.float32)
    if family == "heatmap_peak_component":
        hx, hy = peak_xy(heat)
        binary = prob >= float(cand["prob_threshold"])
        return connected_component_mask(binary, (hx, hy), str(cand.get("component_rule", "seed_nearest")))
    if family == "signed_distance_threshold":
        return (dist >= float(cand["distance_threshold"])).astype(np.float32)
    if family == "hybrid_center_distance_probability":
        center_gate = center_distance_mask(cx, cy, radius, shape)
        return ((prob >= float(cand["prob_threshold"])) & (dist >= float(cand["distance_threshold"])) & (center_gate > 0.5)).astype(np.float32)
    raise ValueError(f"unknown family {family}")


def candidate_name(cand: dict) -> str:
    keys = ["family", "prob_threshold", "distance_threshold", "radius_source", "radius_multiplier", "component_rule", "present_gate"]
    parts = [str(cand.get(k)) for k in keys if k in cand]
    return "|".join(parts)


def candidate_grid() -> list[dict]:
    cands: list[dict] = []
    for thr in np.round(np.linspace(0.20, 0.90, 15), 2):
        cands.append({"family": "global_probability_threshold", "prob_threshold": float(thr)})
    for source in ["radius", "area", "mean"]:
        cands.append({"family": "predicted_center_disk", "radius_source": source, "present_gate": True})
    for source in ["radius", "area", "mean"]:
        for mult in np.round(np.linspace(0.60, 1.80, 9), 2):
            cands.append({"family": "predicted_center_disk_radius_multiplier", "radius_source": source, "radius_multiplier": float(mult), "present_gate": True})
    for thr in [0.35, 0.45, 0.55, 0.65, 0.75]:
        for mult in [0.9, 1.2, 1.5, 1.8, 2.2]:
            cands.append({"family": "center_constrained_probability", "prob_threshold": thr, "radius_source": "mean", "radius_multiplier": mult, "present_gate": True})
    for thr in [0.25, 0.35, 0.45, 0.55, 0.65]:
        for rule in ["seed_nearest", "largest"]:
            cands.append({"family": "heatmap_peak_component", "prob_threshold": thr, "component_rule": rule, "present_gate": True})
    for dthr in np.round(np.linspace(-0.10, 0.08, 10), 3):
        cands.append({"family": "signed_distance_threshold", "distance_threshold": float(dthr), "present_gate": True})
    for pthr in [0.35, 0.45, 0.55, 0.65]:
        for dthr in [-0.08, -0.04, 0.0, 0.04]:
            for mult in [1.0, 1.4, 1.8]:
                cands.append({"family": "hybrid_center_distance_probability", "prob_threshold": pthr, "distance_threshold": dthr, "radius_source": "mean", "radius_multiplier": mult, "present_gate": True})
    for thr in np.round(np.linspace(0.80, 0.99, 20), 2):
        cands.append({"family": "sparse_high_threshold", "prob_threshold": float(thr)})
    for i, cand in enumerate(cands):
        cand["candidate_id"] = f"cand_{i:04d}"
        cand["candidate_name"] = candidate_name(cand)
    return cands


def evaluate_candidate(index: pd.DataFrame, maps: dict[str, np.ndarray], model: str, split: str, cand: dict) -> tuple[dict, pd.DataFrame]:
    sub = index[index["model"].eq(model) & index["split"].eq(split)].copy()
    rows = []
    true_stack = []
    pred_stack = []
    valid_stack = []
    for _, row in sub.iterrows():
        idx = int(row["array_index"])
        pred_mask = candidate_mask(row, maps, cand)
        true_patch = maps["true_patch"][idx]
        valid = maps["valid"][idx]
        pm = patch_metrics(true_patch, pred_mask, valid)
        true_delta = apd90_delta(maps["true_apd90"][idx], true_patch, valid)
        pred_delta = apd90_delta(maps["pred_apd90"][idx], pred_mask, valid)
        area = area_fraction(pred_mask, valid)
        rows.append(
            {
                "model": model,
                "split": split,
                "candidate_id": cand["candidate_id"],
                "candidate_scope": cand.get("candidate_scope", "candidate"),
                "family": cand["family"],
                "candidate_name": cand["candidate_name"],
                "row_id": row["row_id"],
                "base_sample_id": row["base_sample_id"],
                "condition": row["condition"],
                "combined_variant": row["combined_variant"],
                "true_patch_present": row["true_patch_present"],
                "dice": pm["dice"],
                "iou": pm["iou"],
                "precision": pm["precision"],
                "recall": pm["recall"],
                "true_area": area_fraction(true_patch, valid),
                "pred_area": area,
                "true_delta": true_delta,
                "pred_delta": pred_delta,
                "abs_delta_error": abs(pred_delta - true_delta) if np.isfinite(pred_delta) and np.isfinite(true_delta) else math.nan,
                "homogeneous_false_positive_area": area if float(row["true_patch_present"]) < 0.5 else math.nan,
                "global_center_error": row["global_center_error"],
                "heatmap_peak_error": row["heatmap_peak_error"],
                "probability_centroid_error": row["probability_centroid_error"],
                **{k: v for k, v in cand.items() if k not in {"model", "split"}},
            }
        )
        true_stack.append(true_patch)
        pred_stack.append(pred_mask)
        valid_stack.append(valid)
    df = pd.DataFrame(rows)
    non = df[df["true_patch_present"].eq(1)]
    micro = patch_metrics(np.stack(true_stack), np.stack(pred_stack), np.stack(valid_stack).astype(bool)) if rows else {"dice": math.nan, "iou": math.nan, "precision": math.nan, "recall": math.nan}
    rec = {
        "model": model,
        "split": split,
        "candidate_id": cand["candidate_id"],
        "candidate_scope": cand.get("candidate_scope", "candidate"),
        "family": cand["family"],
        "candidate_name": cand["candidate_name"],
        "n": int(len(df)),
        "dice": micro["dice"],
        "row_mean_dice": float(df["dice"].mean(skipna=True)) if len(df) else math.nan,
        "iou": micro["iou"],
        "precision": micro["precision"],
        "recall": micro["recall"],
        "homogeneous_false_positive_area": float(df["homogeneous_false_positive_area"].mean(skipna=True)),
        "patch_area_r2": regression_metrics(non["true_area"], non["pred_area"])["r2"] if len(non) else math.nan,
        "APD90_patch_delta_R2_pred_mask_pred_apd": regression_metrics(non["true_delta"], non["pred_delta"])["r2"] if len(non) else math.nan,
        "APD90_patch_delta_MAE_pred_mask_pred_apd": regression_metrics(non["true_delta"], non["pred_delta"])["mae"] if len(non) else math.nan,
        "mean_global_center_error": float(non["global_center_error"].mean(skipna=True)) if len(non) else math.nan,
    }
    rec.update({k: v for k, v in cand.items() if k not in rec})
    return rec, df


def selection_scores(rec: dict) -> tuple[float, float]:
    delta = rec["APD90_patch_delta_R2_pred_mask_pred_apd"] if np.isfinite(rec["APD90_patch_delta_R2_pred_mask_pred_apd"]) else -2.0
    dice = rec["dice"] if np.isfinite(rec["dice"]) else 0.0
    hfp = rec["homogeneous_false_positive_area"] if np.isfinite(rec["homogeneous_false_positive_area"]) else 1.0
    balanced = delta + 0.35 * dice - 5.0 * max(hfp - TARGET_HFP, 0.0) - 0.75 * max(TARGET_BALANCED_DICE - dice, 0.0)
    sparse = delta - 2.0 * hfp
    return balanced, sparse


def calibrate(index: pd.DataFrame, maps: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict]:
    cands = candidate_grid()
    records = []
    selected_by_family = []
    for model in sorted(index["model"].unique()):
        model_records = []
        for cand in cands:
            rec, _ = evaluate_candidate(index, maps, model, "val", cand)
            rec["selection_score_balanced"], rec["selection_score_sparse"] = selection_scores(rec)
            records.append(rec)
            model_records.append(rec)
        model_df = pd.DataFrame(model_records)
        for family, sub in model_df.groupby("family"):
            if family == "sparse_high_threshold":
                best = sub.sort_values("selection_score_sparse", ascending=False).iloc[0].to_dict()
            else:
                pool = sub[(sub["dice"].ge(0.25)) & (sub["homogeneous_false_positive_area"].le(0.05))]
                if pool.empty:
                    pool = sub
                best = pool.sort_values("selection_score_balanced", ascending=False).iloc[0].to_dict()
            best["candidate_scope"] = "selected_by_family_val"
            selected_by_family.append(best)
            records.append(best)
    val_df = pd.DataFrame(records)
    family_df = pd.DataFrame(selected_by_family)
    balanced_pool = val_df[(val_df["candidate_scope"].eq("candidate")) & (val_df["family"].ne("sparse_high_threshold"))]
    strong_pool = balanced_pool[(balanced_pool["dice"].ge(TARGET_BALANCED_DICE)) & (balanced_pool["homogeneous_false_positive_area"].le(TARGET_HFP))]
    if strong_pool.empty:
        strong_pool = balanced_pool[(balanced_pool["dice"].ge(0.25)) & (balanced_pool["homogeneous_false_positive_area"].le(0.05))]
    if strong_pool.empty:
        strong_pool = balanced_pool
    overall_balanced = strong_pool.sort_values("selection_score_balanced", ascending=False).iloc[0].to_dict()
    sparse_pool = val_df[(val_df["candidate_scope"].eq("candidate")) & (val_df["family"].eq("sparse_high_threshold"))]
    overall_sparse = sparse_pool.sort_values("selection_score_sparse", ascending=False).iloc[0].to_dict()
    overall_balanced["candidate_scope"] = "overall_balanced_val"
    overall_sparse["candidate_scope"] = "overall_sparse_val"
    val_df = pd.concat([val_df, pd.DataFrame([overall_balanced, overall_sparse])], ignore_index=True, sort=False)
    return val_df, {"selected_by_family": selected_by_family, "overall_balanced": overall_balanced, "overall_sparse": overall_sparse}


def rec_to_candidate(rec: dict, scope: str) -> dict:
    keys = ["candidate_id", "candidate_name", "family", "prob_threshold", "distance_threshold", "radius_source", "radius_multiplier", "component_rule", "present_gate", "present_threshold"]
    cand = {k: rec[k] for k in keys if k in rec and pd.notna(rec[k])}
    cand["candidate_scope"] = scope
    return cand


def evaluate_selected(index, maps, selections) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = []
    row_frames = []
    seen = set()
    selected = list(selections["selected_by_family"])
    selected.extend([selections["overall_balanced"], selections["overall_sparse"]])
    for rec in selected:
        scope = rec["candidate_scope"].replace("_val", "_test") if str(rec["candidate_scope"]).endswith("_val") else "selected_by_family_test"
        cand = rec_to_candidate(rec, scope)
        model = rec["model"]
        key = (model, cand["candidate_id"], scope)
        if key in seen:
            continue
        seen.add(key)
        srec, rdf = evaluate_candidate(index, maps, model, "test", cand)
        srec["selected_from_val_score_balanced"] = rec.get("selection_score_balanced", math.nan)
        srec["selected_from_val_score_sparse"] = rec.get("selection_score_sparse", math.nan)
        summary.append(srec)
        row_frames.append(rdf)
    return pd.DataFrame(summary), pd.concat(row_frames, ignore_index=True, sort=False)


def add_condition_variant_summaries(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    focus = rows[rows["candidate_scope"].isin(["overall_balanced_test", "overall_sparse_test", "selected_by_family_test"])].copy()
    for group_col, metric_scope in [("condition", "condition"), ("combined_variant", "combined_variant")]:
        for keys, sub in focus.groupby(["model", "candidate_scope", "family", "candidate_id", group_col]):
            model, scope, family, cand_id, group = keys
            non = sub[sub["true_patch_present"].eq(1)]
            out.append(
                {
                    "model": model,
                    "split": "test",
                    "candidate_id": cand_id,
                    "candidate_scope": scope,
                    "family": family,
                    "metric_scope": metric_scope,
                    group_col: group,
                    "n": int(len(sub)),
                    "row_mean_dice": float(sub["dice"].mean(skipna=True)),
                    "row_mean_precision": float(sub["precision"].mean(skipna=True)),
                    "row_mean_recall": float(sub["recall"].mean(skipna=True)),
                    "homogeneous_false_positive_area": float(sub["homogeneous_false_positive_area"].mean(skipna=True)),
                    "APD90_patch_delta_R2_pred_mask_pred_apd": regression_metrics(non["true_delta"], non["pred_delta"])["r2"] if len(non) else math.nan,
                    "APD90_patch_delta_MAE_pred_mask_pred_apd": regression_metrics(non["true_delta"], non["pred_delta"])["mae"] if len(non) else math.nan,
                }
            )
    return pd.DataFrame(out)


def make_figures(index, maps, test_rows, summary):
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    balanced = test_rows[test_rows["candidate_scope"].eq("overall_balanced_test")].copy()
    sparse = test_rows[test_rows["candidate_scope"].eq("overall_sparse_test")].copy()
    if len(balanced):
        chosen = pd.concat([balanced[balanced["true_patch_present"].eq(1)].sort_values("dice", ascending=False).head(2), balanced[balanced["true_patch_present"].eq(1)].sort_values("dice").head(2)], ignore_index=True)
        fig, axes = plt.subplots(len(chosen), 5, figsize=(14, 3 * max(len(chosen), 1)))
        axes = np.atleast_2d(axes)
        for axrow, row in zip(axes, chosen.itertuples(index=False)):
            src = index[(index["model"].eq(row.model)) & (index["row_id"].eq(row.row_id)) & (index["split"].eq("test"))].iloc[0]
            idx = int(src["array_index"])
            cand = rec_to_candidate(row._asdict(), "tmp")
            mask = candidate_mask(src, maps, cand)
            items = [
                (maps["true_patch"][idx], "true patch", "magma", 0, 1),
                (maps["patch_prob"][idx], "patch prob", "magma", 0, 1),
                (maps["center_prob"][idx], "center heat", "magma", 0, 1),
                (maps["signed_distance"][idx], "signed distance", "coolwarm", None, None),
                (mask, "post mask", "magma", 0, 1),
            ]
            for ax, (arr, title, cmap, vmin, vmax) in zip(axrow, items):
                im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f"{title}\nDice {row.dice:.3f}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.tight_layout()
        fig.savefig(NEURAL_FIG_DIR / "center_mask_postprocess_best_worst_masks.png", dpi=180)
        plt.close(fig)

    if len(balanced) and len(sparse):
        merged = balanced[["row_id", "model", "dice", "pred_delta", "true_delta"]].rename(columns={"model": "balanced_model", "dice": "balanced_dice", "pred_delta": "balanced_delta"}).merge(
            sparse[["row_id", "dice", "pred_delta"]].rename(columns={"dice": "sparse_dice", "pred_delta": "sparse_delta"}), on="row_id", how="inner"
        ).head(4)
        fig, axes = plt.subplots(len(merged), 4, figsize=(11, 3 * max(len(merged), 1)))
        axes = np.atleast_2d(axes)
        bsrc = summary[summary["candidate_scope"].eq("overall_balanced_test")].iloc[0].to_dict()
        ssrc = summary[summary["candidate_scope"].eq("overall_sparse_test")].iloc[0].to_dict()
        bcand = rec_to_candidate(bsrc, "tmp")
        scand = rec_to_candidate(ssrc, "tmp")
        for axrow, row in zip(axes, merged.itertuples(index=False)):
            bidx_row = index[(index["model"].eq(bsrc["model"])) & (index["row_id"].eq(row.row_id)) & (index["split"].eq("test"))].iloc[0]
            sidx_row = index[(index["model"].eq(ssrc["model"])) & (index["row_id"].eq(row.row_id)) & (index["split"].eq("test"))].iloc[0]
            bidx = int(bidx_row["array_index"])
            sidx = int(sidx_row["array_index"])
            items = [
                (maps["true_patch"][bidx], "true patch"),
                (maps["patch_prob"][bidx], "balanced prob"),
                (candidate_mask(bidx_row, maps, bcand), "balanced mask"),
                (candidate_mask(sidx_row, maps, scand), "sparse mask"),
            ]
            for ax, (arr, title) in zip(axrow, items):
                im = ax.imshow(arr, origin="lower", cmap="magma", vmin=0, vmax=1)
                ax.set_title(title, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.tight_layout()
        fig.savefig(NEURAL_FIG_DIR / "center_mask_postprocess_sparse_vs_balanced_masks.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, df, marker in [("balanced", balanced, "o"), ("sparse", sparse, "x")]:
        non = df[df["true_patch_present"].eq(1)]
        ax.scatter(non["true_delta"], non["pred_delta"], s=14, alpha=0.55, label=label, marker=marker)
    ax.set_xlabel("true APD90 patch delta")
    ax.set_ylabel("predicted-mask APD90 patch delta")
    ax.set_title("Center-Mask Postprocess Delta Recovery")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "center_mask_postprocess_apd90_delta_scatter.png", dpi=180)
    plt.close(fig)

    cond = summary[(summary["metric_scope"].eq("condition")) & (summary["candidate_scope"].eq("overall_balanced_test"))].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(cond):
        ax.bar(cond["condition"], cond["row_mean_dice"])
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.set_ylabel("row mean Dice")
    ax.set_title("Center-Mask Postprocess Dice By Condition")
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "center_mask_postprocess_dice_by_condition.png", dpi=180)
    plt.close(fig)

    fail = balanced[(balanced["condition"].eq("conduction_slowing_perturbation")) & (balanced["true_patch_present"].eq(1))].sort_values("dice").head(4)
    if len(fail):
        fig, axes = plt.subplots(len(fail), 4, figsize=(11, 3 * len(fail)))
        axes = np.atleast_2d(axes)
        cand = rec_to_candidate(summary[summary["candidate_scope"].eq("overall_balanced_test")].iloc[0].to_dict(), "tmp")
        for axrow, row in zip(axes, fail.itertuples(index=False)):
            src = index[(index["model"].eq(row.model)) & (index["row_id"].eq(row.row_id)) & (index["split"].eq("test"))].iloc[0]
            idx = int(src["array_index"])
            items = [
                (maps["true_patch"][idx], "true patch", "magma"),
                (maps["patch_prob"][idx], "patch prob", "magma"),
                (maps["center_prob"][idx], "center heat", "magma"),
                (candidate_mask(src, maps, cand), "post mask", "magma"),
            ]
            for ax, (arr, title, cmap) in zip(axrow, items):
                im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=0, vmax=1)
                ax.set_title(f"{title}\nDice {row.dice:.3f}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.tight_layout()
        fig.savefig(NEURAL_FIG_DIR / "center_mask_postprocess_conduction_failures.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    non = balanced[balanced["true_patch_present"].eq(1)].copy()
    if len(non):
        ax.scatter(non["global_center_error"], non["abs_delta_error"], s=18, alpha=0.65)
    ax.set_xlabel("global center error")
    ax.set_ylabel("absolute APD90 delta error")
    ax.set_title("Center Error vs Delta Error")
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "center_mask_postprocess_center_error_vs_delta_error.png", dpi=180)
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
    index, maps = collect_outputs(rows, desc, stats, args, device)
    calibration_df, selections = calibrate(index, maps)
    selected_summary, selected_rows = evaluate_selected(index, maps, selections)
    condition_variant = add_condition_variant_summaries(selected_rows)
    results_df = pd.concat([selected_summary, condition_variant], ignore_index=True, sort=False)
    predictions_df = selected_rows.copy()
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    calibration_df.to_csv(CALIBRATION_CSV, index=False)
    results_df.to_csv(RESULTS_CSV, index=False)
    predictions_df.to_csv(PREDICTIONS_CSV, index=False)
    make_figures(index, maps, predictions_df, results_df)

    balanced = selected_summary[selected_summary["candidate_scope"].eq("overall_balanced_test")].iloc[0].to_dict()
    sparse = selected_summary[selected_summary["candidate_scope"].eq("overall_sparse_test")].iloc[0].to_dict()
    cond = condition_variant[(condition_variant["candidate_scope"].eq("overall_balanced_test")) & (condition_variant["metric_scope"].eq("condition")) & (condition_variant["condition"].eq("conduction_slowing_perturbation"))]
    conduction = cond.iloc[0].to_dict() if len(cond) else {}
    gate = {
        "balanced_pass": bool(
            float(balanced.get("APD90_patch_delta_R2_pred_mask_pred_apd", -math.inf)) > TARGET_BALANCED_DELTA_R2
            and float(balanced.get("dice", 0.0)) > TARGET_BALANCED_DICE
            and float(balanced.get("homogeneous_false_positive_area", math.inf)) < TARGET_HFP
        ),
        "homogeneous_fp_pass": bool(float(balanced.get("homogeneous_false_positive_area", math.inf)) < TARGET_HFP),
        "conduction_pass": bool(float(conduction.get("row_mean_dice", -math.inf)) > TARGET_CONDUCTION_DICE) if conduction else False,
        "sparse_recovery_pass": bool(float(sparse.get("APD90_patch_delta_R2_pred_mask_pred_apd", -math.inf)) > PRIOR_SPARSE_R2),
        "overall_balanced_test": balanced,
        "overall_sparse_test": sparse,
        "conduction_slowing_balanced": conduction,
        "ready_for_pi_deeponet_or_5000": False,
        "reason_not_ready": "Ready only if balanced delta, homogeneous FP, and conduction targets all pass; physics-informed losses and 5,000 scaling remain deferred.",
        "guardrails": {
            "validation_only_calibration": True,
            "grouped_split_preserved": True,
            "no_target_inputs": True,
            "no_condition_or_metadata_extraction_inputs": True,
            "controlled_synthetic_benchmark_only": True,
            "no_physics_informed_losses": True,
            "no_5000_scaleup": True,
        },
    }
    gate["success"] = bool(gate["balanced_pass"] and gate["homogeneous_fp_pass"] and gate["conduction_pass"])
    write_json(GATE_JSON, gate)
    payload = {
        "results_csv": str(RESULTS_CSV),
        "calibration_csv": str(CALIBRATION_CSV),
        "predictions_csv": str(PREDICTIONS_CSV),
        "gate_json": str(GATE_JSON),
        "models": sorted(index["model"].unique().tolist()),
        "candidate_families": sorted({c["family"] for c in candidate_grid()}),
        "overall_balanced_validation_selection": selections["overall_balanced"],
        "overall_sparse_validation_selection": selections["overall_sparse"],
        "overall_balanced_test": balanced,
        "overall_sparse_test": sparse,
        "conduction_slowing_balanced": conduction,
        "figures": [
            "figures/neural_v1e_500/center_mask_postprocess_best_worst_masks.png",
            "figures/neural_v1e_500/center_mask_postprocess_sparse_vs_balanced_masks.png",
            "figures/neural_v1e_500/center_mask_postprocess_apd90_delta_scatter.png",
            "figures/neural_v1e_500/center_mask_postprocess_dice_by_condition.png",
            "figures/neural_v1e_500/center_mask_postprocess_conduction_failures.png",
            "figures/neural_v1e_500/center_mask_postprocess_center_error_vs_delta_error.png",
        ],
        "calibration_guardrail": "All extraction settings are selected using validation rows only. Test labels are used only after selection for final grouped-test evaluation.",
        "extraction_input_guardrail": "Mask extraction uses deployable model outputs only: patch probability, center heatmap, signed distance, predicted center, predicted area/radius, and patch-present probability.",
    }
    write_json(RESULTS_JSON, payload)
    print(json.dumps({"results": str(RESULTS_CSV), "gate": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
