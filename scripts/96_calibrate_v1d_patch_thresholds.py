                      
                                                                          

                                                                               
                                                              
   

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if (PROJECT_ROOT / ".codex_deps" / "sklearn").exists():
    sys.path.append(str(PROJECT_ROOT / ".codex_deps" / "sklearn"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fno_field_data_common import FNOFieldDataset, load_index, load_stats, make_loader, torch
from scripts_78_bridge import FNOFieldReconstructor
from v1d_anomaly_common import ANOMALY_STATS_JSON, finite_regression, make_anomaly_loader
from v1d_field_common import CHECKPOINT_DIR, NEURAL_DIR, NEURAL_FIG_DIR, ROOT, scalar_targets_from_fields, write_json
from v1d_hybrid_common import HYBRID_STATS_JSON, make_hybrid_loader, load_rows_with_hybrid


FIELD_STATS = NEURAL_DIR / "fno_field_train_stats.json"
PATCHAWARE_STATS = NEURAL_DIR / "patchaware_fno_train_stats.json"
CALIB_CSV = NEURAL_DIR / "patch_threshold_calibration.csv"
CALIB_JSON = NEURAL_DIR / "patch_threshold_calibration.json"
TEST_CSV = NEURAL_DIR / "patch_threshold_test_metrics.csv"
SOFT_CSV = NEURAL_DIR / "soft_mask_scalar_metrics.csv"
SPATIAL_CSV = NEURAL_DIR / "patch_probability_spatial_audit.csv"
THRESHOLDS = np.round(np.linspace(0.01, 0.99, 99), 2)


@dataclass
class ModelSpec:
    name: str
    kind: str
    checkpoint: Path
    config: str = ""
    stats_path: Path = FIELD_STATS


MODEL_SPECS = [
    ModelSpec("original_fno_field", "field", CHECKPOINT_DIR / "fno_field_reconstructor.pt", "original", FIELD_STATS),
    ModelSpec("patchaware_fno_scalar_aux", "field", CHECKPOINT_DIR / "patchaware_fno_patch_mask_apd_plus_scalar_aux.pt", "patch_mask_apd_plus_scalar_aux", PATCHAWARE_STATS),
    ModelSpec("anomaly_fno", "anomaly", CHECKPOINT_DIR / "fno_anomaly_localizer.pt", "anomaly", ANOMALY_STATS_JSON),
    ModelSpec("hybrid_fno_geometric_only", "hybrid", CHECKPOINT_DIR / "fno_hybrid_localizer_geometric_only.pt", "geometric_only", HYBRID_STATS_JSON),
    ModelSpec("hybrid_fno_apd_lat_aux", "hybrid", CHECKPOINT_DIR / "fno_hybrid_localizer_geometric_plus_apd_lat_aux.pt", "geometric_plus_apd_lat_aux", HYBRID_STATS_JSON),
    ModelSpec("hybrid_fno_full", "hybrid", CHECKPOINT_DIR / "fno_hybrid_localizer_full_hybrid.pt", "full_hybrid", HYBRID_STATS_JSON),
]


def import_class(path: Path, class_name: str):
    spec = importlib.util.spec_from_file_location(f"_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def load_model(spec: ModelSpec, in_channels: int, device):
    ckpt = torch.load(spec.checkpoint, map_location="cpu")
    args = ckpt.get("args", {})
    if spec.kind == "field":
        model = FNOFieldReconstructor(
            in_channels,
            int(args.get("width", 48)),
            int(args.get("modes", 6)),
            int(args.get("layers", 4)),
            float(args.get("dropout", 0.08)),
        )
    elif spec.kind == "anomaly":
        cls = import_class(ROOT / "scripts" / "87_train_fno_anomaly_localizer_v1d.py", "FNOAnomalyLocalizer")
        model = cls(in_channels, int(args.get("width", 48)), int(args.get("modes", 6)), int(args.get("layers", 4)), float(args.get("dropout", 0.08)))
    elif spec.kind == "hybrid":
        cls = import_class(ROOT / "scripts" / "92_train_fno_hybrid_localizer_v1d.py", "FNOHybridLocalizer")
        model = cls(in_channels, int(args.get("width", 48)), int(args.get("modes", 6)), int(args.get("layers", 4)), float(args.get("dropout", 0.08)))
    else:
        raise ValueError(spec.kind)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def base_truth(row) -> dict:
    valid = np.load(ROOT / row.valid_mask_grid_file).astype(bool)
    patch = np.load(ROOT / row.patch_mask_grid_file).astype(np.float32) > 0.5
    lat = np.load(ROOT / row.LAT_grid_file).astype(np.float32)
    apd90 = np.load(ROOT / row.APD90_grid_file).astype(np.float32)
    return {"valid": valid, "patch": patch, "lat": lat, "apd90": apd90}


def infer_model(spec: ModelSpec, device) -> pd.DataFrame:
    if spec.kind == "hybrid":
        rows = load_rows_with_hybrid()
        stats = load_stats(spec.stats_path)
        sample = next(iter(make_hybrid_loader(rows.head(1), stats, 1, False, "per_sample")))
        loader = make_hybrid_loader(rows, stats, batch_size=16, shuffle=False, input_norm="per_sample")
    elif spec.kind == "anomaly":
        from v1d_anomaly_common import load_rows_with_anomaly

        rows = load_rows_with_anomaly()
        stats = load_stats(spec.stats_path)
        sample = next(iter(make_anomaly_loader(rows.head(1), stats, 1, False, "per_sample")))
        loader = make_anomaly_loader(rows, stats, batch_size=16, shuffle=False, input_norm="per_sample")
    else:
        rows = load_index()
        stats = load_stats(spec.stats_path if spec.stats_path.exists() else FIELD_STATS)
        sample = next(iter(make_loader(rows.head(1), stats, 1, False, "per_sample")))
        loader = make_loader(rows, stats, batch_size=16, shuffle=False, input_norm="per_sample")
    model = load_model(spec, int(sample["x"].shape[1]), device)
    records = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            out = model(batch["x"].to(device).float()).cpu().numpy()
            y = batch["y"].numpy()
            prob = sigmoid(out[:, 3] if spec.kind == "field" else out[:, 0])
            if spec.kind == "field":
                pred_apd = out[:, 2] * stats.target_std[2] + stats.target_mean[2]
            elif out.shape[1] >= 5:
                pred_apd = out[:, 4] * stats.target_std[2] + stats.target_mean[2]
            else:
                pred_apd = np.full_like(prob, np.nan)
            for b in range(prob.shape[0]):
                src = rows.iloc[offset + b]
                truth = base_truth(src)
                records.append(
                    {
                        "model": spec.name,
                        "config": spec.config,
                        "split": src["split"],
                        "field_sample_id": src["field_sample_id"],
                        "base_sample_id": src["base_sample_id"],
                        "condition": src["condition"],
                        "measurement_variant": src["measurement_variant"],
                        "prob": prob[b].astype(np.float32),
                        "pred_apd90": pred_apd[b].astype(np.float32),
                        **truth,
                    }
                )
            offset += prob.shape[0]
    return pd.DataFrame(records)


def binary_metrics(true: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    t = true & valid
    p = pred & valid
    tp = float(np.sum(t & p))
    fp = float(np.sum(~t & p & valid))
    fn = float(np.sum(t & ~p))
    true_area = float(np.mean(t[valid])) if np.any(valid) else math.nan
    pred_area = float(np.mean(p[valid])) if np.any(valid) else math.nan
    centroid = math.nan
    if np.any(t) and np.any(p):
        ty, tx = np.argwhere(t).mean(axis=0)
        py, px = np.argwhere(p).mean(axis=0)
        centroid = float(np.sqrt((tx - px) ** 2 + (ty - py) ** 2))
    return {
        "dice": 2.0 * tp / max(2.0 * tp + fp + fn, 1.0),
        "iou": tp / max(tp + fp + fn, 1.0),
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
        "true_area_fraction": true_area,
        "pred_area_fraction": pred_area,
        "area_bias": pred_area - true_area if np.isfinite(pred_area) and np.isfinite(true_area) else math.nan,
        "area_abs_error": abs(pred_area - true_area) if np.isfinite(pred_area) and np.isfinite(true_area) else math.nan,
        "centroid_error": centroid,
        "false_positive_area": fp / max(float(np.sum(valid)), 1.0),
    }


def patch_delta(apd: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> float:
    v = valid & np.isfinite(apd)
    inside = mask & v
    outside = (~mask) & v
    if not np.any(inside) or not np.any(outside):
        return math.nan
    return float(np.mean(apd[inside]) - np.mean(apd[outside]))


def soft_patch_delta(apd: np.ndarray, prob: np.ndarray, valid: np.ndarray) -> float:
    v = valid & np.isfinite(apd)
    if not np.any(v):
        return math.nan
    p = np.clip(prob, 0.0, 1.0) * v
    q = (1.0 - np.clip(prob, 0.0, 1.0)) * v
    if p.sum() <= 1e-6 or q.sum() <= 1e-6:
        return math.nan
    return float(np.sum(apd * p) / np.sum(p) - np.sum(apd * q) / np.sum(q))


def topk_mask(prob: np.ndarray, valid: np.ndarray, area_fraction: float) -> np.ndarray:
    out = np.zeros_like(valid, dtype=bool)
    idx = np.flatnonzero(valid)
    if idx.size == 0 or area_fraction <= 0:
        return out
    k = int(np.clip(round(area_fraction * idx.size), 1, idx.size))
    vals = prob.ravel()[idx]
    chosen = idx[np.argsort(vals)[-k:]]
    out.ravel()[chosen] = True
    return out


def aggregate(records: list[dict]) -> dict[str, float]:
    if not records:
        return {}
    df = pd.DataFrame(records)
    out = {
        "n": int(len(df)),
        "dice": float(df["dice"].mean()),
        "iou": float(df["iou"].mean()),
        "precision": float(df["precision"].mean()),
        "recall": float(df["recall"].mean()),
        "homogeneous_false_positive_area": float(df[df["condition"].eq("homogeneous")]["false_positive_area"].mean()) if (df["condition"].eq("homogeneous")).any() else math.nan,
        "patch_area_mae": float(df["area_abs_error"].mean()),
        "area_bias": float(df["area_bias"].mean()),
        "centroid_error": float(df["centroid_error"].mean(skipna=True)),
    }
    out.update({f"patch_area_fraction_{k}": v for k, v in finite_regression(df["true_area"].to_numpy(), df["pred_area"].to_numpy()).items()})
    out.update({f"APD90_patch_delta_true_apd_{k}": v for k, v in finite_regression(df["true_delta"].to_numpy(), df["pred_delta_true_apd"].to_numpy()).items()})
    out.update({f"APD90_patch_delta_pred_apd_{k}": v for k, v in finite_regression(df["true_delta"].to_numpy(), df["pred_delta_pred_apd"].to_numpy()).items()})
    return out


def evaluate_threshold(df: pd.DataFrame, threshold_by_key: dict[str, float], key_fn) -> tuple[dict, list[dict]]:
    rows = []
    for row in df.itertuples(index=False):
        key = key_fn(row)
        thr = threshold_by_key.get(key, threshold_by_key.get("all", 0.5))
        pred = row.prob >= thr
        m = binary_metrics(row.patch, pred, row.valid)
        true_scalars = scalar_targets_from_fields(row.lat, row.apd90, row.patch, row.valid)
        rows.append(
            {
                "model": row.model,
                "split": row.split,
                "condition": row.condition,
                "measurement_variant": row.measurement_variant,
                "threshold": thr,
                **m,
                "true_area": true_scalars["patch_area_fraction"],
                "pred_area": m["pred_area_fraction"],
                "true_delta": true_scalars["APD90_patch_delta"],
                "pred_delta_true_apd": patch_delta(row.apd90, pred, row.valid),
                "pred_delta_pred_apd": patch_delta(row.pred_apd90, pred, row.valid),
            }
        )
    return aggregate(rows), rows


def key_for_scheme(row, scheme: str) -> str:
    if scheme == "global":
        return "all"
    if scheme == "per_condition":
        return str(row.condition)
    if scheme == "per_variant":
        return str(row.measurement_variant)
    if scheme == "per_condition_variant":
        return f"{row.condition}||{row.measurement_variant}"
    raise ValueError(scheme)


def calibrate_model(df: pd.DataFrame, scheme: str) -> tuple[list[dict], dict[str, float]]:
    val = df[df["split"].eq("val")].reset_index(drop=True)
    if scheme == "global":
        groups = {"all": val}
    else:
        keys = val.apply(lambda r: key_for_scheme(r, scheme), axis=1)
        groups = {k: val.loc[keys == k].reset_index(drop=True) for k in sorted(keys.unique())}
    cal_rows = []
    selected = {}
    for group_key, sub in groups.items():
        best = None
        for thr in THRESHOLDS:
            metrics, _ = evaluate_threshold(sub, {"all": float(thr)}, lambda _r: "all")
            score = metrics.get("dice", -999) - metrics.get("homogeneous_false_positive_area", 0 if group_key != "all" else 0) - 0.2 * metrics.get("patch_area_mae", 0)
            score += 0.05 * max(metrics.get("APD90_patch_delta_true_apd_r2", -5), -5)
            row = {"model": df["model"].iloc[0], "scheme": scheme, "group_key": group_key, "threshold": float(thr), "selection_score": float(score), **metrics}
            cal_rows.append(row)
            if best is None or score > best["selection_score"]:
                best = row
        selected[group_key] = float(best["threshold"])
    return cal_rows, selected


def area_priors(df: pd.DataFrame, scheme: str) -> dict[str, float]:
    val = df[df["split"].eq("val")]
    priors = {}
    keys = val.apply(lambda r: key_for_scheme(r, scheme), axis=1)
    for key in sorted(keys.unique()):
        sub = val.loc[keys == key]
        areas = [float(np.mean((r.patch & r.valid))) for r in sub.itertuples(index=False)]
        priors[key] = float(np.mean(areas)) if areas else 0.0
    priors["all"] = float(np.mean([np.mean(r.patch & r.valid) for r in val.itertuples(index=False)]))
    return priors


def eval_soft_methods(df: pd.DataFrame, selected_scheme: str, selected_thresholds: dict[str, float]) -> list[dict]:
    test = df[df["split"].eq("test")].reset_index(drop=True)
    priors_global = area_priors(df, "global")
    priors_condition = area_priors(df, "per_condition")
    priors_cv = area_priors(df, "per_condition_variant")
    methods = ["binary_calibrated", "soft_probability", "topk_global_area_prior", "topk_condition_area_prior", "topk_condition_variant_area_prior"]
    rows = []
    for method in methods:
        recs = []
        for row in test.itertuples(index=False):
            if method == "binary_calibrated":
                mask = row.prob >= selected_thresholds.get(key_for_scheme(row, selected_scheme), selected_thresholds.get("all", 0.5))
            elif method == "soft_probability":
                mask = None
            elif method == "topk_global_area_prior":
                mask = topk_mask(row.prob, row.valid, priors_global["all"])
            elif method == "topk_condition_area_prior":
                mask = topk_mask(row.prob, row.valid, priors_condition.get(key_for_scheme(row, "per_condition"), priors_global["all"]))
            else:
                mask = topk_mask(row.prob, row.valid, priors_cv.get(key_for_scheme(row, "per_condition_variant"), priors_condition.get(key_for_scheme(row, "per_condition"), priors_global["all"])))
            true_scalars = scalar_targets_from_fields(row.lat, row.apd90, row.patch, row.valid)
            if method == "soft_probability":
                delta_true = soft_patch_delta(row.apd90, row.prob, row.valid)
                delta_pred = soft_patch_delta(row.pred_apd90, row.prob, row.valid)
                pred_area = float(np.mean(np.clip(row.prob[row.valid], 0, 1))) if np.any(row.valid) else math.nan
                dice = math.nan
            else:
                bm = binary_metrics(row.patch, mask, row.valid)
                delta_true = patch_delta(row.apd90, mask, row.valid)
                delta_pred = patch_delta(row.pred_apd90, mask, row.valid)
                pred_area = bm["pred_area_fraction"]
                dice = bm["dice"]
            recs.append({"true_delta": true_scalars["APD90_patch_delta"], "delta_true_apd": delta_true, "delta_pred_apd": delta_pred, "true_area": true_scalars["patch_area_fraction"], "pred_area": pred_area, "dice": dice})
        rdf = pd.DataFrame(recs)
        out = {"model": df["model"].iloc[0], "method": method, "dice_mean": float(rdf["dice"].mean(skipna=True))}
        out.update({f"APD90_patch_delta_true_apd_{k}": v for k, v in finite_regression(rdf["true_delta"].to_numpy(), rdf["delta_true_apd"].to_numpy()).items()})
        out.update({f"APD90_patch_delta_pred_apd_{k}": v for k, v in finite_regression(rdf["true_delta"].to_numpy(), rdf["delta_pred_apd"].to_numpy()).items()})
        out.update({f"patch_area_fraction_{k}": v for k, v in finite_regression(rdf["true_area"].to_numpy(), rdf["pred_area"].to_numpy()).items()})
        rows.append(out)
    return rows


def spatial_audit(df: pd.DataFrame) -> list[dict]:
    rows = []
    for row in df[df["split"].eq("test")].itertuples(index=False):
        valid = row.valid
        patch = row.patch & valid
        prob = np.clip(row.prob, 0, 1)
        mass_valid = float(np.sum(prob[valid]))
        inside_mass = float(np.sum(prob[patch]))
        outside_mass = float(np.sum(prob[valid & ~patch]))
        centroid_error = math.nan
        if np.any(patch) and mass_valid > 1e-9:
            yy, xx = np.mgrid[0 : prob.shape[0], 0 : prob.shape[1]]
            ty, tx = np.argwhere(patch).mean(axis=0)
            py = float(np.sum(yy[valid] * prob[valid]) / mass_valid)
            px = float(np.sum(xx[valid] * prob[valid]) / mass_valid)
            centroid_error = float(np.sqrt((tx - px) ** 2 + (ty - py) ** 2))
        rows.append(
            {
                "model": row.model,
                "field_sample_id": row.field_sample_id,
                "base_sample_id": row.base_sample_id,
                "condition": row.condition,
                "measurement_variant": row.measurement_variant,
                "prob_centroid_error": centroid_error,
                "prob_mass_inside_true_patch": inside_mass,
                "prob_mass_outside_true_patch": outside_mass,
                "prob_mass_inside_fraction": inside_mass / max(mass_valid, 1e-9),
                "prob_mean_inside_true_patch": float(np.mean(prob[patch])) if np.any(patch) else math.nan,
                "prob_mean_outside_true_patch": float(np.mean(prob[valid & ~patch])) if np.any(valid & ~patch) else math.nan,
            }
        )
    return rows


def plot_outputs(calib: pd.DataFrame, test_metrics: pd.DataFrame, soft: pd.DataFrame, spatial: pd.DataFrame, prob_frames: dict[str, pd.DataFrame], best: dict) -> None:
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    global_cal = calib[calib["scheme"].eq("global")]
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, sub in global_cal.groupby("model"):
        ax.plot(sub["threshold"], sub["dice"], label=model)
    ax.set_xlabel("threshold")
    ax.set_ylabel("validation Dice")
    ax.set_title("Threshold Sweep Dice")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "threshold_sweep_dice.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, sub in global_cal.groupby("model"):
        ax.plot(sub["threshold"], sub["homogeneous_false_positive_area"], label=model)
    ax.set_xlabel("threshold")
    ax.set_ylabel("homogeneous false-positive area")
    ax.set_title("Threshold Sweep False Positive Area")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "threshold_sweep_false_positive_area.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sub = soft[soft["method"].isin(["binary_calibrated", "soft_probability", "topk_condition_area_prior"])]
    for method, chunk in sub.groupby("method"):
        ax.scatter(chunk["model"], chunk["APD90_patch_delta_true_apd_r2"], label=method)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("APD90 patch-delta R2 using true APD90")
    ax.set_title("Soft vs Binary Patch Delta")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "soft_vs_binary_patch_delta.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    summary = spatial.groupby("model")[["prob_mean_inside_true_patch", "prob_mean_outside_true_patch"]].mean().reset_index()
    x = np.arange(len(summary))
    ax.bar(x - 0.18, summary["prob_mean_inside_true_patch"], width=0.36, label="inside")
    ax.bar(x + 0.18, summary["prob_mean_outside_true_patch"], width=0.36, label="outside")
    ax.set_xticks(x, summary["model"], rotation=35, ha="right")
    ax.set_ylabel("mean probability")
    ax.set_title("Probability Mass Inside vs Outside Patch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "probability_mass_inside_outside_patch.png", dpi=180)
    plt.close(fig)

                                                          
    best_row = test_metrics.sort_values("dice", ascending=False).head(1)
    if len(best_row):
        model = best_row["model"].iloc[0]
        scheme = best_row["scheme"].iloc[0]
        frame = prob_frames[model]
        examples = frame[(frame["split"].eq("test")) & (~frame["condition"].eq("homogeneous"))].head(4)
        fig, axes = plt.subplots(len(examples), 4, figsize=(10, 2.5 * max(1, len(examples))))
        if len(examples) == 1:
            axes = axes[None, :]
        for i, row in enumerate(examples.itertuples(index=False)):
            key = key_for_scheme(row, scheme)
            thr = best[model][scheme].get(key, best[model][scheme].get("all", 0.5))
            pred = row.prob >= thr
            for j, (arr, title, cmap) in enumerate([(row.apd90, "APD90", "viridis"), (row.patch, "true patch", "magma"), (row.prob, "probability", "magma"), (pred, "calibrated mask", "magma")]):
                im = axes[i, j].imshow(arr, origin="lower", cmap=cmap)
                axes[i, j].set_title(f"{row.base_sample_id}\n{title}", fontsize=8)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.02)
        fig.tight_layout()
        fig.savefig(NEURAL_FIG_DIR / "calibrated_mask_examples.png", dpi=180)
        plt.close(fig)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is expected for checkpoint inference.")
    device = torch.device("cuda")
    prob_frames = {}
    for spec in MODEL_SPECS:
        if not spec.checkpoint.exists():
            print(f"Skipping missing checkpoint: {spec.checkpoint}", flush=True)
            continue
        print(f"Recomputing probabilities for {spec.name}", flush=True)
        prob_frames[spec.name] = infer_model(spec, device)

    all_calib = []
    best_thresholds: dict[str, dict[str, dict[str, float]]] = {}
    validation_scheme_scores: dict[str, dict[str, dict[str, float]]] = {}
    test_rows = []
    soft_rows = []
    spatial_rows = []
    for model, frame in prob_frames.items():
        best_thresholds[model] = {}
        validation_scheme_scores[model] = {}
        for scheme in ["global", "per_condition", "per_variant", "per_condition_variant"]:
            cal_rows, selected = calibrate_model(frame, scheme)
            all_calib.extend(cal_rows)
            best_thresholds[model][scheme] = selected
            val_metrics, _ = evaluate_threshold(frame[frame["split"].eq("val")].reset_index(drop=True), selected, lambda r, s=scheme: key_for_scheme(r, s))
            val_score = val_metrics.get("dice", -999) - val_metrics.get("homogeneous_false_positive_area", 0) - 0.2 * val_metrics.get("patch_area_mae", 0)
            val_score += 0.05 * max(val_metrics.get("APD90_patch_delta_true_apd_r2", -5), -5)
            validation_scheme_scores[model][scheme] = {"selection_score": float(val_score), **val_metrics}
            metrics, _ = evaluate_threshold(frame[frame["split"].eq("test")].reset_index(drop=True), selected, lambda r, s=scheme: key_for_scheme(r, s))
            test_rows.append({"model": model, "scheme": scheme, **metrics})
        best_scheme = max(validation_scheme_scores[model], key=lambda s: validation_scheme_scores[model][s]["selection_score"])
        soft_rows.extend(eval_soft_methods(frame, best_scheme, best_thresholds[model][best_scheme]))
        spatial_rows.extend(spatial_audit(frame))

    calib_df = pd.DataFrame(all_calib)
    test_df = pd.DataFrame(test_rows)
    soft_df = pd.DataFrame(soft_rows)
    spatial_df = pd.DataFrame(spatial_rows)
    calib_df.to_csv(CALIB_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    soft_df.to_csv(SOFT_CSV, index=False)
    spatial_df.to_csv(SPATIAL_CSV, index=False)
    plot_outputs(calib_df, test_df, soft_df, spatial_df, prob_frames, best_thresholds)
    best_summary = test_df.sort_values("dice", ascending=False).groupby("model").head(1).to_dict(orient="records")
    payload = {
        "best_thresholds": best_thresholds,
        "validation_scheme_scores": validation_scheme_scores,
        "best_test_by_model": best_summary,
        "outputs": {
            "calibration_csv": str(CALIB_CSV.relative_to(ROOT)),
            "test_metrics_csv": str(TEST_CSV.relative_to(ROOT)),
            "soft_mask_scalar_metrics_csv": str(SOFT_CSV.relative_to(ROOT)),
            "spatial_audit_csv": str(SPATIAL_CSV.relative_to(ROOT)),
        },
        "notes": "Thresholds are selected on validation only. Test data is not used for threshold selection.",
    }
    write_json(CALIB_JSON, payload)
    print(json.dumps({"best_test_by_model": best_summary, "json": str(CALIB_JSON)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
