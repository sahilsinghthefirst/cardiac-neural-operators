                                                                         

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if (PROJECT_ROOT / ".local_deps" / "sklearn").exists():
    sys.path.append(str(PROJECT_ROOT / ".local_deps" / "sklearn"))

from v1e_500_common import CHECKPOINT_DIR, NEURAL_DIR, ROOT, SEED
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


CACHE_INDEX = NEURAL_DIR / "deeponet_full_grid_prediction_index.csv"
CACHE_NPZ = NEURAL_DIR / "deeponet_full_grid_prediction_maps.npz"
CACHE_JSON = NEURAL_DIR / "deeponet_full_grid_prediction_cache.json"


def sigmoid_np(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(arr, dtype=np.float64)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    exp_x = np.exp(arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    if np.isscalar(x):
        return float(out)
    return out.astype(np.float32)


def regression_r2(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() < 2 or np.allclose(y[mask], y[mask][0]):
        return math.nan
    ss_res = float(np.sum((y[mask] - p[mask]) ** 2))
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan


def regression_metrics(y, p) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not np.any(mask):
        return {"mae": math.nan, "rmse": math.nan, "r2": math.nan}
    diff = y[mask] - p[mask]
    return {"mae": float(np.mean(np.abs(diff))), "rmse": float(np.sqrt(np.mean(diff * diff))), "r2": regression_r2(y[mask], p[mask])}


def patch_metrics(true_mask: np.ndarray, pred_mask: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    t = (true_mask > 0.5) & valid.astype(bool)
    p = (pred_mask > 0.5) & valid.astype(bool)
    tp = float(np.sum(t & p))
    fp = float(np.sum(~t & p & valid))
    fn = float(np.sum(t & ~p))
    return {
        "dice": (2.0 * tp) / max(2.0 * tp + fp + fn, 1.0),
        "iou": tp / max(tp + fp + fn, 1.0),
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
    }


def scalar_targets(apd90: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> float:
    v = valid.astype(bool)
    inside = (mask > 0.5) & v & np.isfinite(apd90)
    outside = (mask <= 0.5) & v & np.isfinite(apd90)
    return float(np.mean(apd90[inside]) - np.mean(apd90[outside])) if inside.any() and outside.any() else math.nan


def area_fraction(mask: np.ndarray, valid: np.ndarray) -> float:
    v = valid.astype(bool)
    return float(np.mean((mask > 0.5)[v])) if np.any(v) else math.nan


def centroid(mask_or_prob: np.ndarray, valid: np.ndarray, *, weighted: bool) -> tuple[float, float]:
    v = valid.astype(bool)
    yy, xx = np.indices(mask_or_prob.shape)
    if weighted:
        weights = np.where(v, np.clip(mask_or_prob, 0.0, None), 0.0).astype(float)
    else:
        weights = ((mask_or_prob > 0.5) & v).astype(float)
    total = float(np.sum(weights))
    if total <= 0:
        return math.nan, math.nan
    return float(np.sum(xx * weights) / total / max(mask_or_prob.shape[1] - 1, 1)), float(np.sum(yy * weights) / total / max(mask_or_prob.shape[0] - 1, 1))


def centroid_error(pred_prob: np.ndarray, true_mask: np.ndarray, valid: np.ndarray) -> float:
    px, py = centroid(pred_prob, valid, weighted=True)
    tx, ty = centroid(true_mask, valid, weighted=False)
    if not np.isfinite(px) or not np.isfinite(tx):
        return math.nan
    return float(np.sqrt((px - tx) ** 2 + (py - ty) ** 2))


def disk_mask(cx: float, cy: float, radius: float, present: float = 1.0, shape: tuple[int, int] = (32, 32)) -> np.ndarray:
    if present < 0.5 or not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(radius):
        return np.zeros(shape, dtype=np.float32)
    yy, xx = np.indices(shape)
    x = xx / max(shape[1] - 1, 1)
    y = yy / max(shape[0] - 1, 1)
    radius = float(np.clip(radius, 1e-4, 0.75))
    return (((x - np.clip(cx, 0.0, 1.0)) ** 2 + (y - np.clip(cy, 0.0, 1.0)) ** 2) <= radius**2).astype(np.float32)


def topk_mask(score: np.ndarray, valid: np.ndarray, area: float) -> np.ndarray:
    vflat = valid.reshape(-1).astype(bool)
    out = np.zeros(score.size, dtype=np.float32)
    if not np.any(vflat) or not np.isfinite(area) or area <= 0:
        return out.reshape(score.shape)
    k = int(round(float(area) * int(vflat.sum())))
    k = max(1, min(k, int(vflat.sum())))
    valid_idx = np.flatnonzero(vflat)
    vals = score.reshape(-1)[valid_idx]
    order = np.argsort(vals)[-k:]
    out[valid_idx[order]] = 1.0
    return out.reshape(score.shape).astype(np.float32)


def threshold_mask(score: np.ndarray, valid: np.ndarray, threshold: float) -> np.ndarray:
    return ((score >= threshold) & valid.astype(bool)).astype(np.float32)


def soft_delta(apd90: np.ndarray, prob: np.ndarray, valid: np.ndarray) -> float:
    v = valid.astype(bool) & np.isfinite(apd90)
    w_in = np.where(v, np.clip(prob, 0.0, 1.0), 0.0).astype(float)
    w_out = np.where(v, 1.0 - np.clip(prob, 0.0, 1.0), 0.0).astype(float)
    if w_in.sum() <= 1e-12 or w_out.sum() <= 1e-12:
        return math.nan
    return float(np.sum(apd90 * w_in) / np.sum(w_in) - np.sum(apd90 * w_out) / np.sum(w_out))


def load_checkpoint_paths() -> list[Path]:
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
    model.eval()
    return model


def load_or_build_full_grid_cache(splits: tuple[str, ...] = ("val", "test"), batch_size: int = 4, force: bool = False):
                                                              
 
                                                                             
                                                                       
       
    if CACHE_INDEX.exists() and CACHE_NPZ.exists() and not force:
        index = pd.read_csv(CACHE_INDEX)
        maps = dict(np.load(CACHE_NPZ))
        return index, maps

    rows = load_index()
    desc = load_descriptors()
    stats = compute_stats(rows, desc)
    first = next(iter(make_loader(rows.head(1), stats, batch_size=1, shuffle=False, mode="full", descriptors=desc)))
    desc_dim = int(first["descriptor_x"].shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = load_checkpoint_paths()
    if not paths:
        raise FileNotFoundError("No DeepONet checkpoints found")

    records: list[dict] = []
    true_lat = []
    true_apd90 = []
    true_patch = []
    valid_mask = []
    pred_lat = []
    pred_apd90 = []
    pred_patch_prob = []

    for path in paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model_name = str(ckpt.get("model_name", path.stem.replace("deeponet_", "")))
        model = build_model_from_checkpoint(ckpt, desc_dim).to(device)
        for split in splits:
            sub = rows[rows["split"].eq(split)].reset_index(drop=True)
            loader = make_loader(sub, stats, batch_size=batch_size, shuffle=False, mode="full", k_queries=1024, input_norm="per_sample", descriptors=desc, seed=SEED + 909)
            offset = 0
            with torch.no_grad():
                for batch in loader:
                    query, params, area_logits, cond_logits = model(
                        batch["branch_x"].to(device).float(),
                        batch["trunk_xy"].to(device).float(),
                        batch["descriptor_x"].to(device).float(),
                    )
                    query_np = query.detach().cpu().numpy()
                    y_np = batch["y_query"].numpy()
                    valid_np = batch["valid_query_mask"].numpy().astype(bool)
                    params_np = params.detach().cpu().numpy()
                    area_np = area_logits.detach().cpu().numpy()
                    for b in range(query_np.shape[0]):
                        src = sub.iloc[offset + b]
                        y_param = batch["y_param"].numpy()[b]
                        p_mask = batch["param_mask"].numpy()[b].astype(bool)
                        raw_area_radius = batch["raw_area_radius"].numpy()[b]
                        param_unscaled = params_np[b].copy()
                        param_unscaled[1:] = param_unscaled[1:] * stats.param_std[1:] + stats.param_mean[1:]
                        records.append(
                            {
                                "model": model_name,
                                "row_id": src["row_id"],
                                "base_sample_id": src["base_sample_id"],
                                "split": split,
                                "condition": src["condition"],
                                "combined_variant": src["combined_variant"],
                                "true_patch_present": float(y_param[0]),
                                "pred_patch_present_prob": float(sigmoid_np(params_np[b, 0])),
                                "true_center_x_norm": float(y_param[1] * stats.param_std[1] + stats.param_mean[1]) if p_mask[1] else math.nan,
                                "pred_center_x_norm": float(param_unscaled[1]),
                                "true_center_y_norm": float(y_param[2] * stats.param_std[2] + stats.param_mean[2]) if p_mask[2] else math.nan,
                                "pred_center_y_norm": float(param_unscaled[2]),
                                "true_log_area_fraction": float(y_param[3] * stats.param_std[3] + stats.param_mean[3]) if p_mask[3] else math.nan,
                                "pred_log_area_fraction": float(param_unscaled[3]),
                                "true_log_radius": float(y_param[4] * stats.param_std[4] + stats.param_mean[4]) if p_mask[4] else math.nan,
                                "pred_log_radius": float(param_unscaled[4]),
                                "true_patch_area_fraction": float(raw_area_radius[0]) if np.isfinite(raw_area_radius[0]) else math.nan,
                                "pred_patch_area_param": float(np.exp(param_unscaled[3])) if np.isfinite(param_unscaled[3]) else math.nan,
                                "true_radius_param": float(raw_area_radius[1]) if np.isfinite(raw_area_radius[1]) else math.nan,
                                "pred_radius_param": float(np.exp(param_unscaled[4])) if np.isfinite(param_unscaled[4]) else math.nan,
                                "true_area_bin": int(batch["area_bin"].numpy()[b]) if bool(batch["area_bin_mask"].numpy()[b]) else -1,
                                "pred_area_bin": int(np.argmax(area_np[b])),
                            }
                        )
                        true_lat.append((y_np[b, :, 0].reshape(32, 32) * stats.field_std[0] + stats.field_mean[0]).astype(np.float32))
                        true_apd90.append((y_np[b, :, 2].reshape(32, 32) * stats.field_std[2] + stats.field_mean[2]).astype(np.float32))
                        true_patch.append(y_np[b, :, 3].reshape(32, 32).astype(np.float32))
                        valid_mask.append(valid_np[b].reshape(32, 32))
                        pred_lat.append((query_np[b, :, 0].reshape(32, 32) * stats.field_std[0] + stats.field_mean[0]).astype(np.float32))
                        pred_apd90.append((query_np[b, :, 2].reshape(32, 32) * stats.field_std[2] + stats.field_mean[2]).astype(np.float32))
                        pred_patch_prob.append(sigmoid_np(query_np[b, :, 3].reshape(32, 32)).astype(np.float32))
                    offset += query_np.shape[0]
        print(f"cached full-grid maps for {model_name}", flush=True)

    index = pd.DataFrame(records)
    maps = {
        "true_lat": np.stack(true_lat).astype(np.float32),
        "true_apd90": np.stack(true_apd90).astype(np.float32),
        "true_patch": np.stack(true_patch).astype(np.float32),
        "valid_mask": np.stack(valid_mask).astype(bool),
        "pred_lat": np.stack(pred_lat).astype(np.float32),
        "pred_apd90": np.stack(pred_apd90).astype(np.float32),
        "pred_patch_prob": np.stack(pred_patch_prob).astype(np.float32),
    }
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    index.to_csv(CACHE_INDEX, index=False)
    np.savez_compressed(CACHE_NPZ, **maps)
    CACHE_JSON.write_text(
        json.dumps(
            {
                "index": str(CACHE_INDEX),
                "maps": str(CACHE_NPZ),
                "row_count": int(len(index)),
                "models": sorted(index["model"].unique().tolist()),
                "splits": sorted(index["split"].unique().tolist()),
                "leakage_guardrail": "Full-grid cache is generated from checkpoints for analysis only; it is not used as a model input.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return index, maps


def subset_arrays(index: pd.DataFrame, maps: dict[str, np.ndarray], mask: np.ndarray | pd.Series):
    idx = np.flatnonzero(np.asarray(mask))
    return index.iloc[idx].reset_index(drop=True), {k: v[idx] for k, v in maps.items()}
