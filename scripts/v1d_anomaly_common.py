                                                                        

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fno_field_data_common import FNOFieldDataset, FieldStats, compute_stats, load_index, load_stats, save_stats, torch
from v1d_field_common import NEURAL_DIR, PILOT_DIR, READY_DIR, ROOT, scalar_targets_from_fields, write_json


ANOMALY_DIR = PILOT_DIR / "anomaly_targets"
ANOMALY_MANIFEST = PILOT_DIR / "anomaly_target_manifest.csv"
ANOMALY_QUALITY = PILOT_DIR / "anomaly_target_quality_report.json"
ANOMALY_STATS_JSON = NEURAL_DIR / "anomaly_localizer_train_stats.json"


def robust_plane_residual(field: np.ndarray, valid: np.ndarray) -> np.ndarray:
                                                                                 
    h, w = field.shape
    yy, xx = np.mgrid[0:h, 0:w]
    mask = valid & np.isfinite(field)
    if np.sum(mask) < 4:
        return np.zeros_like(field, dtype=np.float32)
    x = (xx[mask].astype(np.float64) / max(w - 1, 1)) * 2.0 - 1.0
    y = (yy[mask].astype(np.float64) / max(h - 1, 1)) * 2.0 - 1.0
    z = field[mask].astype(np.float64)
    a = np.column_stack([np.ones_like(x), x, y])
    keep = np.ones_like(z, dtype=bool)
    for _ in range(3):
        coef, *_ = np.linalg.lstsq(a[keep], z[keep], rcond=None)
        pred = a @ coef
        resid = z - pred
        med = np.median(resid[keep])
        mad = np.median(np.abs(resid[keep] - med)) + 1e-6
        keep = np.abs(resid - med) < 3.5 * 1.4826 * mad
        if np.sum(keep) < 4:
            keep = np.ones_like(z, dtype=bool)
            break
    coef, *_ = np.linalg.lstsq(a[keep], z[keep], rcond=None)
    x_all = (xx.astype(np.float64) / max(w - 1, 1)) * 2.0 - 1.0
    y_all = (yy.astype(np.float64) / max(h - 1, 1)) * 2.0 - 1.0
    baseline = coef[0] + coef[1] * x_all + coef[2] * y_all
    out = field.astype(np.float64) - baseline
    out[~valid] = 0.0
    return out.astype(np.float32)


def soft_from_residual(residual: np.ndarray, valid: np.ndarray, positive_only: bool = False) -> np.ndarray:
    vals = residual[valid & np.isfinite(residual)]
    if vals.size == 0:
        return np.zeros_like(residual, dtype=np.float32)
    if positive_only:
        mag = np.maximum(residual, 0.0)
        scale_vals = np.maximum(vals, 0.0)
    else:
        mag = np.abs(residual)
        scale_vals = np.abs(vals)
    q50 = float(np.nanpercentile(scale_vals, 50))
    q95 = float(np.nanpercentile(scale_vals, 95))
    scale = max(q95 - q50, 1e-6)
    soft = np.clip((mag - q50) / scale, 0.0, 1.0)
    soft[~valid] = 0.0
    return soft.astype(np.float32)


def oriented_soft_from_residual(residual: np.ndarray, valid: np.ndarray, patch: np.ndarray) -> np.ndarray:
                                                                          

                                                                             
                                                                              
                                                                              
                                              
       
    patch_bool = (patch > 0.5) & valid
    if not np.any(patch_bool):
        return soft_from_residual(residual, valid, positive_only=False)
    pos = soft_from_residual(residual, valid, positive_only=True)
    neg = soft_from_residual(-residual, valid, positive_only=True)
    outside = valid & ~patch_bool
    pos_score = float(np.mean(pos[patch_bool]) - np.mean(pos[outside])) if np.any(outside) else float(np.mean(pos[patch_bool]))
    neg_score = float(np.mean(neg[patch_bool]) - np.mean(neg[outside])) if np.any(outside) else float(np.mean(neg[patch_bool]))
    return pos if pos_score >= neg_score else neg


def binary_from_soft(soft: np.ndarray, valid: np.ndarray, target_area_fraction: float | None) -> np.ndarray:
    out = np.zeros_like(soft, dtype=bool)
    mask = valid & np.isfinite(soft)
    if not np.any(mask):
        return out
    vals = soft[mask]
    if target_area_fraction is not None and np.isfinite(target_area_fraction) and target_area_fraction > 0:
        frac = float(np.clip(target_area_fraction, 0.01, 0.30))
        threshold = float(np.quantile(vals, max(0.0, 1.0 - frac)))
        threshold = max(0.25, threshold)
    else:
        threshold = 0.55
    out[mask] = vals >= threshold
    if target_area_fraction and target_area_fraction > 0 and np.sum(out) == 0:
        idx = np.flatnonzero(mask)
        k = max(1, int(round(float(target_area_fraction) * idx.size)))
        top = idx[np.argsort(soft.ravel()[idx])[-k:]]
        out.ravel()[top] = True
    return out


def make_anomaly_targets(lat: np.ndarray, apd90: np.ndarray, patch: np.ndarray, valid: np.ndarray, condition: str) -> dict[str, np.ndarray]:
    patch_area = float(np.mean((patch > 0.5)[valid])) if np.any(valid) else 0.0
    apd_resid = robust_plane_residual(apd90, valid)
    lat_resid = robust_plane_residual(lat, valid)
    apd_soft = oriented_soft_from_residual(apd_resid, valid, patch)
    lat_soft = oriented_soft_from_residual(lat_resid, valid, patch)
    if condition == "homogeneous":
        apd_soft[:] = 0.0
        lat_soft[:] = 0.0
        combined = np.zeros_like(apd_soft, dtype=np.float32)
        binary = np.zeros_like(apd_soft, dtype=bool)
    elif condition == "apd_repolarization_perturbation":
        lat_soft[:] = 0.0
        combined = apd_soft
        binary = binary_from_soft(combined, valid, patch_area)
    elif condition == "conduction_slowing_perturbation":
        apd_soft[:] = 0.0
        combined = lat_soft
        binary = binary_from_soft(combined, valid, patch_area)
    else:
        combined = np.maximum(apd_soft, lat_soft)
        binary = binary_from_soft(combined, valid, patch_area)
    return {
        "apd_anomaly_soft": apd_soft.astype(np.float32),
        "lat_anomaly_soft": lat_soft.astype(np.float32),
        "combined_anomaly_soft": combined.astype(np.float32),
        "anomaly_mask_binary": binary.astype(np.float32),
    }


def dice_iou_precision_recall(true: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    t = (true > 0.5) & valid
    p = (pred > 0.5) & valid
    tp = float(np.sum(t & p))
    fp = float(np.sum(~t & p & valid))
    fn = float(np.sum(t & ~p))
    dice = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
    return {
        "dice": dice,
        "iou": tp / max(tp + fp + fn, 1.0),
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
        "false_positive_area": fp / max(float(np.sum(valid)), 1.0),
        "pred_area_fraction": float(np.mean(p[valid])) if np.any(valid) else math.nan,
        "true_area_fraction": float(np.mean(t[valid])) if np.any(valid) else math.nan,
    }


def finite_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {"mae": math.nan, "rmse": math.nan, "r2": math.nan, "n": 0}
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if y_true.size > 1 and not np.allclose(y_true, y_true[0]) else math.nan,
        "n": int(y_true.size),
    }


def load_anomaly_manifest() -> pd.DataFrame:
    return pd.read_csv(ANOMALY_MANIFEST)


def load_rows_with_anomaly() -> pd.DataFrame:
    rows = load_index()
    manifest = load_anomaly_manifest()
    keep = [
        "base_sample_id",
        "apd_anomaly_soft_file",
        "lat_anomaly_soft_file",
        "combined_anomaly_soft_file",
        "anomaly_mask_binary_file",
        "anomaly_target_metadata_file",
    ]
    return rows.merge(manifest[keep], on="base_sample_id", how="left")


class AnomalyDataset(torch.utils.data.Dataset):
    def __init__(self, rows: pd.DataFrame, stats: FieldStats, input_norm: str = "per_sample") -> None:
        self.rows = rows.reset_index(drop=True)
        self.base_ds = FNOFieldDataset(self.rows, stats, input_norm=input_norm)
        self.stats = stats

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        base = self.base_ds[idx]
        combined = np.load(ROOT / row["combined_anomaly_soft_file"]).astype(np.float32)
        apd = np.load(ROOT / row["apd_anomaly_soft_file"]).astype(np.float32)
        lat = np.load(ROOT / row["lat_anomaly_soft_file"]).astype(np.float32)
        binary = np.load(ROOT / row["anomaly_mask_binary_file"]).astype(np.float32)
        valid = base["valid_mask"].float()
        return {
            **base,
            "anomaly_soft": torch.from_numpy(np.stack([combined, apd, lat], axis=0)),
            "anomaly_binary": torch.from_numpy(binary),
            "valid_mask": valid,
        }


def make_anomaly_loader(rows: pd.DataFrame, stats: FieldStats, batch_size: int, shuffle: bool, input_norm: str = "per_sample"):
    return torch.utils.data.DataLoader(AnomalyDataset(rows, stats, input_norm=input_norm), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def estimate_positive_weight(rows: pd.DataFrame) -> float:
    pos = 0.0
    neg = 0.0
    for row in rows.itertuples(index=False):
        binary = np.load(ROOT / row.anomaly_mask_binary_file).astype(bool)
        valid = np.load(ROOT / row.valid_mask_grid_file).astype(bool)
        pos += float(np.sum(binary & valid))
        neg += float(np.sum(~binary & valid))
    return float(np.clip(neg / max(pos, 1.0), 1.0, 30.0))


def anomaly_loss(pred: torch.Tensor, batch: dict, pos_weight: float, *, aux_field_weight: float = 0.1) -> torch.Tensor:
    valid = batch["valid_mask"].to(pred.device).float()
    soft = batch["anomaly_soft"].to(pred.device).float()
    binary = batch["anomaly_binary"].to(pred.device).float()
    y = batch["y"].to(pred.device).float()
    cond = batch["condition"].to(pred.device)
    pw = torch.tensor(pos_weight, device=pred.device, dtype=pred.dtype)

    total = torch.tensor(0.0, device=pred.device)
                            
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred[:, 0], binary, reduction="none", pos_weight=pw)
    total = total + torch.sum(bce * valid) / (torch.sum(valid) + 1e-6)
    total = total + dice_loss_logits(pred[:, 0], binary, valid)

                                             
    weights_apd = ((cond == 1) | (cond == 3)).float()[:, None, None]
    weights_lat = ((cond == 2) | (cond == 3)).float()[:, None, None]
    weights_hom = (cond == 0).float()[:, None, None]
    apd_weight = valid * (0.25 + weights_apd + weights_hom)
    lat_weight = valid * (0.25 + weights_lat + weights_hom)
    for head, target, weight in [(1, soft[:, 1], apd_weight), (2, soft[:, 2], lat_weight)]:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred[:, head], target, reduction="none")
        total = total + 0.75 * torch.sum(loss * weight) / (torch.sum(weight) + 1e-6)

                                             
    if torch.any(cond == 0):
        hom = cond == 0
        hom_prob = torch.sigmoid(pred[hom, 0])
        hom_valid = valid[hom]
        total = total + 1.5 * torch.sum(hom_prob * hom_valid) / (torch.sum(hom_valid) + 1e-6)

                                                                               
    if pred.shape[1] >= 5 and aux_field_weight > 0:
        lat_loss = torch.nn.functional.smooth_l1_loss(pred[:, 3], y[:, 0], reduction="none", beta=0.75)
        apd_loss = torch.nn.functional.smooth_l1_loss(pred[:, 4], y[:, 2], reduction="none", beta=0.75)
        total = total + aux_field_weight * torch.sum((lat_loss + apd_loss) * valid) / (torch.sum(valid) + 1e-6)
    return total


def dice_loss_logits(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits) * valid
    target = target * valid
    inter = torch.sum(prob * target, dim=(1, 2))
    denom = torch.sum(prob, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
    return torch.mean(1.0 - (2.0 * inter + eps) / (denom + eps))


def evaluate_anomaly_model(model, rows: pd.DataFrame, stats: FieldStats, args, device, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows = []
    pred_rows = []
    for split in ["train", "val", "test"]:
        split_rows = rows[rows["split"].eq(split)].reset_index(drop=True)
        loader = make_anomaly_loader(split_rows, stats, args.batch_size, False, args.input_norm)
        per_sample = []
        offset = 0
        model.eval()
        with torch.no_grad():
            for batch in loader:
                pred = model(batch["x"].to(device).float()).cpu().numpy()
                y = batch["y"].numpy()
                valid = batch["valid_mask"].numpy().astype(bool)
                anomaly_true = batch["anomaly_binary"].numpy()
                prob = 1.0 / (1.0 + np.exp(-pred[:, 0]))
                for b in range(pred.shape[0]):
                    src = split_rows.iloc[offset + b]
                    pred_mask = prob[b] > 0.5
                    true_mask = anomaly_true[b] > 0.5
                    mm = dice_iou_precision_recall(true_mask, pred_mask, valid[b])
                    lat_pred = None
                    apd_pred = None
                    if pred.shape[1] >= 5:
                        lat_pred = pred[b, 3] * stats.target_std[0] + stats.target_mean[0]
                        apd_pred = pred[b, 4] * stats.target_std[2] + stats.target_mean[2]
                    true_lat = y[b, 0] * stats.target_std[0] + stats.target_mean[0]
                    true_apd = y[b, 2] * stats.target_std[2] + stats.target_mean[2]
                    true_patch = y[b, 3] > 0.5
                    true_scalars = scalar_targets_from_fields(true_lat, true_apd, true_patch, valid[b])
                    true_anom_scalars = scalar_targets_from_fields(true_lat, true_apd, true_mask, valid[b])
                    pred_scalars_true_apd = scalar_targets_from_fields(true_lat, true_apd, pred_mask, valid[b])
                    rec = {
                        "model": model_name,
                        "split": split,
                        "field_sample_id": src["field_sample_id"],
                        "base_sample_id": src["base_sample_id"],
                        "condition": src["condition"],
                        "measurement_variant": src["measurement_variant"],
                        **mm,
                        "true_patch_area_fraction": true_scalars["patch_area_fraction"],
                        "true_APD90_patch_delta": true_scalars["APD90_patch_delta"],
                        "pred_area_fraction": pred_scalars_true_apd["patch_area_fraction"],
                        "pred_APD90_delta_pred_mask_true_apd": pred_scalars_true_apd["APD90_patch_delta"],
                        "true_anomaly_area_fraction": true_anom_scalars["patch_area_fraction"],
                    }
                    if apd_pred is not None:
                        pred_scalars_pred_apd = scalar_targets_from_fields(true_lat, apd_pred, pred_mask, valid[b])
                        true_mask_pred_apd = scalar_targets_from_fields(true_lat, apd_pred, true_patch, valid[b])
                        rec["pred_APD90_delta_pred_mask_pred_apd"] = pred_scalars_pred_apd["APD90_patch_delta"]
                        rec["pred_APD90_delta_true_mask_pred_apd"] = true_mask_pred_apd["APD90_patch_delta"]
                    per_sample.append(rec)
                offset += pred.shape[0]
        pred_df = pd.DataFrame(per_sample)
        if len(pred_df) == 0:
            continue
        for group_type, col in [("all", None), ("condition", "condition"), ("measurement_variant", "measurement_variant")]:
            groups = [("all", pred_df)] if col is None else list(pred_df.groupby(col))
            for group_value, sub in groups:
                row = {
                    "model": model_name,
                    "split": split,
                    "group_type": group_type,
                    "group_value": group_value,
                    "n": int(len(sub)),
                    "dice_mean": float(sub["dice"].mean()),
                    "iou_mean": float(sub["iou"].mean()),
                    "precision_mean": float(sub["precision"].mean()),
                    "recall_mean": float(sub["recall"].mean()),
                    "false_positive_area_mean": float(sub["false_positive_area"].mean()),
                }
                for target, pred_col in [
                    ("patch_area_fraction", "pred_area_fraction"),
                    ("APD90_patch_delta_pred_mask_true_apd", "pred_APD90_delta_pred_mask_true_apd"),
                    ("APD90_patch_delta_pred_mask_pred_apd", "pred_APD90_delta_pred_mask_pred_apd"),
                ]:
                    if pred_col in sub.columns:
                        truth = "true_patch_area_fraction" if target == "patch_area_fraction" else "true_APD90_patch_delta"
                        metrics = finite_regression(sub[truth].to_numpy(), sub[pred_col].to_numpy())
                        for k, v in metrics.items():
                            row[f"{target}_{k}"] = v
                result_rows.append(row)
        pred_rows.extend(per_sample)
    return pd.DataFrame(result_rows), pd.DataFrame(pred_rows)


def load_field_model_predictions() -> pd.DataFrame:
    patchaware = NEURAL_DIR / "patchaware_fno_predictions_summary.csv"
    original = NEURAL_DIR / "fno_field_predictions_summary.csv"
    if patchaware.exists():
        df = pd.read_csv(patchaware)
        if "config" in df.columns:
            sub = df[df["config"].eq("patch_mask_apd_plus_scalar_aux")]
            return sub.copy() if len(sub) else df.copy()
        return df.copy()
    return pd.read_csv(original)
