                                                                     

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fno_field_data_common import FNOFieldDataset, FieldStats, load_index, torch
from v1d_anomaly_common import dice_loss_logits, finite_regression
from v1d_field_common import NEURAL_DIR, PILOT_DIR, ROOT, scalar_targets_from_fields


HYBRID_DIR = PILOT_DIR / "hybrid_targets"
HYBRID_MANIFEST = PILOT_DIR / "hybrid_target_manifest.csv"
HYBRID_QUALITY = PILOT_DIR / "hybrid_target_quality_report.json"
HYBRID_STATS_JSON = NEURAL_DIR / "hybrid_localizer_train_stats.json"


@dataclass(frozen=True)
class HybridLossConfig:
    name: str
    geom_bce_weight: float = 1.0
    geom_dice_weight: float = 1.0
    focal_weight: float = 0.0
    hom_negative_weight: float = 0.0
    apd_aux_weight: float = 0.0
    lat_aux_weight: float = 0.0
    field_aux_weight: float = 0.08
    area_calibration_weight: float = 0.0


HYBRID_CONFIGS = [
    HybridLossConfig("geometric_only", field_aux_weight=0.06),
    HybridLossConfig("geometric_plus_homogeneous_negative", hom_negative_weight=2.0, field_aux_weight=0.06),
    HybridLossConfig("geometric_plus_apd_lat_aux", apd_aux_weight=0.35, lat_aux_weight=0.35, field_aux_weight=0.06),
    HybridLossConfig("geometric_plus_aux_plus_area_calibration", apd_aux_weight=0.35, lat_aux_weight=0.35, area_calibration_weight=0.20, field_aux_weight=0.06),
    HybridLossConfig("full_hybrid", hom_negative_weight=2.0, apd_aux_weight=0.35, lat_aux_weight=0.35, area_calibration_weight=0.20, focal_weight=0.25, field_aux_weight=0.06),
]


def load_hybrid_manifest() -> pd.DataFrame:
    return pd.read_csv(HYBRID_MANIFEST)


def load_rows_with_hybrid() -> pd.DataFrame:
    rows = load_index()
    manifest = load_hybrid_manifest()
    keep = [
        "base_sample_id",
        "geometric_patch_mask_file",
        "apd_anomaly_soft_file_hybrid",
        "lat_anomaly_soft_file_hybrid",
        "combined_anomaly_soft_file_hybrid",
        "homogeneous_negative_mask_file",
        "hybrid_target_metadata_file",
    ]
    return rows.merge(manifest[keep], on="base_sample_id", how="left")


class HybridDataset(torch.utils.data.Dataset):
    def __init__(self, rows: pd.DataFrame, stats: FieldStats, input_norm: str = "per_sample") -> None:
        self.rows = rows.reset_index(drop=True)
        self.base_ds = FNOFieldDataset(self.rows, stats, input_norm=input_norm)
        self.stats = stats

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        base = self.base_ds[idx]
        geom = np.load(ROOT / row["geometric_patch_mask_file"]).astype(np.float32)
        apd = np.load(ROOT / row["apd_anomaly_soft_file_hybrid"]).astype(np.float32)
        lat = np.load(ROOT / row["lat_anomaly_soft_file_hybrid"]).astype(np.float32)
        combined = np.load(ROOT / row["combined_anomaly_soft_file_hybrid"]).astype(np.float32)
        negative = np.load(ROOT / row["homogeneous_negative_mask_file"]).astype(np.float32)
        return {
            **base,
            "geometric_patch": torch.from_numpy(geom),
            "hybrid_aux": torch.from_numpy(np.stack([apd, lat, combined], axis=0)),
            "homogeneous_negative": torch.from_numpy(negative),
        }


def make_hybrid_loader(rows: pd.DataFrame, stats: FieldStats, batch_size: int, shuffle: bool, input_norm: str = "per_sample"):
    return torch.utils.data.DataLoader(HybridDataset(rows, stats, input_norm=input_norm), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def estimate_geom_pos_weight(rows: pd.DataFrame) -> float:
    pos = 0.0
    neg = 0.0
    for row in rows.itertuples(index=False):
        geom = np.load(ROOT / row.geometric_patch_mask_file).astype(bool)
        valid = np.load(ROOT / row.valid_mask_grid_file).astype(bool)
        pos += float(np.sum(geom & valid))
        neg += float(np.sum(~geom & valid))
    return float(np.clip(neg / max(pos, 1.0), 1.0, 30.0))


def focal_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, prob, 1.0 - prob)
    focal = ((1.0 - pt) ** gamma) * bce
    return torch.sum(focal * valid) / (torch.sum(valid) + 1e-6)


def hybrid_loss(pred: torch.Tensor, batch: dict, cfg: HybridLossConfig, pos_weight: float, stats: FieldStats) -> torch.Tensor:
    valid = batch["valid_mask"].to(pred.device).float()
    geom = batch["geometric_patch"].to(pred.device).float()
    aux = batch["hybrid_aux"].to(pred.device).float()
    negative = batch["homogeneous_negative"].to(pred.device).float()
    y = batch["y"].to(pred.device).float()
    cond = batch["condition"].to(pred.device)
    pw = torch.tensor(pos_weight, device=pred.device, dtype=pred.dtype)

    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred[:, 0], geom, reduction="none", pos_weight=pw)
    total = cfg.geom_bce_weight * torch.sum(bce * valid) / (torch.sum(valid) + 1e-6)
    total = total + cfg.geom_dice_weight * dice_loss_logits(pred[:, 0], geom, valid)
    if cfg.focal_weight > 0:
        total = total + cfg.focal_weight * focal_bce_with_logits(pred[:, 0], geom, valid)

    if cfg.hom_negative_weight > 0:
        hom = cond == 0
        if torch.any(hom):
            prob = torch.sigmoid(pred[hom, 0])
            neg = negative[hom] * valid[hom]
            total = total + cfg.hom_negative_weight * torch.sum(prob * neg) / (torch.sum(neg) + 1e-6)

    if cfg.apd_aux_weight > 0:
        weights = valid * (((cond == 1) | (cond == 3)).float()[:, None, None] + 0.15)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred[:, 1], aux[:, 0], reduction="none")
        total = total + cfg.apd_aux_weight * torch.sum(loss * weights) / (torch.sum(weights) + 1e-6)
    if cfg.lat_aux_weight > 0:
        weights = valid * (((cond == 2) | (cond == 3)).float()[:, None, None] + 0.15)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred[:, 2], aux[:, 1], reduction="none")
        total = total + cfg.lat_aux_weight * torch.sum(loss * weights) / (torch.sum(weights) + 1e-6)

    if cfg.area_calibration_weight > 0:
        pred_area = torch.sum(torch.sigmoid(pred[:, 0]) * valid, dim=(1, 2)) / (torch.sum(valid, dim=(1, 2)) + 1e-6)
        true_area = torch.sum(geom * valid, dim=(1, 2)) / (torch.sum(valid, dim=(1, 2)) + 1e-6)
        total = total + cfg.area_calibration_weight * torch.nn.functional.smooth_l1_loss(pred_area, true_area, beta=0.02)

    if pred.shape[1] >= 5 and cfg.field_aux_weight > 0:
        lat_loss = torch.nn.functional.smooth_l1_loss(pred[:, 3], y[:, 0], reduction="none", beta=0.75)
        apd_loss = torch.nn.functional.smooth_l1_loss(pred[:, 4], y[:, 2], reduction="none", beta=0.75)
        total = total + cfg.field_aux_weight * torch.sum((lat_loss + apd_loss) * valid) / (torch.sum(valid) + 1e-6)
    return total


def binary_metrics(true: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    t = (true > 0.5) & valid
    p = (pred > 0.5) & valid
    tp = float(np.sum(t & p))
    fp = float(np.sum(~t & p & valid))
    fn = float(np.sum(t & ~p))
    dice = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
    centroid_error = math.nan
    if np.any(t) and np.any(p):
        ty, tx = np.argwhere(t).mean(axis=0)
        py, px = np.argwhere(p).mean(axis=0)
        centroid_error = float(np.sqrt((tx - px) ** 2 + (ty - py) ** 2))
    true_area = float(np.mean(t[valid])) if np.any(valid) else math.nan
    pred_area = float(np.mean(p[valid])) if np.any(valid) else math.nan
    return {
        "dice": dice,
        "iou": tp / max(tp + fp + fn, 1.0),
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
        "centroid_error": centroid_error,
        "true_area_fraction": true_area,
        "pred_area_fraction": pred_area,
        "area_error": pred_area - true_area if np.isfinite(pred_area) and np.isfinite(true_area) else math.nan,
        "false_positive_area": fp / max(float(np.sum(valid)), 1.0),
    }


def evaluate_hybrid_model(model, rows: pd.DataFrame, stats: FieldStats, args, device, model_name: str, config_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows = []
    pred_rows = []
    for split in ["train", "val", "test"]:
        split_rows = rows[rows["split"].eq(split)].reset_index(drop=True)
        loader = make_hybrid_loader(split_rows, stats, args.batch_size, False, args.input_norm)
        per_sample = []
        offset = 0
        model.eval()
        with torch.no_grad():
            for batch in loader:
                out = model(batch["x"].to(device).float()).cpu().numpy()
                y = batch["y"].numpy()
                valid = batch["valid_mask"].numpy().astype(bool)
                geom = batch["geometric_patch"].numpy() > 0.5
                prob = 1.0 / (1.0 + np.exp(-out[:, 0]))
                for b in range(out.shape[0]):
                    src = split_rows.iloc[offset + b]
                    pred_mask = prob[b] > 0.5
                    mm = binary_metrics(geom[b], pred_mask, valid[b])
                    true_lat = y[b, 0] * stats.target_std[0] + stats.target_mean[0]
                    true_apd = y[b, 2] * stats.target_std[2] + stats.target_mean[2]
                    pred_lat = out[b, 3] * stats.target_std[0] + stats.target_mean[0] if out.shape[1] >= 5 else true_lat
                    pred_apd = out[b, 4] * stats.target_std[2] + stats.target_mean[2] if out.shape[1] >= 5 else true_apd
                    true_scalars = scalar_targets_from_fields(true_lat, true_apd, geom[b], valid[b])
                    pred_true_apd = scalar_targets_from_fields(true_lat, true_apd, pred_mask, valid[b])
                    pred_pred_apd = scalar_targets_from_fields(pred_lat, pred_apd, pred_mask, valid[b])
                    true_mask_pred_apd = scalar_targets_from_fields(pred_lat, pred_apd, geom[b], valid[b])
                    soft = prob[b]
                    v = valid[b] & np.isfinite(pred_apd)
                    in_w = soft * v
                    out_w = (1.0 - soft) * v
                    soft_delta = math.nan
                    if in_w.sum() > 1e-6 and out_w.sum() > 1e-6:
                        soft_delta = float((pred_apd * in_w).sum() / in_w.sum() - (pred_apd * out_w).sum() / out_w.sum())
                    per_sample.append(
                        {
                            "model": model_name,
                            "config": config_name,
                            "split": split,
                            "field_sample_id": src["field_sample_id"],
                            "base_sample_id": src["base_sample_id"],
                            "condition": src["condition"],
                            "measurement_variant": src["measurement_variant"],
                            **mm,
                            "true_APD90_patch_delta": true_scalars["APD90_patch_delta"],
                            "true_patch_area_fraction": true_scalars["patch_area_fraction"],
                            "pred_APD90_delta_pred_mask_true_apd": pred_true_apd["APD90_patch_delta"],
                            "pred_APD90_delta_pred_mask_pred_apd": pred_pred_apd["APD90_patch_delta"],
                            "pred_APD90_delta_true_mask_pred_apd": true_mask_pred_apd["APD90_patch_delta"],
                            "pred_patch_area_fraction": pred_true_apd["patch_area_fraction"],
                            "soft_mask_APD90_delta": soft_delta,
                        }
                    )
                offset += out.shape[0]
        pred_df = pd.DataFrame(per_sample)
        pred_rows.extend(per_sample)
        if len(pred_df) == 0:
            continue
        for group_type, col in [("all", None), ("condition", "condition"), ("measurement_variant", "measurement_variant")]:
            groups = [("all", pred_df)] if col is None else list(pred_df.groupby(col))
            for group_value, sub in groups:
                row = {
                    "model": model_name,
                    "config": config_name,
                    "split": split,
                    "group_type": group_type,
                    "group_value": group_value,
                    "n": int(len(sub)),
                    "dice_mean": float(sub["dice"].mean()),
                    "iou_mean": float(sub["iou"].mean()),
                    "precision_mean": float(sub["precision"].mean()),
                    "recall_mean": float(sub["recall"].mean()),
                    "centroid_error_mean": float(sub["centroid_error"].mean(skipna=True)),
                    "area_error_mean": float(sub["area_error"].mean()),
                    "homogeneous_false_positive_area": float(sub[sub["condition"].eq("homogeneous")]["false_positive_area"].mean()) if group_type == "all" else math.nan,
                }
                for target, pred_col in [
                    ("patch_area_fraction", "pred_patch_area_fraction"),
                    ("APD90_delta_pred_mask_true_apd", "pred_APD90_delta_pred_mask_true_apd"),
                    ("APD90_delta_pred_mask_pred_apd", "pred_APD90_delta_pred_mask_pred_apd"),
                    ("APD90_delta_true_mask_pred_apd", "pred_APD90_delta_true_mask_pred_apd"),
                    ("APD90_delta_soft_mask", "soft_mask_APD90_delta"),
                ]:
                    truth = "true_patch_area_fraction" if target == "patch_area_fraction" else "true_APD90_patch_delta"
                    metrics = finite_regression(sub[truth].to_numpy(), sub[pred_col].to_numpy())
                    for k, v in metrics.items():
                        row[f"{target}_{k}"] = v
                result_rows.append(row)
    return pd.DataFrame(result_rows), pd.DataFrame(pred_rows)

