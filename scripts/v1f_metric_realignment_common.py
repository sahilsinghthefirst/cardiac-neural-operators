                                                                        

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from v1e_500_common import PILOT_DIR, ROOT, SEED, write_json


DEEPO_READY = PILOT_DIR / "deeponet_ready"
V1F_OUTPUT_DIR = ROOT / "outputs" / "v1f_metric_realignment"
V1F_TARGET_DIR = PILOT_DIR / "v1f_multichannel_targets"
V1F_FIG_DIR = ROOT / "figures" / "v1f_metric_realignment"

CONDITION_ORDER = [
    "homogeneous",
    "apd_repolarization_perturbation",
    "conduction_slowing_perturbation",
    "mixed_apd_conduction_perturbation",
]
CONDITION_SHORT = {
    "homogeneous": "hom",
    "apd_repolarization_perturbation": "APD",
    "conduction_slowing_perturbation": "cond",
    "mixed_apd_conduction_perturbation": "mixed",
}


def ensure_dirs() -> None:
    V1F_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    V1F_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    V1F_FIG_DIR.mkdir(parents=True, exist_ok=True)


def rel(path: Path | str) -> str:
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_absolute() else str(path)


def load_index() -> pd.DataFrame:
    rows = pd.read_csv(DEEPO_READY / "sample_index.csv")
    rows["split"] = rows["split"].astype(str)
    return rows


def unique_base_rows(rows: pd.DataFrame | None = None) -> pd.DataFrame:
    if rows is None:
        rows = load_index()
    sort_variant = rows["combined_variant"].eq("combined_all_available").astype(int)
    ordered = rows.assign(_preferred=sort_variant).sort_values(["base_sample_id", "_preferred"], ascending=[True, False])
    return ordered.drop_duplicates("base_sample_id", keep="first").drop(columns=["_preferred"]).reset_index(drop=True)


def read_json(path: str | Path) -> dict:
    return json.loads((ROOT / path).read_text())


def load_grid(row: pd.Series, col: str) -> np.ndarray:
    return np.load(ROOT / row[col]).astype(np.float32)


def r2_score_finite(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 2 or np.allclose(y[m], y[m][0]):
        return math.nan
    ss_res = float(np.sum((y[m] - p[m]) ** 2))
    ss_tot = float(np.sum((y[m] - np.mean(y[m])) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan


def regression_summary(y_true, y_pred) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if not np.any(m):
        return {"n": 0, "mae": math.nan, "rmse": math.nan, "r2": math.nan}
    d = y[m] - p[m]
    return {
        "n": int(m.sum()),
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "r2": r2_score_finite(y[m], p[m]),
    }


def patch_delta(field: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> float:
    v = valid.astype(bool) & np.isfinite(field)
    inside = (mask > 0.5) & v
    outside = (mask <= 0.5) & v
    if not inside.any() or not outside.any():
        return math.nan
    return float(np.mean(field[inside]) - np.mean(field[outside]))


def activation_duration(lat: np.ndarray, valid: np.ndarray) -> float:
    vals = lat[valid.astype(bool) & np.isfinite(lat)]
    if vals.size == 0:
        return math.nan
    return float(np.percentile(vals, 95) - np.percentile(vals, 5))


def fit_plane(values: np.ndarray, valid: np.ndarray, background: np.ndarray) -> np.ndarray:
    h, w = values.shape
    yy, xx = np.indices(values.shape)
    x = xx.astype(np.float64) / max(w - 1, 1)
    y = yy.astype(np.float64) / max(h - 1, 1)
    m = valid.astype(bool) & background.astype(bool) & np.isfinite(values)
    if int(m.sum()) < 4:
        med = float(np.nanmedian(values[valid.astype(bool)])) if np.any(valid) else 0.0
        return np.full(values.shape, med, dtype=np.float32)
    design = np.column_stack([np.ones(int(m.sum())), x[m], y[m]])
    coef, *_ = np.linalg.lstsq(design, values[m].astype(np.float64), rcond=None)
    plane = coef[0] + coef[1] * x + coef[2] * y
    return plane.astype(np.float32)


def conduction_proxy_from_lat(lat: np.ndarray, valid: np.ndarray, patch: np.ndarray) -> dict[str, np.ndarray | float]:
    lat_filled = np.asarray(lat, dtype=np.float32).copy()
    if np.any(valid):
        lat_filled[~np.isfinite(lat_filled)] = float(np.nanmedian(lat_filled[valid.astype(bool)]))
    dy = dx = 1.0 / max(lat.shape[0] - 1, 1)
    gy, gx = np.gradient(lat_filled, dy, dx)
    grad_mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    v = valid.astype(bool) & np.isfinite(grad_mag)
    p = (patch > 0.5) & v
    out = (patch <= 0.5) & v
    bg = float(np.nanmedian(grad_mag[out])) if out.any() else float(np.nanmedian(grad_mag[v])) if v.any() else 0.0
    proxy_map = np.where(v, grad_mag - bg, 0.0).astype(np.float32)
    cv_proxy = np.where(v, 1.0 / np.maximum(grad_mag, 1e-6), np.nan)
    return {
        "lat_gradient_magnitude": grad_mag,
        "local_conduction_slowing_proxy": proxy_map,
        "conduction_velocity_proxy_global": float(np.nanmedian(cv_proxy[v])) if v.any() else math.nan,
        "conduction_velocity_proxy_patch": float(np.nanmedian(cv_proxy[p])) if p.any() else math.nan,
        "conduction_severity_scalar": float(np.nanmedian(grad_mag[p]) - bg) if p.any() and out.any() else math.nan,
    }


def anomaly_support(anomaly: np.ndarray, valid: np.ndarray, *, positive_only: bool = True) -> np.ndarray:
    v = valid.astype(bool) & np.isfinite(anomaly)
    vals = anomaly[v]
    if vals.size == 0:
        return np.zeros_like(valid, dtype=bool)
    if positive_only and np.nanmax(vals) > 1e-6:
        pos = vals[vals > 0]
        if pos.size == 0:
            return np.zeros_like(valid, dtype=bool)
        threshold = max(float(np.percentile(pos, 75)) * 0.5, float(np.nanmax(pos)) * 0.15, 1e-6)
        return (anomaly >= threshold) & v
    mag = np.abs(anomaly)
    mag_vals = mag[v]
    threshold = max(float(np.percentile(mag_vals, 90)) * 0.5, float(np.nanmax(mag_vals)) * 0.15, 1e-6)
    return (mag >= threshold) & v


def anomaly_delta(field: np.ndarray, anomaly: np.ndarray, valid: np.ndarray) -> float:
    support = anomaly_support(anomaly, valid)
    return patch_delta(field, support.astype(np.float32), valid)


def compute_physiology(row: pd.Series) -> dict:
    lat = load_grid(row, "LAT_grid_file")
    apd90 = load_grid(row, "APD90_grid_file")
    patch = load_grid(row, "patch_mask_grid_file")
    valid = load_grid(row, "valid_mask_grid_file").astype(bool)
    condition = str(row["condition"])
    patch_bool = (patch > 0.5) & valid
    background = valid & ~patch_bool

    if condition == "homogeneous" or not patch_bool.any():
        apd_anomaly = np.zeros_like(apd90, dtype=np.float32)
        lat_anomaly = np.zeros_like(lat, dtype=np.float32)
    else:
        apd_plane = fit_plane(apd90, valid, background)
        lat_plane = fit_plane(lat, valid, background)
        apd_anomaly = np.where(valid, apd90 - apd_plane, 0.0).astype(np.float32)
        lat_anomaly = np.where(valid, lat - lat_plane, 0.0).astype(np.float32)

    cond_proxy = conduction_proxy_from_lat(lat, valid, patch_bool)
    params = read_json(row["patch_param_targets_file"])
    scalars = {
        "base_sample_id": row["base_sample_id"],
        "split": row["split"],
        "condition": condition,
        "patch_present": int(params.get("patch_present", int(patch_bool.any()))),
        "patch_area_fraction": float(params.get("patch_area_fraction", float(np.mean(patch_bool[valid])) if valid.any() else math.nan)),
        "equivalent_radius_norm": float(params.get("equivalent_radius_norm", math.nan)) if params.get("equivalent_radius_norm") is not None else math.nan,
        "center_x_norm": float(params.get("center_x_norm", math.nan)) if params.get("center_x_norm") is not None else math.nan,
        "center_y_norm": float(params.get("center_y_norm", math.nan)) if params.get("center_y_norm") is not None else math.nan,
        "APD90_patch_delta": patch_delta(apd90, patch_bool, valid),
        "LAT_patch_delay": patch_delta(lat, patch_bool, valid),
        "activation_duration": activation_duration(lat, valid),
        "APD90_valid_mean": float(np.nanmean(apd90[valid])) if valid.any() else math.nan,
        "APD90_valid_std": float(np.nanstd(apd90[valid])) if valid.any() else math.nan,
        "LAT_valid_mean": float(np.nanmean(lat[valid])) if valid.any() else math.nan,
        "LAT_valid_std": float(np.nanstd(lat[valid])) if valid.any() else math.nan,
        "APD_anomaly_mean_abs_valid": float(np.nanmean(np.abs(apd_anomaly[valid]))) if valid.any() else math.nan,
        "APD_anomaly_mean_abs_patch": float(np.nanmean(np.abs(apd_anomaly[patch_bool]))) if patch_bool.any() else math.nan,
        "LAT_anomaly_mean_abs_valid": float(np.nanmean(np.abs(lat_anomaly[valid]))) if valid.any() else math.nan,
        "LAT_anomaly_mean_abs_patch": float(np.nanmean(np.abs(lat_anomaly[patch_bool]))) if patch_bool.any() else math.nan,
        "APD_anomaly_delta": anomaly_delta(apd90, apd_anomaly, valid),
        "LAT_anomaly_delay": anomaly_delta(lat, lat_anomaly, valid),
        "conduction_velocity_proxy_global": float(cond_proxy["conduction_velocity_proxy_global"]),
        "conduction_velocity_proxy_patch": float(cond_proxy["conduction_velocity_proxy_patch"]),
        "conduction_severity_scalar": float(cond_proxy["conduction_severity_scalar"]),
        "condition_label_reporting_only": int(params.get("condition_label", -1)),
    }
    return {
        "LAT": lat,
        "APD90": apd90,
        "patch": patch_bool.astype(np.float32),
        "valid": valid.astype(np.float32),
        "apd_anomaly": apd_anomaly,
        "lat_anomaly": lat_anomaly,
        "local_conduction_slowing_proxy": cond_proxy["local_conduction_slowing_proxy"],
        "lat_gradient_magnitude": cond_proxy["lat_gradient_magnitude"],
        "scalars": scalars,
    }


def center_heatmap_and_distance(patch: np.ndarray, valid: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    patch_bool = (patch > 0.5) & valid.astype(bool)
    h, w = patch.shape
    if not patch_bool.any():
        z = np.zeros((h, w), dtype=np.float32)
        boundary = np.where(valid.astype(bool), 1.0, 0.0).astype(np.float32)
        return z, z, boundary, math.nan, math.nan, math.nan
    yy, xx = np.indices(patch.shape)
    cx = float(params.get("center_x_norm", np.mean(xx[patch_bool]) / max(w - 1, 1))) * max(w - 1, 1)
    cy = float(params.get("center_y_norm", np.mean(yy[patch_bool]) / max(h - 1, 1))) * max(h - 1, 1)
    area = float(params.get("patch_area_fraction", np.mean(patch_bool[valid.astype(bool)])))
    radius_norm = math.sqrt(max(area, 1e-8) / math.pi)
    sigma = max(radius_norm * max(h, w) * 0.45, 1.5)
    heat = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    heat = np.where(valid.astype(bool), heat, 0.0).astype(np.float32)

    dist_in = ndimage.distance_transform_edt(patch_bool)
    dist_out = ndimage.distance_transform_edt(~patch_bool)
    signed_px = dist_in - dist_out
    signed = np.where(valid.astype(bool), signed_px / max(h, w), 0.0).astype(np.float32)
    boundary = np.where(valid.astype(bool), 1.0 + 4.0 * np.exp(-np.abs(signed_px) / 2.0), 0.0).astype(np.float32)
    return heat, signed, boundary, cx, cy, sigma


def build_multichannel_targets_for_row(row: pd.Series) -> dict:
    phys = compute_physiology(row)
    base = str(row["base_sample_id"])
    out_dir = V1F_TARGET_DIR / base
    out_dir.mkdir(parents=True, exist_ok=True)
    params = read_json(row["patch_param_targets_file"])
    heat, signed, boundary, cx, cy, sigma = center_heatmap_and_distance(phys["patch"], phys["valid"], params)
    files = {
        "apd_anomaly_target_file": out_dir / "apd_anomaly_target.npy",
        "lat_anomaly_target_file": out_dir / "lat_anomaly_target.npy",
        "generic_patch_mask_file": out_dir / "generic_patch_mask.npy",
        "center_heatmap_file": out_dir / "center_heatmap.npy",
        "signed_distance_target_file": out_dir / "signed_distance_target.npy",
        "boundary_weight_map_file": out_dir / "boundary_weight_map.npy",
        "valid_mask_file": out_dir / "valid_mask.npy",
        "local_conduction_slowing_proxy_file": out_dir / "local_conduction_slowing_proxy.npy",
    }
    np.save(files["apd_anomaly_target_file"], phys["apd_anomaly"].astype(np.float32))
    np.save(files["lat_anomaly_target_file"], phys["lat_anomaly"].astype(np.float32))
    np.save(files["generic_patch_mask_file"], phys["patch"].astype(np.float32))
    np.save(files["center_heatmap_file"], heat.astype(np.float32))
    np.save(files["signed_distance_target_file"], signed.astype(np.float32))
    np.save(files["boundary_weight_map_file"], boundary.astype(np.float32))
    np.save(files["valid_mask_file"], phys["valid"].astype(np.float32))
    np.save(files["local_conduction_slowing_proxy_file"], phys["local_conduction_slowing_proxy"].astype(np.float32))
    scalar_path = out_dir / "scalar_targets.json"
    scalar_payload = dict(phys["scalars"])
    scalar_payload.update(
        {
            "target_note": "These physiology targets are labels/reporting metadata only, not model inputs.",
            "condition_label_usage": "Condition label is for reporting and possible future loss weighting only.",
        }
    )
    write_json(scalar_path, scalar_payload)
    manifest = {
        "base_sample_id": base,
        "split": row["split"],
        "condition": row["condition"],
        "patch_present": phys["scalars"]["patch_present"],
        "center_x_px": cx,
        "center_y_px": cy,
        "center_sigma_px": sigma,
        "scalar_targets_file": rel(scalar_path),
    }
    manifest.update({k: rel(v) for k, v in files.items()})
    return manifest


def condition_summary(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for cond in CONDITION_ORDER:
        sub = df[df["condition"].eq(cond)]
        if sub.empty:
            continue
        row = {"condition": cond, "n": int(len(sub))}
        for col in value_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(float)
            vals = vals[np.isfinite(vals)]
            row[f"{col}_mean"] = float(np.mean(vals)) if vals.size else math.nan
            row[f"{col}_median"] = float(np.median(vals)) if vals.size else math.nan
            row[f"{col}_std"] = float(np.std(vals)) if vals.size else math.nan
            row[f"{col}_p10"] = float(np.percentile(vals, 10)) if vals.size else math.nan
            row[f"{col}_p90"] = float(np.percentile(vals, 90)) if vals.size else math.nan
        rows.append(row)
    return pd.DataFrame(rows)
