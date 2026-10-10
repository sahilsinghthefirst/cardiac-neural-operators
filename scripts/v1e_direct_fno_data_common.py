                                                                             

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from torch_import_utils import import_torch_prefer_env
from v1e_direct_common import NEURAL_DIR, READY_DIR, ROOT, scalar_targets_from_fields, write_json

torch = import_torch_prefer_env(PROJECT_ROOT)
from torch.utils.data import DataLoader, Dataset


FIELD_TARGETS = ["LAT", "APD80", "APD90"]
PARAM_TARGETS = ["patch_present", "patch_center_x_norm", "patch_center_y_norm", "patch_area_fraction", "patch_radius_equiv_norm"]


@dataclass
class V1EStats:
    input_mean: float
    input_std: float
    target_mean: np.ndarray
    target_std: np.ndarray
    param_mean: np.ndarray
    param_std: np.ndarray


def load_index() -> pd.DataFrame:
    return pd.read_csv(READY_DIR / "sample_index.csv")


def check_no_leakage(rows: pd.DataFrame) -> list[str]:
    leaked = []
    for base_id, sub in rows.groupby("base_sample_id"):
        if sub["split"].nunique() > 1:
            leaked.append(base_id)
    return leaked


def load_param_targets(path: Path) -> dict:
    return json.loads(path.read_text())


def compute_stats(rows: pd.DataFrame, max_rows: int | None = None) -> V1EStats:
    train = rows[rows["split"].eq("train")].reset_index(drop=True)
    if max_rows:
        train = train.head(max_rows)
    total = 0.0
    total_sq = 0.0
    count = 0
    target_vals = [[] for _ in FIELD_TARGETS]
    param_vals = [[] for _ in PARAM_TARGETS]
    for row in train.itertuples(index=False):
        b = np.load(ROOT / row.B_meas_file).astype(np.float64)
        mask = np.load(ROOT / row.sensor_mask_file).astype(bool)
        valid = mask[None, :, :, :].repeat(b.shape[0], axis=0)
        vals = b[valid]
        vals = vals[np.isfinite(vals)]
        total += float(vals.sum())
        total_sq += float(np.square(vals).sum())
        count += int(vals.size)
        valid_grid = np.load(ROOT / row.valid_mask_grid_file).astype(bool)
        for i, col in enumerate(["LAT_grid_file", "APD80_grid_file", "APD90_grid_file"]):
            arr = np.load(ROOT / getattr(row, col)).astype(np.float64)
            target_vals[i].append(arr[valid_grid & np.isfinite(arr)])
        params = load_param_targets(ROOT / row.patch_param_targets_file)
        for i, key in enumerate(PARAM_TARGETS):
            val = params.get(key)
            if val is not None and np.isfinite(val):
                param_vals[i].append(float(val))
    mean = total / max(count, 1)
    var = max(total_sq / max(count, 1) - mean * mean, 0.0)
    std = float(np.sqrt(var)) if var > 0 else 1.0
    tmean = np.zeros(3, dtype=np.float32)
    tstd = np.ones(3, dtype=np.float32)
    for i, chunks in enumerate(target_vals):
        vals = np.concatenate(chunks) if chunks else np.array([0.0])
        tmean[i] = float(np.mean(vals))
        tstd[i] = max(float(np.std(vals)), 1e-6)
    pmean = np.zeros(len(PARAM_TARGETS), dtype=np.float32)
    pstd = np.ones(len(PARAM_TARGETS), dtype=np.float32)
    for i, vals in enumerate(param_vals):
        arr = np.asarray(vals, dtype=np.float32) if vals else np.array([0.0], dtype=np.float32)
        pmean[i] = float(np.mean(arr))
        pstd[i] = max(float(np.std(arr)), 1e-6)
    return V1EStats(mean, std, tmean, tstd, pmean, pstd)


class V1EDirectDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, stats: V1EStats, input_norm: str = "per_sample", include_coords: bool = True) -> None:
        self.rows = rows.reset_index(drop=True)
        self.stats = stats
        self.input_norm = input_norm
        self.include_coords = include_coords

    def __len__(self) -> int:
        return len(self.rows)

    def _load_x(self, row: pd.Series) -> np.ndarray:
        b = np.load(ROOT / row["B_meas_file"]).astype(np.float32)
        mask = np.load(ROOT / row["sensor_mask_file"]).astype(bool)
        valid = mask[None, :, :, :].repeat(b.shape[0], axis=0)
        if self.input_norm == "per_sample":
            vals = b[valid]
            mean = float(np.nanmean(vals))
            std = float(np.nanstd(vals))
        elif self.input_norm == "train_global":
            mean = self.stats.input_mean
            std = self.stats.input_std
        else:
            mean = 0.0
            std = 1.0
        if not np.isfinite(std) or std <= 1e-20:
            std = 1.0
        b = (b - mean) / std
        b = np.where(valid, b, 0.0)
        b = np.where(np.isfinite(b), b, 0.0)
        x = b.transpose(3, 0, 1, 2).reshape(b.shape[0] * b.shape[-1], b.shape[1], b.shape[2])
        if self.include_coords:
            coords = np.load(ROOT / row["coordinate_channels_file"]).astype(np.float32)
            x = np.concatenate([x, coords[:, :, :3].transpose(2, 0, 1)], axis=0)
        return x.astype(np.float32)

    def _load_y(self, row: pd.Series):
        lat = np.load(ROOT / row["LAT_grid_file"]).astype(np.float32)
        apd80 = np.load(ROOT / row["APD80_grid_file"]).astype(np.float32)
        apd90 = np.load(ROOT / row["APD90_grid_file"]).astype(np.float32)
        patch = np.load(ROOT / row["patch_mask_grid_file"]).astype(np.float32)
        valid = np.load(ROOT / row["valid_mask_grid_file"]).astype(bool)
        fields = np.stack([lat, apd80, apd90], axis=0)
        scaled = (fields - self.stats.target_mean[:, None, None]) / self.stats.target_std[:, None, None]
        scaled = np.where(np.isfinite(scaled), scaled, 0.0).astype(np.float32)
        y_fields = np.concatenate([scaled, patch[None, :, :]], axis=0)
        params = load_param_targets(ROOT / row["patch_param_targets_file"])
        y_param = np.array([params.get(k, np.nan) for k in PARAM_TARGETS], dtype=np.float32)
        param_mask = np.isfinite(y_param)
        y_scaled = (np.where(param_mask, y_param, self.stats.param_mean) - self.stats.param_mean) / self.stats.param_std
        y_scaled[0] = np.nan_to_num(y_param[0], nan=0.0)                                 
        param_mask[0] = True
        scalars = scalar_targets_from_fields(lat, apd90, patch, valid)
        return y_fields, valid.astype(np.float32), patch, y_scaled.astype(np.float32), param_mask.astype(bool), scalars

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        y_fields, valid, patch, y_param, param_mask, scalars = self._load_y(row)
        return {
            "x": torch.from_numpy(self._load_x(row)),
            "y_fields": torch.from_numpy(y_fields),
            "valid_mask": torch.from_numpy(valid),
            "patch_mask": torch.from_numpy(patch.astype(np.float32)),
            "y_param": torch.from_numpy(y_param),
            "param_mask": torch.from_numpy(param_mask),
            "condition": torch.tensor(int(row["condition_label"]), dtype=torch.long),
            "lat_range": torch.tensor(float(scalars["LAT_range"]), dtype=torch.float32),
            "apd90_patch_delta": torch.tensor(float(scalars["APD90_patch_delta"]) if np.isfinite(scalars["APD90_patch_delta"]) else np.nan, dtype=torch.float32),
            "patch_area_fraction": torch.tensor(float(scalars["patch_area_fraction"]), dtype=torch.float32),
            "row_id": row["row_id"],
            "base_sample_id": row["base_sample_id"],
            "measurement_variant": row["measurement_variant"],
            "condition_name": row["condition"],
        }


def make_loader(rows: pd.DataFrame, stats: V1EStats, batch_size: int, shuffle: bool, input_norm: str = "per_sample") -> DataLoader:
    return DataLoader(V1EDirectDataset(rows, stats, input_norm=input_norm), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def save_stats(path: Path, stats: V1EStats) -> None:
    write_json(
        path,
        {
            "input_mean": stats.input_mean,
            "input_std": stats.input_std,
            "target_mean": stats.target_mean.tolist(),
            "target_std": stats.target_std.tolist(),
            "param_mean": stats.param_mean.tolist(),
            "param_std": stats.param_std.tolist(),
        },
    )
