                                                                 

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
from v1d_field_common import NEURAL_DIR, READY_DIR, ROOT, scalar_targets_from_fields, write_json

torch = import_torch_prefer_env(PROJECT_ROOT)
from torch.utils.data import DataLoader, Dataset


FIELD_TARGETS = ["LAT", "APD80", "APD90"]


@dataclass
class FieldStats:
    input_mean: float
    input_std: float
    target_mean: np.ndarray
    target_std: np.ndarray


def load_index() -> pd.DataFrame:
    path = READY_DIR / "sample_index.csv"
    rows = pd.read_csv(path)
    split_path = READY_DIR / "splits" / "grouped_train_val_test.csv"
    if "split" not in rows.columns and split_path.exists():
        rows = rows.merge(pd.read_csv(split_path)[["field_sample_id", "split"]], on="field_sample_id", how="left")
    return rows


def check_no_leakage(rows: pd.DataFrame) -> list[str]:
    leaked = []
    for base_id, sub in rows.groupby("base_sample_id"):
        if sub["split"].nunique() > 1:
            leaked.append(base_id)
    return leaked


def compute_stats(rows: pd.DataFrame, max_rows: int | None = None) -> FieldStats:
    train = rows[rows["split"].eq("train")].reset_index(drop=True)
    if max_rows:
        train = train.head(max_rows)
    total = 0.0
    total_sq = 0.0
    count = 0
    target_vals = [[] for _ in FIELD_TARGETS]
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
            tv = arr[valid_grid & np.isfinite(arr)]
            target_vals[i].append(tv)
    mean = total / max(count, 1)
    var = max(total_sq / max(count, 1) - mean * mean, 0.0)
    std = float(np.sqrt(var)) if var > 0 else 1.0
    tmean = np.zeros(3, dtype=np.float32)
    tstd = np.ones(3, dtype=np.float32)
    for i, chunks in enumerate(target_vals):
        vals = np.concatenate(chunks) if chunks else np.array([0.0])
        tmean[i] = float(np.mean(vals))
        tstd[i] = float(np.std(vals))
        if tstd[i] < 1e-8:
            tstd[i] = 1.0
    return FieldStats(mean, std, tmean, tstd)


class FNOFieldDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        stats: FieldStats,
        input_norm: str = "per_sample",
        include_coords: bool = True,
        include_sensor_mask_channel: bool = False,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.stats = stats
        self.input_norm = input_norm
        self.include_coords = include_coords
        self.include_sensor_mask_channel = include_sensor_mask_channel

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
            keep = 4 if self.include_sensor_mask_channel else 3
            coord_ch = coords[:, :, :keep].transpose(2, 0, 1)
            x = np.concatenate([x, coord_ch], axis=0)
        return x.astype(np.float32)

    def _load_y(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        lat = np.load(ROOT / row["LAT_grid_file"]).astype(np.float32)
        apd80 = np.load(ROOT / row["APD80_grid_file"]).astype(np.float32)
        apd90 = np.load(ROOT / row["APD90_grid_file"]).astype(np.float32)
        patch = np.load(ROOT / row["patch_mask_grid_file"]).astype(np.float32)
        valid = np.load(ROOT / row["valid_mask_grid_file"]).astype(bool)
        fields = np.stack([lat, apd80, apd90], axis=0)
        scaled = (fields - self.stats.target_mean[:, None, None]) / self.stats.target_std[:, None, None]
        scaled = np.where(np.isfinite(scaled), scaled, 0.0).astype(np.float32)
        y = np.concatenate([scaled, patch[None, :, :]], axis=0)
        scalars = scalar_targets_from_fields(lat, apd90, patch, valid)
        return y, valid.astype(np.float32), patch.astype(np.float32), scalars

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        y, valid, patch, scalars = self._load_y(row)
        return {
            "x": torch.from_numpy(self._load_x(row)),
            "y": torch.from_numpy(y),
            "valid_mask": torch.from_numpy(valid),
            "patch_mask": torch.from_numpy(patch),
            "condition": torch.tensor(int(row.get("condition_label", -1)) if "condition_label" in row else -1, dtype=torch.long),
            "lat_range": torch.tensor(float(scalars["LAT_range"]), dtype=torch.float32),
            "apd90_patch_delta": torch.tensor(float(scalars["APD90_patch_delta"]) if np.isfinite(scalars["APD90_patch_delta"]) else np.nan, dtype=torch.float32),
            "patch_area_fraction": torch.tensor(float(scalars["patch_area_fraction"]), dtype=torch.float32),
            "field_sample_id": row["field_sample_id"],
            "base_sample_id": row["base_sample_id"],
            "measurement_variant": row["measurement_variant"],
            "condition_name": row["condition"],
        }


def make_loader(rows: pd.DataFrame, stats: FieldStats, batch_size: int, shuffle: bool, input_norm: str = "per_sample") -> DataLoader:
    return DataLoader(FNOFieldDataset(rows, stats, input_norm=input_norm), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def save_stats(path: Path, stats: FieldStats) -> None:
    write_json(
        path,
        {
            "input_mean": stats.input_mean,
            "input_std": stats.input_std,
            "target_mean": stats.target_mean.tolist(),
            "target_std": stats.target_std.tolist(),
        },
    )


def load_stats(path: Path) -> FieldStats:
    data = json.loads(path.read_text())
    return FieldStats(float(data["input_mean"]), float(data["input_std"]), np.array(data["target_mean"], dtype=np.float32), np.array(data["target_std"], dtype=np.float32))

