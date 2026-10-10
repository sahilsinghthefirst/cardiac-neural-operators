                                                                  

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
from v1e_500_common import PILOT_DIR, ROOT, write_json


torch = import_torch_prefer_env(ROOT)
from torch.utils.data import DataLoader, Dataset


DEEPO_READY = PILOT_DIR / "deeponet_ready"
FIELD_NAMES = ["LAT", "APD80", "APD90"]
PARAM_KEYS = ["patch_present", "center_x_norm", "center_y_norm", "log_area_fraction", "log_radius"]
RAW_PARAM_KEYS = ["patch_area_fraction", "equivalent_radius_norm"]


@dataclass
class DeepONetStats:
    input_mean: float
    input_std: float
    field_mean: np.ndarray
    field_std: np.ndarray
    param_mean: np.ndarray
    param_std: np.ndarray
    descriptor_mean: np.ndarray
    descriptor_std: np.ndarray
    descriptor_columns: list[str]


def load_index() -> pd.DataFrame:
    rows = pd.read_csv(DEEPO_READY / "sample_index.csv")
    if "combined_variant" not in rows.columns and "measurement_variant" in rows.columns:
        rows["combined_variant"] = rows["measurement_variant"]
    return rows


def load_descriptors() -> pd.DataFrame:
    return pd.read_csv(DEEPO_READY / "direct_descriptors.csv")


def load_query_points() -> np.ndarray:
    return np.load(DEEPO_READY / "query_points_32x32.npy").astype(np.float32)


def check_no_leakage(rows: pd.DataFrame) -> list[str]:
    return [base for base, sub in rows.groupby("base_sample_id") if sub["split"].nunique() > 1]


def load_params(path: str | Path) -> dict:
    return json.loads((ROOT / path).read_text())


def scalar_targets_from_fields(lat: np.ndarray, apd90: np.ndarray, patch: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    v = valid.astype(bool)
    lat_vals = lat[v & np.isfinite(lat)]
    out = {
        "LAT_range": float(np.nanmax(lat_vals) - np.nanmin(lat_vals)) if lat_vals.size else np.nan,
        "patch_area_fraction": float(np.mean((patch > 0.5)[v])) if np.any(v) else np.nan,
    }
    inside = (patch > 0.5) & v & np.isfinite(apd90)
    outside = (patch <= 0.5) & v & np.isfinite(apd90)
    out["APD90_patch_delta"] = float(np.mean(apd90[inside]) - np.mean(apd90[outside])) if inside.any() and outside.any() else np.nan
    return out


def compute_stats(rows: pd.DataFrame, descriptors: pd.DataFrame | None = None) -> DeepONetStats:
    train = rows[rows["split"].eq("train")].reset_index(drop=True)
    total = 0.0
    total_sq = 0.0
    count = 0
    field_vals = [[] for _ in FIELD_NAMES]
    param_vals = [[] for _ in PARAM_KEYS]
    for row in train.itertuples(index=False):
        b = np.load(ROOT / row.B_meas_file).astype(np.float64)
        mask = np.load(ROOT / row.sensor_mask_file).astype(bool)
        valid_b = mask[None, :, :, :].repeat(b.shape[0], axis=0)
        vals = b[valid_b]
        vals = vals[np.isfinite(vals)]
        total += float(vals.sum())
        total_sq += float(np.square(vals).sum())
        count += int(vals.size)
        valid_grid = np.load(ROOT / row.valid_mask_grid_file).astype(bool)
        for i, col in enumerate(["LAT_grid_file", "APD80_grid_file", "APD90_grid_file"]):
            arr = np.load(ROOT / getattr(row, col)).astype(np.float64)
            field_vals[i].append(arr[valid_grid & np.isfinite(arr)])
        params = load_params(row.patch_param_targets_file)
        for i, key in enumerate(PARAM_KEYS):
            val = params.get(key)
            if val is not None and np.isfinite(val):
                param_vals[i].append(float(val))
    mean = total / max(count, 1)
    var = max(total_sq / max(count, 1) - mean * mean, 0.0)
    std = float(np.sqrt(var)) if var > 0 else 1.0
    fmean = np.zeros(len(FIELD_NAMES), dtype=np.float32)
    fstd = np.ones(len(FIELD_NAMES), dtype=np.float32)
    for i, chunks in enumerate(field_vals):
        vals = np.concatenate(chunks) if chunks else np.array([0.0])
        fmean[i] = float(np.mean(vals))
        fstd[i] = max(float(np.std(vals)), 1e-6)
    pmean = np.zeros(len(PARAM_KEYS), dtype=np.float32)
    pstd = np.ones(len(PARAM_KEYS), dtype=np.float32)
    for i, vals in enumerate(param_vals):
        arr = np.asarray(vals, dtype=np.float32) if vals else np.array([0.0], dtype=np.float32)
        pmean[i] = float(np.mean(arr))
        pstd[i] = max(float(np.std(arr)), 1e-6)
    pmean[0] = 0.0
    pstd[0] = 1.0

    desc_cols: list[str] = []
    dmean = np.zeros(0, dtype=np.float32)
    dstd = np.ones(0, dtype=np.float32)
    if descriptors is not None:
        skip = {"row_id", "base_sample_id", "condition", "combined_variant", "measurement_variant", "split"}
        desc_cols = [c for c in descriptors.columns if c not in skip]
        train_ids = set(train["row_id"])
        dtrain = descriptors[descriptors["row_id"].isin(train_ids)][desc_cols].to_numpy(np.float32)
        dmean = np.nanmean(dtrain, axis=0).astype(np.float32)
        dstd = np.nanstd(dtrain, axis=0).astype(np.float32)
        dstd[dstd < 1e-6] = 1.0
    return DeepONetStats(mean, std, fmean, fstd, pmean, pstd, dmean, dstd, desc_cols)


class V1EDeepONetDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        stats: DeepONetStats,
        *,
        mode: str = "random",
        k_queries: int = 256,
        input_norm: str = "per_sample",
        descriptors: pd.DataFrame | None = None,
        seed: int = 20260525,
    ) -> None:
        if mode not in {"random", "full"}:
            raise ValueError(f"mode must be random or full, got {mode}")
        self.rows = rows.reset_index(drop=True)
        self.stats = stats
        self.mode = mode
        self.k_queries = int(k_queries)
        self.input_norm = input_norm
        self.query_points = load_query_points()
        self.rng = np.random.default_rng(seed)
        self.desc_map: dict[str, np.ndarray] = {}
        if descriptors is not None and stats.descriptor_columns:
            for row in descriptors.itertuples(index=False):
                vals = np.asarray([getattr(row, col) for col in stats.descriptor_columns], dtype=np.float32)
                vals = (vals - stats.descriptor_mean) / stats.descriptor_std
                self.desc_map[getattr(row, "row_id")] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_branch(self, row: pd.Series) -> np.ndarray:
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
        b = np.where(valid, (b - mean) / std, 0.0)
        b = np.where(np.isfinite(b), b, 0.0)
        return b.transpose(3, 0, 1, 2).astype(np.float32)

    def _query_indices(self, valid: np.ndarray) -> np.ndarray:
        valid_flat = np.flatnonzero(valid.reshape(-1).astype(bool))
        if self.mode == "full":
            return np.arange(self.query_points.shape[0], dtype=np.int64)
        if valid_flat.size == 0:
            return self.rng.choice(self.query_points.shape[0], size=self.k_queries, replace=True).astype(np.int64)
        replace = valid_flat.size < self.k_queries
        return self.rng.choice(valid_flat, size=self.k_queries, replace=replace).astype(np.int64)

    def _load_targets(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, dict[str, float]]:
        lat = np.load(ROOT / row["LAT_grid_file"]).astype(np.float32)
        apd80 = np.load(ROOT / row["APD80_grid_file"]).astype(np.float32)
        apd90 = np.load(ROOT / row["APD90_grid_file"]).astype(np.float32)
        patch = np.load(ROOT / row["patch_mask_grid_file"]).astype(np.float32)
        valid = np.load(ROOT / row["valid_mask_grid_file"]).astype(bool)
        query_idx = self._query_indices(valid)
        fields = np.stack([lat, apd80, apd90], axis=-1).reshape(-1, 3)
        scaled_fields = (fields - self.stats.field_mean[None, :]) / self.stats.field_std[None, :]
        scaled_fields = np.nan_to_num(scaled_fields, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        patch_flat = patch.reshape(-1, 1).astype(np.float32)
        y_query = np.concatenate([scaled_fields, patch_flat], axis=1)[query_idx]
        valid_query = valid.reshape(-1)[query_idx].astype(np.float32)
        xy = self.query_points[query_idx].astype(np.float32)

        params = load_params(row["patch_param_targets_file"])
        y_param_raw = np.asarray([params.get(k, np.nan) for k in PARAM_KEYS], dtype=np.float32)
        p_mask = np.isfinite(y_param_raw)
        y_scaled = (np.where(p_mask, y_param_raw, self.stats.param_mean) - self.stats.param_mean) / self.stats.param_std
        y_scaled[0] = np.nan_to_num(y_param_raw[0], nan=0.0)
        p_mask[0] = True
        area_bin = int(params.get("area_bin", -1))
        raw_area_radius = np.asarray([params.get(k, np.nan) for k in RAW_PARAM_KEYS], dtype=np.float32)
        scalars = scalar_targets_from_fields(lat, apd90, patch, valid)
        sampled_all_valid = bool(np.all(valid.reshape(-1)[query_idx])) if self.mode == "random" else True
        return xy, y_query, valid_query, y_scaled.astype(np.float32), p_mask.astype(bool), raw_area_radius, area_bin, sampled_all_valid, scalars

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        xy, y_query, valid_query, y_param, p_mask, raw_area_radius, area_bin, sampled_all_valid, scalars = self._load_targets(row)
        desc = self.desc_map.get(row["row_id"], np.zeros(len(self.stats.descriptor_columns), dtype=np.float32))
        return {
            "branch_x": torch.from_numpy(self._load_branch(row)),
            "descriptor_x": torch.from_numpy(desc),
            "trunk_xy": torch.from_numpy(xy),
            "y_query": torch.from_numpy(y_query),
            "valid_query_mask": torch.from_numpy(valid_query),
            "y_param": torch.from_numpy(y_param),
            "param_mask": torch.from_numpy(p_mask),
            "area_bin": torch.tensor(max(area_bin, 0), dtype=torch.long),
            "area_bin_mask": torch.tensor(area_bin >= 0, dtype=torch.bool),
            "raw_area_radius": torch.from_numpy(raw_area_radius),
            "condition_label": torch.tensor(int(row["condition_label"]), dtype=torch.long),
            "lat_range": torch.tensor(float(scalars["LAT_range"]), dtype=torch.float32),
            "apd90_patch_delta": torch.tensor(float(scalars["APD90_patch_delta"]) if np.isfinite(scalars["APD90_patch_delta"]) else np.nan, dtype=torch.float32),
            "patch_area_fraction": torch.tensor(float(scalars["patch_area_fraction"]), dtype=torch.float32),
            "sampled_all_queries_valid": torch.tensor(sampled_all_valid, dtype=torch.bool),
            "metadata": {
                "row_id": row["row_id"],
                "base_sample_id": row["base_sample_id"],
                "split": row["split"],
                "condition": row["condition"],
                "combined_variant": row["combined_variant"],
            },
        }


def make_loader(
    rows: pd.DataFrame,
    stats: DeepONetStats,
    *,
    batch_size: int,
    shuffle: bool,
    mode: str = "random",
    k_queries: int = 256,
    input_norm: str = "per_sample",
    descriptors: pd.DataFrame | None = None,
    seed: int = 20260525,
) -> DataLoader:
    return DataLoader(
        V1EDeepONetDataset(
            rows,
            stats,
            mode=mode,
            k_queries=k_queries,
            input_norm=input_norm,
            descriptors=descriptors,
            seed=seed,
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def save_stats(path: Path, stats: DeepONetStats) -> None:
    write_json(
        path,
        {
            "input_mean": stats.input_mean,
            "input_std": stats.input_std,
            "field_mean": stats.field_mean.tolist(),
            "field_std": stats.field_std.tolist(),
            "param_mean": stats.param_mean.tolist(),
            "param_std": stats.param_std.tolist(),
            "descriptor_mean": stats.descriptor_mean.tolist(),
            "descriptor_std": stats.descriptor_std.tolist(),
            "descriptor_columns": stats.descriptor_columns,
            "normalization_policy": "input/field/param/descriptor statistics are fit on train split only; per-sample branch normalization is also supported.",
        },
    )
