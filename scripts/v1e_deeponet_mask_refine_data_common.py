from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from v1e_500_common import PILOT_DIR, ROOT, SEED
from v1e_deeponet_data_common import (
    PARAM_KEYS,
    RAW_PARAM_KEYS,
    DeepONetStats,
    check_no_leakage,
    compute_stats,
    load_descriptors,
    load_index,
    load_params,
    load_query_points,
    scalar_targets_from_fields,
    torch,
)

from torch.utils.data import DataLoader, Dataset


MASK_TARGET_DIR = PILOT_DIR / "deeponet_mask_refine_targets"
MASK_TARGET_MANIFEST = MASK_TARGET_DIR / "target_manifest.csv"


def load_mask_refine_manifest() -> pd.DataFrame:
    return pd.read_csv(MASK_TARGET_MANIFEST)


def merge_rows_with_refine_targets(rows: pd.DataFrame | None = None) -> pd.DataFrame:
    if rows is None:
        rows = load_index()
    targets = load_mask_refine_manifest()
    target_cols = [
        "row_id",
        "center_heatmap_file",
        "signed_distance_target_file",
        "boundary_weight_map_file",
        "valid_mask_file",
        "center_x_px",
        "center_y_px",
        "center_sigma_px",
    ]
    merged = rows.merge(targets[target_cols], on="row_id", how="left", validate="one_to_one")
    missing = merged["center_heatmap_file"].isna().sum()
    if missing:
        raise ValueError(f"Missing mask-refinement targets for {missing} rows")
    return merged


class V1EMaskRefineDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        stats: DeepONetStats,
        *,
        mode: str = "random",
        k_queries: int = 256,
        input_norm: str = "per_sample",
        descriptors: pd.DataFrame | None = None,
        seed: int = SEED,
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

    def _load_targets(self, row: pd.Series):
        lat = np.load(ROOT / row["LAT_grid_file"]).astype(np.float32)
        apd90 = np.load(ROOT / row["APD90_grid_file"]).astype(np.float32)
        patch = np.load(ROOT / row["patch_mask_grid_file"]).astype(np.float32)
        valid = np.load(ROOT / row["valid_mask_grid_file"]).astype(bool)
        center_heat = np.load(ROOT / row["center_heatmap_file"]).astype(np.float32)
        distance = np.load(ROOT / row["signed_distance_target_file"]).astype(np.float32)
        boundary_weight = np.load(ROOT / row["boundary_weight_map_file"]).astype(np.float32)
        query_idx = self._query_indices(valid)

        fields = np.stack([lat, apd90], axis=-1).reshape(-1, 2)
        field_mean = self.stats.field_mean[[0, 2]]
        field_std = self.stats.field_std[[0, 2]]
        scaled_fields = (fields - field_mean[None, :]) / field_std[None, :]
        scaled_fields = np.nan_to_num(scaled_fields, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        params = load_params(row["patch_param_targets_file"])
        y_param_raw = np.asarray([params.get(k, np.nan) for k in PARAM_KEYS], dtype=np.float32)
        p_mask = np.isfinite(y_param_raw)
        y_scaled = (np.where(p_mask, y_param_raw, self.stats.param_mean) - self.stats.param_mean) / self.stats.param_std
        y_scaled[0] = np.nan_to_num(y_param_raw[0], nan=0.0)
        p_mask[0] = True
        area_bin = int(params.get("area_bin", -1))
        raw_area_radius = np.asarray([params.get(k, np.nan) for k in RAW_PARAM_KEYS], dtype=np.float32)
        scalars = scalar_targets_from_fields(lat, apd90, patch, valid)
        full_valid = valid.reshape(-1)
        return {
            "trunk_xy": self.query_points[query_idx].astype(np.float32),
            "y_field": scaled_fields[query_idx],
            "patch_query": patch.reshape(-1)[query_idx].astype(np.float32),
            "center_heatmap_query": center_heat.reshape(-1)[query_idx].astype(np.float32),
            "distance_query": distance.reshape(-1)[query_idx].astype(np.float32),
            "boundary_weight_query": boundary_weight.reshape(-1)[query_idx].astype(np.float32),
            "valid_query_mask": full_valid[query_idx].astype(np.float32),
            "y_param": y_scaled.astype(np.float32),
            "param_mask": p_mask.astype(bool),
            "area_bin": area_bin,
            "raw_area_radius": raw_area_radius,
            "lat_range": scalars["LAT_range"],
            "apd90_patch_delta": scalars["APD90_patch_delta"],
            "patch_area_fraction": scalars["patch_area_fraction"],
            "sampled_all_queries_valid": bool(np.all(full_valid[query_idx])) if self.mode == "random" else True,
        }

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        targets = self._load_targets(row)
        desc = self.desc_map.get(row["row_id"], np.zeros(len(self.stats.descriptor_columns), dtype=np.float32))
        return {
            "branch_x": torch.from_numpy(self._load_branch(row)),
            "descriptor_x": torch.from_numpy(desc),
            "trunk_xy": torch.from_numpy(targets["trunk_xy"]),
            "y_field": torch.from_numpy(targets["y_field"]),
            "patch_query": torch.from_numpy(targets["patch_query"]),
            "center_heatmap_query": torch.from_numpy(targets["center_heatmap_query"]),
            "distance_query": torch.from_numpy(targets["distance_query"]),
            "boundary_weight_query": torch.from_numpy(targets["boundary_weight_query"]),
            "valid_query_mask": torch.from_numpy(targets["valid_query_mask"]),
            "y_param": torch.from_numpy(targets["y_param"]),
            "param_mask": torch.from_numpy(targets["param_mask"]),
            "area_bin": torch.tensor(max(targets["area_bin"], 0), dtype=torch.long),
            "area_bin_mask": torch.tensor(targets["area_bin"] >= 0, dtype=torch.bool),
            "raw_area_radius": torch.from_numpy(targets["raw_area_radius"]),
            "condition_label": torch.tensor(int(row["condition_label"]), dtype=torch.long),
            "lat_range": torch.tensor(float(targets["lat_range"]), dtype=torch.float32),
            "apd90_patch_delta": torch.tensor(float(targets["apd90_patch_delta"]) if np.isfinite(targets["apd90_patch_delta"]) else np.nan, dtype=torch.float32),
            "patch_area_fraction": torch.tensor(float(targets["patch_area_fraction"]), dtype=torch.float32),
            "sampled_all_queries_valid": torch.tensor(targets["sampled_all_queries_valid"], dtype=torch.bool),
            "metadata": {
                "row_id": row["row_id"],
                "base_sample_id": row["base_sample_id"],
                "split": row["split"],
                "condition": row["condition"],
                "combined_variant": row["combined_variant"],
            },
        }


def make_mask_refine_loader(
    rows: pd.DataFrame,
    stats: DeepONetStats,
    *,
    batch_size: int,
    shuffle: bool,
    mode: str = "random",
    k_queries: int = 256,
    input_norm: str = "per_sample",
    descriptors: pd.DataFrame | None = None,
    seed: int = SEED,
) -> DataLoader:
    return DataLoader(
        V1EMaskRefineDataset(
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
