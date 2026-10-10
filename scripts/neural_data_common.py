                      
                                                                       

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
from torch_import_utils import import_torch_prefer_env

torch = import_torch_prefer_env(PROJECT_ROOT)
from torch.utils.data import Dataset


READY_DIR = PROJECT_ROOT / "outputs" / "dataset_v1c_5000_augmented_ready"
SAMPLE_INDEX = READY_DIR / "sample_index_augmented.csv"
TARGETS = READY_DIR / "targets_ep_augmented.csv"
SPLITS = READY_DIR / "splits" / "grouped_train_val_test.json"

ID_COLUMNS = {
    "augmented_sample_id",
    "base_sample_id",
    "variant",
    "sample_id",
    "condition",
    "notes",
}
TARGET_COLUMNS = {
    "condition_label",
    "y_condition",
    "y_apd90_patch_delta",
    "y_lat_range",
    "y_patch_area_fraction",
}


def load_split_map(path: Path = SPLITS) -> dict[str, str]:
    data = json.loads(path.read_text())
    base_split = data.get("base_split")
    if not isinstance(base_split, dict):
        raise ValueError(f"{path} does not contain a base_split mapping")
    return base_split


def attach_splits(df: pd.DataFrame, split_map: dict[str, str] | None = None) -> pd.DataFrame:
    out = df.copy()
    split_map = split_map or load_split_map()
    out["split"] = out["base_sample_id"].map(split_map)
    if out["split"].isna().any():
        missing = out.loc[out["split"].isna(), "base_sample_id"].unique()[:10]
        raise ValueError(f"Rows missing grouped split assignment: {missing}")
    return out


def check_no_group_leakage(df: pd.DataFrame) -> list[str]:
    counts = df.groupby("base_sample_id")["split"].nunique()
    leaked = counts[counts > 1]
    return leaked.index.astype(str).tolist()


def select_deployable_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(ID_COLUMNS) | set(TARGET_COLUMNS)
    cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if col.endswith("_start_ms") or col.endswith("_end_ms") or col.endswith("_n_timepoints"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def load_neural_index() -> pd.DataFrame:
    index = pd.read_csv(SAMPLE_INDEX)
    targets = pd.read_csv(TARGETS)
    needed_targets = [
        "augmented_sample_id",
        "base_sample_id",
        "variant",
        "condition",
        "y_condition",
        "y_lat_range",
        "y_apd90_patch_delta",
        "y_patch_area_fraction",
    ]
    merged = index.merge(
        targets[needed_targets],
        on=["augmented_sample_id", "base_sample_id", "variant", "condition"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(index) or len(merged) != len(targets):
        raise ValueError(f"Index/target row mismatch: index={len(index)} targets={len(targets)} merged={len(merged)}")
    return attach_splits(merged)


def load_bz_as_grid(
    bz_path: str | Path,
    mask_path: str | Path | None = None,
    grid_shape: tuple[int, int] = (8, 8),
    normalize: bool = True,
) -> np.ndarray:
    bz = np.load(PROJECT_ROOT / bz_path if not Path(bz_path).is_absolute() else bz_path).astype(np.float32)
    if bz.ndim != 2 or bz.shape[1] != grid_shape[0] * grid_shape[1]:
        raise ValueError(f"Expected Bz [time, {grid_shape[0] * grid_shape[1]}], got {bz.shape} at {bz_path}")

    valid_mask = np.ones(bz.shape[1], dtype=bool)
    if mask_path is not None and str(mask_path) and str(mask_path) != "nan":
        path = PROJECT_ROOT / mask_path if not Path(mask_path).is_absolute() else Path(mask_path)
        if path.exists():
            valid_mask = np.load(path).astype(bool)
            if valid_mask.shape != (bz.shape[1],):
                raise ValueError(f"Expected sensor mask shape {(bz.shape[1],)}, got {valid_mask.shape}")
            bz[:, ~valid_mask] = np.nan

    valid_values = bz[:, valid_mask]
    finite = np.isfinite(valid_values)
    if not finite.any():
        raise ValueError(f"No finite valid Bz values in {bz_path}")

    if normalize:
        center = float(np.nanmean(valid_values))
        scale = float(np.nanstd(valid_values))
        if scale <= 0 or not np.isfinite(scale):
            scale = 1.0
        bz = (bz - center) / scale

    bz = np.where(np.isfinite(bz), bz, 0.0).astype(np.float32)
    return bz.reshape(bz.shape[0], grid_shape[0], grid_shape[1])


class BzTensorDataset(Dataset):
                                                             
 
                               
 
                                               
                                                                      
                                                                                       
                                                      
                                                   
       

    def __init__(
        self,
        rows: pd.DataFrame | None = None,
        split: str | None = None,
        feature_frame: pd.DataFrame | None = None,
        feature_columns: Iterable[str] | None = None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        normalize_bz: bool = True,
    ) -> None:
        df = rows.copy() if rows is not None else load_neural_index()
        if split is not None:
            df = df[df["split"] == split].copy()
        if feature_frame is not None and feature_columns is not None:
            feature_cols = list(feature_columns)
            df = df.merge(
                feature_frame[["augmented_sample_id", *feature_cols]],
                on="augmented_sample_id",
                how="left",
                validate="one_to_one",
            )
            if df[feature_cols].isna().any().any():
                raise ValueError("Missing deployable feature values after merge")
            self.feature_columns = feature_cols
            features = df[feature_cols].to_numpy(np.float32)
            if feature_mean is not None and feature_std is not None:
                features = (features - feature_mean.astype(np.float32)) / feature_std.astype(np.float32)
            self.features = features.astype(np.float32)
        else:
            self.feature_columns = []
            self.features = np.zeros((len(df), 0), dtype=np.float32)
        self.rows = df.reset_index(drop=True)
        self.normalize_bz = normalize_bz

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.rows.iloc[idx]
        x_bz = load_bz_as_grid(row["Bz_file"], row.get("sensor_mask_file"), normalize=self.normalize_bz)
        targets = np.array(
            [
                row["y_condition"],
                row["y_lat_range"],
                row["y_apd90_patch_delta"],
                row["y_patch_area_fraction"],
            ],
            dtype=np.float32,
        )
        target_mask = np.isfinite(targets)
        targets = np.where(target_mask, targets, 0.0).astype(np.float32)
        return {
            "x_bz": torch.from_numpy(x_bz),
            "features": torch.from_numpy(self.features[idx]),
            "targets": torch.from_numpy(targets),
            "target_mask": torch.from_numpy(target_mask),
            "metadata": {
                "augmented_sample_id": row["augmented_sample_id"],
                "base_sample_id": row["base_sample_id"],
                "variant": row["variant"],
                "condition": row["condition"],
                "split": row["split"],
            },
        }


class FusionDataset(BzTensorDataset):
                                                           
 
                                                                             
                                                                               
                                                                                
                         
       

    def __init__(
        self,
        rows: pd.DataFrame | None = None,
        split: str | None = None,
        feature_frame: pd.DataFrame | None = None,
        feature_columns: Iterable[str] | None = None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        normalize_bz: bool = True,
    ) -> None:
        if feature_frame is None or feature_columns is None:
            raise ValueError("FusionDataset requires deployable feature_frame and feature_columns")
        super().__init__(
            rows=rows,
            split=split,
            feature_frame=feature_frame,
            feature_columns=feature_columns,
            feature_mean=feature_mean,
            feature_std=feature_std,
            normalize_bz=normalize_bz,
        )
