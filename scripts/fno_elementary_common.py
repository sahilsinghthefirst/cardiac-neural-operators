                                                                   

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from torch_import_utils import import_torch_prefer_env

torch = import_torch_prefer_env(PROJECT_ROOT)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from neural_data_common import check_no_group_leakage, load_neural_index, select_deployable_feature_columns


READY_DIR = PROJECT_ROOT / "outputs" / "dataset_v1c_5000_augmented_ready"
FIXED_FEATURES = READY_DIR / "features_mig_deployable_fixed_augmented.csv"
MAGNETIC_FEATURES = READY_DIR / "features_mig_deployable_magnetic_augmented.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "neural_v1c"
FIG_DIR = PROJECT_ROOT / "figures" / "neural_v1c"
CKPT_DIR = OUT_DIR / "checkpoints"
STATS_JSON = OUT_DIR / "cnn_bz_global_train_stats.json"
SEED = 20260525
REG_TARGETS = ["y_lat_range", "y_apd90_patch_delta", "y_patch_area_fraction"]
VARIANT_ORDER = [
    "clean",
    "noise_low",
    "noise_medium",
    "noise_high",
    "dropout_10pct",
    "dropout_25pct",
    "jitter_1mm",
    "jitter_3mm",
    "height_30mm",
    "height_70mm",
    "combined_medium",
]
BENCHMARKS = {
    "y_condition_macro_f1": 0.787,
    "y_lat_range_r2": 0.9918,
    "y_apd90_patch_delta_r2": 0.787,
    "y_patch_area_fraction_r2": 0.450,
}


@dataclass(frozen=True)
class FnoConfig:
    name: str
    bz_norm: str
    hidden_width: int = 48
    modes_x: int = 4
    modes_y: int = 4
    layers: int = 4
    dropout: float = 0.1
    lr: float = 8e-4
    weight_decay: float = 1e-4
    batch_size: int = 1024


@dataclass(frozen=True)
class FnoFusionConfig:
    name: str
    bz_norm: str
    feature_kind: str
    feature_columns: tuple[str, ...]
    hidden_width: int = 48
    modes_x: int = 4
    modes_y: int = 4
    layers: int = 4
    dropout: float = 0.12
    lr: float = 8e-4
    weight_decay: float = 1e-4
    batch_size: int = 1024


@dataclass(frozen=True)
class Fno3dConfig:
    name: str
    bz_norm: str
    hidden_width: int = 24
    modes_t: int = 16
    modes_x: int = 4
    modes_y: int = 4
    layers: int = 3
    dropout: float = 0.08
    lr: float = 8e-4
    weight_decay: float = 1e-4
    batch_size: int = 256


@dataclass(frozen=True)
class Fno3dFusionConfig:
    name: str
    bz_norm: str
    feature_kind: str
    feature_columns: tuple[str, ...]
    hidden_width: int = 24
    modes_t: int = 16
    modes_x: int = 4
    modes_y: int = 4
    layers: int = 3
    dropout: float = 0.1
    lr: float = 8e-4
    weight_decay: float = 1e-4
    batch_size: int = 256


class BzFnoDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, bz_norm: str, global_mean: float, global_std: float) -> None:
        self.rows = rows.reset_index(drop=True)
        self.bz_norm = bz_norm
        self.global_mean = global_mean
        self.global_std = global_std

    def __len__(self) -> int:
        return len(self.rows)

    def _load_bz(self, row: pd.Series) -> np.ndarray:
        bz = np.load(PROJECT_ROOT / row["Bz_file"]).astype(np.float32)
        mask = np.load(PROJECT_ROOT / row["sensor_mask_file"]).astype(bool)
        if bz.shape != (301, 64):
            raise ValueError(f"Expected Bz shape [301,64], got {bz.shape} for {row['augmented_sample_id']}")
        bz[:, ~mask] = np.nan
        valid = bz[:, mask]
        if self.bz_norm == "per_sample":
            mean = float(np.nanmean(valid))
            std = float(np.nanstd(valid))
        elif self.bz_norm == "train_global":
            mean = float(self.global_mean)
            std = float(self.global_std)
        else:
            mean = 0.0
            std = 1.0
        if not np.isfinite(std) or std <= 1e-12:
            std = 1.0
        bz = (bz - mean) / std
        bz = np.where(np.isfinite(bz), bz, 0.0).astype(np.float32)
        return bz.reshape(301, 8, 8)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows.iloc[idx]
        y_reg = np.array([row[target] for target in REG_TARGETS], dtype=np.float32)
        reg_mask = np.isfinite(y_reg)
        y_reg = np.where(reg_mask, y_reg, 0.0).astype(np.float32)
        return {
            "x_bz": torch.from_numpy(self._load_bz(row)),
            "y_class": torch.tensor(int(row["y_condition"]), dtype=torch.long),
            "y_reg": torch.from_numpy(y_reg),
            "reg_mask": torch.from_numpy(reg_mask),
        }


class FnoFusionDataset(BzFnoDataset):
                                                        
 
                                                                              
                                                                             
       

    def __init__(self, rows: pd.DataFrame, config: FnoFusionConfig, global_mean: float, global_std: float) -> None:
        super().__init__(rows, config.bz_norm, global_mean, global_std)
        self.feature_columns = list(config.feature_columns)
        self.features = self.rows[self.feature_columns].to_numpy(np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        item["x_features"] = torch.from_numpy(self.features[idx])
        return item


class BzFno3dDataset(BzFnoDataset):
                                                   
 
                                                                           
                                          
       

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        item["x_bz"] = item["x_bz"].unsqueeze(0)
        return item


class Fno3dFusionDataset(BzFno3dDataset):
                                                                  

    def __init__(self, rows: pd.DataFrame, config: Fno3dFusionConfig, global_mean: float, global_std: float) -> None:
        super().__init__(rows, config.bz_norm, global_mean, global_std)
        self.feature_columns = list(config.feature_columns)
        self.features = self.rows[self.feature_columns].to_numpy(np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        item["x_features"] = torch.from_numpy(self.features[idx])
        return item


class SpectralConv2d(nn.Module):
                                                                 

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1 / math.sqrt(max(1, in_channels * out_channels))
        self.weights_pos = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat))
        self.weights_neg = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat))

    @staticmethod
    def compl_mul2d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        if self.modes_x > height // 2 or self.modes_y > width // 2:
            raise ValueError(f"FNO modes ({self.modes_x},{self.modes_y}) exceed conservative Nyquist for grid {height}x{width}")
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes_x, : self.modes_y] = self.compl_mul2d(
            x_ft[:, :, : self.modes_x, : self.modes_y], self.weights_pos
        )
        out_ft[:, :, -self.modes_x :, : self.modes_y] = self.compl_mul2d(
            x_ft[:, :, -self.modes_x :, : self.modes_y], self.weights_neg
        )
        return torch.fft.irfft2(out_ft, s=(height, width), dim=(-2, -1))


class FNOBlock2d(nn.Module):
    def __init__(self, width: int, modes_x: int, modes_y: int, dropout: float) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_x, modes_y)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.spectral(x) + self.pointwise(x)))


class ElementaryFNO2dTimeChannels(nn.Module):
                                                                         

    def __init__(self, hidden_width: int = 48, modes_x: int = 4, modes_y: int = 4, layers: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_width = hidden_width
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.layers = layers
        self.lift = nn.Conv2d(301, hidden_width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock2d(hidden_width, modes_x, modes_y, dropout) for _ in range(layers)])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_width, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.class_head = nn.Linear(64, 4)
        self.reg_head = nn.Linear(64, 3)

    def forward(self, x_bz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.lift(x_bz)
        for block in self.blocks:
            x = block(x)
        h = self.head(self.pool(x))
        return self.class_head(h), self.reg_head(h)


class FNO2dEncoderTimeChannels(nn.Module):
                                                         

    def __init__(self, hidden_width: int = 48, modes_x: int = 4, modes_y: int = 4, layers: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_width = hidden_width
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.layers = layers
        self.lift = nn.Conv2d(301, hidden_width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock2d(hidden_width, modes_x, modes_y, dropout) for _ in range(layers)])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_width, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 64),
            nn.GELU(),
        )
        self.out_dim = 64

    def forward(self, x_bz: torch.Tensor) -> torch.Tensor:
        x = self.lift(x_bz)
        for block in self.blocks:
            x = block(x)
        return self.project(self.pool(x))


class FNOFeatureFusionNet(nn.Module):
                                                               

    def __init__(
        self,
        feature_dim: int,
        hidden_width: int = 48,
        modes_x: int = 4,
        modes_y: int = 4,
        layers: int = 4,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        self.bz_encoder = FNO2dEncoderTimeChannels(hidden_width, modes_x, modes_y, layers, dropout)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.bz_encoder.out_dim + 64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.class_head = nn.Linear(64, 4)
        self.reg_head = nn.Linear(64, 3)

    def forward(self, x_bz: torch.Tensor, x_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bz_h = self.bz_encoder(x_bz)
        feat_h = self.feature_encoder(x_features)
        h = self.fusion(torch.cat([bz_h, feat_h], dim=1))
        return self.class_head(h), self.reg_head(h)


class SpectralConv3d(nn.Module):
                                                              

    def __init__(self, in_channels: int, out_channels: int, modes_t: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_t = modes_t
        self.modes_x = modes_x
        self.modes_y = modes_y
        scale = 1 / math.sqrt(max(1, in_channels * out_channels))
        shape = (in_channels, out_channels, modes_t, modes_x, modes_y)
        self.weights_pp = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights_np = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights_pn = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights_nn = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def compl_mul3d(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bitxy,iotxy->botxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, time, height, width = x.shape
        if self.modes_t > time // 2 or self.modes_x > height // 2 or self.modes_y > width // 2:
            raise ValueError(
                f"FNO3d modes ({self.modes_t},{self.modes_x},{self.modes_y}) exceed conservative "
                f"Nyquist for grid {time}x{height}x{width}"
            )
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            time,
            height,
            width // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        mt, mx, my = self.modes_t, self.modes_x, self.modes_y
        out_ft[:, :, :mt, :mx, :my] = self.compl_mul3d(x_ft[:, :, :mt, :mx, :my], self.weights_pp)
        out_ft[:, :, -mt:, :mx, :my] = self.compl_mul3d(x_ft[:, :, -mt:, :mx, :my], self.weights_np)
        out_ft[:, :, :mt, -mx:, :my] = self.compl_mul3d(x_ft[:, :, :mt, -mx:, :my], self.weights_pn)
        out_ft[:, :, -mt:, -mx:, :my] = self.compl_mul3d(x_ft[:, :, -mt:, -mx:, :my], self.weights_nn)
        return torch.fft.irfftn(out_ft, s=(time, height, width), dim=(-3, -2, -1))


class FNOBlock3d(nn.Module):
    def __init__(self, width: int, modes_t: int, modes_x: int, modes_y: int, dropout: float) -> None:
        super().__init__()
        self.spectral = SpectralConv3d(width, width, modes_t, modes_x, modes_y)
        self.pointwise = nn.Conv3d(width, width, kernel_size=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(self.spectral(x) + self.pointwise(x)))


class FNO3dEncoder(nn.Module):
                                                                           

    def __init__(
        self,
        hidden_width: int = 24,
        modes_t: int = 16,
        modes_x: int = 4,
        modes_y: int = 4,
        layers: int = 3,
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        self.hidden_width = hidden_width
        self.modes_t = modes_t
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.layers = layers
        self.lift = nn.Conv3d(1, hidden_width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock3d(hidden_width, modes_t, modes_x, modes_y, dropout) for _ in range(layers)])
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_width, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, 64),
            nn.GELU(),
        )
        self.out_dim = 64

    def forward(self, x_bz: torch.Tensor) -> torch.Tensor:
        x = self.lift(x_bz)
        for block in self.blocks:
            x = block(x)
        return self.project(self.pool(x))


class SpatiotemporalFNO3d(nn.Module):
                                                                              

    def __init__(
        self,
        hidden_width: int = 24,
        modes_t: int = 16,
        modes_x: int = 4,
        modes_y: int = 4,
        layers: int = 3,
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        self.encoder = FNO3dEncoder(hidden_width, modes_t, modes_x, modes_y, layers, dropout)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.class_head = nn.Linear(64, 4)
        self.reg_head = nn.Linear(64, 3)

    def forward(self, x_bz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.head(self.encoder(x_bz))
        return self.class_head(h), self.reg_head(h)


class SpatiotemporalFNO3dFusionNet(nn.Module):
                                                                                

    def __init__(
        self,
        feature_dim: int,
        hidden_width: int = 24,
        modes_t: int = 16,
        modes_x: int = 4,
        modes_y: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.bz_encoder = FNO3dEncoder(hidden_width, modes_t, modes_x, modes_y, layers, dropout)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.bz_encoder.out_dim + 64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.class_head = nn.Linear(64, 4)
        self.reg_head = nn.Linear(64, 3)

    def forward(self, x_bz: torch.Tensor, x_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bz_h = self.bz_encoder(x_bz)
        feat_h = self.feature_encoder(x_features)
        h = self.fusion(torch.cat([bz_h, feat_h], dim=1))
        return self.class_head(h), self.reg_head(h)


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)


def load_rows() -> pd.DataFrame:
    rows = load_neural_index()
    leakage = check_no_group_leakage(rows)
    if leakage:
        raise ValueError(f"Grouped split leakage detected: {leakage[:10]}")
    return rows


def load_rows_and_deployable_features() -> tuple[pd.DataFrame, dict[str, list[str]]]:
                                                                       
    rows = load_rows()
    fixed = pd.read_csv(FIXED_FEATURES)
    magnetic = pd.read_csv(MAGNETIC_FEATURES)
    fixed_cols = select_deployable_feature_columns(fixed)
    magnetic_cols = select_deployable_feature_columns(magnetic)
    fixed = fixed[["augmented_sample_id", *fixed_cols]].rename(columns={col: f"fixed__{col}" for col in fixed_cols})
    magnetic = magnetic[["augmented_sample_id", *magnetic_cols]].rename(
        columns={col: f"magnetic__{col}" for col in magnetic_cols}
    )
    rows = rows.merge(fixed, on="augmented_sample_id", how="inner", validate="one_to_one")
    rows = rows.merge(magnetic, on="augmented_sample_id", how="inner", validate="one_to_one")
    if len(rows) != 55000:
        raise ValueError(f"Expected 55000 rows after feature merge, got {len(rows)}")
    groups = {
        "fixed": [f"fixed__{col}" for col in fixed_cols],
        "magnetic": [f"magnetic__{col}" for col in magnetic_cols],
        "fixed_magnetic": [f"fixed__{col}" for col in fixed_cols] + [f"magnetic__{col}" for col in magnetic_cols],
    }
    train = rows["split"].eq("train")
    for cols in groups.values():
        values = rows[cols].to_numpy(np.float32)
        mean = rows.loc[train, cols].to_numpy(np.float32).mean(axis=0)
        std = rows.loc[train, cols].to_numpy(np.float32).std(axis=0)
        std[std < 1e-8] = 1.0
        rows.loc[:, cols] = ((values - mean) / std).astype(np.float32)
    return rows, groups


def compute_train_global_stats(train_rows: pd.DataFrame) -> tuple[float, float]:
    if STATS_JSON.exists():
        data = json.loads(STATS_JSON.read_text())
        return float(data["mean"]), float(data["std"])
    total = 0.0
    total_sq = 0.0
    count = 0
    for row in train_rows.itertuples(index=False):
        bz = np.load(PROJECT_ROOT / row.Bz_file).astype(np.float64)
        mask = np.load(PROJECT_ROOT / row.sensor_mask_file).astype(bool)
        vals = bz[:, mask]
        vals = vals[np.isfinite(vals)]
        total += float(vals.sum())
        total_sq += float(np.square(vals).sum())
        count += int(vals.size)
    mean = total / max(count, 1)
    var = max(total_sq / max(count, 1) - mean * mean, 0.0)
    std = math.sqrt(var) if var > 0 else 1.0
    STATS_JSON.write_text(json.dumps({"mean": mean, "std": std, "count": count}, indent=2))
    return mean, std


def target_scaler(train_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(3, dtype=np.float32)
    std = np.ones(3, dtype=np.float32)
    for i, target in enumerate(REG_TARGETS):
        vals = train_rows[target].to_numpy(np.float32)
        vals = vals[np.isfinite(vals)]
        mean[i] = float(vals.mean())
        std[i] = float(vals.std())
        if std[i] < 1e-8:
            std[i] = 1.0
    return mean, std


def class_weights(train_rows: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = np.bincount(train_rows["y_condition"].to_numpy(int), minlength=4).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def collate_scaled(batch: list[dict[str, torch.Tensor]], reg_mean: np.ndarray, reg_std: np.ndarray) -> dict[str, torch.Tensor]:
    x_bz = torch.stack([item["x_bz"] for item in batch])
    y_class = torch.stack([item["y_class"] for item in batch])
    y_reg = torch.stack([item["y_reg"] for item in batch]).float()
    reg_mask = torch.stack([item["reg_mask"] for item in batch]).bool()
    mean = torch.tensor(reg_mean, dtype=torch.float32)
    std = torch.tensor(reg_std, dtype=torch.float32)
    y_reg_scaled = torch.where(reg_mask, (y_reg - mean) / std, torch.zeros_like(y_reg))
    return {"x_bz": x_bz, "y_class": y_class, "y_reg": y_reg, "y_reg_scaled": y_reg_scaled, "reg_mask": reg_mask}


def make_loader(
    rows: pd.DataFrame,
    config: FnoConfig,
    global_mean: float,
    global_std: float,
    reg_mean: np.ndarray,
    reg_std: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        BzFnoDataset(rows, config.bz_norm, global_mean, global_std),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda batch: collate_scaled(batch, reg_mean, reg_std),
    )


def make_fusion_loader(
    rows: pd.DataFrame,
    config: FnoFusionConfig,
    global_mean: float,
    global_std: float,
    reg_mean: np.ndarray,
    reg_std: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        FnoFusionDataset(rows, config, global_mean, global_std),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda batch: collate_fusion_scaled(batch, reg_mean, reg_std),
    )


def make_3d_loader(
    rows: pd.DataFrame,
    config: Fno3dConfig,
    global_mean: float,
    global_std: float,
    reg_mean: np.ndarray,
    reg_std: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        BzFno3dDataset(rows, config.bz_norm, global_mean, global_std),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda batch: collate_scaled(batch, reg_mean, reg_std),
    )


def make_3d_fusion_loader(
    rows: pd.DataFrame,
    config: Fno3dFusionConfig,
    global_mean: float,
    global_std: float,
    reg_mean: np.ndarray,
    reg_std: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        Fno3dFusionDataset(rows, config, global_mean, global_std),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda batch: collate_fusion_scaled(batch, reg_mean, reg_std),
    )


def collate_fusion_scaled(batch: list[dict[str, torch.Tensor]], reg_mean: np.ndarray, reg_std: np.ndarray) -> dict[str, torch.Tensor]:
    out = collate_scaled(batch, reg_mean, reg_std)
    out["x_features"] = torch.stack([item["x_features"] for item in batch])
    return out


def multitask_loss(
    logits: torch.Tensor,
    reg_pred: torch.Tensor,
    y_class: torch.Tensor,
    y_reg_scaled: torch.Tensor,
    reg_mask: torch.Tensor,
    ce_loss: nn.Module,
) -> torch.Tensor:
    loss = ce_loss(logits, y_class)
    reg_losses = []
    for i in range(3):
        mask = reg_mask[:, i]
        if mask.any():
            reg_losses.append(nn.functional.smooth_l1_loss(reg_pred[mask, i], y_reg_scaled[mask, i], beta=0.75))
    if reg_losses:
        loss = loss + sum(reg_losses) / len(reg_losses)
    return loss


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
