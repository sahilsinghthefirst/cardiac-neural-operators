                      
                                                         
 
                                                                          
                                                                             
                                                                        
   

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if importlib.util.find_spec("sklearn") is None and (PROJECT_ROOT / ".local_deps" / "sklearn").exists():
    sys.path.insert(0, str(PROJECT_ROOT / ".local_deps" / "sklearn"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch_import_utils import import_torch_prefer_env
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

torch = import_torch_prefer_env(PROJECT_ROOT)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from neural_data_common import check_no_group_leakage, load_neural_index


OUT_DIR = PROJECT_ROOT / "outputs" / "neural_v1c"
FIG_DIR = PROJECT_ROOT / "figures" / "neural_v1c"
CKPT_DIR = OUT_DIR / "checkpoints"
RESULTS_CSV = OUT_DIR / "cnn_sweep_results.csv"
RESULTS_JSON = OUT_DIR / "cnn_sweep_results.json"
PREDICTIONS_CSV = OUT_DIR / "cnn_sweep_predictions.csv"
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
    "y_condition_macro_f1": 0.764,
    "y_lat_range_r2": 0.991,
    "y_apd90_patch_delta_r2": 0.723,
    "y_patch_area_fraction_r2": 0.398,
}


@dataclass(frozen=True)
class CnnSweepConfig:
    name: str
    architecture: str
    bz_norm: str
    dropout: float = 0.2
    lr: float = 8e-4
    weight_decay: float = 1e-4
    batch_cpu: int = 384
    batch_gpu: int = 768


class BzSweepDataset(Dataset):
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


class TimeChannel2dBackbone(nn.Module):
    out_dim = 96

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(301, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.4),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.4),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalPooled2dBackbone(nn.Module):
    out_dim = 96

    def __init__(self, dropout: float, bins: int = 48) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(bins)
        self.spatial = nn.Sequential(
            nn.Conv2d(bins, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.4),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.4),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        sensors = x.reshape(b, 301, 64).transpose(1, 2)
        pooled = self.pool(sensors).transpose(1, 2).reshape(b, -1, 8, 8)
        return self.spatial(pooled)


class Light3dBackbone(nn.Module):
    out_dim = 96

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(9, 3, 3), stride=(3, 1, 1), padding=(4, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Dropout3d(dropout * 0.3),
            nn.Conv3d(8, 16, kernel_size=(7, 3, 3), stride=(3, 1, 1), padding=(3, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Dropout3d(dropout * 0.3),
            nn.Conv3d(16, 32, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(2, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(32, 96),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(1))


class SeparableTemporalSpatialBackbone(nn.Module):
    out_dim = 96

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=11, padding=5),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.4),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        by_sensor = x.reshape(b, 301, 64).permute(0, 2, 1).reshape(b * 64, 1, 301)
        temporal = self.temporal(by_sensor).reshape(b, 64, 16).permute(0, 2, 1).reshape(b, 16, 8, 8)
        return self.spatial(temporal)


def make_backbone(architecture: str, dropout: float) -> nn.Module:
    if architecture == "time_channel_2d":
        return TimeChannel2dBackbone(dropout)
    if architecture == "temporal_pooled_2d":
        return TemporalPooled2dBackbone(dropout)
    if architecture == "lightweight_3d":
        return Light3dBackbone(dropout)
    if architecture == "separable_temporal_spatial":
        return SeparableTemporalSpatialBackbone(dropout)
    raise ValueError(f"Unknown architecture {architecture}")


class MultiTaskCnn(nn.Module):
    def __init__(self, architecture: str, dropout: float) -> None:
        super().__init__()
        self.backbone = make_backbone(architecture, dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.out_dim, 96), nn.ReLU(), nn.Dropout(dropout))
        self.class_head = nn.Linear(96, 4)
        self.reg_head = nn.Linear(96, 3)

    def forward(self, x_bz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.head(self.backbone(x_bz))
        return self.class_head(h), self.reg_head(h)


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


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
    config: CnnSweepConfig,
    global_mean: float,
    global_std: float,
    reg_mean: np.ndarray,
    reg_std: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        BzSweepDataset(rows, config.bz_norm, global_mean, global_std),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda batch: collate_scaled(batch, reg_mean, reg_std),
    )


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


def train_config(
    rows: pd.DataFrame,
    config: CnnSweepConfig,
    device: torch.device,
    global_mean: float,
    global_std: float,
    max_epochs: int,
    patience: int,
) -> tuple[MultiTaskCnn, list[dict[str, float]], np.ndarray, np.ndarray]:
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    val_rows = rows[rows["split"].eq("val")].reset_index(drop=True)
    reg_mean, reg_std = target_scaler(train_rows)
    batch_size = config.batch_gpu if device.type == "cuda" else config.batch_cpu
    if config.architecture == "lightweight_3d":
        batch_size = min(batch_size, 192 if device.type == "cpu" else 384)
    train_loader = make_loader(train_rows, config, global_mean, global_std, reg_mean, reg_std, batch_size, True)
    val_loader = make_loader(val_rows, config, global_mean, global_std, reg_mean, reg_std, batch_size, False)
    model = MultiTaskCnn(config.architecture, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights(train_rows, device))
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            x_bz = batch["x_bz"].to(device).float()
            y_class = batch["y_class"].to(device)
            y_reg_scaled = batch["y_reg_scaled"].to(device)
            reg_mask = batch["reg_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, reg_pred = model(x_bz)
            loss = multitask_loss(logits, reg_pred, y_class, y_reg_scaled, reg_mask, ce_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                logits, reg_pred = model(batch["x_bz"].to(device).float())
                loss = multitask_loss(
                    logits,
                    reg_pred,
                    batch["y_class"].to(device),
                    batch["y_reg_scaled"].to(device),
                    batch["reg_mask"].to(device),
                    ce_loss,
                )
                val_losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        history.append(row)
        print(f"{config.name} epoch {epoch}: train={row['train_loss']:.4f} val={row['val_loss']:.4f}", flush=True)
        if row["val_loss"] < best_val - 1e-5:
            best_val = row["val_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "reg_mean": reg_mean,
            "reg_std": reg_std,
            "history": history,
            "architecture": config.architecture,
        },
        CKPT_DIR / f"cnn_sweep_{config.name}.pt",
    )
    return model, history, reg_mean, reg_std


def predict_rows(
    model: MultiTaskCnn,
    rows: pd.DataFrame,
    config: CnnSweepConfig,
    device: torch.device,
    global_mean: float,
    global_std: float,
    reg_mean: np.ndarray,
    reg_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = config.batch_gpu if device.type == "cuda" else config.batch_cpu
    if config.architecture == "lightweight_3d":
        batch_size = min(batch_size, 192 if device.type == "cpu" else 384)
    loader = make_loader(rows.reset_index(drop=True), config, global_mean, global_std, reg_mean, reg_std, batch_size, False)
    probs = []
    regs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits, reg_scaled = model(batch["x_bz"].to(device).float())
            probs.append(logits.softmax(dim=1).cpu().numpy())
            regs.append(reg_scaled.cpu().numpy())
    prob = np.vstack(probs)
    reg = np.vstack(regs) * reg_std[None, :] + reg_mean[None, :]
    return prob, reg


def finite_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return math.nan
    return float(pearsonr(y_true, y_pred).statistic)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else math.nan,
        "pearson_r": finite_pearson(y_true, y_pred),
    }


def evaluate_config(
    rows: pd.DataFrame,
    config: CnnSweepConfig,
    model: MultiTaskCnn,
    device: torch.device,
    global_mean: float,
    global_std: float,
    reg_mean: np.ndarray,
    reg_std: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    result_rows = []
    prediction_rows = []
    for split in ["train", "val", "test"]:
        split_rows = rows[rows["split"].eq(split)].reset_index(drop=True)
        probs, reg_pred = predict_rows(model, split_rows, config, device, global_mean, global_std, reg_mean, reg_std)
        y_class = split_rows["y_condition"].to_numpy(int)
        class_pred = probs.argmax(axis=1)
        cm = confusion_matrix(y_class, class_pred, labels=[0, 1, 2, 3])
        result_rows.append(
            {
                "task": "classification",
                "target": "y_condition",
                "config": config.name,
                "architecture": config.architecture,
                "bz_norm": config.bz_norm,
                "model": "cnn_bz_sweep",
                "split": split,
                "variant": "all",
                "n_eval": int(len(split_rows)),
                "accuracy": float(accuracy_score(y_class, class_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_class, class_pred)),
                "macro_f1": float(f1_score(y_class, class_pred, average="macro")),
                "mae": math.nan,
                "rmse": math.nan,
                "r2": math.nan,
                "pearson_r": math.nan,
                "confusion_matrix": json.dumps(cm.tolist()),
            }
        )
        for i, row in split_rows.iterrows():
            prediction_rows.append(
                {
                    "task": "classification",
                    "target": "y_condition",
                    "config": config.name,
                    "architecture": config.architecture,
                    "bz_norm": config.bz_norm,
                    "model": "cnn_bz_sweep",
                    "split": split,
                    "augmented_sample_id": row["augmented_sample_id"],
                    "base_sample_id": row["base_sample_id"],
                    "variant": row["variant"],
                    "condition": row["condition"],
                    "y_true": int(y_class[i]),
                    "y_pred": int(class_pred[i]),
                    "residual": math.nan,
                }
            )
        for j, target in enumerate(REG_TARGETS):
            y_true_all = split_rows[target].to_numpy(float)
            mask = np.isfinite(y_true_all)
            result_rows.append(
                {
                    "task": "regression",
                    "target": target,
                    "config": config.name,
                    "architecture": config.architecture,
                    "bz_norm": config.bz_norm,
                    "model": "cnn_bz_sweep",
                    "split": split,
                    "variant": "all",
                    "n_eval": int(mask.sum()),
                    "accuracy": math.nan,
                    "balanced_accuracy": math.nan,
                    "macro_f1": math.nan,
                    **regression_metrics(y_true_all[mask], reg_pred[mask, j]),
                    "confusion_matrix": "",
                }
            )
            for row, true_value, pred_value in zip(split_rows[mask].itertuples(index=False), y_true_all[mask], reg_pred[mask, j]):
                prediction_rows.append(
                    {
                        "task": "regression",
                        "target": target,
                        "config": config.name,
                        "architecture": config.architecture,
                        "bz_norm": config.bz_norm,
                        "model": "cnn_bz_sweep",
                        "split": split,
                        "augmented_sample_id": row.augmented_sample_id,
                        "base_sample_id": row.base_sample_id,
                        "variant": row.variant,
                        "condition": row.condition,
                        "y_true": float(true_value),
                        "y_pred": float(pred_value),
                        "residual": float(pred_value - true_value),
                    }
                )
        if split == "test":
            for variant in VARIANT_ORDER:
                variant_rows = split_rows[split_rows["variant"].eq(variant)].reset_index(drop=True)
                probs_v, reg_v = predict_rows(model, variant_rows, config, device, global_mean, global_std, reg_mean, reg_std)
                yv = variant_rows["y_condition"].to_numpy(int)
                pv = probs_v.argmax(axis=1)
                result_rows.append(
                    {
                        "task": "classification",
                        "target": "y_condition",
                        "config": config.name,
                        "architecture": config.architecture,
                        "bz_norm": config.bz_norm,
                        "model": "cnn_bz_sweep",
                        "split": "test",
                        "variant": variant,
                        "n_eval": int(len(variant_rows)),
                        "accuracy": float(accuracy_score(yv, pv)),
                        "balanced_accuracy": float(balanced_accuracy_score(yv, pv)),
                        "macro_f1": float(f1_score(yv, pv, average="macro")),
                        "mae": math.nan,
                        "rmse": math.nan,
                        "r2": math.nan,
                        "pearson_r": math.nan,
                        "confusion_matrix": "",
                    }
                )
                for j, target in enumerate(REG_TARGETS):
                    y_true_all = variant_rows[target].to_numpy(float)
                    mask = np.isfinite(y_true_all)
                    result_rows.append(
                        {
                            "task": "regression",
                            "target": target,
                            "config": config.name,
                            "architecture": config.architecture,
                            "bz_norm": config.bz_norm,
                            "model": "cnn_bz_sweep",
                            "split": "test",
                            "variant": variant,
                            "n_eval": int(mask.sum()),
                            "accuracy": math.nan,
                            "balanced_accuracy": math.nan,
                            "macro_f1": math.nan,
                            **regression_metrics(y_true_all[mask], reg_v[mask, j]),
                            "confusion_matrix": "",
                        }
                    )
    return result_rows, prediction_rows


def json_ready(records: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for record in records:
        out.append({key: None if isinstance(value, float) and math.isnan(value) else value for key, value in record.items()})
    return out


def default_configs() -> list[CnnSweepConfig]:
    configs = []
    for architecture in ["time_channel_2d", "temporal_pooled_2d", "lightweight_3d", "separable_temporal_spatial"]:
        for norm in ["per_sample", "train_global"]:
            name = f"{architecture}_{norm}"
            configs.append(CnnSweepConfig(name=name, architecture=architecture, bz_norm=norm))
    return configs


def plot_figures(results: pd.DataFrame, predictions: pd.DataFrame, histories: dict[str, list[dict[str, float]]]) -> None:
    test = results[(results["split"].eq("test")) & (results["variant"].eq("all"))].copy()
    comparisons = [
        ("y_condition", "macro_f1", BENCHMARKS["y_condition_macro_f1"]),
        ("y_lat_range", "r2", BENCHMARKS["y_lat_range_r2"]),
        ("y_apd90_patch_delta", "r2", BENCHMARKS["y_apd90_patch_delta_r2"]),
        ("y_patch_area_fraction", "r2", BENCHMARKS["y_patch_area_fraction_r2"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, (target, metric, benchmark) in zip(axes.ravel(), comparisons):
        sub = test[test["target"].eq(target)].sort_values(metric, ascending=False)
        labels = sub["config"].str.replace("_", "\n")
        ax.bar(np.arange(len(sub)), sub[metric], color="#4c78a8")
        ax.axhline(benchmark, color="#f58518", linestyle="--", label="best prior")
        ax.set_xticks(np.arange(len(sub)), labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(target)
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cnn_sweep_summary.png", dpi=200)
    plt.close(fig)

    for target, filename in [
        ("y_apd90_patch_delta", "cnn_sweep_apd90_delta.png"),
        ("y_patch_area_fraction", "cnn_sweep_patch_area.png"),
    ]:
        best = test[test["target"].eq(target)].sort_values("r2", ascending=False).iloc[0]
        sub = predictions[
            predictions["target"].eq(target)
            & predictions["config"].eq(best["config"])
            & predictions["split"].eq("test")
        ]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(sub["y_true"], sub["y_pred"], s=8, alpha=0.35, linewidths=0)
        lo = min(sub["y_true"].min(), sub["y_pred"].min())
        hi = max(sub["y_true"].max(), sub["y_pred"].max())
        pad = 0.05 * (hi - lo if hi > lo else 1)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linestyle="--", linewidth=1)
        ax.set_xlabel(f"True {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(f"{target}: {best['config']} R2={best['r2']:.3f}")
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    per_variant = results[(results["split"].eq("test")) & (~results["variant"].eq("all"))].copy()
    for ax, (target, metric, benchmark) in zip(axes.ravel(), comparisons):
        best = test[test["target"].eq(target)].sort_values(metric, ascending=False).iloc[0]
        sub = per_variant[(per_variant["target"].eq(target)) & (per_variant["config"].eq(best["config"]))].copy()
        sub["variant"] = pd.Categorical(sub["variant"], categories=VARIANT_ORDER, ordered=True)
        sub = sub.sort_values("variant")
        ax.bar(sub["variant"].astype(str), sub[metric], color="#72b7b2")
        ax.axhline(benchmark, color="#f58518", linestyle="--")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{target}: {best['config']}")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cnn_sweep_per_variant_metrics.png", dpi=200)
    plt.close(fig)


def run_sweep(max_epochs: int | None = None, patience: int | None = None) -> dict[str, object]:
    seed_everything()
    ensure_dirs()
    rows = load_rows()
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    global_mean, global_std = compute_train_global_stats(train_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_epochs = max_epochs if max_epochs is not None else (16 if device.type == "cpu" else 32)
    patience = patience if patience is not None else (4 if device.type == "cpu" else 7)
    all_results: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    configs = default_configs()
    for config in configs:
        print(f"Training {config.name} on {device}", flush=True)
        model, history, reg_mean, reg_std = train_config(rows, config, device, global_mean, global_std, max_epochs, patience)
        histories[config.name] = history
        results, predictions = evaluate_config(rows, config, model, device, global_mean, global_std, reg_mean, reg_std)
        all_results.extend(results)
        all_predictions.extend(predictions)

    results_df = pd.DataFrame(all_results)
    predictions_df = pd.DataFrame(all_predictions)
    results_df.to_csv(RESULTS_CSV, index=False)
    predictions_df.to_csv(PREDICTIONS_CSV, index=False)
    metadata = {
        "seed": SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_path": str(Path(torch.__file__).resolve()),
        "configs": [config.__dict__ for config in configs],
        "benchmarks": BENCHMARKS,
        "histories": histories,
        "results": json_ready(results_df.to_dict(orient="records")),
        "leakage_policy": "Bz tensors only; grouped base_sample_id split; no oracle windows or metadata inputs",
    }
    RESULTS_JSON.write_text(json.dumps(metadata, indent=2))
    plot_figures(results_df, predictions_df, histories)
    test = results_df[(results_df["split"].eq("test")) & (results_df["variant"].eq("all"))]
    summary = {
        "results_csv": str(RESULTS_CSV.relative_to(PROJECT_ROOT)),
        "predictions_csv": str(PREDICTIONS_CSV.relative_to(PROJECT_ROOT)),
        "device": str(device),
        "best_test": {
            "condition": test[test["target"].eq("y_condition")].sort_values("macro_f1", ascending=False).head(1).to_dict(orient="records"),
            "lat_range": test[test["target"].eq("y_lat_range")].sort_values("r2", ascending=False).head(1).to_dict(orient="records"),
            "apd90_patch_delta": test[test["target"].eq("y_apd90_patch_delta")].sort_values("r2", ascending=False).head(1).to_dict(orient="records"),
            "patch_area_fraction": test[test["target"].eq("y_patch_area_fraction")].sort_values("r2", ascending=False).head(1).to_dict(orient="records"),
        },
    }
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    args = parser.parse_args()
    run_sweep(args.max_epochs, args.patience)


if __name__ == "__main__":
    main()
