                      
                                                                
 
                                                                        
                                                                             
                                                         
   

from __future__ import annotations

import json
import math
import random
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if importlib.util.find_spec("sklearn") is None and (PROJECT_ROOT / ".local_deps" / "sklearn").exists():
    sys.path.insert(0, str(PROJECT_ROOT / ".local_deps" / "sklearn"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

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

from neural_data_common import check_no_group_leakage, load_neural_index, select_deployable_feature_columns


READY_DIR = PROJECT_ROOT / "outputs" / "dataset_v1c_5000_augmented_ready"
OUT_DIR = PROJECT_ROOT / "outputs" / "neural_v1c"
FIG_DIR = PROJECT_ROOT / "figures" / "neural_v1c"
CKPT_DIR = OUT_DIR / "checkpoints"
FIXED = READY_DIR / "features_mig_deployable_fixed_augmented.csv"
MAGNETIC = READY_DIR / "features_mig_deployable_magnetic_augmented.csv"
MLP_RESULTS = OUT_DIR / "mlp_feature_baseline_results.csv"

RESULTS_CSV = OUT_DIR / "fusion_baseline_results.csv"
RESULTS_JSON = OUT_DIR / "fusion_baseline_results.json"
PREDICTIONS_CSV = OUT_DIR / "fusion_predictions.csv"
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
class FusionConfig:
    name: str
    feature_kind: str
    bz_norm: str
    feature_columns: tuple[str, ...]
    train: bool = True


class FusionBzFeatureDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        feature_columns: list[str],
        bz_norm: str,
        global_mean: float | None,
        global_std: float | None,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.bz_norm = bz_norm
        self.global_mean = global_mean
        self.global_std = global_std
        self.features = (
            self.rows[feature_columns].to_numpy(np.float32)
            if feature_columns
            else np.zeros((len(self.rows), 0), dtype=np.float32)
        )

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
            "x_features": torch.from_numpy(self.features[idx]),
            "y_class": torch.tensor(int(row["y_condition"]), dtype=torch.long),
            "y_reg": torch.from_numpy(y_reg),
            "reg_mask": torch.from_numpy(reg_mask),
        }


class FusionNet(nn.Module):
    def __init__(self, feature_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.bz_branch = nn.Sequential(
            nn.Conv2d(301, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(48, 64),
            nn.ReLU(),
        )
        self.feature_dim = feature_dim
        self.feature_branch = (
            nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            if feature_dim > 0
            else None
        )
        fusion_dim = 128 if feature_dim > 0 else 64
        self.fusion = nn.Sequential(nn.Dropout(dropout), nn.Linear(fusion_dim, 64), nn.ReLU(), nn.Dropout(dropout))
        self.class_head = nn.Linear(64, 4)
        self.reg_head = nn.Linear(64, 3)

    def forward(self, x_bz: torch.Tensor, x_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bz_h = self.bz_branch(x_bz)
        if self.feature_branch is not None:
            feat_h = self.feature_branch(x_features)
            h = torch.cat([bz_h, feat_h], dim=1)
        else:
            h = bz_h
        h = self.fusion(h)
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


def load_rows_and_features() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rows = load_neural_index()
    leakage = check_no_group_leakage(rows)
    if leakage:
        raise ValueError(f"Grouped split leakage detected: {leakage[:10]}")
    fixed = pd.read_csv(FIXED)
    magnetic = pd.read_csv(MAGNETIC)
    fixed_cols = select_deployable_feature_columns(fixed)
    magnetic_cols = select_deployable_feature_columns(magnetic)
    fixed = fixed[["augmented_sample_id", *fixed_cols]].rename(columns={col: f"fixed__{col}" for col in fixed_cols})
    magnetic = magnetic[["augmented_sample_id", *magnetic_cols]].rename(
        columns={col: f"magnetic__{col}" for col in magnetic_cols}
    )
    rows = rows.merge(fixed, on="augmented_sample_id", how="inner", validate="one_to_one")
    rows = rows.merge(magnetic, on="augmented_sample_id", how="inner", validate="one_to_one")
    if len(rows) != 55000:
        raise ValueError(f"Expected 55000 fusion rows, got {len(rows)}")

    feature_groups = {
        "none": [],
        "fixed": [f"fixed__{col}" for col in fixed_cols],
        "magnetic": [f"magnetic__{col}" for col in magnetic_cols],
        "fixed_magnetic": [f"fixed__{col}" for col in fixed_cols] + [f"magnetic__{col}" for col in magnetic_cols],
    }
    train = rows["split"].eq("train")
    for cols in feature_groups.values():
        if not cols:
            continue
        mean = rows.loc[train, cols].to_numpy(np.float32).mean(axis=0)
        std = rows.loc[train, cols].to_numpy(np.float32).std(axis=0)
        std[std < 1e-8] = 1.0
        rows.loc[:, cols] = ((rows[cols].to_numpy(np.float32) - mean) / std).astype(np.float32)
    return rows, feature_groups


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
    x_features = torch.stack([item["x_features"] for item in batch])
    y_class = torch.stack([item["y_class"] for item in batch])
    y_reg = torch.stack([item["y_reg"] for item in batch]).float()
    reg_mask = torch.stack([item["reg_mask"] for item in batch]).bool()
    mean = torch.tensor(reg_mean, dtype=torch.float32)
    std = torch.tensor(reg_std, dtype=torch.float32)
    y_reg_scaled = torch.where(reg_mask, (y_reg - mean) / std, torch.zeros_like(y_reg))
    return {
        "x_bz": x_bz,
        "x_features": x_features,
        "y_class": y_class,
        "y_reg": y_reg,
        "y_reg_scaled": y_reg_scaled,
        "reg_mask": reg_mask,
    }


def make_loader(
    rows: pd.DataFrame,
    config: FusionConfig,
    global_mean: float,
    global_std: float,
    reg_mean: np.ndarray,
    reg_std: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = FusionBzFeatureDataset(
        rows,
        feature_columns=list(config.feature_columns),
        bz_norm=config.bz_norm,
        global_mean=global_mean,
        global_std=global_std,
    )
    return DataLoader(
        dataset,
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
    config: FusionConfig,
    device: torch.device,
    global_mean: float,
    global_std: float,
    max_epochs: int,
    patience: int,
) -> tuple[FusionNet, list[dict[str, float]], np.ndarray, np.ndarray]:
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    val_rows = rows[rows["split"].eq("val")].reset_index(drop=True)
    reg_mean, reg_std = target_scaler(train_rows)
    batch_size = 256 if device.type == "cpu" else 512
    train_loader = make_loader(train_rows, config, global_mean, global_std, reg_mean, reg_std, batch_size, True)
    val_loader = make_loader(val_rows, config, global_mean, global_std, reg_mean, reg_std, batch_size, False)
    model = FusionNet(feature_dim=len(config.feature_columns)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
            x_features = batch["x_features"].to(device).float()
            y_class = batch["y_class"].to(device)
            y_reg_scaled = batch["y_reg_scaled"].to(device)
            reg_mask = batch["reg_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, reg_pred = model(x_bz, x_features)
            loss = multitask_loss(logits, reg_pred, y_class, y_reg_scaled, reg_mask, ce_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                logits, reg_pred = model(batch["x_bz"].to(device).float(), batch["x_features"].to(device).float())
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
        print(f"{config.name} epoch {epoch}: train={row['train_loss']:.4f} val={row['val_loss']:.4f}")
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
            "architecture": "Bz CNN branch + deployable MIG feature MLP branch + fusion heads",
        },
        CKPT_DIR / f"fusion_{config.name}.pt",
    )
    return model, history, reg_mean, reg_std


def predict_rows(
    model: FusionNet,
    rows: pd.DataFrame,
    config: FusionConfig,
    device: torch.device,
    global_mean: float,
    global_std: float,
    reg_mean: np.ndarray,
    reg_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(rows.reset_index(drop=True), config, global_mean, global_std, reg_mean, reg_std, 512, False)
    probs = []
    regs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits, reg_scaled = model(batch["x_bz"].to(device).float(), batch["x_features"].to(device).float())
            probs.append(logits.softmax(dim=1).cpu().numpy())
            regs.append(reg_scaled.cpu().numpy())
    prob = np.vstack(probs)
    reg = np.vstack(regs) * reg_std[None, :] + reg_mean[None, :]
    return prob, reg


def finite_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return math.nan
    if np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
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
    config: FusionConfig,
    model: FusionNet,
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
                "feature_kind": config.feature_kind,
                "bz_norm": config.bz_norm,
                "model": "cnn_mig_fusion",
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
                    "feature_kind": config.feature_kind,
                    "bz_norm": config.bz_norm,
                    "model": "cnn_mig_fusion",
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
                    "feature_kind": config.feature_kind,
                    "bz_norm": config.bz_norm,
                    "model": "cnn_mig_fusion",
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
                        "feature_kind": config.feature_kind,
                        "bz_norm": config.bz_norm,
                        "model": "cnn_mig_fusion",
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
                        "feature_kind": config.feature_kind,
                        "bz_norm": config.bz_norm,
                        "model": "cnn_mig_fusion",
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
                            "feature_kind": config.feature_kind,
                            "bz_norm": config.bz_norm,
                            "model": "cnn_mig_fusion",
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
    cleaned = []
    for record in records:
        row = {}
        for key, value in record.items():
            row[key] = None if isinstance(value, float) and math.isnan(value) else value
        cleaned.append(row)
    return cleaned


def previous_mlp_rows() -> pd.DataFrame:
    if not MLP_RESULTS.exists():
        return pd.DataFrame()
    df = pd.read_csv(MLP_RESULTS)
    df = df.copy()
    df["config"] = "feature_only_mlp_previous_" + df["feature_setting"].astype(str)
    df["feature_kind"] = df["feature_setting"].astype(str)
    df["bz_norm"] = "none"
    df["variant"] = "all"
    df["model"] = "mlp_feature_previous"
    return df[
        [
            "task",
            "target",
            "config",
            "feature_kind",
            "bz_norm",
            "model",
            "split",
            "variant",
            "n_eval",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "mae",
            "rmse",
            "r2",
            "pearson_r",
        ]
    ]


def plot_figures(results: pd.DataFrame, predictions: pd.DataFrame, histories: dict[str, list[dict[str, float]]]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, hist in histories.items():
        h = pd.DataFrame(hist)
        ax.plot(h["epoch"], h["train_loss"], label=f"{name} train", alpha=0.75)
        ax.plot(h["epoch"], h["val_loss"], label=f"{name} val", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Fusion Training Curves")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fusion_training_curves.png", dpi=200)
    plt.close(fig)

    test = results[(results["split"].eq("test")) & (results["variant"].eq("all"))].copy()
    comparisons = [
        ("y_condition", "macro_f1", BENCHMARKS["y_condition_macro_f1"]),
        ("y_lat_range", "r2", BENCHMARKS["y_lat_range_r2"]),
        ("y_apd90_patch_delta", "r2", BENCHMARKS["y_apd90_patch_delta_r2"]),
        ("y_patch_area_fraction", "r2", BENCHMARKS["y_patch_area_fraction_r2"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (target, metric, benchmark) in zip(axes.ravel(), comparisons):
        sub = test[test["target"].eq(target)].copy()
        sub = sub.sort_values(metric, ascending=False).head(8)
        labels = sub["config"].str.replace("_", "\n")
        ax.bar(np.arange(len(sub)), sub[metric], color="#4c78a8")
        ax.axhline(benchmark, color="#f58518", linestyle="--", label="best prior")
        ax.set_xticks(np.arange(len(sub)), labels, rotation=25, ha="right", fontsize=7)
        ax.set_title(target)
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fusion_vs_cnn_vs_mlp_summary.png", dpi=200)
    plt.close(fig)

    fusion_test = test[test["model"].eq("cnn_mig_fusion")].copy()
    best_cls = fusion_test[fusion_test["target"].eq("y_condition")].sort_values("macro_f1", ascending=False).iloc[0]
    pred_cls = predictions[
        predictions["target"].eq("y_condition")
        & predictions["config"].eq(best_cls["config"])
        & predictions["split"].eq("test")
    ]
    labels = sorted(pred_cls["y_true"].astype(int).unique())
    cm = confusion_matrix(pred_cls["y_true"].astype(int), pred_cls["y_pred"].astype(int), labels=labels)
    names = [pred_cls[pred_cls["y_true"].astype(int).eq(label)]["condition"].mode().iloc[0].replace("_", "\n") for label in labels]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Fusion Confusion Matrix: {best_cls['config']} macro F1={best_cls['macro_f1']:.3f}")
    threshold = cm.max() * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fusion_confusion_matrix.png", dpi=200)
    plt.close(fig)

    for target, filename in [
        ("y_apd90_patch_delta", "fusion_regression_apd90_delta.png"),
        ("y_patch_area_fraction", "fusion_regression_patch_area.png"),
    ]:
        best = fusion_test[fusion_test["target"].eq(target)].sort_values("r2", ascending=False).iloc[0]
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

    per_variant = results[
        (results["split"].eq("test"))
        & (~results["variant"].eq("all"))
        & (results["model"].eq("cnn_mig_fusion"))
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    for ax, (target, metric, benchmark) in zip(axes.ravel(), comparisons):
        best = fusion_test[fusion_test["target"].eq(target)].sort_values(metric, ascending=False).iloc[0]
        sub = per_variant[(per_variant["target"].eq(target)) & (per_variant["config"].eq(best["config"]))].copy()
        sub["variant"] = pd.Categorical(sub["variant"], categories=VARIANT_ORDER, ordered=True)
        sub = sub.sort_values("variant")
        ax.bar(sub["variant"].astype(str), sub[metric], color="#72b7b2")
        ax.axhline(benchmark, color="#f58518", linestyle="--")
        ax.set_title(target)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=35)
        ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fusion_per_variant_metrics.png", dpi=200)
    plt.close(fig)


def main() -> None:
    seed_everything()
    ensure_dirs()
    rows, feature_groups = load_rows_and_features()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_mean, global_std = compute_train_global_stats(rows[rows["split"].eq("train")])
    configs = [
        FusionConfig("bz_only_per_sample", "none", "per_sample", tuple(feature_groups["none"])),
        FusionConfig("bz_only_train_global", "none", "train_global", tuple(feature_groups["none"])),
        FusionConfig("fusion_fixed_train_global", "fixed", "train_global", tuple(feature_groups["fixed"])),
        FusionConfig("fusion_magnetic_train_global", "magnetic", "train_global", tuple(feature_groups["magnetic"])),
        FusionConfig("fusion_fixed_magnetic_per_sample", "fixed_magnetic", "per_sample", tuple(feature_groups["fixed_magnetic"])),
        FusionConfig("fusion_fixed_magnetic_train_global", "fixed_magnetic", "train_global", tuple(feature_groups["fixed_magnetic"])),
    ]
    max_epochs = 12 if device.type == "cpu" else 24
    patience = 3 if device.type == "cpu" else 5
    all_results: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    for config in configs:
        print(f"Training {config.name} on {device} with {len(config.feature_columns)} features")
        model, history, reg_mean, reg_std = train_config(rows, config, device, global_mean, global_std, max_epochs, patience)
        histories[config.name] = history
        results, predictions = evaluate_config(rows, config, model, device, global_mean, global_std, reg_mean, reg_std)
        all_results.extend(results)
        all_predictions.extend(predictions)

    trained_results = pd.DataFrame(all_results)
    mlp_rows = previous_mlp_rows()
    combined_results = pd.concat([trained_results, mlp_rows], ignore_index=True, sort=False) if not mlp_rows.empty else trained_results
    predictions = pd.DataFrame(all_predictions)
    combined_results.to_csv(RESULTS_CSV, index=False)
    predictions.to_csv(PREDICTIONS_CSV, index=False)
    metadata = {
        "seed": SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "architecture": "Bz CNN branch plus deployable MIG feature MLP branch with fused multitask heads",
        "configs_trained": [config.__dict__ for config in configs],
        "previous_mlp_loaded": bool(not mlp_rows.empty),
        "benchmarks": BENCHMARKS,
        "split_policy": "grouped by base_sample_id",
        "leakage_policy": "Bz tensors plus deployable MIG features only; no oracle windows or metadata inputs",
        "histories": histories,
        "results": json_ready(combined_results.to_dict(orient="records")),
    }
    RESULTS_JSON.write_text(json.dumps(metadata, indent=2))
    plot_figures(combined_results, predictions, histories)
    test = combined_results[(combined_results["split"].eq("test")) & (combined_results["variant"].eq("all"))]
    summary = {
        "results_csv": str(RESULTS_CSV.relative_to(PROJECT_ROOT)),
        "results_json": str(RESULTS_JSON.relative_to(PROJECT_ROOT)),
        "predictions_csv": str(PREDICTIONS_CSV.relative_to(PROJECT_ROOT)),
        "device": str(device),
        "best_test": {
            "condition": test[test["target"].eq("y_condition")].sort_values("macro_f1", ascending=False).head(1).to_dict(orient="records"),
            "lat_range": test[test["target"].eq("y_lat_range")].sort_values("r2", ascending=False).head(1).to_dict(orient="records"),
            "apd90_patch_delta": test[test["target"].eq("y_apd90_patch_delta")].sort_values("r2", ascending=False).head(1).to_dict(orient="records"),
            "patch_area_fraction": test[test["target"].eq("y_patch_area_fraction")].sort_values("r2", ascending=False).head(1).to_dict(orient="records"),
        },
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
