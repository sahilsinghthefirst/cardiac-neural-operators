                      
                                                                
 
                                                                        
                                                                           
                                      
   

from __future__ import annotations

import json
import math
import random
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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

from neural_data_common import check_no_group_leakage, load_neural_index


OUT_DIR = PROJECT_ROOT / "outputs" / "neural_v1c"
FIG_DIR = PROJECT_ROOT / "figures" / "neural_v1c"
CKPT_DIR = OUT_DIR / "checkpoints"
RESULTS_CSV = OUT_DIR / "cnn_bz_baseline_results.csv"
RESULTS_JSON = OUT_DIR / "cnn_bz_baseline_results.json"
PREDICTIONS_CSV = OUT_DIR / "cnn_bz_predictions.csv"
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
    "y_apd90_patch_delta_r2": 0.670,
    "y_patch_area_fraction_r2": 0.279,
}


@dataclass(frozen=True)
class NormalizationSetting:
    name: str
    mode: str
    mean: float | None = None
    std: float | None = None


class BzCnnDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        normalization: NormalizationSetting,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.rows)

    def _load_x(self, row: pd.Series) -> np.ndarray:
        bz = np.load(PROJECT_ROOT / row["Bz_file"]).astype(np.float32)
        mask = np.load(PROJECT_ROOT / row["sensor_mask_file"]).astype(bool)
        if bz.shape != (301, 64):
            raise ValueError(f"Expected Bz shape [301,64], got {bz.shape} for {row['augmented_sample_id']}")
        bz[:, ~mask] = np.nan
        valid = bz[:, mask]
        finite = np.isfinite(valid)
        if not finite.any():
            raise ValueError(f"No finite valid Bz values for {row['augmented_sample_id']}")
        if self.normalization.mode == "per_sample":
            center = float(np.nanmean(valid))
            scale = float(np.nanstd(valid))
        elif self.normalization.mode == "train_global":
            center = float(self.normalization.mean)
            scale = float(self.normalization.std)
        else:
            center = 0.0
            scale = 1.0
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        bz = (bz - center) / scale
        bz = np.where(np.isfinite(bz), bz, 0.0).astype(np.float32)
        return bz.reshape(301, 8, 8)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.rows.iloc[idx]
        x = self._load_x(row)
        y_reg = np.array([row[target] for target in REG_TARGETS], dtype=np.float32)
        reg_mask = np.isfinite(y_reg)
        y_reg = np.where(reg_mask, y_reg, 0.0).astype(np.float32)
        return {
            "x": torch.from_numpy(x),
            "y_class": torch.tensor(int(row["y_condition"]), dtype=torch.long),
            "y_reg": torch.from_numpy(y_reg),
            "reg_mask": torch.from_numpy(reg_mask),
            "row_index": torch.tensor(idx, dtype=torch.long),
        }


class SmallBzCnn(nn.Module):
                                                           

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
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
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(48, 64), nn.ReLU(), nn.Dropout(dropout))
        self.class_head = nn.Linear(64, 4)
        self.reg_head = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(self.trunk(x))
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
    df = load_neural_index()
    leakage = check_no_group_leakage(df)
    if leakage:
        raise ValueError(f"Grouped split leakage detected: {leakage[:10]}")
    return df


def compute_train_global_stats(train_rows: pd.DataFrame, max_rows: int | None = None) -> tuple[float, float]:
    if STATS_JSON.exists():
        data = json.loads(STATS_JSON.read_text())
        return float(data["mean"]), float(data["std"])
    rows = train_rows
    if max_rows is not None and len(rows) > max_rows:
        rows = rows.sample(n=max_rows, random_state=SEED)
    total = 0.0
    total_sq = 0.0
    count = 0
    for row in rows.itertuples(index=False):
        bz = np.load(PROJECT_ROOT / row.Bz_file).astype(np.float64)
        mask = np.load(PROJECT_ROOT / row.sensor_mask_file).astype(bool)
        vals = bz[:, mask]
        vals = vals[np.isfinite(vals)]
        total += float(vals.sum())
        total_sq += float(np.square(vals).sum())
        count += int(vals.size)
    mean = total / max(count, 1)
    variance = max(total_sq / max(count, 1) - mean * mean, 0.0)
    std = math.sqrt(variance) if variance > 0 else 1.0
    STATS_JSON.write_text(
        json.dumps(
            {
                "mean": mean,
                "std": std,
                "count": count,
                "train_rows_used": int(len(rows)),
                "policy": "computed from train split valid sensors only",
            },
            indent=2,
        )
    )
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


def collate_with_scaled_targets(batch: list[dict[str, object]], reg_mean: np.ndarray, reg_std: np.ndarray) -> dict[str, torch.Tensor]:
    x = torch.stack([item["x"] for item in batch])
    y_class = torch.stack([item["y_class"] for item in batch])
    y_reg = torch.stack([item["y_reg"] for item in batch]).float()
    reg_mask = torch.stack([item["reg_mask"] for item in batch]).bool()
    row_index = torch.stack([item["row_index"] for item in batch])
    mean = torch.tensor(reg_mean, dtype=torch.float32)
    std = torch.tensor(reg_std, dtype=torch.float32)
    y_reg_scaled = (y_reg - mean) / std
    y_reg_scaled = torch.where(reg_mask, y_reg_scaled, torch.zeros_like(y_reg_scaled))
    return {"x": x, "y_class": y_class, "y_reg": y_reg, "y_reg_scaled": y_reg_scaled, "reg_mask": reg_mask, "row_index": row_index}


def make_loader(
    rows: pd.DataFrame,
    normalization: NormalizationSetting,
    reg_mean: np.ndarray,
    reg_std: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = BzCnnDataset(rows, normalization)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda batch: collate_with_scaled_targets(batch, reg_mean, reg_std),
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


def train_one(
    rows: pd.DataFrame,
    normalization: NormalizationSetting,
    device: torch.device,
    max_epochs: int = 24,
    patience: int = 5,
) -> tuple[SmallBzCnn, list[dict[str, float]], np.ndarray, np.ndarray]:
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    val_rows = rows[rows["split"].eq("val")].reset_index(drop=True)
    reg_mean, reg_std = target_scaler(train_rows)
    batch_size = 256 if device.type == "cpu" else 512
    train_loader = make_loader(train_rows, normalization, reg_mean, reg_std, batch_size, shuffle=True)
    val_loader = make_loader(val_rows, normalization, reg_mean, reg_std, batch_size, shuffle=False)
    model = SmallBzCnn().to(device)
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
            x = batch["x"].to(device).float()
            y_class = batch["y_class"].to(device)
            y_reg_scaled = batch["y_reg_scaled"].to(device)
            reg_mask = batch["reg_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, reg_pred = model(x)
            loss = multitask_loss(logits, reg_pred, y_class, y_reg_scaled, reg_mask, ce_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device).float()
                y_class = batch["y_class"].to(device)
                y_reg_scaled = batch["y_reg_scaled"].to(device)
                reg_mask = batch["reg_mask"].to(device)
                logits, reg_pred = model(x)
                loss = multitask_loss(logits, reg_pred, y_class, y_reg_scaled, reg_mask, ce_loss)
                val_losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        history.append(row)
        print(f"{normalization.name} epoch {epoch}: train={row['train_loss']:.4f} val={row['val_loss']:.4f}")
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
            "normalization": normalization.__dict__,
            "reg_mean": reg_mean,
            "reg_std": reg_std,
            "history": history,
            "architecture": "2D CNN with time as channels [301,8,8]",
        },
        CKPT_DIR / f"cnn_bz_{normalization.name}.pt",
    )
    return model, history, reg_mean, reg_std


def predict_split(
    model: SmallBzCnn,
    rows: pd.DataFrame,
    normalization: NormalizationSetting,
    reg_mean: np.ndarray,
    reg_std: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(rows.reset_index(drop=True), normalization, reg_mean, reg_std, batch_size=512, shuffle=False)
    class_probs = []
    regs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device).float()
            logits, reg_scaled = model(x)
            class_probs.append(logits.softmax(dim=1).cpu().numpy())
            regs.append(reg_scaled.cpu().numpy())
    probs = np.vstack(class_probs)
    reg_scaled = np.vstack(regs)
    reg = reg_scaled * reg_std[None, :] + reg_mean[None, :]
    return probs, reg


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


def evaluate_setting(
    rows: pd.DataFrame,
    normalization: NormalizationSetting,
    model: SmallBzCnn,
    reg_mean: np.ndarray,
    reg_std: np.ndarray,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    result_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for split in ["train", "val", "test"]:
        split_rows = rows[rows["split"].eq(split)].reset_index(drop=True)
        probs, reg_pred = predict_split(model, split_rows, normalization, reg_mean, reg_std, device)
        class_pred = probs.argmax(axis=1)
        y_class = split_rows["y_condition"].to_numpy(int)
        cm = confusion_matrix(y_class, class_pred, labels=[0, 1, 2, 3])
        result_rows.append(
            {
                "task": "classification",
                "target": "y_condition",
                "normalization": normalization.name,
                "model": "small_bz_2d_cnn",
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
                    "normalization": normalization.name,
                    "model": "small_bz_2d_cnn",
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
            y_true = y_true_all[mask]
            y_pred = reg_pred[mask, j]
            result = {
                "task": "regression",
                "target": target,
                "normalization": normalization.name,
                "model": "small_bz_2d_cnn",
                "split": split,
                "variant": "all",
                "n_eval": int(mask.sum()),
                "accuracy": math.nan,
                "balanced_accuracy": math.nan,
                "macro_f1": math.nan,
                **regression_metrics(y_true, y_pred),
                "confusion_matrix": "",
            }
            result_rows.append(result)
            for row, true_value, pred_value in zip(split_rows[mask].itertuples(index=False), y_true, y_pred):
                prediction_rows.append(
                    {
                        "task": "regression",
                        "target": target,
                        "normalization": normalization.name,
                        "model": "small_bz_2d_cnn",
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
                if variant_rows.empty:
                    continue
                probs_v, reg_v = predict_split(model, variant_rows, normalization, reg_mean, reg_std, device)
                yv = variant_rows["y_condition"].to_numpy(int)
                pv = probs_v.argmax(axis=1)
                result_rows.append(
                    {
                        "task": "classification",
                        "target": "y_condition",
                        "normalization": normalization.name,
                        "model": "small_bz_2d_cnn",
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
                            "normalization": normalization.name,
                            "model": "small_bz_2d_cnn",
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
        row = {}
        for key, value in record.items():
            row[key] = None if isinstance(value, float) and math.isnan(value) else value
        out.append(row)
    return out


def plot_figures(results: pd.DataFrame, predictions: pd.DataFrame, histories: dict[str, list[dict[str, float]]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, hist in histories.items():
        h = pd.DataFrame(hist)
        ax.plot(h["epoch"], h["train_loss"], label=f"{name} train")
        ax.plot(h["epoch"], h["val_loss"], label=f"{name} val", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("CNN Bz Training Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cnn_training_curves.png", dpi=200)
    plt.close(fig)

    test = results[(results["split"].eq("test")) & (results["variant"].eq("all"))].copy()
    comparisons = [
        ("y_condition", "macro_f1", BENCHMARKS["y_condition_macro_f1"]),
        ("y_lat_range", "r2", BENCHMARKS["y_lat_range_r2"]),
        ("y_apd90_patch_delta", "r2", BENCHMARKS["y_apd90_patch_delta_r2"]),
        ("y_patch_area_fraction", "r2", BENCHMARKS["y_patch_area_fraction_r2"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (target, metric, baseline) in zip(axes.ravel(), comparisons):
        sub = test[test["target"].eq(target)].sort_values("normalization")
        ax.bar(sub["normalization"], sub[metric], color="#4c78a8")
        ax.axhline(baseline, color="#f58518", linestyle="--", label="benchmark")
        ax.set_title(target)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=15)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cnn_vs_mlp_vs_classical_summary.png", dpi=200)
    plt.close(fig)

    best_cls = test[test["target"].eq("y_condition")].sort_values("macro_f1", ascending=False).iloc[0]
    sub = predictions[
        predictions["target"].eq("y_condition")
        & predictions["normalization"].eq(best_cls["normalization"])
        & predictions["split"].eq("test")
    ]
    labels = sorted(sub["y_true"].astype(int).unique())
    cm = confusion_matrix(sub["y_true"].astype(int), sub["y_pred"].astype(int), labels=labels)
    names = [sub[sub["y_true"].astype(int).eq(label)]["condition"].mode().iloc[0].replace("_", "\n") for label in labels]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"CNN Confusion Matrix: {best_cls['normalization']} macro F1={best_cls['macro_f1']:.3f}")
    threshold = cm.max() * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cnn_confusion_matrix.png", dpi=200)
    plt.close(fig)

    for target, filename in [
        ("y_apd90_patch_delta", "cnn_regression_apd90_delta.png"),
        ("y_patch_area_fraction", "cnn_regression_patch_area.png"),
    ]:
        best = test[test["target"].eq(target)].sort_values("r2", ascending=False).iloc[0]
        sub = predictions[
            predictions["target"].eq(target)
            & predictions["normalization"].eq(best["normalization"])
            & predictions["split"].eq("test")
        ].copy()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(sub["y_true"], sub["y_pred"], s=8, alpha=0.35, linewidths=0)
        lo = min(sub["y_true"].min(), sub["y_pred"].min())
        hi = max(sub["y_true"].max(), sub["y_pred"].max())
        pad = 0.05 * (hi - lo if hi > lo else 1)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linestyle="--", linewidth=1)
        ax.set_xlabel(f"True {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(f"{target}: {best['normalization']} R2={best['r2']:.3f}")
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=200)
        plt.close(fig)

    per_variant = results[(results["split"].eq("test")) & (~results["variant"].eq("all"))].copy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    specs = [
        ("y_condition", "macro_f1", "Condition macro F1"),
        ("y_lat_range", "r2", "LAT range R2"),
        ("y_apd90_patch_delta", "r2", "APD90 patch delta R2"),
        ("y_patch_area_fraction", "r2", "Patch area fraction R2"),
    ]
    for ax, (target, metric, title) in zip(axes.ravel(), specs):
        best_norm = test[test["target"].eq(target)].sort_values(metric, ascending=False).iloc[0]["normalization"]
        sub = per_variant[(per_variant["target"].eq(target)) & (per_variant["normalization"].eq(best_norm))].copy()
        sub["variant"] = pd.Categorical(sub["variant"], categories=VARIANT_ORDER, ordered=True)
        sub = sub.sort_values("variant")
        ax.bar(sub["variant"].astype(str), sub[metric], color="#72b7b2")
        ax.axhline(BENCHMARKS["y_condition_macro_f1"] if target == "y_condition" else BENCHMARKS[f"{target}_r2"], color="#f58518", linestyle="--")
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=35)
        ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cnn_per_variant_metrics.png", dpi=200)
    plt.close(fig)


def main() -> None:
    seed_everything()
    ensure_dirs()
    rows = load_rows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = rows[rows["split"].eq("train")]
    mean, std = compute_train_global_stats(train_rows)
    settings = [
        NormalizationSetting("per_sample", "per_sample"),
        NormalizationSetting("train_global", "train_global", mean, std),
    ]
    all_results: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    for setting in settings:
        print(f"Training CNN normalization={setting.name} device={device}")
        model, history, reg_mean, reg_std = train_one(rows, setting, device)
        histories[setting.name] = history
        results, predictions = evaluate_setting(rows, setting, model, reg_mean, reg_std, device)
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
        "architecture": "small 2D CNN with time as channels [301,8,8], shared trunk, classification/regression heads",
        "normalization_settings": [setting.__dict__ for setting in settings],
        "benchmarks": BENCHMARKS,
        "split_policy": "grouped by base_sample_id",
        "leakage_policy": "Bz tensors only; no oracle LAT/RT/APD-windowed features or metadata inputs",
        "histories": histories,
        "results": json_ready(all_results),
    }
    RESULTS_JSON.write_text(json.dumps(metadata, indent=2))
    plot_figures(results_df, predictions_df, histories)
    test = results_df[(results_df["split"].eq("test")) & (results_df["variant"].eq("all"))]
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
