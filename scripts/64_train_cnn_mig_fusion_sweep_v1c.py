                      
                                                                          
 
                                                                            
                                                                            
                                                          
   

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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

torch = import_torch_prefer_env(PROJECT_ROOT)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from neural_data_common import check_no_group_leakage, load_neural_index, select_deployable_feature_columns

_sweep_spec = importlib.util.spec_from_file_location("cnn_bz_sweep_v1c", PROJECT_ROOT / "scripts" / "63_train_cnn_bz_sweep_v1c.py")
if _sweep_spec is None or _sweep_spec.loader is None:
    raise ImportError("Could not load scripts/63_train_cnn_bz_sweep_v1c.py")
_sweep = importlib.util.module_from_spec(_sweep_spec)
sys.modules["cnn_bz_sweep_v1c"] = _sweep
_sweep_spec.loader.exec_module(_sweep)

BENCHMARKS = _sweep.BENCHMARKS
CKPT_DIR = _sweep.CKPT_DIR
FIG_DIR = _sweep.FIG_DIR
OUT_DIR = _sweep.OUT_DIR
REG_TARGETS = _sweep.REG_TARGETS
SEED = _sweep.SEED
VARIANT_ORDER = _sweep.VARIANT_ORDER
class_weights = _sweep.class_weights
compute_train_global_stats = _sweep.compute_train_global_stats
json_ready = _sweep.json_ready
make_backbone = _sweep.make_backbone
multitask_loss = _sweep.multitask_loss
regression_metrics = _sweep.regression_metrics
seed_everything = _sweep.seed_everything
target_scaler = _sweep.target_scaler


READY_DIR = PROJECT_ROOT / "outputs" / "dataset_v1c_5000_augmented_ready"
FIXED = READY_DIR / "features_mig_deployable_fixed_augmented.csv"
MAGNETIC = READY_DIR / "features_mig_deployable_magnetic_augmented.csv"
CNN_SWEEP_RESULTS = OUT_DIR / "cnn_sweep_results.csv"
RESULTS_CSV = OUT_DIR / "fusion_sweep_results.csv"
RESULTS_JSON = OUT_DIR / "fusion_sweep_results.json"
PREDICTIONS_CSV = OUT_DIR / "fusion_sweep_predictions.csv"
MLP_RESULTS = OUT_DIR / "mlp_feature_baseline_results.csv"


@dataclass(frozen=True)
class FusionSweepConfig:
    name: str
    architecture: str
    bz_norm: str
    feature_kind: str
    feature_columns: tuple[str, ...]
    dropout: float = 0.22
    lr: float = 8e-4
    weight_decay: float = 1e-4
    batch_cpu: int = 384
    batch_gpu: int = 768


class FusionSweepDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        config: FusionSweepConfig,
        global_mean: float,
        global_std: float,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.config = config
        self.global_mean = global_mean
        self.global_std = global_std
        self.features = (
            self.rows[list(config.feature_columns)].to_numpy(np.float32)
            if config.feature_columns
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
        if self.config.bz_norm == "per_sample":
            mean = float(np.nanmean(valid))
            std = float(np.nanstd(valid))
        elif self.config.bz_norm == "train_global":
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


class FusionSweepNet(nn.Module):
    def __init__(self, architecture: str, feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.bz_branch = make_backbone(architecture, dropout)
        self.bz_project = nn.Sequential(nn.Linear(self.bz_branch.out_dim, 96), nn.ReLU(), nn.Dropout(dropout))
        self.feature_dim = feature_dim
        self.feature_branch = (
            nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 96),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            if feature_dim > 0
            else None
        )
        fusion_dim = 192 if feature_dim > 0 else 96
        self.fusion = nn.Sequential(nn.Linear(fusion_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 96), nn.ReLU())
        self.class_head = nn.Linear(96, 4)
        self.reg_head = nn.Linear(96, 3)

    def forward(self, x_bz: torch.Tensor, x_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bz_h = self.bz_project(self.bz_branch(x_bz))
        if self.feature_branch is not None:
            feat_h = self.feature_branch(x_features)
            h = torch.cat([bz_h, feat_h], dim=1)
        else:
            h = bz_h
        h = self.fusion(h)
        return self.class_head(h), self.reg_head(h)


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
        raise ValueError(f"Expected 55000 rows, got {len(rows)}")
    feature_groups = {
        "fixed": [f"fixed__{col}" for col in fixed_cols],
        "magnetic": [f"magnetic__{col}" for col in magnetic_cols],
        "fixed_magnetic": [f"fixed__{col}" for col in fixed_cols] + [f"magnetic__{col}" for col in magnetic_cols],
    }
    train = rows["split"].eq("train")
    for cols in feature_groups.values():
        mean = rows.loc[train, cols].to_numpy(np.float32).mean(axis=0)
        std = rows.loc[train, cols].to_numpy(np.float32).std(axis=0)
        std[std < 1e-8] = 1.0
        rows.loc[:, cols] = ((rows[cols].to_numpy(np.float32) - mean) / std).astype(np.float32)
    return rows, feature_groups


def select_best_cnn_config() -> tuple[str, str, dict[str, float]]:
    if not CNN_SWEEP_RESULTS.exists():
        raise FileNotFoundError(f"Missing {CNN_SWEEP_RESULTS}; run scripts/63_train_cnn_bz_sweep_v1c.py first")
    results = pd.read_csv(CNN_SWEEP_RESULTS)
    test = results[(results["split"].eq("test")) & (results["variant"].eq("all"))]
    rows = []
    for config, group in test.groupby("config"):
        apd = group[group["target"].eq("y_apd90_patch_delta")]["r2"]
        patch = group[group["target"].eq("y_patch_area_fraction")]["r2"]
        cond = group[group["target"].eq("y_condition")]["macro_f1"]
        lat = group[group["target"].eq("y_lat_range")]["r2"]
        if apd.empty or patch.empty:
            continue
        score = float(apd.iloc[0]) + float(patch.iloc[0])
        rows.append(
            {
                "config": config,
                "architecture": group["architecture"].iloc[0],
                "bz_norm": group["bz_norm"].iloc[0],
                "score": score,
                "apd90_r2": float(apd.iloc[0]),
                "patch_r2": float(patch.iloc[0]),
                "condition_macro_f1": float(cond.iloc[0]) if not cond.empty else math.nan,
                "lat_range_r2": float(lat.iloc[0]) if not lat.empty else math.nan,
            }
        )
    if not rows:
        raise ValueError("Could not select a CNN config from sweep results")
    best = sorted(rows, key=lambda row: row["score"], reverse=True)[0]
    return str(best["architecture"]), str(best["bz_norm"]), best


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
    config: FusionSweepConfig,
    global_mean: float,
    global_std: float,
    reg_mean: np.ndarray,
    reg_std: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        FusionSweepDataset(rows, config, global_mean, global_std),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda batch: collate_scaled(batch, reg_mean, reg_std),
    )


def train_config(
    rows: pd.DataFrame,
    config: FusionSweepConfig,
    device: torch.device,
    global_mean: float,
    global_std: float,
    max_epochs: int,
    patience: int,
) -> tuple[FusionSweepNet, list[dict[str, float]], np.ndarray, np.ndarray]:
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    val_rows = rows[rows["split"].eq("val")].reset_index(drop=True)
    reg_mean, reg_std = target_scaler(train_rows)
    batch_size = config.batch_gpu if device.type == "cuda" else config.batch_cpu
    if config.architecture == "lightweight_3d":
        batch_size = min(batch_size, 192 if device.type == "cpu" else 384)
    train_loader = make_loader(train_rows, config, global_mean, global_std, reg_mean, reg_std, batch_size, True)
    val_loader = make_loader(val_rows, config, global_mean, global_std, reg_mean, reg_std, batch_size, False)
    model = FusionSweepNet(config.architecture, len(config.feature_columns), config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights(train_rows, device))
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, reg_pred = model(batch["x_bz"].to(device).float(), batch["x_features"].to(device).float())
            loss = multitask_loss(
                logits,
                reg_pred,
                batch["y_class"].to(device),
                batch["y_reg_scaled"].to(device),
                batch["reg_mask"].to(device),
                ce_loss,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
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
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": float(np.mean(val_losses))}
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
            "architecture": "best CNN sweep backbone plus deployable MIG feature branch",
        },
        CKPT_DIR / f"fusion_sweep_{config.name}.pt",
    )
    return model, history, reg_mean, reg_std


def predict_rows(
    model: FusionSweepNet,
    rows: pd.DataFrame,
    config: FusionSweepConfig,
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
            logits, reg_scaled = model(batch["x_bz"].to(device).float(), batch["x_features"].to(device).float())
            probs.append(logits.softmax(dim=1).cpu().numpy())
            regs.append(reg_scaled.cpu().numpy())
    prob = np.vstack(probs)
    reg = np.vstack(regs) * reg_std[None, :] + reg_mean[None, :]
    return prob, reg


def evaluate_config(
    rows: pd.DataFrame,
    config: FusionSweepConfig,
    model: FusionSweepNet,
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
                "feature_kind": config.feature_kind,
                "bz_norm": config.bz_norm,
                "model": "cnn_mig_fusion_sweep",
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
                    "feature_kind": config.feature_kind,
                    "bz_norm": config.bz_norm,
                    "model": "cnn_mig_fusion_sweep",
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
                    "feature_kind": config.feature_kind,
                    "bz_norm": config.bz_norm,
                    "model": "cnn_mig_fusion_sweep",
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
                        "feature_kind": config.feature_kind,
                        "bz_norm": config.bz_norm,
                        "model": "cnn_mig_fusion_sweep",
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
                        "feature_kind": config.feature_kind,
                        "bz_norm": config.bz_norm,
                        "model": "cnn_mig_fusion_sweep",
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
                            "feature_kind": config.feature_kind,
                            "bz_norm": config.bz_norm,
                            "model": "cnn_mig_fusion_sweep",
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


def previous_rows() -> pd.DataFrame:
    frames = []
    if CNN_SWEEP_RESULTS.exists():
        cnn = pd.read_csv(CNN_SWEEP_RESULTS)
        frames.append(cnn[(cnn["split"].eq("test")) & (cnn["variant"].eq("all"))].copy())
    if MLP_RESULTS.exists():
        mlp = pd.read_csv(MLP_RESULTS)
        if "variant" not in mlp.columns:
            mlp["variant"] = "all"
        mlp = mlp[(mlp["split"].eq("test")) & (mlp["variant"].eq("all"))].copy()
        mlp["architecture"] = "feature_only"
        mlp["bz_norm"] = "none"
        frames.append(mlp)
    if frames:
        return pd.concat(frames, ignore_index=True, sort=False)
    return pd.DataFrame()


def plot_figures(results: pd.DataFrame, predictions: pd.DataFrame) -> None:
    test = results[(results["split"].eq("test")) & (results["variant"].eq("all"))].copy()
    comparisons = [
        ("y_condition", "macro_f1", BENCHMARKS["y_condition_macro_f1"]),
        ("y_lat_range", "r2", BENCHMARKS["y_lat_range_r2"]),
        ("y_apd90_patch_delta", "r2", BENCHMARKS["y_apd90_patch_delta_r2"]),
        ("y_patch_area_fraction", "r2", BENCHMARKS["y_patch_area_fraction_r2"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
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
    fig.savefig(FIG_DIR / "fusion_sweep_summary.png", dpi=200)
    plt.close(fig)

    fusion_test = test[test["model"].eq("cnn_mig_fusion_sweep")]
    per_variant = results[
        (results["split"].eq("test"))
        & (~results["variant"].eq("all"))
        & (results["model"].eq("cnn_mig_fusion_sweep"))
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    for ax, (target, metric, benchmark) in zip(axes.ravel(), comparisons):
        best = fusion_test[fusion_test["target"].eq(target)].sort_values(metric, ascending=False).iloc[0]
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
    fig.savefig(FIG_DIR / "fusion_sweep_per_variant_metrics.png", dpi=200)
    plt.close(fig)


def run_sweep(max_epochs: int | None = None, patience: int | None = None) -> dict[str, object]:
    random.seed(SEED)
    seed_everything()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    rows, feature_groups = load_rows_and_features()
    architecture, bz_norm, selected = select_best_cnn_config()
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    global_mean, global_std = compute_train_global_stats(train_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_epochs = max_epochs if max_epochs is not None else (16 if device.type == "cpu" else 32)
    patience = patience if patience is not None else (4 if device.type == "cpu" else 7)
    configs = [
        FusionSweepConfig(f"fusion_sweep_fixed_{bz_norm}", architecture, bz_norm, "fixed", tuple(feature_groups["fixed"])),
        FusionSweepConfig(f"fusion_sweep_magnetic_{bz_norm}", architecture, bz_norm, "magnetic", tuple(feature_groups["magnetic"])),
        FusionSweepConfig(
            f"fusion_sweep_fixed_magnetic_{bz_norm}",
            architecture,
            bz_norm,
            "fixed_magnetic",
            tuple(feature_groups["fixed_magnetic"]),
        ),
    ]
    all_results: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    for config in configs:
        print(f"Training {config.name} on {device}; backbone={architecture}/{bz_norm}", flush=True)
        model, history, reg_mean, reg_std = train_config(rows, config, device, global_mean, global_std, max_epochs, patience)
        histories[config.name] = history
        results, predictions = evaluate_config(rows, config, model, device, global_mean, global_std, reg_mean, reg_std)
        all_results.extend(results)
        all_predictions.extend(predictions)

    trained_results = pd.DataFrame(all_results)
    comparisons = previous_rows()
    combined = pd.concat([trained_results, comparisons], ignore_index=True, sort=False) if not comparisons.empty else trained_results
    predictions = pd.DataFrame(all_predictions)
    combined.to_csv(RESULTS_CSV, index=False)
    predictions.to_csv(PREDICTIONS_CSV, index=False)
    metadata = {
        "seed": SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_path": str(Path(torch.__file__).resolve()),
        "selected_cnn_backbone": selected,
        "configs": [config.__dict__ for config in configs],
        "benchmarks": BENCHMARKS,
        "histories": histories,
        "results": json_ready(combined.to_dict(orient="records")),
        "leakage_policy": "Bz tensors plus deployable MIG features only; grouped base_sample_id split; no oracle windows or metadata inputs",
    }
    RESULTS_JSON.write_text(json.dumps(metadata, indent=2))
    plot_figures(combined, predictions)
    test = combined[(combined["split"].eq("test")) & (combined["variant"].eq("all"))]
    summary = {
        "results_csv": str(RESULTS_CSV.relative_to(PROJECT_ROOT)),
        "predictions_csv": str(PREDICTIONS_CSV.relative_to(PROJECT_ROOT)),
        "device": str(device),
        "selected_cnn_backbone": selected,
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
