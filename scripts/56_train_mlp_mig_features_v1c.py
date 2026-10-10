                      
                                                             
 
                                                                              
                                                        
   

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
from torch.utils.data import DataLoader, TensorDataset

from neural_data_common import attach_splits, load_split_map, select_deployable_feature_columns

READY_DIR = PROJECT_ROOT / "outputs" / "dataset_v1c_5000_augmented_ready"
OUT_DIR = PROJECT_ROOT / "outputs" / "neural_v1c"
FIG_DIR = PROJECT_ROOT / "figures" / "neural_v1c"
CKPT_DIR = OUT_DIR / "checkpoints"

FIXED = READY_DIR / "features_mig_deployable_fixed_augmented.csv"
MAG = READY_DIR / "features_mig_deployable_magnetic_augmented.csv"
TARGETS = READY_DIR / "targets_ep_augmented.csv"

RESULTS_CSV = OUT_DIR / "mlp_feature_baseline_results.csv"
RESULTS_JSON = OUT_DIR / "mlp_feature_baseline_results.json"
PREDICTIONS_CSV = OUT_DIR / "mlp_feature_predictions.csv"

SEED = 20260525
CLASSICAL = {
    "y_condition_macro_f1": 0.743,
    "y_lat_range_r2": 0.991,
    "y_apd90_patch_delta_r2": 0.561,
    "y_patch_area_fraction_r2": 0.275,
}
REG_TARGETS = ["y_lat_range", "y_apd90_patch_delta", "y_patch_area_fraction"]


@dataclass(frozen=True)
class FeatureSet:
    name: str
    df: pd.DataFrame
    columns: list[str]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int = 128, hidden2: int = 64, dropout: float = 0.15) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.class_head = nn.Linear(hidden2, 4)
        self.reg_head = nn.Linear(hidden2, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.class_head(h), self.reg_head(h)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)


def load_feature_sets() -> tuple[list[FeatureSet], pd.DataFrame]:
    split_map = load_split_map()
    targets = attach_splits(pd.read_csv(TARGETS), split_map)
    target_cols = [
        "augmented_sample_id",
        "base_sample_id",
        "variant",
        "condition",
        "split",
        "y_condition",
        "y_lat_range",
        "y_apd90_patch_delta",
        "y_patch_area_fraction",
    ]

    fixed = pd.read_csv(FIXED)
    fixed_cols = select_deployable_feature_columns(fixed)
    magnetic = pd.read_csv(MAG)
    magnetic_cols = select_deployable_feature_columns(magnetic)

    base = targets[target_cols].copy()
    fixed_joined = base.merge(
        fixed[["augmented_sample_id", *fixed_cols]], on="augmented_sample_id", how="inner", validate="one_to_one"
    )
    magnetic_joined = base.merge(
        magnetic[["augmented_sample_id", *magnetic_cols]], on="augmented_sample_id", how="inner", validate="one_to_one"
    )

    fixed_prefixed = fixed[["augmented_sample_id", *fixed_cols]].rename(
        columns={c: f"fixed__{c}" for c in fixed_cols}
    )
    mag_prefixed = magnetic[["augmented_sample_id", *magnetic_cols]].rename(
        columns={c: f"magnetic__{c}" for c in magnetic_cols}
    )
    concat = base.merge(fixed_prefixed, on="augmented_sample_id", how="inner", validate="one_to_one")
    concat = concat.merge(mag_prefixed, on="augmented_sample_id", how="inner", validate="one_to_one")

    for name, df in [("fixed", fixed_joined), ("magnetic", magnetic_joined), ("fixed_magnetic", concat)]:
        if len(df) != 55000:
            raise ValueError(f"{name} feature join produced {len(df)} rows instead of 55000")

    return [
        FeatureSet("fixed", fixed_joined, fixed_cols),
        FeatureSet("magnetic", magnetic_joined, magnetic_cols),
        FeatureSet("fixed_magnetic", concat, [f"fixed__{c}" for c in fixed_cols] + [f"magnetic__{c}" for c in magnetic_cols]),
    ], targets


def standardize_train_only(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = df["split"].eq("train")
    x = df[columns].to_numpy(np.float32)
    mean = x[train.to_numpy()].mean(axis=0)
    std = x[train.to_numpy()].std(axis=0)
    std[std < 1e-8] = 1.0
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def prepare_tensors(df: pd.DataFrame, columns: list[str]) -> dict[str, object]:
    x, feature_mean, feature_std = standardize_train_only(df, columns)
    y_class = df["y_condition"].to_numpy(np.int64)
    y_reg = df[REG_TARGETS].to_numpy(np.float32)
    reg_mask = np.isfinite(y_reg)
    reg_mean = np.zeros(3, dtype=np.float32)
    reg_std = np.ones(3, dtype=np.float32)
    train = df["split"].eq("train").to_numpy()
    y_scaled = np.zeros_like(y_reg, dtype=np.float32)
    for i in range(3):
        mask = train & reg_mask[:, i]
        reg_mean[i] = float(y_reg[mask, i].mean())
        reg_std[i] = float(y_reg[mask, i].std())
        if reg_std[i] < 1e-8:
            reg_std[i] = 1.0
        finite = reg_mask[:, i]
        y_scaled[finite, i] = (y_reg[finite, i] - reg_mean[i]) / reg_std[i]
    return {
        "x": x,
        "y_class": y_class,
        "y_reg": y_reg,
        "y_reg_scaled": y_scaled,
        "reg_mask": reg_mask,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "reg_mean": reg_mean,
        "reg_std": reg_std,
    }


def make_loader(data: dict[str, object], indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(data["x"][indices]),
        torch.from_numpy(data["y_class"][indices]),
        torch.from_numpy(data["y_reg_scaled"][indices]),
        torch.from_numpy(data["reg_mask"][indices].astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def class_weights(y: np.ndarray, train_idx: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y[train_idx], minlength=4).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def multitask_loss(
    logits: torch.Tensor,
    reg_pred: torch.Tensor,
    y_class: torch.Tensor,
    y_reg_scaled: torch.Tensor,
    reg_mask: torch.Tensor,
    ce: nn.Module,
    beta: float = 0.75,
) -> torch.Tensor:
    loss = ce(logits, y_class)
    reg_losses = []
    for i in range(3):
        mask = reg_mask[:, i] > 0
        if mask.any():
            reg_losses.append(nn.functional.smooth_l1_loss(reg_pred[mask, i], y_reg_scaled[mask, i], beta=beta))
    if reg_losses:
        loss = loss + sum(reg_losses) / len(reg_losses)
    return loss


def train_one(feature_set: FeatureSet, max_epochs: int = 80, patience: int = 10) -> tuple[MLP, dict[str, object], list[dict[str, float]]]:
    df = feature_set.df
    data = prepare_tensors(df, feature_set.columns)
    train_idx = np.flatnonzero(df["split"].eq("train").to_numpy())
    val_idx = np.flatnonzero(df["split"].eq("val").to_numpy())
    train_loader = make_loader(data, train_idx, batch_size=512, shuffle=True)
    val_loader = make_loader(data, val_idx, batch_size=1024, shuffle=False)

    model = MLP(input_dim=len(feature_set.columns))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(weight=class_weights(data["y_class"], train_idx))
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yc, yr, mask in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, reg = model(xb.float())
            loss = multitask_loss(logits, reg, yc.long(), yr.float(), mask.float(), ce)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yc, yr, mask in val_loader:
                logits, reg = model(xb.float())
                loss = multitask_loss(logits, reg, yc.long(), yr.float(), mask.float(), ce)
                val_losses.append(float(loss.detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        history.append(row)
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
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": len(feature_set.columns),
        "feature_columns": feature_set.columns,
        "feature_mean": data["feature_mean"],
        "feature_std": data["feature_std"],
        "reg_mean": data["reg_mean"],
        "reg_std": data["reg_std"],
        "history": history,
    }
    torch.save(checkpoint, CKPT_DIR / f"mlp_{feature_set.name}.pt")
    return model, data, history


def predict_all(model: MLP, data: dict[str, object], batch_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    x = torch.from_numpy(data["x"])
    class_parts = []
    reg_parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits, reg = model(x[start : start + batch_size].float())
            class_parts.append(logits.softmax(dim=1).cpu().numpy())
            reg_parts.append(reg.cpu().numpy())
    class_prob = np.vstack(class_parts)
    reg_scaled = np.vstack(reg_parts)
    reg = reg_scaled * data["reg_std"][None, :] + data["reg_mean"][None, :]
    return class_prob, reg


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


def evaluate(
    feature_set: FeatureSet,
    data: dict[str, object],
    class_prob: np.ndarray,
    reg_pred: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    df = feature_set.df.reset_index(drop=True)
    y_class = data["y_class"]
    y_pred_class = class_prob.argmax(axis=1)
    y_reg = data["y_reg"]
    reg_mask = data["reg_mask"]
    result_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    for split in ["train", "val", "test"]:
        idx = np.flatnonzero(df["split"].eq(split).to_numpy())
        yt = y_class[idx]
        yp = y_pred_class[idx]
        result_rows.append(
            {
                "task": "classification",
                "target": "y_condition",
                "feature_setting": feature_set.name,
                "model": "mlp_multitask",
                "split": split,
                "n_eval": int(len(idx)),
                "accuracy": float(accuracy_score(yt, yp)),
                "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
                "macro_f1": float(f1_score(yt, yp, average="macro")),
                "mae": math.nan,
                "rmse": math.nan,
                "r2": math.nan,
                "pearson_r": math.nan,
                "classical_benchmark": CLASSICAL["y_condition_macro_f1"],
            }
        )
        for row_idx, true_value, pred_value in zip(idx, yt, yp):
            meta = df.iloc[row_idx]
            pred_rows.append(
                {
                    "task": "classification",
                    "target": "y_condition",
                    "feature_setting": feature_set.name,
                    "model": "mlp_multitask",
                    "split": split,
                    "augmented_sample_id": meta["augmented_sample_id"],
                    "base_sample_id": meta["base_sample_id"],
                    "variant": meta["variant"],
                    "condition": meta["condition"],
                    "y_true": int(true_value),
                    "y_pred": int(pred_value),
                    "residual": math.nan,
                }
            )
        for j, target in enumerate(REG_TARGETS):
            mask = reg_mask[idx, j].astype(bool)
            eval_idx = idx[mask]
            yt_reg = y_reg[eval_idx, j].astype(float)
            yp_reg = reg_pred[eval_idx, j].astype(float)
            metrics = regression_metrics(yt_reg, yp_reg)
            result_rows.append(
                {
                    "task": "regression",
                    "target": target,
                    "feature_setting": feature_set.name,
                    "model": "mlp_multitask",
                    "split": split,
                    "n_eval": int(len(eval_idx)),
                    "accuracy": math.nan,
                    "balanced_accuracy": math.nan,
                    "macro_f1": math.nan,
                    **metrics,
                    "classical_benchmark": CLASSICAL[f"{target}_r2"],
                }
            )
            for row_idx, true_value, pred_value in zip(eval_idx, yt_reg, yp_reg):
                meta = df.iloc[row_idx]
                pred_rows.append(
                    {
                        "task": "regression",
                        "target": target,
                        "feature_setting": feature_set.name,
                        "model": "mlp_multitask",
                        "split": split,
                        "augmented_sample_id": meta["augmented_sample_id"],
                        "base_sample_id": meta["base_sample_id"],
                        "variant": meta["variant"],
                        "condition": meta["condition"],
                        "y_true": float(true_value),
                        "y_pred": float(pred_value),
                        "residual": float(pred_value - true_value),
                    }
                )
    return result_rows, pred_rows


def save_figures(results: pd.DataFrame, predictions: pd.DataFrame, histories: dict[str, list[dict[str, float]]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, hist in histories.items():
        h = pd.DataFrame(hist)
        ax.plot(h["epoch"], h["train_loss"], label=f"{name} train", alpha=0.7)
        ax.plot(h["epoch"], h["val_loss"], label=f"{name} val", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("MLP Training Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mlp_training_curves.png", dpi=200)
    plt.close(fig)

    test = results[results["split"].eq("test")].copy()
    rows = [
        ("y_condition", "macro_f1", CLASSICAL["y_condition_macro_f1"]),
        ("y_lat_range", "r2", CLASSICAL["y_lat_range_r2"]),
        ("y_apd90_patch_delta", "r2", CLASSICAL["y_apd90_patch_delta_r2"]),
        ("y_patch_area_fraction", "r2", CLASSICAL["y_patch_area_fraction_r2"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (target, metric, classical) in zip(axes.ravel(), rows):
        sub = test[test["target"].eq(target)].sort_values("feature_setting")
        ax.bar(sub["feature_setting"], sub[metric], color="#4c78a8", label="MLP")
        ax.axhline(classical, color="#f58518", linestyle="--", label="classical")
        ax.set_title(target)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mlp_vs_classical_summary.png", dpi=200)
    plt.close(fig)

    for target, filename in [
        ("y_apd90_patch_delta", "mlp_regression_apd90_delta.png"),
        ("y_patch_area_fraction", "mlp_regression_patch_area.png"),
    ]:
        best = test[test["target"].eq(target)].sort_values("r2", ascending=False).iloc[0]
        sub = predictions[
            predictions["target"].eq(target)
            & predictions["feature_setting"].eq(best["feature_setting"])
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
        ax.set_title(f"{target}: {best['feature_setting']} R2={best['r2']:.3f}")
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=200)
        plt.close(fig)

    best_cls = test[test["target"].eq("y_condition")].sort_values("macro_f1", ascending=False).iloc[0]
    sub = predictions[
        predictions["target"].eq("y_condition")
        & predictions["feature_setting"].eq(best_cls["feature_setting"])
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
    ax.set_title(f"MLP Confusion Matrix: {best_cls['feature_setting']} macro F1={best_cls['macro_f1']:.3f}")
    thresh = cm.max() * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mlp_confusion_matrix.png", dpi=200)
    plt.close(fig)


def json_clean(records: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for record in records:
        row = {}
        for k, v in record.items():
            row[k] = None if isinstance(v, float) and math.isnan(v) else v
        out.append(row)
    return out


def main() -> None:
    seed_everything()
    ensure_dirs()
    feature_sets, _ = load_feature_sets()
    all_results: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    metadata: dict[str, object] = {
        "seed": SEED,
        "architecture": "input_dim -> 128 -> 64 -> classification/regression heads",
        "loss": "cross entropy plus masked SmoothL1 regression losses on train-standardized targets",
        "split_policy": "grouped by base_sample_id",
        "feature_policy": "deployable fixed, deployable magnetic, and concatenated deployable features only",
        "classical_benchmarks": CLASSICAL,
        "torch_version": torch.__version__,
    }

    for feature_set in feature_sets:
        print(f"Training MLP feature set: {feature_set.name} ({len(feature_set.columns)} features)")
        model, data, history = train_one(feature_set)
        histories[feature_set.name] = history
        class_prob, reg_pred = predict_all(model, data)
        results, predictions = evaluate(feature_set, data, class_prob, reg_pred)
        all_results.extend(results)
        all_predictions.extend(predictions)
        metadata[f"{feature_set.name}_epochs"] = len(history)

    results_df = pd.DataFrame(all_results)
    preds_df = pd.DataFrame(all_predictions)
    results_df.to_csv(RESULTS_CSV, index=False)
    preds_df.to_csv(PREDICTIONS_CSV, index=False)
    RESULTS_JSON.write_text(json.dumps({"metadata": metadata, "results": json_clean(all_results)}, indent=2))
    save_figures(results_df, preds_df, histories)

    test = results_df[results_df["split"].eq("test")]
    summary = {
        "results_csv": str(RESULTS_CSV.relative_to(PROJECT_ROOT)),
        "results_json": str(RESULTS_JSON.relative_to(PROJECT_ROOT)),
        "predictions_csv": str(PREDICTIONS_CSV.relative_to(PROJECT_ROOT)),
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
