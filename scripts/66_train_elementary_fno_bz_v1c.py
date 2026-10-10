                      
                                                       
 
                                                                                
                                                                        
                                                                                 
   

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if (PROJECT_ROOT / ".local_deps" / "sklearn").exists():
    sys.path.append(str(PROJECT_ROOT / ".local_deps" / "sklearn"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

from fno_elementary_common import (
    BENCHMARKS,
    CKPT_DIR,
    FIG_DIR,
    OUT_DIR,
    REG_TARGETS,
    SEED,
    VARIANT_ORDER,
    ElementaryFNO2dTimeChannels,
    FnoConfig,
    class_weights,
    compute_train_global_stats,
    ensure_dirs,
    load_rows,
    make_loader,
    multitask_loss,
    parameter_count,
    seed_everything,
    target_scaler,
    torch,
)


RESULTS_CSV = OUT_DIR / "fno_elementary_results.csv"
RESULTS_JSON = OUT_DIR / "fno_elementary_results.json"
PREDICTIONS_CSV = OUT_DIR / "fno_elementary_predictions.csv"


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


def train_one(
    rows: pd.DataFrame,
    config: FnoConfig,
    device: torch.device,
    global_mean: float,
    global_std: float,
    max_epochs: int,
    patience: int,
) -> tuple[ElementaryFNO2dTimeChannels, list[dict[str, float]], np.ndarray, np.ndarray]:
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    val_rows = rows[rows["split"].eq("val")].reset_index(drop=True)
    reg_mean, reg_std = target_scaler(train_rows)
    train_loader = make_loader(train_rows, config, global_mean, global_std, reg_mean, reg_std, shuffle=True)
    val_loader = make_loader(val_rows, config, global_mean, global_std, reg_mean, reg_std, shuffle=False)
    model = ElementaryFNO2dTimeChannels(
        hidden_width=config.hidden_width,
        modes_x=config.modes_x,
        modes_y=config.modes_y,
        layers=config.layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights(train_rows, device))
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits, reg_pred = model(batch["x_bz"].to(device).float())
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
            "architecture": "elementary_fno2d_time_channels",
            "parameter_count": parameter_count(model),
        },
        CKPT_DIR / f"fno_elementary_{config.name}.pt",
    )
    return model, history, reg_mean, reg_std


def predict_rows(
    model: ElementaryFNO2dTimeChannels,
    rows: pd.DataFrame,
    config: FnoConfig,
    device: torch.device,
    global_mean: float,
    global_std: float,
    reg_mean: np.ndarray,
    reg_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(rows.reset_index(drop=True), config, global_mean, global_std, reg_mean, reg_std, shuffle=False)
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


def evaluate_config(
    rows: pd.DataFrame,
    config: FnoConfig,
    model: ElementaryFNO2dTimeChannels,
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
                "model": "elementary_fno2d_time_channels",
                "bz_norm": config.bz_norm,
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
                    "model": "elementary_fno2d_time_channels",
                    "bz_norm": config.bz_norm,
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
                    "model": "elementary_fno2d_time_channels",
                    "bz_norm": config.bz_norm,
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
                        "model": "elementary_fno2d_time_channels",
                        "bz_norm": config.bz_norm,
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
                        "model": "elementary_fno2d_time_channels",
                        "bz_norm": config.bz_norm,
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
                            "model": "elementary_fno2d_time_channels",
                            "bz_norm": config.bz_norm,
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
    clean = []
    for record in records:
        clean.append({key: None if isinstance(value, float) and math.isnan(value) else value for key, value in record.items()})
    return clean


def plot_figures(results: pd.DataFrame, predictions: pd.DataFrame, histories: dict[str, list[dict[str, float]]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, hist in histories.items():
        h = pd.DataFrame(hist)
        ax.plot(h["epoch"], h["train_loss"], label=f"{name} train")
        ax.plot(h["epoch"], h["val_loss"], label=f"{name} val", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Elementary FNO Training Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fno_elementary_training_curves.png", dpi=200)
    plt.close(fig)

    test = results[(results["split"].eq("test")) & (results["variant"].eq("all"))].copy()
    comparisons = [
        ("y_condition", "macro_f1", BENCHMARKS["y_condition_macro_f1"]),
        ("y_lat_range", "r2", BENCHMARKS["y_lat_range_r2"]),
        ("y_apd90_patch_delta", "r2", BENCHMARKS["y_apd90_patch_delta_r2"]),
        ("y_patch_area_fraction", "r2", BENCHMARKS["y_patch_area_fraction_r2"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (target, metric, benchmark) in zip(axes.ravel(), comparisons):
        sub = test[test["target"].eq(target)].sort_values(metric, ascending=False)
        ax.bar(sub["config"], sub[metric], color="#4c78a8")
        ax.axhline(benchmark, color="#f58518", linestyle="--", label="CNN/fusion best")
        ax.set_title(target)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fno_elementary_vs_cnn_fusion_summary.png", dpi=200)
    plt.close(fig)

    best_cls = test[test["target"].eq("y_condition")].sort_values("macro_f1", ascending=False).iloc[0]
    sub = predictions[
        predictions["target"].eq("y_condition")
        & predictions["config"].eq(best_cls["config"])
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
    ax.set_title(f"Elementary FNO Confusion Matrix: {best_cls['config']} macro F1={best_cls['macro_f1']:.3f}")
    threshold = cm.max() * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fno_elementary_confusion_matrix.png", dpi=200)
    plt.close(fig)

    for target, filename in [
        ("y_apd90_patch_delta", "fno_elementary_apd90_delta.png"),
        ("y_patch_area_fraction", "fno_elementary_patch_area.png"),
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

    per_variant = results[(results["split"].eq("test")) & (~results["variant"].eq("all"))].copy()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
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
    fig.savefig(FIG_DIR / "fno_elementary_per_variant_metrics.png", dpi=200)
    plt.close(fig)


def run(max_epochs: int, patience: int) -> dict[str, object]:
    seed_everything()
    ensure_dirs()
    if not torch.cuda.is_available():
        diagnostic = {
            "torch_file": str(Path(torch.__file__).resolve()),
            "torch_version": torch.__version__,
            "torch_version_cuda": torch.version.cuda,
            "cuda_available": False,
            "message": "CUDA unavailable; refusing to train CPU FNO.",
        }
        RESULTS_JSON.write_text(json.dumps(diagnostic, indent=2))
        raise SystemExit("CUDA unavailable; refusing to train CPU FNO")
    device = torch.device("cuda")
    rows = load_rows()
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    global_mean, global_std = compute_train_global_stats(train_rows)
    configs = [
        FnoConfig(name="elementary_fno2d_per_sample", bz_norm="per_sample"),
        FnoConfig(name="elementary_fno2d_train_global", bz_norm="train_global"),
    ]
    all_results: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    for config in configs:
        print(f"Training {config.name} on {device}; modes=({config.modes_x},{config.modes_y}) width={config.hidden_width}", flush=True)
        model, history, reg_mean, reg_std = train_one(rows, config, device, global_mean, global_std, max_epochs, patience)
        histories[config.name] = history
        results, predictions = evaluate_config(rows, config, model, device, global_mean, global_std, reg_mean, reg_std)
        all_results.extend(results)
        all_predictions.extend(predictions)

    results_df = pd.DataFrame(all_results)
    predictions_df = pd.DataFrame(all_predictions)
    results_df.to_csv(RESULTS_CSV, index=False)
    predictions_df.to_csv(PREDICTIONS_CSV, index=False)
    plot_figures(results_df, predictions_df, histories)
    test = results_df[(results_df["split"].eq("test")) & (results_df["variant"].eq("all"))]
    metadata = {
        "seed": SEED,
        "device": str(device),
        "torch_file": str(Path(torch.__file__).resolve()),
        "torch_version": torch.__version__,
        "torch_version_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "neuraloperator": "unavailable; used internal minimal FNO2d",
        "architecture": "elementary_fno2d_time_channels",
        "input_shape": "[batch, 301, 8, 8]",
        "spectral_modes": {"modes_x": 4, "modes_y": 4, "grid": [8, 8]},
        "benchmarks": BENCHMARKS,
        "histories": histories,
        "results": json_ready(results_df.to_dict(orient="records")),
        "best_test": {
            "condition": test[test["target"].eq("y_condition")].sort_values("macro_f1", ascending=False).head(1).to_dict(orient="records"),
            "lat_range": test[test["target"].eq("y_lat_range")].sort_values("r2", ascending=False).head(1).to_dict(orient="records"),
            "apd90_patch_delta": test[test["target"].eq("y_apd90_patch_delta")].sort_values("r2", ascending=False).head(1).to_dict(orient="records"),
            "patch_area_fraction": test[test["target"].eq("y_patch_area_fraction")].sort_values("r2", ascending=False).head(1).to_dict(orient="records"),
        },
    }
    RESULTS_JSON.write_text(json.dumps(metadata, indent=2))
    summary = {key: value for key, value in metadata.items() if key not in {"results", "histories"}}
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()
    run(args.max_epochs, args.patience)


if __name__ == "__main__":
    main()
