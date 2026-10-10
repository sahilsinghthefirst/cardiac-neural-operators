                      
                                                                   

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if (PROJECT_ROOT / ".codex_deps" / "sklearn").exists():
    sys.path.append(str(PROJECT_ROOT / ".codex_deps" / "sklearn"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fno_elementary_common import FNOBlock2d
from fno_field_data_common import compute_stats, load_index, make_loader, save_stats, torch
from v1d_field_common import CHECKPOINT_DIR, NEURAL_DIR, NEURAL_FIG_DIR, require_cuda, scalar_targets_from_fields, write_json


RESULTS_CSV = NEURAL_DIR / "fno_field_results.csv"
RESULTS_JSON = NEURAL_DIR / "fno_field_results.json"
PRED_SUMMARY = NEURAL_DIR / "fno_field_predictions_summary.csv"
STATS_JSON = NEURAL_DIR / "fno_field_train_stats.json"
SEED = 20260525


class FNOFieldReconstructor(torch.nn.Module):
    def __init__(self, in_channels: int, width: int, modes: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lift = torch.nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = torch.nn.ModuleList([FNOBlock2d(width, modes, modes, dropout) for _ in range(layers)])
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(width, width, kernel_size=1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(dropout),
            torch.nn.Conv2d(width, 4, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def seed_all() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    prob = prob * valid
    target = target * valid
    inter = torch.sum(prob * target, dim=(1, 2))
    denom = torch.sum(prob, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
    return torch.mean(1.0 - (2.0 * inter + eps) / (denom + eps))


def field_loss(pred: torch.Tensor, batch: dict) -> torch.Tensor:
    valid = batch["valid_mask"].to(pred.device).float()
    y = batch["y"].to(pred.device).float()
    loss = torch.tensor(0.0, device=pred.device)
    for i in range(3):
        diff = torch.nn.functional.smooth_l1_loss(pred[:, i], y[:, i], reduction="none", beta=0.75)
        loss = loss + torch.sum(diff * valid) / (torch.sum(valid) + 1e-6)
    patch_target = y[:, 3]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred[:, 3], patch_target, reduction="none")
    patch_loss = torch.sum(bce * valid) / (torch.sum(valid) + 1e-6)
    return loss / 3.0 + 0.5 * patch_loss + 0.5 * dice_loss(pred[:, 3], patch_target, valid)


def finite_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return math.nan
    return float(pearsonr(y_true, y_pred).statistic)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else math.nan,
        "pearson_r": finite_pearson(y_true, y_pred),
    }


def patch_metrics(true: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    t = (true > 0.5) & valid
    p = (pred > 0.5) & valid
    tp = float(np.sum(t & p))
    fp = float(np.sum(~t & p & valid))
    fn = float(np.sum(t & ~p))
    dice = (2 * tp) / max(2 * tp + fp + fn, 1.0)
    iou = tp / max(tp + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    area_error = float(np.mean(p[valid]) - np.mean(t[valid])) if np.any(valid) else math.nan
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall, "area_error": area_error}


def train_model(rows: pd.DataFrame, args, device: torch.device):
    stats = compute_stats(rows)
    save_stats(STATS_JSON, stats)
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    val_rows = rows[rows["split"].eq("val")].reset_index(drop=True)
    sample_ds = make_loader(train_rows.head(1), stats, batch_size=1, shuffle=False, input_norm=args.input_norm)
    in_channels = next(iter(sample_ds))["x"].shape[1]
    model = FNOFieldReconstructor(in_channels, args.width, args.modes, args.layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_rows, stats, args.batch_size, True, args.input_norm)
    val_loader = make_loader(val_rows, stats, args.batch_size, False, args.input_norm)
    best_state = None
    best_val = float("inf")
    bad = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(batch["x"].to(device).float())
            loss = field_loss(pred, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["x"].to(device).float())
                val_losses.append(float(field_loss(pred, batch).detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        history.append(row)
        print(f"fno_field epoch {epoch}: train={row['train_loss']:.4f} val={row['val_loss']:.4f}", flush=True)
        if row["val_loss"] < best_val - 1e-5:
            best_val = row["val_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    ckpt = CHECKPOINT_DIR / "fno_field_reconstructor.pt"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "history": history, "in_channels": in_channels}, ckpt)
    return model, stats, history, ckpt, in_channels


def evaluate(model, stats, rows: pd.DataFrame, args, device: torch.device):
    results = []
    pred_rows = []
    target_names = ["LAT", "APD80", "APD90"]
    for split in ["train", "val", "test"]:
        split_rows = rows[rows["split"].eq(split)].reset_index(drop=True)
        loader = make_loader(split_rows, stats, args.batch_size, False, args.input_norm)
        field_true = {k: [] for k in target_names}
        field_pred = {k: [] for k in target_names}
        patch_true_all = []
        patch_pred_all = []
        valid_all = []
        scalar_true = {"LAT_range": [], "APD90_patch_delta": [], "patch_area_fraction": []}
        scalar_pred = {"LAT_range": [], "APD90_patch_delta": [], "patch_area_fraction": []}
        meta_rows = []
        model.eval()
        offset = 0
        with torch.no_grad():
            for batch in loader:
                pred = model(batch["x"].to(device).float()).cpu().numpy()
                y = batch["y"].numpy()
                valid = batch["valid_mask"].numpy().astype(bool)
                patch_true = y[:, 3]
                patch_pred = 1.0 / (1.0 + np.exp(-pred[:, 3]))
                for i, name in enumerate(target_names):
                    t = y[:, i] * stats.target_std[i] + stats.target_mean[i]
                    p = pred[:, i] * stats.target_std[i] + stats.target_mean[i]
                    field_true[name].append(t)
                    field_pred[name].append(p)
                patch_true_all.append(patch_true)
                patch_pred_all.append(patch_pred)
                valid_all.append(valid)
                for b in range(pred.shape[0]):
                    lat_p = pred[b, 0] * stats.target_std[0] + stats.target_mean[0]
                    apd_p = pred[b, 2] * stats.target_std[2] + stats.target_mean[2]
                    patch_bin = (patch_pred[b] > 0.5).astype(np.float32)
                    pred_scalars = scalar_targets_from_fields(lat_p, apd_p, patch_bin, valid[b])
                    true_lat = y[b, 0] * stats.target_std[0] + stats.target_mean[0]
                    true_apd = y[b, 2] * stats.target_std[2] + stats.target_mean[2]
                    true_scalars = scalar_targets_from_fields(true_lat, true_apd, patch_true[b], valid[b])
                    for key in scalar_true:
                        scalar_true[key].append(true_scalars[key])
                        scalar_pred[key].append(pred_scalars[key])
                    src = split_rows.iloc[offset + b]
                    meta_rows.append(
                        {
                            "field_sample_id": src["field_sample_id"],
                            "base_sample_id": src["base_sample_id"],
                            "split": split,
                            "condition": src["condition"],
                            "measurement_variant": src["measurement_variant"],
                            **{f"true_{k}": true_scalars[k] for k in scalar_true},
                            **{f"pred_{k}": pred_scalars[k] for k in scalar_pred},
                        }
                    )
                offset += pred.shape[0]
        valid_cat = np.concatenate(valid_all, axis=0).astype(bool)
        patch_true_cat = np.concatenate(patch_true_all, axis=0)
        patch_pred_cat = np.concatenate(patch_pred_all, axis=0)
        pm = patch_metrics(patch_true_cat, patch_pred_cat, valid_cat)
        results.append({"split": split, "target": "patch_mask", "metric_scope": "field", "variant": "all", "n": int(np.sum(valid_cat)), **pm})
        for name in target_names:
            t = np.concatenate(field_true[name], axis=0)[valid_cat]
            p = np.concatenate(field_pred[name], axis=0)[valid_cat]
            results.append({"split": split, "target": name, "metric_scope": "field", "variant": "all", "n": int(t.size), **regression_metrics(t, p)})
        pred_df_split = pd.DataFrame(meta_rows)
        for key in scalar_true:
            mask = np.isfinite(pred_df_split[f"true_{key}"]) & np.isfinite(pred_df_split[f"pred_{key}"])
            if mask.any():
                results.append(
                    {
                        "split": split,
                        "target": key,
                        "metric_scope": "derived_scalar",
                        "variant": "all",
                        "n": int(mask.sum()),
                        **regression_metrics(pred_df_split.loc[mask, f"true_{key}"].to_numpy(), pred_df_split.loc[mask, f"pred_{key}"].to_numpy()),
                    }
                )
        if split == "test":
            for variant, sub in pred_df_split.groupby("measurement_variant"):
                for key in scalar_true:
                    mask = np.isfinite(sub[f"true_{key}"]) & np.isfinite(sub[f"pred_{key}"])
                    if mask.any():
                        results.append(
                            {
                                "split": split,
                                "target": key,
                                "metric_scope": "derived_scalar",
                                "variant": variant,
                                "n": int(mask.sum()),
                                **regression_metrics(sub.loc[mask, f"true_{key}"].to_numpy(), sub.loc[mask, f"pred_{key}"].to_numpy()),
                            }
                        )
        pred_rows.extend(meta_rows)
    return pd.DataFrame(results), pd.DataFrame(pred_rows)


def plot_outputs(results: pd.DataFrame, preds: pd.DataFrame, history: list[dict]) -> None:
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist["epoch"], hist["train_loss"], label="train")
    ax.plot(hist["epoch"], hist["val_loss"], label="val")
    ax.set_title("FNO Field Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "fno_field_training_curves.png", dpi=180)
    plt.close(fig)
    for key, fname in [
        ("APD90_patch_delta", "derived_scalar_from_fields.png"),
        ("APD90_patch_delta", "fno_predicted_vs_true_apd90.png"),
        ("LAT_range", "fno_predicted_vs_true_lat.png"),
        ("patch_area_fraction", "fno_predicted_vs_true_patch_mask.png"),
    ]:
        sub = preds[preds["split"].eq("test")].copy()
        mask = np.isfinite(sub[f"true_{key}"]) & np.isfinite(sub[f"pred_{key}"])
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(sub.loc[mask, f"true_{key}"], sub.loc[mask, f"pred_{key}"], s=8, alpha=0.35)
        if mask.any():
            lo = min(sub.loc[mask, f"true_{key}"].min(), sub.loc[mask, f"pred_{key}"].min())
            hi = max(sub.loc[mask, f"true_{key}"].max(), sub.loc[mask, f"pred_{key}"].max())
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel(f"True {key}")
        ax.set_ylabel(f"Pred {key}")
        ax.set_title(f"FNO derived {key}")
        fig.tight_layout()
        fig.savefig(NEURAL_FIG_DIR / fname, dpi=180)
        plt.close(fig)
    test = results[results["split"].eq("test") & results["metric_scope"].eq("field")].copy()
    fig, ax = plt.subplots(figsize=(8, 4))
    metric = []
    labels = []
    for target in ["LAT", "APD80", "APD90"]:
        row = test[test["target"].eq(target)].head(1)
        if len(row):
            metric.append(float(row["r2"].iloc[0]))
            labels.append(target)
    ax.bar(labels, metric, color="#4c78a8")
    ax.set_ylabel("pixel R2")
    ax.set_title("FNO Field Reconstruction Summary")
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "fno_vs_cnn_field_summary.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-norm", choices=["per_sample", "train_global"], default="per_sample")
    args = parser.parse_args()
    seed_all()
    require_cuda(torch)
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_index()
    device = torch.device("cuda")
    model, stats, history, ckpt, in_channels = train_model(rows, args, device)
    results, preds = evaluate(model, stats, rows, args, device)
    results.to_csv(RESULTS_CSV, index=False)
    preds.to_csv(PRED_SUMMARY, index=False)
    plot_outputs(results, preds, history)
    payload = {
        "seed": SEED,
        "architecture": "fno2d_field_reconstructor",
        "input_channels": int(in_channels),
        "output_channels": ["LAT", "APD80", "APD90", "patch_logit"],
        "checkpoint": str(ckpt),
        "history": history,
        "results": results.replace({np.nan: None}).to_dict(orient="records"),
        "cuda": {"torch_version": torch.__version__, "torch_version_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)},
    }
    write_json(RESULTS_JSON, payload)
    print(json.dumps({"rows": len(rows), "checkpoint": str(ckpt), "test": results[results["split"].eq("test")].head(12).to_dict(orient="records")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
