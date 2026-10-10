                      
                                                           

from __future__ import annotations

import argparse
import json
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

from fno_elementary_common import FNOBlock2d
from fno_field_data_common import compute_stats, save_stats, torch
from scripts_78_bridge import seed_all
from v1d_anomaly_common import (
    ANOMALY_STATS_JSON,
    anomaly_loss,
    estimate_positive_weight,
    evaluate_anomaly_model,
    load_rows_with_anomaly,
    make_anomaly_loader,
)
from v1d_field_common import CHECKPOINT_DIR, NEURAL_DIR, NEURAL_FIG_DIR, ROOT, require_cuda, write_json


RESULTS_CSV = NEURAL_DIR / "fno_anomaly_localizer_results.csv"
RESULTS_JSON = NEURAL_DIR / "fno_anomaly_localizer_results.json"
PRED_CSV = NEURAL_DIR / "fno_anomaly_localizer_predictions.csv"


class FNOAnomalyLocalizer(torch.nn.Module):
    def __init__(self, in_channels: int, width: int, modes: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lift = torch.nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = torch.nn.ModuleList([FNOBlock2d(width, modes, modes, dropout) for _ in range(layers)])
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(width, width, kernel_size=1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(dropout),
            torch.nn.Conv2d(width, 5, kernel_size=1),
        )

    def forward(self, x):
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def validation_metrics(model, rows, stats, args, device):
    results, _preds = evaluate_anomaly_model(model, rows, stats, args, device, "fno_anomaly_localizer")
    sub = results[(results["split"].eq("val")) & (results["group_type"].eq("all")) & (results["group_value"].eq("all"))]
    if len(sub) == 0:
        return {"dice": 0.0, "iou": 0.0}
    return {"dice": float(sub["dice_mean"].iloc[0]), "iou": float(sub["iou_mean"].iloc[0])}


def plot_training(history: list[dict]) -> None:
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(hist["epoch"], hist["train_loss"], label="train")
    axes[0].plot(hist["epoch"], hist["val_loss"], label="val")
    axes[0].set_title("FNO Anomaly Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(hist["epoch"], hist["val_dice"], label="val Dice")
    axes[1].plot(hist["epoch"], hist["val_iou"], label="val IoU")
    axes[1].set_title("Validation Localization")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "fno_anomaly_localizer_training_curves.png", dpi=180)
    plt.close(fig)


def plot_summary(results: pd.DataFrame, preds: pd.DataFrame) -> None:
    test = results[(results["split"].eq("test")) & (results["group_type"].isin(["condition", "measurement_variant"]))].copy()
    for group_type, fname, title in [
        ("condition", "fno_anomaly_dice_by_condition.png", "FNO Anomaly Dice by Condition"),
        ("measurement_variant", "fno_anomaly_dice_by_variant.png", "FNO Anomaly Dice by Measurement Variant"),
    ]:
        sub = test[test["group_type"].eq(group_type)].sort_values("dice_mean")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(sub["group_value"].astype(str), sub["dice_mean"], color="#4c78a8")
        ax.set_ylabel("mean Dice")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(NEURAL_FIG_DIR / fname, dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 5))
    sub = preds[preds["split"].eq("test")]
    mask = np.isfinite(sub["true_patch_area_fraction"]) & np.isfinite(sub["pred_area_fraction"])
    ax.scatter(sub.loc[mask, "true_patch_area_fraction"], sub.loc[mask, "pred_area_fraction"], s=9, alpha=0.35)
    ax.set_xlabel("true geometric patch area fraction")
    ax.set_ylabel("predicted anomaly area fraction")
    ax.set_title("FNO Anomaly Area")
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "fno_anomaly_patch_area.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=16)
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
    rows = load_rows_with_anomaly()
    if rows["anomaly_mask_binary_file"].isna().any():
        raise RuntimeError("Missing anomaly target paths; run scripts/86_build_anomaly_localization_targets_v1d.py first")
    stats = compute_stats(rows)
    save_stats(ANOMALY_STATS_JSON, stats)
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    val_rows = rows[rows["split"].eq("val")].reset_index(drop=True)
    device = torch.device("cuda")
    in_channels = int(next(iter(make_anomaly_loader(train_rows.head(1), stats, 1, False, args.input_norm)))["x"].shape[1])
    model = FNOAnomalyLocalizer(in_channels, args.width, args.modes, args.layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_anomaly_loader(train_rows, stats, args.batch_size, True, args.input_norm)
    val_loader = make_anomaly_loader(val_rows, stats, args.batch_size, False, args.input_norm)
    pos_weight = estimate_positive_weight(train_rows)
    best_state = None
    best_dice = -1.0
    bad = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(batch["x"].to(device).float())
            loss = anomaly_loss(pred, batch, pos_weight, aux_field_weight=0.08)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["x"].to(device).float())
                val_losses.append(float(anomaly_loss(pred, batch, pos_weight, aux_field_weight=0.08).detach().cpu()))
        vm = validation_metrics(model, rows, stats, args, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": float(np.mean(val_losses)),
            "val_dice": vm["dice"],
            "val_iou": vm["iou"],
        }
        history.append(row)
        print(f"fno_anomaly epoch {epoch}: train={row['train_loss']:.4f} val={row['val_loss']:.4f} dice={row['val_dice']:.4f}", flush=True)
        if row["val_dice"] > best_dice + 1e-4:
            best_dice = row["val_dice"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINT_DIR / "fno_anomaly_localizer.pt"
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "history": history, "in_channels": in_channels, "pos_weight": pos_weight}, ckpt)
    results, preds = evaluate_anomaly_model(model, rows, stats, args, device, "fno_anomaly_localizer")
    results.to_csv(RESULTS_CSV, index=False)
    preds.to_csv(PRED_CSV, index=False)
    plot_training(history)
    plot_summary(results, preds)
    payload = {
        "architecture": "fno2d_anomaly_localizer",
        "input_channels": in_channels,
        "output_channels": ["combined_anomaly_logit", "apd_anomaly_logit", "lat_anomaly_logit", "LAT_aux", "APD90_aux"],
        "checkpoint": str(ckpt.relative_to(ROOT)),
        "history": history,
        "pos_weight": pos_weight,
        "results": results.replace({np.nan: None}).to_dict(orient="records"),
        "cuda": {"torch_version": torch.__version__, "torch_version_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)},
    }
    write_json(RESULTS_JSON, payload)
    print(json.dumps({"checkpoint": str(ckpt), "test": results[results["split"].eq("test")].head(12).to_dict(orient="records")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
