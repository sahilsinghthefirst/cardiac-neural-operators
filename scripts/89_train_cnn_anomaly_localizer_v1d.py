                      
                                                                 

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

from fno_field_data_common import compute_stats, save_stats, torch
from scripts_78_bridge import seed_all
from v1d_anomaly_common import anomaly_loss, estimate_positive_weight, evaluate_anomaly_model, load_rows_with_anomaly, make_anomaly_loader
from v1d_field_common import CHECKPOINT_DIR, NEURAL_DIR, NEURAL_FIG_DIR, ROOT, require_cuda, write_json


RESULTS_CSV = NEURAL_DIR / "cnn_anomaly_localizer_results.csv"
RESULTS_JSON = NEURAL_DIR / "cnn_anomaly_localizer_results.json"
PRED_CSV = NEURAL_DIR / "cnn_anomaly_localizer_predictions.csv"
STATS_JSON = NEURAL_DIR / "cnn_anomaly_localizer_train_stats.json"


class ConvBlock(torch.nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(cin, cout, 3, padding=1),
            torch.nn.BatchNorm2d(cout),
            torch.nn.GELU(),
            torch.nn.Dropout2d(dropout),
            torch.nn.Conv2d(cout, cout, 3, padding=1),
            torch.nn.BatchNorm2d(cout),
            torch.nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class AnomalyUNet(torch.nn.Module):
    def __init__(self, in_channels: int, base: int, dropout: float) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base, dropout)
        self.down1 = torch.nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc2 = ConvBlock(base * 2, base * 2, dropout)
        self.down2 = torch.nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.mid = ConvBlock(base * 4, base * 4, dropout)
        self.up2 = torch.nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2, dropout)
        self.up1 = torch.nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base, dropout)
        self.out = torch.nn.Conv2d(base, 5, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        mid = self.mid(self.down2(e2))
        d2 = self.up2(mid)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def quick_val_dice(model, rows, stats, args, device):
    results, _ = evaluate_anomaly_model(model, rows, stats, args, device, "cnn_anomaly_localizer")
    sub = results[(results["split"].eq("val")) & (results["group_type"].eq("all")) & (results["group_value"].eq("all"))]
    return float(sub["dice_mean"].iloc[0]) if len(sub) else 0.0


def plot_training(history: list[dict]) -> None:
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(hist["epoch"], hist["train_loss"], label="train")
    axes[0].plot(hist["epoch"], hist["val_loss"], label="val")
    axes[0].set_title("CNN Anomaly Loss")
    axes[0].legend()
    axes[1].plot(hist["epoch"], hist["val_dice"])
    axes[1].set_title("Validation Dice")
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "cnn_anomaly_localizer_training_curves.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-norm", choices=["per_sample", "train_global"], default="per_sample")
    args = parser.parse_args()
    seed_all()
    require_cuda(torch)
    rows = load_rows_with_anomaly()
    stats = compute_stats(rows)
    save_stats(STATS_JSON, stats)
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    val_rows = rows[rows["split"].eq("val")].reset_index(drop=True)
    device = torch.device("cuda")
    in_channels = int(next(iter(make_anomaly_loader(train_rows.head(1), stats, 1, False, args.input_norm)))["x"].shape[1])
    model = AnomalyUNet(in_channels, args.base, args.dropout).to(device)
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
        train_losses = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(batch["x"].to(device).float())
            loss = anomaly_loss(pred, batch, pos_weight, aux_field_weight=0.08)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                val_losses.append(float(anomaly_loss(model(batch["x"].to(device).float()), batch, pos_weight, aux_field_weight=0.08).detach().cpu()))
        val_dice = quick_val_dice(model, rows, stats, args, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses)), "val_dice": val_dice}
        history.append(row)
        print(f"cnn_anomaly epoch {epoch}: train={row['train_loss']:.4f} val={row['val_loss']:.4f} dice={val_dice:.4f}", flush=True)
        if val_dice > best_dice + 1e-4:
            best_dice = val_dice
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINT_DIR / "cnn_anomaly_localizer.pt"
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "history": history, "in_channels": in_channels, "pos_weight": pos_weight}, ckpt)
    results, preds = evaluate_anomaly_model(model, rows, stats, args, device, "cnn_anomaly_localizer")
    results.to_csv(RESULTS_CSV, index=False)
    preds.to_csv(PRED_CSV, index=False)
    plot_training(history)
    payload = {
        "architecture": "unet_anomaly_localizer",
        "input_channels": in_channels,
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
