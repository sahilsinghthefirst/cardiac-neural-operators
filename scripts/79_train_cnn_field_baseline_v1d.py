                      
                                                      

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

from fno_field_data_common import compute_stats, load_index, make_loader, save_stats, torch
from v1d_field_common import CHECKPOINT_DIR, NEURAL_DIR, NEURAL_FIG_DIR, require_cuda, write_json
from scripts_78_bridge import evaluate, field_loss, seed_all


RESULTS_CSV = NEURAL_DIR / "cnn_field_results.csv"
RESULTS_JSON = NEURAL_DIR / "cnn_field_results.json"
STATS_JSON = NEURAL_DIR / "cnn_field_train_stats.json"


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


class SmallUNet(torch.nn.Module):
    def __init__(self, in_channels: int, base: int = 48, dropout: float = 0.08) -> None:
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
        self.out = torch.nn.Conv2d(base, 4, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        mid = self.mid(self.down2(e2))
        d2 = self.up2(mid)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def plot_training(history):
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h["epoch"], h["train_loss"], label="train")
    ax.plot(h["epoch"], h["val_loss"], label="val")
    ax.set_title("CNN Field Baseline Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "cnn_field_training_curves.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=18)
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
    rows = load_index()
    stats = compute_stats(rows)
    save_stats(STATS_JSON, stats)
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    val_rows = rows[rows["split"].eq("val")].reset_index(drop=True)
    in_channels = next(iter(make_loader(train_rows.head(1), stats, 1, False, args.input_norm)))["x"].shape[1]
    device = torch.device("cuda")
    model = SmallUNet(int(in_channels), args.base, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_rows, stats, args.batch_size, True, args.input_norm)
    val_loader = make_loader(val_rows, stats, args.batch_size, False, args.input_norm)
    best = float("inf")
    best_state = None
    bad = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(batch["x"].to(device).float())
            loss = field_loss(pred, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        vals = []
        with torch.no_grad():
            for batch in val_loader:
                vals.append(float(field_loss(model(batch["x"].to(device).float()), batch).detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": float(np.mean(vals))}
        history.append(row)
        print(f"cnn_field epoch {epoch}: train={row['train_loss']:.4f} val={row['val_loss']:.4f}", flush=True)
        if row["val_loss"] < best - 1e-5:
            best = row["val_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    ckpt = CHECKPOINT_DIR / "cnn_field_unet.pt"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "history": history, "in_channels": int(in_channels)}, ckpt)
    results, preds = evaluate(model, stats, rows, args, device)
    results.to_csv(RESULTS_CSV, index=False)
    preds.to_csv(NEURAL_DIR / "cnn_field_predictions_summary.csv", index=False)
    plot_training(history)
    payload = {
        "architecture": "small_unet_field_baseline",
        "input_channels": int(in_channels),
        "checkpoint": str(ckpt),
        "history": history,
        "results": results.replace({np.nan: None}).to_dict(orient="records"),
        "cuda": {"torch_version": torch.__version__, "torch_version_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)},
    }
    write_json(RESULTS_JSON, payload)
    print(json.dumps({"checkpoint": str(ckpt), "test": results[results["split"].eq("test")].head(12).to_dict(orient="records")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
