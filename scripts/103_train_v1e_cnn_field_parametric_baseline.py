                      
                                                               

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v1e_direct_common import CHECKPOINT_DIR, NEURAL_DIR, NEURAL_FIG_DIR, require_cuda, write_json
from v1e_direct_fno_data_common import compute_stats, load_index, make_loader, save_stats, torch


SPEC = importlib.util.spec_from_file_location("_v1e_fno_train", PROJECT_ROOT / "scripts" / "102_train_v1e_fno_field_parametric.py")
if SPEC is None or SPEC.loader is None:
    raise ImportError("Could not load script 102 helpers")
FNO_TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FNO_TRAIN)

RESULTS_CSV = NEURAL_DIR / "v1e_cnn_field_parametric_results.csv"
RESULTS_JSON = NEURAL_DIR / "v1e_cnn_field_parametric_results.json"
PRED_CSV = NEURAL_DIR / "v1e_cnn_field_parametric_predictions.csv"


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


class V1ESmallUNetParametric(torch.nn.Module):
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
        self.field_head = torch.nn.Conv2d(base, 4, 1)
        self.param_head = torch.nn.Sequential(torch.nn.Linear(base * 4, base * 2), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(base * 2, 5))
        self.condition_head = torch.nn.Linear(base * 4, 4)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        mid = self.mid(self.down2(e2))
        pooled = mid.mean(dim=(2, 3))
        d2 = self.up2(mid)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.field_head(d1), self.param_head(pooled), self.condition_head(pooled)


def plot_training(history):
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h["epoch"], h["train_loss"], label="train")
    ax.plot(h["epoch"], h["val_loss"], label="val")
    ax.set_title("v1e CNN/UNet Field + Parametric Training")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "v1e_cnn_training_curves.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=14)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-norm", choices=["per_sample", "train_global"], default="per_sample")
    parser.add_argument("--variants", nargs="*", default=["direct_bz_50mm", "direct_bz_noise_medium", "direct_bz_masked_noise", "direct_axial_grad_30mm_baseline"])
    args = parser.parse_args()
    FNO_TRAIN.seed_all()
    require_cuda(torch)
    rows = load_index()
    rows = rows[rows["measurement_variant"].isin(args.variants)].reset_index(drop=True)
    stats = compute_stats(rows)
    save_stats(NEURAL_DIR / "v1e_cnn_field_parametric_stats.json", stats)
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    val_rows = rows[rows["split"].eq("val")].reset_index(drop=True)
    in_channels = next(iter(make_loader(train_rows.head(1), stats, 1, False, args.input_norm)))["x"].shape[1]
    device = torch.device("cuda")
    model = V1ESmallUNetParametric(int(in_channels), args.base, args.dropout).to(device)
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
            loss = FNO_TRAIN.field_param_loss(*model(batch["x"].to(device).float()), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        vals = []
        with torch.no_grad():
            for batch in val_loader:
                vals.append(float(FNO_TRAIN.field_param_loss(*model(batch["x"].to(device).float()), batch).detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": float(np.mean(vals))}
        history.append(row)
        print(f"v1e_cnn epoch {epoch}: train={row['train_loss']:.4f} val={row['val_loss']:.4f}", flush=True)
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
    ckpt = CHECKPOINT_DIR / "v1e_cnn_unet_parametric.pt"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "history": history, "in_channels": int(in_channels)}, ckpt)
    results, preds = FNO_TRAIN.evaluate_model(model, stats, rows, args, device, "cnn_unet_direct_parametric")
    results.to_csv(RESULTS_CSV, index=False)
    preds.to_csv(PRED_CSV, index=False)
    plot_training(history)
    payload = {
        "architecture": "small_unet_field_parametric_baseline",
        "checkpoint": str(ckpt),
        "history": history,
        "results": results.replace({np.nan: None}).to_dict(orient="records"),
        "cuda": {"torch_version": torch.__version__, "torch_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)},
    }
    write_json(RESULTS_JSON, payload)
    print(json.dumps({"results": str(RESULTS_CSV), "predictions": str(PRED_CSV)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
