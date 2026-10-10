                      
                                                        

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
from v1d_field_common import CHECKPOINT_DIR, NEURAL_DIR, NEURAL_FIG_DIR, ROOT, require_cuda, write_json
from v1d_hybrid_common import HYBRID_CONFIGS, estimate_geom_pos_weight, evaluate_hybrid_model, hybrid_loss, load_rows_with_hybrid, make_hybrid_loader


RESULTS_CSV = NEURAL_DIR / "cnn_hybrid_localizer_results.csv"
RESULTS_JSON = NEURAL_DIR / "cnn_hybrid_localizer_results.json"
PRED_CSV = NEURAL_DIR / "cnn_hybrid_localizer_predictions.csv"
STATS_JSON = NEURAL_DIR / "cnn_hybrid_localizer_train_stats.json"


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


class HybridUNet(torch.nn.Module):
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


def val_dice(model, rows, stats, args, device, cfg_name):
    results, _ = evaluate_hybrid_model(model, rows, stats, args, device, "cnn_hybrid_localizer", cfg_name)
    sub = results[(results["split"].eq("val")) & (results["group_type"].eq("all")) & (results["group_value"].eq("all"))]
    return float(sub["dice_mean"].iloc[0]) if len(sub) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-norm", choices=["per_sample", "train_global"], default="per_sample")
    parser.add_argument("--config", default="full_hybrid")
    args = parser.parse_args()
    seed_all()
    require_cuda(torch)
    rows = load_rows_with_hybrid()
    stats = compute_stats(rows)
    save_stats(STATS_JSON, stats)
    cfg = next((c for c in HYBRID_CONFIGS if c.name == args.config), HYBRID_CONFIGS[-1])
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    val_rows = rows[rows["split"].eq("val")].reset_index(drop=True)
    in_channels = int(next(iter(make_hybrid_loader(train_rows.head(1), stats, 1, False, args.input_norm)))["x"].shape[1])
    device = torch.device("cuda")
    model = HybridUNet(in_channels, args.base, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = estimate_geom_pos_weight(train_rows)
    train_loader = make_hybrid_loader(train_rows, stats, args.batch_size, True, args.input_norm)
    val_loader = make_hybrid_loader(val_rows, stats, args.batch_size, False, args.input_norm)
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
            loss = hybrid_loss(pred, batch, cfg, pos_weight, stats)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                val_losses.append(float(hybrid_loss(model(batch["x"].to(device).float()), batch, cfg, pos_weight, stats).detach().cpu()))
        vd = val_dice(model, rows, stats, args, device, cfg.name)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses)), "val_dice": vd}
        history.append(row)
        print(f"cnn_hybrid {cfg.name} epoch {epoch}: train={row['train_loss']:.4f} val={row['val_loss']:.4f} dice={vd:.4f}", flush=True)
        if vd > best_dice + 1e-4:
            best_dice = vd
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    ckpt = CHECKPOINT_DIR / f"cnn_hybrid_localizer_{cfg.name}.pt"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "loss_config": cfg.__dict__, "history": history, "in_channels": in_channels, "pos_weight": pos_weight}, ckpt)
    results, preds = evaluate_hybrid_model(model, rows, stats, args, device, "cnn_hybrid_localizer", cfg.name)
    results.to_csv(RESULTS_CSV, index=False)
    preds.to_csv(PRED_CSV, index=False)
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist["epoch"], hist["val_dice"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation Dice")
    ax.set_title("CNN Hybrid Validation Dice")
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "cnn_hybrid_training_curves.png", dpi=180)
    plt.close(fig)
    payload = {
        "architecture": "unet_hybrid_geometric_primary_localizer",
        "config": cfg.__dict__,
        "checkpoint": str(ckpt.relative_to(ROOT)),
        "history": history,
        "pos_weight": pos_weight,
        "results": results.replace({np.nan: None}).to_dict(orient="records"),
        "cuda": {"torch_version": torch.__version__, "torch_version_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)},
    }
    write_json(RESULTS_JSON, payload)
    print(json.dumps({"checkpoint": str(ckpt), "test": results[(results.split.eq('test')) & (results.group_type.eq('all'))].to_dict(orient="records")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
