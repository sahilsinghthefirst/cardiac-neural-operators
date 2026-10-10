                      
                                                                         

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
from v1d_field_common import CHECKPOINT_DIR, NEURAL_DIR, NEURAL_FIG_DIR, ROOT, require_cuda, write_json
from v1d_hybrid_common import (
    HYBRID_CONFIGS,
    HYBRID_STATS_JSON,
    HybridLossConfig,
    estimate_geom_pos_weight,
    evaluate_hybrid_model,
    hybrid_loss,
    load_rows_with_hybrid,
    make_hybrid_loader,
)


RESULTS_CSV = NEURAL_DIR / "fno_hybrid_localizer_results.csv"
RESULTS_JSON = NEURAL_DIR / "fno_hybrid_localizer_results.json"
PRED_CSV = NEURAL_DIR / "fno_hybrid_localizer_predictions.csv"


class FNOHybridLocalizer(torch.nn.Module):
    def __init__(self, in_channels: int, width: int, modes: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lift = torch.nn.Conv2d(in_channels, width, 1)
        self.blocks = torch.nn.ModuleList([FNOBlock2d(width, modes, modes, dropout) for _ in range(layers)])
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(width, width, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(dropout),
            torch.nn.Conv2d(width, 5, 1),
        )

    def forward(self, x):
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h)


def val_score(model, rows, stats, args, device, cfg_name):
    results, _ = evaluate_hybrid_model(model, rows, stats, args, device, "fno_hybrid_localizer", cfg_name)
    sub = results[(results["split"].eq("val")) & (results["group_type"].eq("all")) & (results["group_value"].eq("all"))]
    if len(sub) == 0:
        return {"dice": 0.0, "apd_delta_r2": -999.0}
    r = sub.iloc[0]
    return {
        "dice": float(r["dice_mean"]),
        "apd_delta_r2": float(r.get("APD90_delta_pred_mask_pred_apd_r2", -999.0)),
    }


def train_one(rows, stats, args, cfg: HybridLossConfig, device, pos_weight):
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    val_rows = rows[rows["split"].eq("val")].reset_index(drop=True)
    in_channels = int(next(iter(make_hybrid_loader(train_rows.head(1), stats, 1, False, args.input_norm)))["x"].shape[1])
    model = FNOHybridLocalizer(in_channels, args.width, args.modes, args.layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_hybrid_loader(train_rows, stats, args.batch_size, True, args.input_norm)
    val_loader = make_hybrid_loader(val_rows, stats, args.batch_size, False, args.input_norm)
    best_state = None
    best_score = -1e9
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
        vs = val_score(model, rows, stats, args, device, cfg.name)
        score = vs["dice"] + 0.05 * max(vs["apd_delta_r2"], -5.0)
        row = {"config": cfg.name, "epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses)), "val_dice": vs["dice"], "val_apd_delta_r2": vs["apd_delta_r2"]}
        history.append(row)
        print(f"fno_hybrid {cfg.name} epoch {epoch}: train={row['train_loss']:.4f} val={row['val_loss']:.4f} dice={row['val_dice']:.4f} apdR2={row['val_apd_delta_r2']:.4f}", flush=True)
        if score > best_score + 1e-4:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            break
    if best_state:
        model.load_state_dict(best_state)
    ckpt = CHECKPOINT_DIR / f"fno_hybrid_localizer_{cfg.name}.pt"
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "loss_config": cfg.__dict__, "history": history, "in_channels": in_channels, "pos_weight": pos_weight}, ckpt)
    return model, history, ckpt, in_channels


def plot_outputs(results: pd.DataFrame, histories: list[dict]) -> None:
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    test = results[(results["split"].eq("test")) & (results["group_type"].eq("all"))].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, title in [
        (axes[0], "dice_mean", "Geometric Dice"),
        (axes[1], "homogeneous_false_positive_area", "Homogeneous FP Area"),
        (axes[2], "APD90_delta_pred_mask_pred_apd_r2", "APD90 Delta R2"),
    ]:
        ax.bar(test["config"], test[col], color="#4c78a8")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "fno_hybrid_localizer_summary.png", dpi=180)
    plt.close(fig)
    hist = pd.DataFrame(histories)
    fig, ax = plt.subplots(figsize=(8, 4))
    for cfg, sub in hist.groupby("config"):
        ax.plot(sub["epoch"], sub["val_dice"], label=cfg)
    ax.set_ylabel("validation Dice")
    ax.set_xlabel("epoch")
    ax.set_title("FNO Hybrid Validation Dice")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "fno_hybrid_training_curves.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-norm", choices=["per_sample", "train_global"], default="per_sample")
    parser.add_argument("--configs", nargs="*", default=[cfg.name for cfg in HYBRID_CONFIGS])
    args = parser.parse_args()
    seed_all()
    require_cuda(torch)
    rows = load_rows_with_hybrid()
    if rows["geometric_patch_mask_file"].isna().any():
        raise RuntimeError("Missing hybrid target paths; run script 91 first.")
    stats = compute_stats(rows)
    save_stats(HYBRID_STATS_JSON, stats)
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    pos_weight = estimate_geom_pos_weight(train_rows)
    selected = [cfg for cfg in HYBRID_CONFIGS if cfg.name in set(args.configs)]
    device = torch.device("cuda")
    all_results = []
    all_preds = []
    histories = []
    checkpoints = []
    for cfg in selected:
        model, hist, ckpt, in_channels = train_one(rows, stats, args, cfg, device, pos_weight)
        results, preds = evaluate_hybrid_model(model, rows, stats, args, device, "fno_hybrid_localizer", cfg.name)
        all_results.append(results)
        all_preds.append(preds)
        histories.extend(hist)
        checkpoints.append({"config": cfg.name, "checkpoint": str(ckpt.relative_to(ROOT)), "in_channels": in_channels, "history": hist})
    results_df = pd.concat(all_results, ignore_index=True)
    preds_df = pd.concat(all_preds, ignore_index=True)
    results_df.to_csv(RESULTS_CSV, index=False)
    preds_df.to_csv(PRED_CSV, index=False)
    plot_outputs(results_df, histories)
    payload = {
        "architecture": "fno2d_hybrid_geometric_primary_localizer",
        "configs": [cfg.__dict__ for cfg in selected],
        "checkpoints": checkpoints,
        "pos_weight": pos_weight,
        "results": results_df.replace({np.nan: None}).to_dict(orient="records"),
        "cuda": {"torch_version": torch.__version__, "torch_version_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)},
    }
    write_json(RESULTS_JSON, payload)
    print(json.dumps({"results_csv": str(RESULTS_CSV), "test": results_df[(results_df.split.eq('test')) & (results_df.group_type.eq('all'))].to_dict(orient="records")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
