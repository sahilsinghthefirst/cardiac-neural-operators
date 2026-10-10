                      
                                                                  

                                                                       
                                                                        
                                                             
   

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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
from scripts_78_bridge import FNOFieldReconstructor, evaluate, seed_all
from v1d_field_common import CHECKPOINT_DIR, NEURAL_DIR, NEURAL_FIG_DIR, ROOT, require_cuda, write_json


RESULTS_CSV = NEURAL_DIR / "patchaware_fno_results.csv"
RESULTS_JSON = NEURAL_DIR / "patchaware_fno_results.json"
PRED_CSV = NEURAL_DIR / "patchaware_fno_predictions_summary.csv"
STATS_JSON = NEURAL_DIR / "patchaware_fno_train_stats.json"


@dataclass(frozen=True)
class LossConfig:
    name: str
    patch_bce_weight: float = 0.5
    patch_dice_weight: float = 0.5
    patch_apd_weight: float = 0.0
    pos_weight_scale: float = 1.0
    scalar_aux_weight: float = 0.0


CONFIGS = [
    LossConfig("baseline", pos_weight_scale=0.0),
    LossConfig("patch_mask_weight_up", patch_bce_weight=1.0, patch_dice_weight=1.0, pos_weight_scale=0.0),
    LossConfig("patch_apd_weight_up", patch_apd_weight=4.0, pos_weight_scale=0.0),
    LossConfig("patch_mask_and_apd_weight_up", patch_bce_weight=1.0, patch_dice_weight=1.0, patch_apd_weight=4.0, pos_weight_scale=0.0),
    LossConfig(
        "patch_mask_apd_plus_scalar_aux",
        patch_bce_weight=1.0,
        patch_dice_weight=1.0,
        patch_apd_weight=4.0,
        pos_weight_scale=0.0,
        scalar_aux_weight=0.05,
    ),
]


def dice_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits) * valid
    target = target * valid
    inter = torch.sum(prob * target, dim=(1, 2))
    denom = torch.sum(prob, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
    return torch.mean(1.0 - (2.0 * inter + eps) / (denom + eps))


def estimate_patch_pos_weight(train_rows: pd.DataFrame) -> float:
    pos = 0.0
    neg = 0.0
    for row in train_rows.itertuples(index=False):
        patch = np.load(ROOT / row.patch_mask_grid_file).astype(bool)
        valid = np.load(ROOT / row.valid_mask_grid_file).astype(bool)
        pos += float(np.sum(patch & valid))
        neg += float(np.sum(~patch & valid))
    if pos <= 0:
        return 1.0
    return float(np.clip(neg / pos, 1.0, 20.0))


def masked_range(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    high_mask = torch.where(valid > 0.5, x, torch.tensor(-1.0e9, device=x.device, dtype=x.dtype))
    low_mask = torch.where(valid > 0.5, x, torch.tensor(1.0e9, device=x.device, dtype=x.dtype))
    return torch.amax(high_mask, dim=(1, 2)) - torch.amin(low_mask, dim=(1, 2))


def patch_delta(x: torch.Tensor, patch: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inside = (patch > 0.5).float() * valid
    outside = (patch <= 0.5).float() * valid
    inside_count = torch.sum(inside, dim=(1, 2))
    outside_count = torch.sum(outside, dim=(1, 2))
    ok = (inside_count > 0.5) & (outside_count > 0.5)
    delta = torch.sum(x * inside, dim=(1, 2)) / torch.clamp(inside_count, min=1.0)
    delta = delta - torch.sum(x * outside, dim=(1, 2)) / torch.clamp(outside_count, min=1.0)
    return delta, ok


def patchaware_loss(pred: torch.Tensor, batch: dict, stats, cfg: LossConfig, pos_weight: float) -> torch.Tensor:
    valid = batch["valid_mask"].to(pred.device).float()
    y = batch["y"].to(pred.device).float()
    patch = y[:, 3]
    total = torch.tensor(0.0, device=pred.device)
    for i in range(3):
        weight = valid
        if i in (1, 2) and cfg.patch_apd_weight > 0:
            weight = valid * (1.0 + cfg.patch_apd_weight * patch)
        diff = torch.nn.functional.smooth_l1_loss(pred[:, i], y[:, i], reduction="none", beta=0.75)
        total = total + torch.sum(diff * weight) / (torch.sum(weight) + 1e-6)
    field_term = total / 3.0

    effective_pos_weight = 1.0 if cfg.pos_weight_scale <= 0 else pos_weight * cfg.pos_weight_scale
    pos_w = torch.tensor(effective_pos_weight, device=pred.device, dtype=pred.dtype)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred[:, 3], patch, reduction="none", pos_weight=pos_w)
    patch_bce = torch.sum(bce * valid) / (torch.sum(valid) + 1e-6)
    patch_term = cfg.patch_bce_weight * patch_bce + cfg.patch_dice_weight * dice_loss(pred[:, 3], patch, valid)

    aux = torch.tensor(0.0, device=pred.device)
    if cfg.scalar_aux_weight > 0:
        mean = torch.tensor(stats.target_mean, device=pred.device, dtype=pred.dtype)
        std = torch.tensor(stats.target_std, device=pred.device, dtype=pred.dtype)
        lat_pred = pred[:, 0] * std[0] + mean[0]
        lat_true = y[:, 0] * std[0] + mean[0]
        apd_pred = pred[:, 2] * std[2] + mean[2]
        apd_true = y[:, 2] * std[2] + mean[2]
        aux = aux + torch.nn.functional.smooth_l1_loss(masked_range(lat_pred, valid), masked_range(lat_true, valid), beta=1.0)
        pred_delta, ok = patch_delta(apd_pred, patch, valid)
        true_delta, _ = patch_delta(apd_true, patch, valid)
        if torch.any(ok):
            aux = aux + torch.nn.functional.smooth_l1_loss(pred_delta[ok], true_delta[ok], beta=1.0)
        pred_area = torch.sum(torch.sigmoid(pred[:, 3]) * valid, dim=(1, 2)) / (torch.sum(valid, dim=(1, 2)) + 1e-6)
        true_area = torch.sum(patch * valid, dim=(1, 2)) / (torch.sum(valid, dim=(1, 2)) + 1e-6)
        aux = aux + torch.nn.functional.smooth_l1_loss(pred_area, true_area, beta=0.02)
    return field_term + patch_term + cfg.scalar_aux_weight * aux


def validation_loss(model, loader, stats, cfg: LossConfig, device: torch.device, pos_weight: float) -> float:
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["x"].to(device).float())
            losses.append(float(patchaware_loss(pred, batch, stats, cfg, pos_weight).detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def train_one(rows: pd.DataFrame, stats, args, cfg: LossConfig, device: torch.device, pos_weight: float):
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    val_rows = rows[rows["split"].eq("val")].reset_index(drop=True)
    probe = make_loader(train_rows.head(1), stats, batch_size=1, shuffle=False, input_norm=args.input_norm)
    in_channels = int(next(iter(probe))["x"].shape[1])
    model = FNOFieldReconstructor(in_channels, args.width, args.modes, args.layers, args.dropout).to(device)
    init_source = None
    init_path = Path(args.init_checkpoint)
    if init_path.exists():
        ckpt = torch.load(init_path, map_location="cpu")
        if int(ckpt.get("in_channels", -1)) == in_channels:
            model.load_state_dict(ckpt["model_state_dict"])
            init_source = str(init_path.relative_to(ROOT))
        else:
            print(f"Skipping init checkpoint with incompatible in_channels: {ckpt.get('in_channels')} != {in_channels}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(train_rows, stats, args.batch_size, True, args.input_norm)
    val_loader = make_loader(val_rows, stats, args.batch_size, False, args.input_norm)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = validation_loss(model, val_loader, stats, cfg, device, pos_weight)
    bad = 0
    history = [{"config": cfg.name, "epoch": 0, "train_loss": math.nan, "val_loss": best_val, "init_source": init_source}]
    print(f"patchaware_fno {cfg.name} epoch 0: init_val={best_val:.4f} init={init_source}", flush=True)
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(batch["x"].to(device).float())
            loss = patchaware_loss(pred, batch, stats, cfg, pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["x"].to(device).float())
                val_losses.append(float(patchaware_loss(pred, batch, stats, cfg, pos_weight).detach().cpu()))
        row = {"config": cfg.name, "epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        history.append(row)
        print(f"patchaware_fno {cfg.name} epoch {epoch}: train={row['train_loss']:.4f} val={row['val_loss']:.4f}", flush=True)
        if row["val_loss"] < best_val - 1e-5:
            best_val = row["val_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            break
    model.load_state_dict(best_state)
    ckpt = CHECKPOINT_DIR / f"patchaware_fno_{cfg.name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "loss_config": cfg.__dict__,
            "history": history,
            "in_channels": in_channels,
            "pos_weight": pos_weight,
            "init_source": init_source,
        },
        ckpt,
    )
    return model, history, ckpt, in_channels


def get_metric(results: pd.DataFrame, target: str, scope: str = "derived_scalar", metric: str = "r2") -> float:
    sub = results[(results["split"].eq("test")) & (results["target"].eq(target)) & (results["metric_scope"].eq(scope)) & (results["variant"].eq("all"))]
    if len(sub) == 0 or metric not in sub:
        return math.nan
    return float(sub[metric].iloc[0])


def plot_outputs(results: pd.DataFrame, histories: list[dict]) -> None:
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    test = results[(results["split"].eq("test")) & (results["variant"].eq("all"))].copy()
    metrics = []
    for cfg in test["config"].unique():
        sub = test[test["config"].eq(cfg)]
        metrics.append(
            {
                "config": cfg,
                "APD90_field_R2": float(sub[(sub["target"].eq("APD90")) & (sub["metric_scope"].eq("field"))]["r2"].iloc[0]),
                "patch_Dice": float(sub[(sub["target"].eq("patch_mask")) & (sub["metric_scope"].eq("field"))]["dice"].iloc[0]),
                "APD90_delta_R2": float(sub[(sub["target"].eq("APD90_patch_delta")) & (sub["metric_scope"].eq("derived_scalar"))]["r2"].iloc[0]),
                "patch_area_R2": float(sub[(sub["target"].eq("patch_area_fraction")) & (sub["metric_scope"].eq("derived_scalar"))]["r2"].iloc[0]),
            }
        )
    summary = pd.DataFrame(metrics)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col, title in zip(
        axes.ravel(),
        ["APD90_field_R2", "patch_Dice", "APD90_delta_R2", "patch_area_R2"],
        ["APD90 field R2", "Patch Dice", "APD90 patch-delta R2", "Patch area fraction R2"],
    ):
        ax.bar(summary["config"], summary[col], color="#4c78a8")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "patchaware_fno_summary.png", dpi=180)
    plt.close(fig)

    hist = pd.DataFrame(histories)
    fig, ax = plt.subplots(figsize=(8, 4))
    for cfg, sub in hist.groupby("config"):
        ax.plot(sub["epoch"], sub["val_loss"], label=cfg)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss")
    ax.set_title("Patch-aware FNO Validation Loss")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "patchaware_fno_training_curves.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-norm", choices=["per_sample", "train_global"], default="per_sample")
    parser.add_argument("--configs", nargs="*", default=[cfg.name for cfg in CONFIGS])
    parser.add_argument("--init-checkpoint", default=str(CHECKPOINT_DIR / "fno_field_reconstructor.pt"))
    args = parser.parse_args()

    seed_all()
    require_cuda(torch)
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_index()
    stats = compute_stats(rows)
    save_stats(STATS_JSON, stats)
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    pos_weight = estimate_patch_pos_weight(train_rows)
    selected = [cfg for cfg in CONFIGS if cfg.name in set(args.configs)]
    device = torch.device("cuda")
    all_results = []
    all_preds = []
    all_histories = []
    checkpoints = []
    for cfg in selected:
        model, history, ckpt, in_channels = train_one(rows, stats, args, cfg, device, pos_weight)
        eval_args = SimpleNamespace(batch_size=args.batch_size, input_norm=args.input_norm)
        results, preds = evaluate(model, stats, rows, eval_args, device)
        results.insert(0, "config", cfg.name)
        preds.insert(0, "config", cfg.name)
        all_results.append(results)
        all_preds.append(preds)
        all_histories.extend(history)
        checkpoints.append({"config": cfg.name, "checkpoint": str(ckpt.relative_to(ROOT)), "in_channels": int(in_channels), "history": history})
    results_df = pd.concat(all_results, ignore_index=True)
    preds_df = pd.concat(all_preds, ignore_index=True)
    results_df.to_csv(RESULTS_CSV, index=False)
    preds_df.to_csv(PRED_CSV, index=False)
    plot_outputs(results_df, all_histories)
    test_summary = []
    for cfg in selected:
        sub = results_df[results_df["config"].eq(cfg.name)]
        test_summary.append(
            {
                "config": cfg.name,
                "APD90_field_R2": get_metric(sub, "APD90", "field"),
                "patch_Dice": get_metric(sub, "patch_mask", "field", "dice"),
                "APD90_patch_delta_R2": get_metric(sub, "APD90_patch_delta"),
                "patch_area_fraction_R2": get_metric(sub, "patch_area_fraction"),
                "LAT_range_R2": get_metric(sub, "LAT_range"),
            }
        )
    payload = {
        "seed": 20260525,
        "architecture": "patchaware_fno2d_field_reconstructor",
        "input_note": "Only magnetic measurements and coordinate channels are model inputs; patch masks are target-side loss weights.",
        "patch_positive_weight": pos_weight,
        "configs": [cfg.__dict__ for cfg in selected],
        "checkpoints": checkpoints,
        "test_summary": test_summary,
        "results": results_df.replace({np.nan: None}).to_dict(orient="records"),
        "cuda": {"torch_version": torch.__version__, "torch_version_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)},
    }
    write_json(RESULTS_JSON, payload)
    print(json.dumps({"results_csv": str(RESULTS_CSV), "test_summary": test_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
