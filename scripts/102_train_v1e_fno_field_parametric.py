                      
                                                                              

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
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from fno_elementary_common import FNOBlock2d
from v1e_direct_common import CHECKPOINT_DIR, NEURAL_DIR, NEURAL_FIG_DIR, require_cuda, scalar_targets_from_fields, write_json
from v1e_direct_fno_data_common import PARAM_TARGETS, compute_stats, load_index, make_loader, save_stats, torch


RESULTS_CSV = NEURAL_DIR / "v1e_fno_field_parametric_results.csv"
RESULTS_JSON = NEURAL_DIR / "v1e_fno_field_parametric_results.json"
PRED_CSV = NEURAL_DIR / "v1e_fno_field_parametric_predictions.csv"
STATS_JSON = NEURAL_DIR / "v1e_fno_field_parametric_stats.json"
SEED = 20260525


CONFIGS = {
    "direct_bz_only_fno_parametric": ["direct_bz_50mm"],
    "direct_bz_plus_grad_fno_parametric": ["direct_bz_50mm", "direct_axial_grad_30mm_baseline"],
    "direct_bz_with_noise_training": ["direct_bz_50mm", "direct_bz_noise_medium", "direct_bz_masked_noise"],
}


class V1EFNOFieldParametric(torch.nn.Module):
    def __init__(self, in_channels: int, width: int, modes: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lift = torch.nn.Conv2d(in_channels, width, 1)
        self.blocks = torch.nn.ModuleList([FNOBlock2d(width, modes, modes, dropout) for _ in range(layers)])
        self.field_head = torch.nn.Sequential(
            torch.nn.Conv2d(width, width, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(dropout),
            torch.nn.Conv2d(width, 4, 1),
        )
        self.param_head = torch.nn.Sequential(
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(width, len(PARAM_TARGETS)),
        )
        self.condition_head = torch.nn.Linear(width, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        pooled = h.mean(dim=(2, 3))
        return self.field_head(h), self.param_head(pooled), self.condition_head(pooled)


def seed_all() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def dice_loss(logits, target, valid, eps: float = 1e-6):
    prob = torch.sigmoid(logits) * valid
    target = target * valid
    inter = torch.sum(prob * target, dim=(1, 2))
    denom = torch.sum(prob, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
    return torch.mean(1.0 - (2.0 * inter + eps) / (denom + eps))


def field_param_loss(field_pred, param_pred, cond_logits, batch, weights=None):
    if weights is None:
        weights = {}
    valid = batch["valid_mask"].to(field_pred.device).float()
    y = batch["y_fields"].to(field_pred.device).float()
    loss = torch.tensor(0.0, device=field_pred.device)
    for i in range(3):
        diff = torch.nn.functional.smooth_l1_loss(field_pred[:, i], y[:, i], reduction="none", beta=0.75)
        loss = loss + torch.sum(diff * valid) / (torch.sum(valid) + 1e-6)
    loss = weights.get("field", 1.0) * loss / 3.0
    patch_target = y[:, 3]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(field_pred[:, 3], patch_target, reduction="none")
    patch_loss = torch.sum(bce * valid) / (torch.sum(valid) + 1e-6) + dice_loss(field_pred[:, 3], patch_target, valid)
    loss = loss + weights.get("patch", 1.0) * patch_loss
    y_param = batch["y_param"].to(field_pred.device).float()
    p_mask = batch["param_mask"].to(field_pred.device).bool()
    present_loss = torch.nn.functional.binary_cross_entropy_with_logits(param_pred[:, 0], y_param[:, 0])
    loss = loss + weights.get("present", 0.5) * present_loss
    if torch.any(p_mask[:, 1:]):
        reg = torch.nn.functional.smooth_l1_loss(param_pred[:, 1:], y_param[:, 1:], reduction="none", beta=0.5)
        loss = loss + weights.get("params", 0.75) * torch.sum(reg * p_mask[:, 1:].float()) / (torch.sum(p_mask[:, 1:].float()) + 1e-6)
    loss = loss + weights.get("condition", 0.15) * torch.nn.functional.cross_entropy(cond_logits, batch["condition"].to(field_pred.device))
    return loss


def finite_pearson(y_true, y_pred):
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        return math.nan
    return float(pearsonr(y_true, y_pred).statistic)


def regression_metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else math.nan,
        "pearson_r": finite_pearson(y_true, y_pred),
    }


def patch_metrics(true, pred, valid):
    t = (true > 0.5) & valid
    p = (pred > 0.5) & valid
    tp = float(np.sum(t & p))
    fp = float(np.sum(~t & p & valid))
    fn = float(np.sum(t & ~p))
    return {
        "dice": (2 * tp) / max(2 * tp + fp + fn, 1.0),
        "iou": tp / max(tp + fp + fn, 1.0),
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
        "homogeneous_false_positive_area": float(np.mean(p[valid])) if np.any(valid) else math.nan,
    }


def param_mask(center_x, center_y, radius, present, shape):
    if present < 0.5 or not np.isfinite(center_x) or not np.isfinite(center_y) or not np.isfinite(radius):
        return np.zeros(shape, dtype=np.float32)
    yy, xx = np.indices(shape)
    x = xx / max(shape[1] - 1, 1)
    y = yy / max(shape[0] - 1, 1)
    return (((x - center_x) ** 2 + (y - center_y) ** 2) <= radius**2).astype(np.float32)


def train_one(config_name: str, rows: pd.DataFrame, args, device):
    variants = CONFIGS[config_name]
    cfg_rows = rows[rows["measurement_variant"].isin(variants)].reset_index(drop=True)
    stats = compute_stats(cfg_rows)
    save_stats(NEURAL_DIR / f"v1e_fno_{config_name}_stats.json", stats)
    train_rows = cfg_rows[cfg_rows["split"].eq("train")].reset_index(drop=True)
    val_rows = cfg_rows[cfg_rows["split"].eq("val")].reset_index(drop=True)
    in_channels = next(iter(make_loader(train_rows.head(1), stats, 1, False, args.input_norm)))["x"].shape[1]
    model = V1EFNOFieldParametric(int(in_channels), args.width, args.modes, args.layers, args.dropout).to(device)
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
            loss = field_param_loss(*pred, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        vals = []
        with torch.no_grad():
            for batch in val_loader:
                vals.append(float(field_param_loss(*model(batch["x"].to(device).float()), batch).detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": float(np.mean(vals))}
        history.append(row)
        print(f"v1e_fno {config_name} epoch {epoch}: train={row['train_loss']:.4f} val={row['val_loss']:.4f}", flush=True)
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
    ckpt = CHECKPOINT_DIR / f"v1e_fno_{config_name}.pt"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "history": history, "in_channels": int(in_channels), "config_name": config_name}, ckpt)
    return model, stats, history, ckpt, cfg_rows


def evaluate_model(model, stats, rows: pd.DataFrame, args, device, model_name: str):
    results = []
    pred_rows = []
    for split in ["train", "val", "test"]:
        sub = rows[rows["split"].eq(split)].reset_index(drop=True)
        loader = make_loader(sub, stats, args.batch_size, False, args.input_norm)
        field_true = {k: [] for k in ["LAT", "APD80", "APD90"]}
        field_pred = {k: [] for k in ["LAT", "APD80", "APD90"]}
        patch_true_all = []
        patch_pred_all = []
        valid_all = []
        y_class = []
        p_class = []
        meta = []
        model.eval()
        offset = 0
        with torch.no_grad():
            for batch in loader:
                field, params, cond = model(batch["x"].to(device).float())
                field = field.cpu().numpy()
                params = params.cpu().numpy()
                cond_np = cond.cpu().numpy()
                y = batch["y_fields"].numpy()
                valid = batch["valid_mask"].numpy().astype(bool)
                param_mask_np = batch["param_mask"].numpy().astype(bool)
                patch_prob = 1.0 / (1.0 + np.exp(-field[:, 3]))
                patch_true = y[:, 3]
                for i, name in enumerate(["LAT", "APD80", "APD90"]):
                    field_true[name].append(y[:, i] * stats.target_std[i] + stats.target_mean[i])
                    field_pred[name].append(field[:, i] * stats.target_std[i] + stats.target_mean[i])
                patch_true_all.append(patch_true)
                patch_pred_all.append(patch_prob)
                valid_all.append(valid)
                y_class.extend(batch["condition"].numpy().tolist())
                p_class.extend(np.argmax(cond_np, axis=1).tolist())
                y_param = batch["y_param"].numpy()
                for b in range(field.shape[0]):
                    src = sub.iloc[offset + b]
                    lat_p = field[b, 0] * stats.target_std[0] + stats.target_mean[0]
                    apd_p = field[b, 2] * stats.target_std[2] + stats.target_mean[2]
                    patch_bin = (patch_prob[b] > 0.5).astype(np.float32)
                    present = float(1.0 / (1.0 + np.exp(-params[b, 0])))
                    unscaled = params[b].copy()
                    unscaled[1:] = params[b, 1:] * stats.param_std[1:] + stats.param_mean[1:]
                    p_mask = param_mask(unscaled[1], unscaled[2], unscaled[4], present, patch_bin.shape)
                    true_lat = y[b, 0] * stats.target_std[0] + stats.target_mean[0]
                    true_apd = y[b, 2] * stats.target_std[2] + stats.target_mean[2]
                    true_scalars = scalar_targets_from_fields(true_lat, true_apd, patch_true[b], valid[b])
                    pix_scalars = scalar_targets_from_fields(lat_p, apd_p, patch_bin, valid[b])
                    prm_scalars = scalar_targets_from_fields(lat_p, apd_p, p_mask, valid[b])
                    meta.append(
                        {
                            "model": model_name,
                            "row_id": src["row_id"],
                            "base_sample_id": src["base_sample_id"],
                            "split": split,
                            "condition": src["condition"],
                            "measurement_variant": src["measurement_variant"],
                            "true_LAT_range": true_scalars["LAT_range"],
                            "pred_LAT_range": pix_scalars["LAT_range"],
                            "true_APD90_patch_delta": true_scalars["APD90_patch_delta"],
                            "pred_APD90_patch_delta_pixel_mask": pix_scalars["APD90_patch_delta"],
                            "pred_APD90_patch_delta_param_mask": prm_scalars["APD90_patch_delta"],
                            "true_patch_area_fraction": true_scalars["patch_area_fraction"],
                            "pred_patch_area_fraction_pixel_mask": pix_scalars["patch_area_fraction"],
                            "pred_patch_area_fraction_param_mask": prm_scalars["patch_area_fraction"],
                            "true_patch_present": float(y_param[b, 0]),
                            "pred_patch_present_prob": present,
                            "true_patch_center_x_norm": np.nan if not param_mask_np[b, 1] else y_param[b, 1] * stats.param_std[1] + stats.param_mean[1],
                            "pred_patch_center_x_norm": unscaled[1],
                            "true_patch_center_y_norm": np.nan if not param_mask_np[b, 2] else y_param[b, 2] * stats.param_std[2] + stats.param_mean[2],
                            "pred_patch_center_y_norm": unscaled[2],
                            "true_patch_area_param": np.nan if not param_mask_np[b, 3] else y_param[b, 3] * stats.param_std[3] + stats.param_mean[3],
                            "pred_patch_area_param": unscaled[3],
                            "true_patch_radius_norm": np.nan if not param_mask_np[b, 4] else y_param[b, 4] * stats.param_std[4] + stats.param_mean[4],
                            "pred_patch_radius_norm": unscaled[4],
                        }
                    )
                offset += field.shape[0]
        valid_cat = np.concatenate(valid_all, axis=0).astype(bool)
        patch_true_cat = np.concatenate(patch_true_all, axis=0)
        patch_pred_cat = np.concatenate(patch_pred_all, axis=0)
        pm = patch_metrics(patch_true_cat, patch_pred_cat, valid_cat)
        if split != "test":
            pm.pop("homogeneous_false_positive_area", None)
        results.append({"model": model_name, "split": split, "target": "patch_mask", "metric_scope": "field", "n": int(np.sum(valid_cat)), **pm})
        for name in ["LAT", "APD80", "APD90"]:
            t = np.concatenate(field_true[name], axis=0)[valid_cat]
            p = np.concatenate(field_pred[name], axis=0)[valid_cat]
            results.append({"model": model_name, "split": split, "target": name, "metric_scope": "field", "n": int(t.size), **regression_metrics(t, p)})
        results.append({"model": model_name, "split": split, "target": "condition", "metric_scope": "classification", "n": len(y_class), "accuracy": float(accuracy_score(y_class, p_class)), "macro_f1": float(f1_score(y_class, p_class, average="macro"))})
        pdf = pd.DataFrame(meta)
        for true_col, pred_col, target in [
            ("true_LAT_range", "pred_LAT_range", "LAT_range"),
            ("true_APD90_patch_delta", "pred_APD90_patch_delta_pixel_mask", "APD90_patch_delta_pixel_mask"),
            ("true_APD90_patch_delta", "pred_APD90_patch_delta_param_mask", "APD90_patch_delta_param_mask"),
            ("true_patch_area_fraction", "pred_patch_area_fraction_pixel_mask", "patch_area_fraction_pixel_mask"),
            ("true_patch_area_fraction", "pred_patch_area_fraction_param_mask", "patch_area_fraction_param_mask"),
            ("true_patch_area_param", "pred_patch_area_param", "patch_area_param_head"),
        ]:
            mask = np.isfinite(pdf[true_col]) & np.isfinite(pdf[pred_col])
            if mask.any():
                results.append({"model": model_name, "split": split, "target": target, "metric_scope": "derived_or_param_scalar", "n": int(mask.sum()), **regression_metrics(pdf.loc[mask, true_col], pdf.loc[mask, pred_col])})
        pred_rows.extend(meta)
    return pd.DataFrame(results), pd.DataFrame(pred_rows)


def plot_training(histories):
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, hist in histories.items():
        h = pd.DataFrame(hist)
        ax.plot(h["epoch"], h["val_loss"], label=name)
    ax.set_title("v1e FNO Validation Loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "v1e_fno_training_curves.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--max-epochs", type=int, default=14)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-norm", choices=["per_sample", "train_global"], default="per_sample")
    args = parser.parse_args()
    seed_all()
    require_cuda(torch)
    rows = load_index()
    grid = int(rows["sensor_grid"].iloc[0])
    if args.modes > grid // 2:
        args.modes = max(4, grid // 2)
    device = torch.device("cuda")
    all_results = []
    all_preds = []
    histories = {}
    checkpoints = {}
    for config in args.configs:
        model, stats, history, ckpt, cfg_rows = train_one(config, rows, args, device)
        res, pred = evaluate_model(model, stats, cfg_rows, args, device, config)
        all_results.append(res)
        all_preds.append(pred)
        histories[config] = history
        checkpoints[config] = str(ckpt)
    results = pd.concat(all_results, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_CSV, index=False)
    preds.to_csv(PRED_CSV, index=False)
    plot_training(histories)
    payload = {
        "architecture": "v1e_fno2d_field_parametric",
        "configs": {k: CONFIGS[k] for k in args.configs},
        "checkpoints": checkpoints,
        "histories": histories,
        "results": results.replace({np.nan: None}).to_dict(orient="records"),
        "cuda": {"torch_version": torch.__version__, "torch_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)},
        "leakage_guardrail": "Inputs are direct magnetic tensors and coordinate channels only; target fields/patch params are labels.",
    }
    write_json(RESULTS_JSON, payload)
    print(json.dumps({"results": str(RESULTS_CSV), "predictions": str(PRED_CSV)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
