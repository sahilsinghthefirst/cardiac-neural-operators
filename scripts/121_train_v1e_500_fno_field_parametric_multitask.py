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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from fno import FNOFieldParametric
from v1e_500_common import CHECKPOINT_DIR, NEURAL_DIR, NEURAL_FIG_DIR, SEED, require_cuda, write_json
from v1e_field_data_common import FIELD_NAMES, PARAM_KEYS, compute_stats, load_descriptors, load_index, make_loader, save_stats, scalar_targets_from_fields, torch


RESULTS_CSV = NEURAL_DIR / "fno_field_parametric_results.csv"
RESULTS_JSON = NEURAL_DIR / "fno_field_parametric_results.json"
PRED_CSV = NEURAL_DIR / "fno_field_parametric_predictions.csv"

FNO_CONFIGS = {
    "fno_field_only": {"descriptors": False, "param": 0.0, "area": 0.0, "scalar": 0.0, "consistency": 0.0},
    "fno_field_plus_param_heads": {"descriptors": False, "param": 0.75, "area": 0.4, "scalar": 0.0, "consistency": 0.02},
    "fno_field_plus_param_heads_plus_descriptors": {"descriptors": True, "param": 0.75, "area": 0.4, "scalar": 0.0, "consistency": 0.02},
    "fno_field_plus_param_heads_plus_descriptors_area_calibrated": {"descriptors": True, "param": 0.9, "area": 0.6, "scalar": 0.06, "consistency": 0.05},
}


def seed_all() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def finite_r2(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 2 or np.allclose(y[m], y[m][0]):
        return math.nan
    return float(r2_score(y[m], p[m]))


def finite_pearson(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 2 or np.allclose(y[m], y[m][0]) or np.allclose(p[m], p[m][0]):
        return math.nan
    return float(np.corrcoef(y[m], p[m])[0, 1])


def regression_metrics(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if not np.any(m):
        return {"mae": math.nan, "rmse": math.nan, "r2": math.nan, "pearson_r": math.nan}
    return {
        "mae": float(mean_absolute_error(y[m], p[m])),
        "rmse": float(np.sqrt(mean_squared_error(y[m], p[m]))),
        "r2": finite_r2(y[m], p[m]),
        "pearson_r": finite_pearson(y[m], p[m]),
    }


def dice_loss(logits, target, valid, eps: float = 1e-6):
    prob = torch.sigmoid(logits) * valid
    target = target * valid
    inter = torch.sum(prob * target, dim=(1, 2))
    denom = torch.sum(prob, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
    return torch.mean(1.0 - (2.0 * inter + eps) / (denom + eps))


def patch_metrics(true, pred_prob, valid):
    t = (true > 0.5) & valid
    p = (pred_prob > 0.5) & valid
    tp = float(np.sum(t & p))
    fp = float(np.sum(~t & p & valid))
    fn = float(np.sum(t & ~p))
    return {
        "dice": (2 * tp) / max(2 * tp + fp + fn, 1.0),
        "iou": tp / max(tp + fp + fn, 1.0),
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
    }


def param_mask(cx, cy, radius, present, shape):
    if present < 0.5 or not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(radius):
        return np.zeros(shape, dtype=np.float32)
    yy, xx = np.indices(shape)
    x = xx / max(shape[1] - 1, 1)
    y = yy / max(shape[0] - 1, 1)
    radius = float(np.clip(radius, 1e-4, 0.75))
    return (((x - np.clip(cx, 0, 1)) ** 2 + (y - np.clip(cy, 0, 1)) ** 2) <= radius**2).astype(np.float32)


def multitask_loss(field_pred, param_pred, area_logits, cond_logits, batch, stats, weights):
    valid = batch["valid_mask"].to(field_pred.device).float()
    y = batch["y_fields"].to(field_pred.device).float()
    loss = torch.tensor(0.0, device=field_pred.device)
    for i in range(3):
        diff = torch.nn.functional.smooth_l1_loss(field_pred[:, i], y[:, i], reduction="none", beta=0.75)
        loss = loss + torch.sum(diff * valid) / (torch.sum(valid) + 1e-6)
    loss = weights.get("field", 1.0) * loss / 3.0
    patch_target = y[:, 3]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(field_pred[:, 3], patch_target, reduction="none")
    present = batch["y_param"].to(field_pred.device).float()[:, 0]
    neg_weight = torch.where(present[:, None, None] < 0.5, torch.full_like(valid, 2.0), torch.ones_like(valid))
    patch_loss = torch.sum(bce * valid * neg_weight) / (torch.sum(valid * neg_weight) + 1e-6) + dice_loss(field_pred[:, 3], patch_target, valid)
    loss = loss + weights.get("patch", 1.0) * patch_loss
    if weights.get("param", 0.0) > 0:
        y_param = batch["y_param"].to(field_pred.device).float()
        p_mask = batch["param_mask"].to(field_pred.device).bool()
        bce_p = torch.nn.functional.binary_cross_entropy_with_logits(param_pred[:, 0], y_param[:, 0], reduction="none")
        loss = loss + weights.get("present", 0.5) * torch.mean(bce_p * torch.where(y_param[:, 0] < 0.5, 2.0, 1.0))
        non = p_mask[:, 1:].any(dim=1)
        if torch.any(non):
            reg = torch.nn.functional.smooth_l1_loss(param_pred[non, 1:], y_param[non, 1:], reduction="none", beta=0.5)
            loss = loss + weights["param"] * torch.mean(reg)
            if weights.get("area", 0.0) > 0:
                loss = loss + weights["area"] * torch.nn.functional.cross_entropy(area_logits[non], batch["area_bin"].to(field_pred.device)[non])
            if weights.get("consistency", 0.0) > 0:
                mean = torch.as_tensor(stats.param_mean, dtype=torch.float32, device=field_pred.device)
                std = torch.as_tensor(stats.param_std, dtype=torch.float32, device=field_pred.device)
                unscaled = param_pred * std[None, :] + mean[None, :]
                consistency = unscaled[non, 3] - (math.log(math.pi) + 2.0 * unscaled[non, 4])
                loss = loss + weights["consistency"] * torch.mean(consistency * consistency)
        loss = loss + weights.get("condition", 0.12) * torch.nn.functional.cross_entropy(cond_logits, batch["condition"].to(field_pred.device))
    if weights.get("scalar", 0.0) > 0:

        apd_pred = field_pred[:, 2]
        apd_true_scale = torch.as_tensor(stats.field_std[2], dtype=torch.float32, device=field_pred.device)
        patch = y[:, 3] > 0.5
        outside = (y[:, 3] <= 0.5) & (valid > 0.5)
        inside = patch & (valid > 0.5)
        deltas = []
        truth = []
        for b in range(field_pred.shape[0]):
            if torch.any(inside[b]) and torch.any(outside[b]):
                deltas.append((torch.mean(apd_pred[b][inside[b]]) - torch.mean(apd_pred[b][outside[b]])) * apd_true_scale)
                truth.append((torch.mean(y[b, 2][inside[b]]) - torch.mean(y[b, 2][outside[b]])) * apd_true_scale)
        if deltas:
            loss = loss + weights["scalar"] * torch.nn.functional.smooth_l1_loss(torch.stack(deltas), torch.stack(truth), beta=1.0)
    return loss


def train_model(model_name, model, rows, desc, stats, args, device, weights):
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    val_rows = rows[rows["split"].eq("val")].reset_index(drop=True)
    train_loader = make_loader(train_rows, stats, args.batch_size, True, args.input_norm, desc)
    val_loader = make_loader(val_rows, stats, args.batch_size, False, args.input_norm, desc)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    best_state = None
    bad = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(batch["x"].to(device).float(), batch["desc"].to(device).float())
            loss = multitask_loss(*pred, batch, stats, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["x"].to(device).float(), batch["desc"].to(device).float())
                val_losses.append(float(multitask_loss(*pred, batch, stats, weights).detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        history.append(row)
        print(f"{model_name} epoch {epoch}: train={row['train_loss']:.4f} val={row['val_loss']:.4f}", flush=True)
        if row["val_loss"] < best - 1e-5:
            best = row["val_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINT_DIR / f"v1e_500_{model_name}.pt"
    torch.save({"model_state_dict": model.state_dict(), "history": history, "args": vars(args), "weights": weights}, ckpt)
    return model, history, ckpt


def evaluate_model(model, stats, rows, desc, args, device, model_name):
    results = []
    pred_rows = []
    for split in ["train", "val", "test"]:
        sub = rows[rows["split"].eq(split)].reset_index(drop=True)
        loader = make_loader(sub, stats, args.batch_size, False, args.input_norm, desc)
        field_true = {k: [] for k in FIELD_NAMES}
        field_pred = {k: [] for k in FIELD_NAMES}
        patch_true_all = []
        patch_pred_all = []
        valid_all = []
        y_class = []
        p_class = []
        y_present = []
        p_present = []
        meta = []
        model.eval()
        offset = 0
        with torch.no_grad():
            for batch in loader:
                field, params, area_logits, cond = model(batch["x"].to(device).float(), batch["desc"].to(device).float())
                field = field.cpu().numpy()
                params = params.cpu().numpy()
                area_np = area_logits.cpu().numpy()
                cond_np = cond.cpu().numpy()
                y = batch["y_fields"].numpy()
                valid = batch["valid_mask"].numpy().astype(bool)
                param_mask_np = batch["param_mask"].numpy().astype(bool)
                y_param = batch["y_param"].numpy()
                patch_prob = 1.0 / (1.0 + np.exp(-field[:, 3]))
                for i, name in enumerate(FIELD_NAMES):
                    field_true[name].append(y[:, i] * stats.field_std[i] + stats.field_mean[i])
                    field_pred[name].append(field[:, i] * stats.field_std[i] + stats.field_mean[i])
                patch_true_all.append(y[:, 3])
                patch_pred_all.append(patch_prob)
                valid_all.append(valid)
                y_class.extend(batch["condition"].numpy().tolist())
                p_class.extend(np.argmax(cond_np, axis=1).tolist())
                y_present.extend(y_param[:, 0].tolist())
                p_present.extend((1.0 / (1.0 + np.exp(-params[:, 0]))).tolist())
                for b in range(field.shape[0]):
                    src = sub.iloc[offset + b]
                    lat_p = field[b, 0] * stats.field_std[0] + stats.field_mean[0]
                    apd_p = field[b, 2] * stats.field_std[2] + stats.field_mean[2]
                    true_lat = y[b, 0] * stats.field_std[0] + stats.field_mean[0]
                    true_apd = y[b, 2] * stats.field_std[2] + stats.field_mean[2]
                    patch_true = y[b, 3]
                    patch_bin = (patch_prob[b] > 0.5).astype(np.float32)
                    present = float(1.0 / (1.0 + np.exp(-params[b, 0])))
                    unscaled = params[b].copy()
                    unscaled[1:] = params[b, 1:] * stats.param_std[1:] + stats.param_mean[1:]
                    pred_radius = float(np.exp(unscaled[4])) if np.isfinite(unscaled[4]) else np.nan
                    pred_area = float(np.exp(unscaled[3])) if np.isfinite(unscaled[3]) else np.nan
                    p_mask = param_mask(unscaled[1], unscaled[2], pred_radius, present, patch_bin.shape)
                    true_scalars = scalar_targets_from_fields(true_lat, true_apd, patch_true, valid[b])
                    pix_pred_scalars = scalar_targets_from_fields(lat_p, apd_p, patch_bin, valid[b])
                    prm_pred_scalars = scalar_targets_from_fields(lat_p, apd_p, p_mask, valid[b])
                    prm_true_apd = scalar_targets_from_fields(true_lat, true_apd, p_mask, valid[b])
                    true_mask_pred_apd = scalar_targets_from_fields(lat_p, apd_p, patch_true, valid[b])
                    true_log_area = y_param[b, 3] * stats.param_std[3] + stats.param_mean[3] if param_mask_np[b, 3] else np.nan
                    true_log_radius = y_param[b, 4] * stats.param_std[4] + stats.param_mean[4] if param_mask_np[b, 4] else np.nan
                    meta.append(
                        {
                            "model": model_name,
                            "row_id": src["row_id"],
                            "base_sample_id": src["base_sample_id"],
                            "split": split,
                            "condition": src["condition"],
                            "combined_variant": src["combined_variant"],
                            "true_LAT_range": true_scalars["LAT_range"],
                            "pred_LAT_range": pix_pred_scalars["LAT_range"],
                            "true_APD90_patch_delta": true_scalars["APD90_patch_delta"],
                            "pred_APD90_patch_delta_pixel_mask_pred_apd": pix_pred_scalars["APD90_patch_delta"],
                            "pred_APD90_patch_delta_param_mask_pred_apd": prm_pred_scalars["APD90_patch_delta"],
                            "pred_APD90_patch_delta_param_mask_true_apd": prm_true_apd["APD90_patch_delta"],
                            "pred_APD90_patch_delta_true_mask_pred_apd": true_mask_pred_apd["APD90_patch_delta"],
                            "true_patch_area_fraction": true_scalars["patch_area_fraction"],
                            "pred_patch_area_fraction_pixel_mask": pix_pred_scalars["patch_area_fraction"],
                            "pred_patch_area_fraction_param_mask": prm_pred_scalars["patch_area_fraction"],
                            "true_patch_present": float(y_param[b, 0]),
                            "pred_patch_present_prob": present,
                            "true_center_x_norm": np.nan if not param_mask_np[b, 1] else y_param[b, 1] * stats.param_std[1] + stats.param_mean[1],
                            "pred_center_x_norm": unscaled[1],
                            "true_center_y_norm": np.nan if not param_mask_np[b, 2] else y_param[b, 2] * stats.param_std[2] + stats.param_mean[2],
                            "pred_center_y_norm": unscaled[2],
                            "true_log_area_fraction": true_log_area,
                            "pred_log_area_fraction": unscaled[3],
                            "true_log_radius": true_log_radius,
                            "pred_log_radius": unscaled[4],
                            "true_patch_area_param": float(np.exp(true_log_area)) if np.isfinite(true_log_area) else np.nan,
                            "pred_patch_area_param": pred_area,
                            "true_radius_param": float(np.exp(true_log_radius)) if np.isfinite(true_log_radius) else np.nan,
                            "pred_radius_param": pred_radius,
                            "true_area_bin": int(batch["area_bin"].numpy()[b]) if bool(batch["area_bin_mask"].numpy()[b]) else -1,
                            "pred_area_bin": int(np.argmax(area_np[b])),
                            "pred_condition_label": int(np.argmax(cond_np[b])),
                        }
                    )
                offset += field.shape[0]
        valid_cat = np.concatenate(valid_all, axis=0).astype(bool)
        patch_true_cat = np.concatenate(patch_true_all, axis=0)
        patch_pred_cat = np.concatenate(patch_pred_all, axis=0)
        pm = patch_metrics(patch_true_cat, patch_pred_cat, valid_cat)
        results.append({"model": model_name, "split": split, "target": "patch_mask", "metric_scope": "field", "n": int(np.sum(valid_cat)), **pm})
        for name in FIELD_NAMES:
            t = np.concatenate(field_true[name], axis=0)[valid_cat]
            p = np.concatenate(field_pred[name], axis=0)[valid_cat]
            results.append({"model": model_name, "split": split, "target": name, "metric_scope": "field", "n": int(t.size), **regression_metrics(t, p)})
        results.append({"model": model_name, "split": split, "target": "condition", "metric_scope": "classification", "n": len(y_class), "accuracy": float(accuracy_score(y_class, p_class)), "balanced_accuracy": float(balanced_accuracy_score(y_class, p_class)), "macro_f1": float(f1_score(y_class, p_class, average="macro"))})
        pred_present_bin = (np.asarray(p_present) >= 0.5).astype(int)
        present_row = {"model": model_name, "split": split, "target": "patch_present", "metric_scope": "classification", "n": len(y_present), "accuracy": float(accuracy_score(y_present, pred_present_bin)), "balanced_accuracy": float(balanced_accuracy_score(y_present, pred_present_bin)), "macro_f1": float(f1_score(y_present, pred_present_bin, average="macro"))}
        try:
            present_row["auroc"] = float(roc_auc_score(y_present, p_present)) if len(np.unique(y_present)) > 1 else math.nan
        except Exception:
            present_row["auroc"] = math.nan
        results.append(present_row)
        pdf = pd.DataFrame(meta)
        hom = pdf[pdf["true_patch_present"].eq(0)]
        if len(hom):
            results.append({"model": model_name, "split": split, "target": "homogeneous_false_positive_area_pixel_mask", "metric_scope": "patch", "n": int(len(hom)), "area": float(hom["pred_patch_area_fraction_pixel_mask"].mean())})
            results.append({"model": model_name, "split": split, "target": "homogeneous_false_positive_area_param_mask", "metric_scope": "patch", "n": int(len(hom)), "area": float(hom["pred_patch_area_fraction_param_mask"].mean())})
        for true_col, pred_col, target in [
            ("true_LAT_range", "pred_LAT_range", "LAT_range"),
            ("true_APD90_patch_delta", "pred_APD90_patch_delta_pixel_mask_pred_apd", "APD90_patch_delta_pixel_mask_pred_apd"),
            ("true_APD90_patch_delta", "pred_APD90_patch_delta_param_mask_pred_apd", "APD90_patch_delta_param_mask_pred_apd"),
            ("true_APD90_patch_delta", "pred_APD90_patch_delta_param_mask_true_apd", "APD90_patch_delta_param_mask_true_apd"),
            ("true_APD90_patch_delta", "pred_APD90_patch_delta_true_mask_pred_apd", "APD90_patch_delta_true_mask_pred_apd"),
            ("true_patch_area_fraction", "pred_patch_area_fraction_pixel_mask", "patch_area_fraction_pixel_mask"),
            ("true_patch_area_fraction", "pred_patch_area_fraction_param_mask", "patch_area_fraction_param_mask"),
            ("true_patch_area_param", "pred_patch_area_param", "patch_area_param_head"),
            ("true_log_area_fraction", "pred_log_area_fraction", "log_area_param_head"),
            ("true_radius_param", "pred_radius_param", "radius_param_head"),
            ("true_log_radius", "pred_log_radius", "log_radius_param_head"),
        ]:
            mask = np.isfinite(pdf[true_col]) & np.isfinite(pdf[pred_col])
            if mask.any():
                results.append({"model": model_name, "split": split, "target": target, "metric_scope": "derived_or_param_scalar", "n": int(mask.sum()), **regression_metrics(pdf.loc[mask, true_col], pdf.loc[mask, pred_col])})
        non = pdf[pdf["true_area_bin"].ge(0)]
        if len(non):
            results.append({"model": model_name, "split": split, "target": "area_bin", "metric_scope": "classification", "n": int(len(non)), "accuracy": float(accuracy_score(non["true_area_bin"], non["pred_area_bin"])), "macro_f1": float(f1_score(non["true_area_bin"], non["pred_area_bin"], average="macro"))})
        pred_rows.extend(meta)
    return pd.DataFrame(results), pd.DataFrame(pred_rows)


def plot_training(histories: dict[str, list[dict]], filename: str) -> None:
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, hist in histories.items():
        h = pd.DataFrame(hist)
        ax.plot(h["epoch"], h["val_loss"], label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss")
    ax.set_title("v1e 500 FNO Field + Parametric")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / filename, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", choices=list(FNO_CONFIGS), default=list(FNO_CONFIGS))
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--width", type=int, default=64)
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
    desc = load_descriptors()
    stats = compute_stats(rows, desc)
    save_stats(NEURAL_DIR / "v1e_500_fno_field_parametric_stats.json", stats)
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    first = next(iter(make_loader(train_rows.head(1), stats, 1, False, args.input_norm, desc)))
    in_channels = int(first["x"].shape[1])
    desc_dim = int(first["desc"].shape[1])
    device = torch.device("cuda")
    all_results = []
    all_preds = []
    histories = {}
    checkpoints = {}
    for config in args.configs:
        weights = {"field": 1.0, "patch": 1.0, "present": 0.5, "condition": 0.12, **FNO_CONFIGS[config]}
        model = FNOFieldParametric(in_channels, desc_dim, args.width, min(args.modes, 16), args.layers, args.dropout, weights["descriptors"]).to(device)
        model, hist, ckpt = train_model(config, model, rows, desc, stats, args, device, weights)
        res, pred = evaluate_model(model, stats, rows, desc, args, device, config)
        all_results.append(res)
        all_preds.append(pred)
        histories[config] = hist
        checkpoints[config] = str(ckpt)
    results = pd.concat(all_results, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_CSV, index=False)
    preds.to_csv(PRED_CSV, index=False)
    plot_training(histories, "v1e_500_fno_field_training_curves.png")
    payload = {
        "architecture": "fno2d_field_parametric_multitask",
        "configs": FNO_CONFIGS,
        "checkpoints": checkpoints,
        "histories": histories,
        "input_channels": in_channels,
        "descriptor_dim": desc_dim,
        "results": results.replace({np.nan: None}).to_dict(orient="records"),
        "cuda": {"torch_version": torch.__version__, "torch_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)},
        "leakage_guardrail": "Inputs are combined magnetic tensors, coordinate channels, and optional B_meas-derived descriptors only; target fields and patch params are labels.",
    }
    write_json(RESULTS_JSON, payload)
    print(json.dumps({"results": str(RESULTS_CSV), "predictions": str(PRED_CSV), "configs": args.configs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
