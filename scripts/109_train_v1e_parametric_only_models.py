                      
                                                                             

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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from fno_elementary_common import FNOBlock2d
from v1e_direct_common import CHECKPOINT_DIR, COMBINED_READY_DIR, NEURAL_DIR, NEURAL_FIG_DIR, ROOT, require_cuda, write_json
from v1e_direct_fno_data_common import PARAM_TARGETS, compute_stats, make_loader, torch


RESULTS_CSV = NEURAL_DIR / "parametric_only_results.csv"
RESULTS_JSON = NEURAL_DIR / "parametric_only_results.json"
PRED_CSV = NEURAL_DIR / "parametric_only_predictions.csv"
SEED = 20260525


def load_rows() -> pd.DataFrame:
    return pd.read_csv(COMBINED_READY_DIR / "sample_index.csv")


def descriptor_features(b: np.ndarray, mask: np.ndarray, max_channels: int = 4) -> np.ndarray:
    valid = mask[None, :, :, :].repeat(b.shape[0], axis=0)
    filled = np.where(valid, b, 0.0).astype(np.float64)
    feats = []
    for c in range(max_channels):
        if c >= b.shape[-1]:
            feats.extend([0.0] * 11)
            continue
        primary = filled[:, :, :, c]
        m = mask[:, :, c]
        vals = primary[:, m]
        rms_t = np.sqrt(np.mean(vals * vals, axis=1)) if vals.size else np.zeros(primary.shape[0])
        energy = np.mean(primary * primary, axis=0)
        total = float(np.sum(energy))
        g = primary.shape[1]
        yy, xx = np.indices((g, g))
        if total > 0:
            cx = float(np.sum(xx * energy) / total / max(g - 1, 1))
            cy = float(np.sum(yy * energy) / total / max(g - 1, 1))
            p = (energy.ravel() / total).clip(1e-30, None)
            entropy = float(-np.sum(p * np.log(p)) / np.log(p.size))
            concentration = float(np.max(p))
        else:
            cx = cy = entropy = concentration = 0.0
        left = energy[:, : g // 2].sum()
        right = energy[:, g // 2 :].sum()
        bottom = energy[: g // 2, :].sum()
        top = energy[g // 2 :, :].sum()
        feats.extend(
            [
                float(np.sqrt(np.mean(vals * vals))) if vals.size else 0.0,
                float(np.max(np.abs(vals))) if vals.size else 0.0,
                float(np.mean(rms_t[:25])),
                float(np.mean(rms_t[150:280])) if len(rms_t) > 150 else 0.0,
                float(np.argmax(rms_t)),
                cx,
                cy,
                entropy,
                concentration,
                float((right - left) / (right + left + 1e-30)),
                float((top - bottom) / (top + bottom + 1e-30)),
            ]
        )
    return np.asarray(feats, dtype=np.float32)


def load_param_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def build_tabular(rows: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    feats = []
    labels = []
    for _, row in rows.iterrows():
        b = np.load(ROOT / row["B_meas_file"]).astype(np.float32)
        mask = np.load(ROOT / row["sensor_mask_file"]).astype(bool)
        params = load_param_json(row["patch_param_targets_file"])
        feats.append(descriptor_features(b, mask))
        labels.append(
            {
                "row_id": row["row_id"],
                "base_sample_id": row["base_sample_id"],
                "split": row["split"],
                "condition": row["condition"],
                "measurement_variant": row["measurement_variant"],
                "condition_label": row["condition_label"],
                **{k: params.get(k, np.nan) for k in PARAM_TARGETS},
            }
        )
    return np.vstack(feats), pd.DataFrame(labels)


class CNNParametric(torch.nn.Module):
    def __init__(self, in_channels: int, base: int = 48, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, base, 3, padding=1),
            torch.nn.BatchNorm2d(base),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(base, base * 2, 3, padding=1),
            torch.nn.BatchNorm2d(base * 2),
            torch.nn.GELU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(base * 2, base * 3, 3, padding=1),
            torch.nn.GELU(),
        )
        self.head = torch.nn.Sequential(torch.nn.Linear(base * 3, base * 2), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(base * 2, 9))

    def forward(self, x):
        h = self.net(x).mean(dim=(2, 3))
        out = self.head(h)
        return out[:, :5], out[:, 5:]


class FNOParametric(torch.nn.Module):
    def __init__(self, in_channels: int, width: int = 48, modes: int = 12, layers: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.lift = torch.nn.Conv2d(in_channels, width, 1)
        self.blocks = torch.nn.ModuleList([FNOBlock2d(width, modes, modes, dropout) for _ in range(layers)])
        self.head = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(width, 9))

    def forward(self, x):
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        out = self.head(h.mean(dim=(2, 3)))
        return out[:, :5], out[:, 5:]


def param_loss(param_pred, cond_logits, batch):
    y = batch["y_param"].to(param_pred.device).float()
    mask = batch["param_mask"].to(param_pred.device).bool()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(param_pred[:, 0], y[:, 0])
    if torch.any(mask[:, 1:]):
        reg = torch.nn.functional.smooth_l1_loss(param_pred[:, 1:], y[:, 1:], reduction="none", beta=0.5)
        loss = loss + 1.0 * torch.sum(reg * mask[:, 1:].float()) / (torch.sum(mask[:, 1:].float()) + 1e-6)
    loss = loss + 0.25 * torch.nn.functional.cross_entropy(cond_logits, batch["condition"].to(param_pred.device))
    return loss


def train_torch_model(model_name: str, model, rows: pd.DataFrame, stats, args, device):
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    val_rows = rows[rows["split"].eq("val")].reset_index(drop=True)
    train_loader = make_loader(train_rows, stats, args.batch_size, True, args.input_norm)
    val_loader = make_loader(val_rows, stats, args.batch_size, False, args.input_norm)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    best_state = None
    bad = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = param_loss(*model(batch["x"].to(device).float()), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        vals = []
        with torch.no_grad():
            for batch in val_loader:
                vals.append(float(param_loss(*model(batch["x"].to(device).float()), batch).detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": float(np.mean(vals))}
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
    if best_state:
        model.load_state_dict(best_state)
    ckpt = CHECKPOINT_DIR / f"v1e_parametric_only_{model_name}.pt"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "history": history, "args": vars(args)}, ckpt)
    return model, history, ckpt


def param_mask(cx, cy, radius, present, shape=(32, 32)):
    if present < 0.5 or not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(radius):
        return np.zeros(shape, dtype=bool)
    yy, xx = np.indices(shape)
    x = xx / max(shape[1] - 1, 1)
    y = yy / max(shape[0] - 1, 1)
    return ((x - cx) ** 2 + (y - cy) ** 2) <= radius**2


def dice_from_params(row):
    true = param_mask(row.patch_center_x_norm, row.patch_center_y_norm, row.patch_radius_equiv_norm, row.patch_present)
    pred = param_mask(row.pred_patch_center_x_norm, row.pred_patch_center_y_norm, row.pred_patch_radius_equiv_norm, row.pred_patch_present_prob)
    tp = float(np.sum(true & pred))
    fp = float(np.sum(~true & pred))
    fn = float(np.sum(true & ~pred))
    return (2 * tp) / max(2 * tp + fp + fn, 1.0)


def summarize_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    test = preds[preds["split"].eq("test")].copy()
    for model, sub in test.groupby("model"):
        y = sub["patch_present"].astype(int)
        p_prob = sub["pred_patch_present_prob"].astype(float)
        p = (p_prob >= 0.5).astype(int)
        rec = {
            "model": model,
            "split": "test",
            "target": "patch_present",
            "n": int(len(sub)),
            "accuracy": float(accuracy_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro")),
        }
        try:
            rec["auroc"] = float(roc_auc_score(y, p_prob)) if len(np.unique(y)) > 1 else math.nan
        except Exception:
            rec["auroc"] = math.nan
        rows.append(rec)
        non = sub[sub["patch_present"].eq(1)].copy()
        if len(non):
            center_err = np.sqrt((non["patch_center_x_norm"] - non["pred_patch_center_x_norm"]) ** 2 + (non["patch_center_y_norm"] - non["pred_patch_center_y_norm"]) ** 2)
            rows.append({"model": model, "split": "test", "target": "patch_center_error", "n": int(len(non)), "mae": float(np.mean(center_err)), "rmse": float(np.sqrt(np.mean(center_err * center_err)))})
            for target, pred in [
                ("patch_area_fraction", "pred_patch_area_fraction"),
                ("patch_radius_equiv_norm", "pred_patch_radius_equiv_norm"),
            ]:
                rows.append(
                    {
                        "model": model,
                        "split": "test",
                        "target": target,
                        "n": int(len(non)),
                        "mae": float(mean_absolute_error(non[target], non[pred])),
                        "rmse": float(np.sqrt(mean_squared_error(non[target], non[pred]))),
                        "r2": float(r2_score(non[target], non[pred])) if len(non) > 1 else math.nan,
                    }
                )
            dices = non.apply(dice_from_params, axis=1)
            rows.append({"model": model, "split": "test", "target": "parametric_mask_dice", "n": int(len(non)), "dice": float(np.mean(dices))})
        hom = sub[sub["patch_present"].eq(0)].copy()
        if len(hom):
            fp_areas = []
            for row in hom.itertuples(index=False):
                fp_areas.append(float(np.mean(param_mask(row.pred_patch_center_x_norm, row.pred_patch_center_y_norm, row.pred_patch_radius_equiv_norm, row.pred_patch_present_prob))))
            rows.append({"model": model, "split": "test", "target": "homogeneous_false_positive_area", "n": int(len(hom)), "area": float(np.mean(fp_areas))})
    return pd.DataFrame(rows)


def eval_torch(model_name, model, rows, stats, args, device):
    pred_rows = []
    model.eval()
    for split in ["train", "val", "test"]:
        sub = rows[rows["split"].eq(split)].reset_index(drop=True)
        loader = make_loader(sub, stats, args.batch_size, False, args.input_norm)
        offset = 0
        with torch.no_grad():
            for batch in loader:
                param, cond = model(batch["x"].to(device).float())
                param = param.cpu().numpy()
                cond = cond.cpu().numpy()
                y = batch["y_param"].numpy()
                mask = batch["param_mask"].numpy().astype(bool)
                for b in range(param.shape[0]):
                    src = sub.iloc[offset + b]
                    pred = param[b].copy()
                    true = y[b].copy()
                    pred[1:] = pred[1:] * stats.param_std[1:] + stats.param_mean[1:]
                    true[1:] = true[1:] * stats.param_std[1:] + stats.param_mean[1:]
                    pred_present = float(1 / (1 + np.exp(-pred[0])))
                    pred_rows.append(
                        {
                            "model": model_name,
                            "row_id": src["row_id"],
                            "base_sample_id": src["base_sample_id"],
                            "split": split,
                            "condition": src["condition"],
                            "measurement_variant": src["measurement_variant"],
                            "patch_present": float(y[b, 0]),
                            "patch_center_x_norm": np.nan if not mask[b, 1] else true[1],
                            "patch_center_y_norm": np.nan if not mask[b, 2] else true[2],
                            "patch_area_fraction": np.nan if not mask[b, 3] else true[3],
                            "patch_radius_equiv_norm": np.nan if not mask[b, 4] else true[4],
                            "pred_patch_present_prob": pred_present,
                            "pred_condition": int(np.argmax(cond[b])),
                            "pred_patch_center_x_norm": float(pred[1]),
                            "pred_patch_center_y_norm": float(pred[2]),
                            "pred_patch_area_fraction": float(pred[3]),
                            "pred_patch_radius_equiv_norm": float(pred[4]),
                        }
                    )
                offset += param.shape[0]
    return pd.DataFrame(pred_rows)


def run_tabular(rows):
    X, labels = build_tabular(rows)
    train = labels["split"].eq("train")
    scaler = StandardScaler().fit(X[train])
    Xs = scaler.transform(X)
    clf = RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced", min_samples_leaf=2)
    clf.fit(Xs[train], labels.loc[train, "patch_present"].astype(int))
    non_train = train & labels["patch_present"].eq(1)
    reg_targets = ["patch_center_x_norm", "patch_center_y_norm", "patch_area_fraction", "patch_radius_equiv_norm"]
    reg = RandomForestRegressor(n_estimators=300, random_state=SEED, min_samples_leaf=2)
    reg.fit(Xs[non_train], labels.loc[non_train, reg_targets])
    probs = clf.predict_proba(Xs)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(Xs)
    reg_pred = reg.predict(Xs)
    out = labels.copy()
    out["model"] = "tabular_rf_signal_descriptors"
    out["pred_patch_present_prob"] = probs
    out["pred_condition"] = -1
    for i, col in enumerate(reg_targets):
        out[f"pred_{col}"] = reg_pred[:, i]
    return out


def make_figures(results, preds):
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    test = preds[preds["split"].eq("test")]
    fig, ax = plt.subplots(figsize=(6, 4))
    vals = results[(results["split"].eq("test")) & (results["target"].eq("patch_present"))]
    ax.bar(vals["model"], vals["macro_f1"])
    ax.set_ylabel("macro F1")
    ax.set_title("Patch Present")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "parametric_only_patch_present.png", dpi=180)
    plt.close(fig)

    center = results[(results["split"].eq("test")) & (results["target"].eq("patch_center_error"))]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(center["model"], center["mae"])
    ax.set_ylabel("normalized center error")
    ax.set_title("Patch Center Error")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "parametric_only_center_error.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for model, sub in test[test["patch_present"].eq(1)].groupby("model"):
        ax.scatter(sub["patch_area_fraction"], sub["pred_patch_area_fraction"], s=12, alpha=0.7, label=model)
    ax.set_xlabel("true area")
    ax.set_ylabel("predicted area")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "parametric_only_area_predictions.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    metric = results[(results["split"].eq("test")) & (results["target"].isin(["parametric_mask_dice", "patch_area_fraction", "patch_radius_equiv_norm"]))]
    labels = metric["model"] + "\n" + metric["target"]
    vals = metric["dice"].fillna(metric["r2"]).fillna(0)
    ax.bar(range(len(metric)), vals)
    ax.set_xticks(range(len(metric)))
    ax.set_xticklabels(labels, rotation=75, fontsize=7)
    ax.set_title("Parametric-only FNO vs CNN")
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "parametric_only_fno_vs_cnn.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-norm", choices=["per_sample", "train_global"], default="per_sample")
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--modes", type=int, default=12)
    args = parser.parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    require_cuda(torch)
    rows = load_rows()
    stats = compute_stats(rows)
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    in_channels = next(iter(make_loader(train_rows.head(1), stats, 1, False, args.input_norm)))["x"].shape[1]
    device = torch.device("cuda")
    tab_preds = run_tabular(rows)
    cnn = CNNParametric(int(in_channels), base=40).to(device)
    fno = FNOParametric(int(in_channels), width=args.width, modes=min(args.modes, 16)).to(device)
    cnn, cnn_hist, cnn_ckpt = train_torch_model("cnn_encoder_parametric", cnn, rows, stats, args, device)
    fno, fno_hist, fno_ckpt = train_torch_model("fno2d_encoder_parametric", fno, rows, stats, args, device)
    preds = pd.concat([tab_preds, eval_torch("cnn_encoder_parametric", cnn, rows, stats, args, device), eval_torch("fno2d_encoder_parametric", fno, rows, stats, args, device)], ignore_index=True)
    results = summarize_predictions(preds)
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_CSV, index=False)
    preds.to_csv(PRED_CSV, index=False)
    make_figures(results, preds)
    payload = {
        "models": ["tabular_rf_signal_descriptors", "cnn_encoder_parametric", "fno2d_encoder_parametric"],
        "cnn_checkpoint": str(cnn_ckpt),
        "fno_checkpoint": str(fno_ckpt),
        "cnn_history": cnn_hist,
        "fno_history": fno_hist,
        "results": results.replace({np.nan: None}).to_dict(orient="records"),
        "cuda": {"torch_version": torch.__version__, "torch_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)},
        "leakage_guardrail": "Inputs are combined magnetic tensors or descriptors computed from them only; target fields/patch params are labels.",
    }
    write_json(RESULTS_JSON, payload)
    print(json.dumps({"results": str(RESULTS_CSV), "predictions": str(PRED_CSV)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
