                      
                                                           
 
                                                                          
                                                                           
                                              
   

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if (PROJECT_ROOT / ".local_deps" / "sklearn").exists():
    sys.path.append(str(PROJECT_ROOT / ".local_deps" / "sklearn"))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from fno_elementary_common import FNOBlock2d
from torch_import_utils import import_torch_prefer_env
from v1e_500_common import CHECKPOINT_DIR, NEURAL_DIR, READY_DIR, ROOT, SEED, require_cuda, write_json


torch = import_torch_prefer_env(ROOT)
PRED_CSV = NEURAL_DIR / "v1e_500_parametric_predictions_raw.csv"
SUMMARY_JSON = NEURAL_DIR / "v1e_500_parametric_training_summary.json"
PARAM_KEYS = ["patch_present", "center_x_norm", "center_y_norm", "log_area_fraction", "log_radius"]


@dataclass
class TrainStats:
    input_mean: float
    input_std: float
    param_mean: np.ndarray
    param_std: np.ndarray


def load_param_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def load_rows() -> pd.DataFrame:
    return pd.read_csv(READY_DIR / "sample_index.csv")


def descriptor_features(b: np.ndarray, mask: np.ndarray, max_channels: int = 4) -> np.ndarray:
    valid = mask[None, :, :, :].repeat(b.shape[0], axis=0)
    filled = np.where(valid, b, 0.0).astype(np.float64)
    feats: list[float] = []
    for c in range(max_channels):
        if c >= b.shape[-1] or not np.any(mask[:, :, c]):
            feats.extend([0.0] * 11)
            continue
        primary = filled[:, :, :, c]
        vals = primary[:, mask[:, :, c]]
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


def label_row(row: pd.Series) -> dict:
    params = load_param_json(row["patch_param_targets_file"])
    return {
        "row_id": row["row_id"],
        "base_sample_id": row["base_sample_id"],
        "split": row["split"],
        "condition": row["condition"],
        "measurement_variant": row["measurement_variant"],
        "condition_label": int(row["condition_label"]),
        "patch_present": float(params["patch_present"]),
        "center_x_norm": params.get("center_x_norm", np.nan),
        "center_y_norm": params.get("center_y_norm", np.nan),
        "patch_area_fraction": float(params.get("patch_area_fraction", 0.0)),
        "equivalent_radius_norm": params.get("equivalent_radius_norm", np.nan),
        "log_area_fraction": params.get("log_area_fraction", np.nan),
        "log_radius": params.get("log_radius", np.nan),
        "area_bin": int(params.get("area_bin", -1)),
        "area_bin_label": params.get("area_bin_label", "none"),
    }


def compute_stats(rows: pd.DataFrame, input_norm: str) -> TrainStats:
    train = rows[rows["split"].eq("train")]
    total = 0.0
    total_sq = 0.0
    count = 0
    param_vals = [[] for _ in PARAM_KEYS]
    for _, row in train.iterrows():
        b = np.load(ROOT / row["B_meas_file"]).astype(np.float64)
        mask = np.load(ROOT / row["sensor_mask_file"]).astype(bool)
        valid = mask[None, :, :, :].repeat(b.shape[0], axis=0)
        vals = b[valid]
        vals = vals[np.isfinite(vals)]
        total += float(vals.sum())
        total_sq += float(np.square(vals).sum())
        count += int(vals.size)
        params = label_row(row)
        for i, key in enumerate(PARAM_KEYS):
            val = params[key]
            if val is not None and np.isfinite(val):
                param_vals[i].append(float(val))
    mean = total / max(count, 1) if input_norm == "train_global" else 0.0
    var = max(total_sq / max(count, 1) - mean * mean, 0.0)
    std = float(np.sqrt(var)) if var > 0 else 1.0
    pmean = np.zeros(len(PARAM_KEYS), dtype=np.float32)
    pstd = np.ones(len(PARAM_KEYS), dtype=np.float32)
    for i, vals in enumerate(param_vals):
        arr = np.asarray(vals, dtype=np.float32) if vals else np.array([0.0], dtype=np.float32)
        pmean[i] = float(np.mean(arr))
        pstd[i] = max(float(np.std(arr)), 1e-6)
    pmean[0] = 0.0
    pstd[0] = 1.0
    return TrainStats(mean, std, pmean, pstd)


class ParametricDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, stats: TrainStats, input_norm: str = "per_sample") -> None:
        self.rows = rows.reset_index(drop=True)
        self.stats = stats
        self.input_norm = input_norm

    def __len__(self) -> int:
        return len(self.rows)

    def _load_x(self, row: pd.Series) -> np.ndarray:
        b = np.load(ROOT / row["B_meas_file"]).astype(np.float32)
        mask = np.load(ROOT / row["sensor_mask_file"]).astype(bool)
        valid = mask[None, :, :, :].repeat(b.shape[0], axis=0)
        if self.input_norm == "per_sample":
            vals = b[valid]
            mean = float(np.nanmean(vals))
            std = float(np.nanstd(vals))
        elif self.input_norm == "train_global":
            mean = self.stats.input_mean
            std = self.stats.input_std
        else:
            mean = 0.0
            std = 1.0
        if not np.isfinite(std) or std <= 1e-20:
            std = 1.0
        b = np.where(valid, (b - mean) / std, 0.0)
        b = np.where(np.isfinite(b), b, 0.0)
        x = b.transpose(3, 0, 1, 2).reshape(b.shape[0] * b.shape[-1], b.shape[1], b.shape[2])
        coords = np.load(ROOT / row["coordinate_channels_file"]).astype(np.float32)[:, :, :3]
        return np.concatenate([x, coords.transpose(2, 0, 1)], axis=0).astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows.iloc[idx]
        labels = label_row(row)
        y = np.asarray([labels[k] for k in PARAM_KEYS], dtype=np.float32)
        mask = np.isfinite(y)
        y_scaled = (np.where(mask, y, self.stats.param_mean) - self.stats.param_mean) / self.stats.param_std
        y_scaled[0] = np.nan_to_num(y[0], nan=0.0)
        mask[0] = True
        return {
            "x": torch.from_numpy(self._load_x(row)),
            "y_param": torch.from_numpy(y_scaled.astype(np.float32)),
            "param_mask": torch.from_numpy(mask.astype(bool)),
            "area_bin": torch.tensor(max(labels["area_bin"], 0), dtype=torch.long),
            "area_bin_mask": torch.tensor(labels["area_bin"] >= 0, dtype=torch.bool),
            "condition": torch.tensor(labels["condition_label"], dtype=torch.long),
            "row_id": row["row_id"],
        }


def make_loader(rows: pd.DataFrame, stats: TrainStats, batch_size: int, shuffle: bool, input_norm: str):
    return DataLoader(ParametricDataset(rows, stats, input_norm), batch_size=batch_size, shuffle=shuffle, num_workers=0)


class CNNParametric(torch.nn.Module):
    def __init__(self, in_channels: int, base: int = 40, dropout: float = 0.1) -> None:
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
        self.head = torch.nn.Sequential(torch.nn.Linear(base * 3, base * 2), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(base * 2, 16))

    def forward(self, x):
        h = self.net(x).mean(dim=(2, 3))
        out = self.head(h)
        return out[:, :5], out[:, 5:8], out[:, 8:12]


class FNOParametric(torch.nn.Module):
    def __init__(self, in_channels: int, width: int = 48, modes: int = 12, layers: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.lift = torch.nn.Conv2d(in_channels, width, 1)
        self.blocks = torch.nn.ModuleList([FNOBlock2d(width, modes, modes, dropout) for _ in range(layers)])
        self.head = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(width, 16))

    def forward(self, x):
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        out = self.head(h.mean(dim=(2, 3)))
        return out[:, :5], out[:, 5:8], out[:, 8:12]


def multitask_loss(param_pred, area_logits, cond_logits, batch, stats: TrainStats, consistency_weight: float):
    y = batch["y_param"].to(param_pred.device).float()
    mask = batch["param_mask"].to(param_pred.device).bool()
    y_present = y[:, 0]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(param_pred[:, 0], y_present, reduction="none")
    weights = torch.where(y_present < 0.5, torch.full_like(y_present, 2.0), torch.ones_like(y_present))
    loss = torch.mean(bce * weights)
    non = mask[:, 1:].any(dim=1)
    if torch.any(non):
        reg = torch.nn.functional.smooth_l1_loss(param_pred[non, 1:], y[non, 1:], reduction="none", beta=0.5)
        loss = loss + torch.mean(reg)
        loss = loss + 0.6 * torch.nn.functional.cross_entropy(area_logits[non], batch["area_bin"].to(param_pred.device)[non])
        if consistency_weight > 0:
            mean = torch.as_tensor(stats.param_mean, dtype=torch.float32, device=param_pred.device)
            std = torch.as_tensor(stats.param_std, dtype=torch.float32, device=param_pred.device)
            unscaled = param_pred * std[None, :] + mean[None, :]
            consistency = unscaled[non, 3] - (math.log(math.pi) + 2.0 * unscaled[non, 4])
            loss = loss + consistency_weight * torch.mean(consistency * consistency)
    loss = loss + 0.15 * torch.nn.functional.cross_entropy(cond_logits, batch["condition"].to(param_pred.device))
    return loss


def train_torch_model(name: str, model, rows: pd.DataFrame, stats: TrainStats, args, device):
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    val_rows = rows[rows["split"].eq("val")].reset_index(drop=True)
    train_loader = make_loader(train_rows, stats, args.batch_size, True, args.input_norm)
    val_loader = make_loader(val_rows, stats, args.batch_size, False, args.input_norm)
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
            loss = multitask_loss(*model(batch["x"].to(device).float()), batch, stats, args.consistency_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                val_losses.append(float(multitask_loss(*model(batch["x"].to(device).float()), batch, stats, args.consistency_weight).detach().cpu()))
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        history.append(row)
        print(f"{name} epoch {epoch}: train={row['train_loss']:.4f} val={row['val_loss']:.4f}", flush=True)
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
    ckpt = CHECKPOINT_DIR / f"v1e_500_{name}.pt"
    torch.save({"model_state_dict": model.state_dict(), "history": history, "args": vars(args)}, ckpt)
    return model, history, ckpt


def build_tabular(rows: pd.DataFrame):
    feats = []
    labels = []
    for _, row in rows.iterrows():
        b = np.load(ROOT / row["B_meas_file"]).astype(np.float32)
        mask = np.load(ROOT / row["sensor_mask_file"]).astype(bool)
        feats.append(descriptor_features(b, mask))
        labels.append(label_row(row))
    return np.vstack(feats), pd.DataFrame(labels)


def run_tabular(rows: pd.DataFrame) -> pd.DataFrame:
    X, labels = build_tabular(rows)
    train = labels["split"].eq("train")
    scaler = StandardScaler().fit(X[train])
    Xs = scaler.transform(X)
    clf = RandomForestClassifier(n_estimators=400, random_state=SEED, class_weight="balanced", min_samples_leaf=2)
    clf.fit(Xs[train], labels.loc[train, "patch_present"].astype(int))
    cond = RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced", min_samples_leaf=2)
    cond.fit(Xs[train], labels.loc[train, "condition_label"].astype(int))
    non_train = train & labels["patch_present"].eq(1)
    reg_targets = ["center_x_norm", "center_y_norm", "log_area_fraction", "log_radius"]
    reg = RandomForestRegressor(n_estimators=400, random_state=SEED, min_samples_leaf=2)
    reg.fit(Xs[non_train], labels.loc[non_train, reg_targets])
    area = RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced", min_samples_leaf=2)
    area.fit(Xs[non_train], labels.loc[non_train, "area_bin"].astype(int))
    probs = clf.predict_proba(Xs)[:, 1]
    reg_pred = reg.predict(Xs)
    area_pred = area.predict(Xs)
    out = labels.copy()
    out["model"] = "tabular_rf_signal_descriptors"
    out["pred_patch_present_prob"] = probs
    out["pred_center_x_norm"] = reg_pred[:, 0]
    out["pred_center_y_norm"] = reg_pred[:, 1]
    out["pred_log_area_fraction"] = reg_pred[:, 2]
    out["pred_log_radius"] = reg_pred[:, 3]
    out["pred_area_bin"] = area_pred
    out["pred_condition_label"] = cond.predict(Xs)
    return out


def eval_torch_model(name: str, model, rows: pd.DataFrame, stats: TrainStats, args, device) -> pd.DataFrame:
    pred_rows = []
    model.eval()
    for split in ["train", "val", "test"]:
        sub = rows[rows["split"].eq(split)].reset_index(drop=True)
        loader = make_loader(sub, stats, args.batch_size, False, args.input_norm)
        offset = 0
        with torch.no_grad():
            for batch in loader:
                p, area, cond = model(batch["x"].to(device).float())
                p = p.cpu().numpy()
                area = area.cpu().numpy()
                cond = cond.cpu().numpy()
                for b in range(p.shape[0]):
                    src = sub.iloc[offset + b]
                    labels = label_row(src)
                    pred = p[b].copy()
                    pred[1:] = pred[1:] * stats.param_std[1:] + stats.param_mean[1:]
                    pred_present = float(1.0 / (1.0 + np.exp(-pred[0])))
                    pred_rows.append(
                        {
                            **labels,
                            "model": name,
                            "pred_patch_present_prob": pred_present,
                            "pred_center_x_norm": float(pred[1]),
                            "pred_center_y_norm": float(pred[2]),
                            "pred_log_area_fraction": float(pred[3]),
                            "pred_log_radius": float(pred[4]),
                            "pred_area_bin": int(np.argmax(area[b])),
                            "pred_condition_label": int(np.argmax(cond[b])),
                        }
                    )
                offset += p.shape[0]
    return pd.DataFrame(pred_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-norm", choices=["per_sample", "train_global"], default="per_sample")
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--consistency-weight", type=float, default=0.03)
    args = parser.parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    require_cuda(torch)
    rows = load_rows()
    stats = compute_stats(rows, args.input_norm)
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    first_batch = next(iter(make_loader(train_rows.head(1), stats, 1, False, args.input_norm)))
    in_channels = int(first_batch["x"].shape[1])
    device = torch.device("cuda")
    tabular_preds = run_tabular(rows)
    cnn = CNNParametric(in_channels, base=40).to(device)
    fno = FNOParametric(in_channels, width=args.width, modes=min(args.modes, 16)).to(device)
    cnn, cnn_history, cnn_ckpt = train_torch_model("cnn_encoder_parametric", cnn, rows, stats, args, device)
    fno, fno_history, fno_ckpt = train_torch_model("fno2d_encoder_parametric", fno, rows, stats, args, device)
    preds = pd.concat(
        [
            tabular_preds,
            eval_torch_model("cnn_encoder_parametric", cnn, rows, stats, args, device),
            eval_torch_model("fno2d_encoder_parametric", fno, rows, stats, args, device),
        ],
        ignore_index=True,
    )
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(PRED_CSV, index=False)
    summary = {
        "prediction_file": str(PRED_CSV),
        "models": ["tabular_rf_signal_descriptors", "cnn_encoder_parametric", "fno2d_encoder_parametric"],
        "cnn_checkpoint": str(cnn_ckpt),
        "fno_checkpoint": str(fno_ckpt),
        "cnn_history": cnn_history,
        "fno_history": fno_history,
        "input_channels": in_channels,
        "cuda": {"torch_version": torch.__version__, "torch_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)},
        "leakage_guardrail": "Inputs are direct combined magnetic tensors or descriptors computed from them only; patch parameters and target fields are labels.",
    }
    write_json(SUMMARY_JSON, summary)
    print(json.dumps({"predictions": str(PRED_CSV), "input_channels": in_channels, "cuda": summary["cuda"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
