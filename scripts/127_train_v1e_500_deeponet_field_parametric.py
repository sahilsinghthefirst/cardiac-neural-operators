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

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from v1e_500_common import CHECKPOINT_DIR, NEURAL_DIR, SEED, require_cuda, write_json
from v1e_deeponet_data_common import PARAM_KEYS, compute_stats, load_descriptors, load_index, make_loader, save_stats, torch


TRAIN_LOG = NEURAL_DIR / "deeponet_training_log.csv"
TRAIN_EVAL = NEURAL_DIR / "deeponet_training_eval.csv"
SUMMARY_JSON = NEURAL_DIR / "deeponet_training_summary.json"
STATS_JSON = NEURAL_DIR / "v1e_500_deeponet_stats.json"

DEEPNET_CONFIGS = {
    "deeponet_cnn_branch": {"descriptors": False, "fourier": False},
    "deeponet_cnn_branch_plus_descriptors": {"descriptors": True, "fourier": False},
    "deeponet_cnn_branch_fourier_trunk": {"descriptors": False, "fourier": True},
    "deeponet_cnn_branch_plus_descriptors_fourier_trunk": {"descriptors": True, "fourier": True},
}


def seed_all() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def finite_r2(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() < 2 or np.allclose(y[mask], y[mask][0]):
        return math.nan
    return float(r2_score(y[mask], p[mask]))


def regression_metrics(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if not np.any(mask):
        return {"mae": math.nan, "rmse": math.nan, "r2": math.nan}
    return {
        "mae": float(mean_absolute_error(y[mask], p[mask])),
        "rmse": float(np.sqrt(mean_squared_error(y[mask], p[mask]))),
        "r2": finite_r2(y[mask], p[mask]),
    }


def dice_loss_logits(logits, target, valid, eps: float = 1e-6):
    prob = torch.sigmoid(logits) * valid
    target = target * valid
    inter = torch.sum(prob * target, dim=1)
    denom = torch.sum(prob, dim=1) + torch.sum(target, dim=1)
    return torch.mean(1.0 - (2.0 * inter + eps) / (denom + eps))


def dice_metric_np(true: np.ndarray, prob: np.ndarray, valid: np.ndarray, threshold: float = 0.5) -> dict:
    t = (true > 0.5) & valid
    p = (prob >= threshold) & valid
    tp = float(np.sum(t & p))
    fp = float(np.sum(~t & p & valid))
    fn = float(np.sum(t & ~p))
    return {
        "dice": (2.0 * tp) / max(2.0 * tp + fp + fn, 1.0),
        "iou": tp / max(tp + fp + fn, 1.0),
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
    }


class CNNBranchEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 48, kernel_size=1),
            torch.nn.BatchNorm2d(48),
            torch.nn.GELU(),
            torch.nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.GELU(),
            torch.nn.Dropout2d(dropout),
            torch.nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(96),
            torch.nn.GELU(),
            torch.nn.Conv2d(96, latent_dim, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
        )

    def forward(self, branch_x):
        batch = branch_x.shape[0]
        x = branch_x.reshape(batch, branch_x.shape[1] * branch_x.shape[2], branch_x.shape[3], branch_x.shape[4])
        return self.net(x)


class TrunkEncoder(torch.nn.Module):
    def __init__(self, basis_dim: int, hidden: int, dropout: float, fourier: bool, fourier_bands: int = 4) -> None:
        super().__init__()
        self.fourier = fourier
        self.fourier_bands = fourier_bands
        in_dim = 2
        if fourier:
            in_dim += 4 * fourier_bands
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, basis_dim),
        )

    def _features(self, xy):
        if not self.fourier:
            return xy
        feats = [xy]
        for band in range(self.fourier_bands):
            freq = float(2**band) * math.pi
            feats.append(torch.sin(freq * xy))
            feats.append(torch.cos(freq * xy))
        return torch.cat(feats, dim=-1)

    def forward(self, xy):
        return self.net(self._features(xy))


class DeepONetFieldParametric(torch.nn.Module):
    def __init__(
        self,
        branch_in_channels: int,
        desc_dim: int,
        *,
        basis_dim: int,
        branch_latent: int,
        trunk_hidden: int,
        dropout: float,
        use_descriptors: bool,
        fourier_trunk: bool,
    ) -> None:
        super().__init__()
        self.use_descriptors = bool(use_descriptors and desc_dim > 0)
        self.branch = CNNBranchEncoder(branch_in_channels, branch_latent, dropout)
        if self.use_descriptors:
            self.desc_encoder = torch.nn.Sequential(
                torch.nn.Linear(desc_dim, 96),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(96, 64),
                torch.nn.GELU(),
            )
            latent_total = branch_latent + 64
        else:
            self.desc_encoder = None
            latent_total = branch_latent
        self.branch_project = torch.nn.Sequential(
            torch.nn.Linear(latent_total, branch_latent),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.coeff_head = torch.nn.Linear(branch_latent, 4 * basis_dim)
        self.output_bias = torch.nn.Linear(branch_latent, 4)
        self.trunk = TrunkEncoder(basis_dim, trunk_hidden, dropout, fourier_trunk)
        self.param_head = torch.nn.Sequential(torch.nn.Linear(branch_latent, branch_latent), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(branch_latent, len(PARAM_KEYS)))
        self.area_head = torch.nn.Linear(branch_latent, 3)
        self.condition_head = torch.nn.Linear(branch_latent, 4)
        self.basis_dim = basis_dim

    def encode_branch(self, branch_x, descriptor_x=None):
        h = self.branch(branch_x)
        if self.desc_encoder is not None and descriptor_x is not None and descriptor_x.shape[1] > 0:
            h = torch.cat([h, self.desc_encoder(descriptor_x)], dim=1)
        return self.branch_project(h)

    def forward(self, branch_x, trunk_xy, descriptor_x=None):
        branch_h = self.encode_branch(branch_x, descriptor_x)
        coeff = self.coeff_head(branch_h).reshape(branch_x.shape[0], 4, self.basis_dim)
        bias = self.output_bias(branch_h)[:, None, :]
        trunk = self.trunk(trunk_xy)
        query_out = torch.einsum("boc,bkc->bko", coeff, trunk) / math.sqrt(float(self.basis_dim))
        query_out = query_out + bias
        return query_out, self.param_head(branch_h), self.area_head(branch_h), self.condition_head(branch_h)


def multitask_loss(query_pred, param_pred, area_logits, cond_logits, batch, stats, weights):
    valid = batch["valid_query_mask"].to(query_pred.device).float()
    y = batch["y_query"].to(query_pred.device).float()
    loss = torch.tensor(0.0, device=query_pred.device)
    for i in range(3):
        diff = torch.nn.functional.smooth_l1_loss(query_pred[:, :, i], y[:, :, i], reduction="none", beta=0.75)
        loss = loss + torch.sum(diff * valid) / (torch.sum(valid) + 1e-6)
    loss = weights["field"] * loss / 3.0
    patch_target = y[:, :, 3]
    present = batch["y_param"].to(query_pred.device).float()[:, 0]
    neg_weight = torch.where(present[:, None] < 0.5, torch.full_like(valid, 2.0), torch.ones_like(valid))
    bce = torch.nn.functional.binary_cross_entropy_with_logits(query_pred[:, :, 3], patch_target, reduction="none")
    patch_loss = torch.sum(bce * valid * neg_weight) / (torch.sum(valid * neg_weight) + 1e-6)
    patch_loss = patch_loss + dice_loss_logits(query_pred[:, :, 3], patch_target, valid)
    loss = loss + weights["patch"] * patch_loss

    y_param = batch["y_param"].to(query_pred.device).float()
    p_mask = batch["param_mask"].to(query_pred.device).bool()
    present_loss = torch.nn.functional.binary_cross_entropy_with_logits(param_pred[:, 0], y_param[:, 0], reduction="none")
    loss = loss + weights["present"] * torch.mean(present_loss * torch.where(y_param[:, 0] < 0.5, 2.0, 1.0))
    non = p_mask[:, 1:].any(dim=1)
    if torch.any(non):
        reg = torch.nn.functional.smooth_l1_loss(param_pred[non, 1:], y_param[non, 1:], reduction="none", beta=0.5)
        loss = loss + weights["param"] * torch.mean(reg)
        loss = loss + weights["area"] * torch.nn.functional.cross_entropy(area_logits[non], batch["area_bin"].to(query_pred.device)[non])
        if weights["consistency"] > 0:
            mean = torch.as_tensor(stats.param_mean, dtype=torch.float32, device=query_pred.device)
            std = torch.as_tensor(stats.param_std, dtype=torch.float32, device=query_pred.device)
            unscaled = param_pred * std[None, :] + mean[None, :]
            consistency = unscaled[non, 3] - (math.log(math.pi) + 2.0 * unscaled[non, 4])
            loss = loss + weights["consistency"] * torch.mean(consistency * consistency)
    loss = loss + weights["condition"] * torch.nn.functional.cross_entropy(cond_logits, batch["condition_label"].to(query_pred.device))

    if weights["scalar"] > 0:
        apd_pred = query_pred[:, :, 2]
        apd_true = y[:, :, 2]
        patch = y[:, :, 3] > 0.5
        inside = patch & (valid > 0.5)
        outside = (~patch) & (valid > 0.5)
        pred_deltas = []
        true_deltas = []
        for idx in range(query_pred.shape[0]):
            if torch.any(inside[idx]) and torch.any(outside[idx]):
                pred_deltas.append(torch.mean(apd_pred[idx][inside[idx]]) - torch.mean(apd_pred[idx][outside[idx]]))
                true_deltas.append(torch.mean(apd_true[idx][inside[idx]]) - torch.mean(apd_true[idx][outside[idx]]))
        if pred_deltas:
            loss = loss + weights["scalar"] * torch.nn.functional.smooth_l1_loss(torch.stack(pred_deltas), torch.stack(true_deltas), beta=0.5)
    return loss


def train_one_config(name, config, rows, desc, stats, args, device):
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    val_rows = rows[rows["split"].eq("val")].reset_index(drop=True)
    train_loader = make_loader(
        train_rows,
        stats,
        batch_size=args.batch_size,
        shuffle=True,
        mode="random",
        k_queries=args.k_queries,
        input_norm=args.input_norm,
        descriptors=desc,
        seed=SEED,
    )
    val_loader = make_loader(
        val_rows,
        stats,
        batch_size=args.batch_size,
        shuffle=False,
        mode="random",
        k_queries=args.k_queries,
        input_norm=args.input_norm,
        descriptors=desc,
        seed=SEED + 101,
    )
    first = next(iter(make_loader(train_rows.head(1), stats, batch_size=1, shuffle=False, mode="random", descriptors=desc)))
    branch_in = int(first["branch_x"].shape[1] * first["branch_x"].shape[2])
    desc_dim = int(first["descriptor_x"].shape[1])
    model = DeepONetFieldParametric(
        branch_in,
        desc_dim,
        basis_dim=args.basis_dim,
        branch_latent=args.branch_latent,
        trunk_hidden=args.trunk_hidden,
        dropout=args.dropout,
        use_descriptors=config["descriptors"],
        fourier_trunk=config["fourier"],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = {
        "field": 1.0,
        "patch": 1.0,
        "present": 0.45,
        "param": 0.7,
        "area": 0.35,
        "condition": 0.08,
        "consistency": 0.02,
        "scalar": 0.04,
    }
    best_score = float("inf")
    best_state = None
    bad = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(
                batch["branch_x"].to(device).float(),
                batch["trunk_xy"].to(device).float(),
                batch["descriptor_x"].to(device).float(),
            )
            loss = multitask_loss(*pred, batch, stats, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        dice_vals = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(
                    batch["branch_x"].to(device).float(),
                    batch["trunk_xy"].to(device).float(),
                    batch["descriptor_x"].to(device).float(),
                )
                loss = multitask_loss(*pred, batch, stats, weights)
                val_losses.append(float(loss.detach().cpu()))
                prob = torch.sigmoid(pred[0][:, :, 3]).detach().cpu().numpy()
                true = batch["y_query"][:, :, 3].numpy()
                valid = batch["valid_query_mask"].numpy().astype(bool)
                dice_vals.append(dice_metric_np(true, prob, valid)["dice"])
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_dice = float(np.mean(dice_vals)) if dice_vals else 0.0
        val_score = val_loss + 0.2 * (1.0 - val_dice)
        row = {"model": name, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_patch_dice_sampled": val_dice, "val_score": val_score}
        history.append(row)
        print(f"{name} epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} dice={val_dice:.3f} score={val_score:.4f}", flush=True)
        if val_score < best_score - 1e-5:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINT_DIR / f"deeponet_{name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": name,
            "config": config,
            "args": vars(args),
            "weights": weights,
            "branch_in_channels": branch_in,
            "descriptor_dim": desc_dim,
            "basis_dim": args.basis_dim,
            "branch_latent": args.branch_latent,
            "trunk_hidden": args.trunk_hidden,
            "dropout": args.dropout,
            "stats": {
                "input_mean": stats.input_mean,
                "input_std": stats.input_std,
                "field_mean": stats.field_mean,
                "field_std": stats.field_std,
                "param_mean": stats.param_mean,
                "param_std": stats.param_std,
                "descriptor_mean": stats.descriptor_mean,
                "descriptor_std": stats.descriptor_std,
                "descriptor_columns": stats.descriptor_columns,
            },
            "history": history,
            "leakage_guardrail": "Inputs are B_meas, optional B_meas-derived descriptors, and query coordinates only; target fields and patch params are labels.",
        },
        ckpt,
    )
    return model, history, ckpt


def evaluate_full_grid(model, name, rows, desc, stats, args, device):
    out = []
    model.eval()
    for split in ["train", "val", "test"]:
        sub = rows[rows["split"].eq(split)].reset_index(drop=True)
        loader = make_loader(
            sub,
            stats,
            batch_size=args.eval_batch_size,
            shuffle=False,
            mode="full",
            k_queries=1024,
            input_norm=args.input_norm,
            descriptors=desc,
            seed=SEED + 303,
        )
        field_true = {k: [] for k in ["LAT", "APD80", "APD90"]}
        field_pred = {k: [] for k in ["LAT", "APD80", "APD90"]}
        patch_true = []
        patch_prob = []
        valid_all = []
        y_present = []
        p_present = []
        y_condition = []
        p_condition = []
        with torch.no_grad():
            for batch in loader:
                query, params, area_logits, cond_logits = model(
                    batch["branch_x"].to(device).float(),
                    batch["trunk_xy"].to(device).float(),
                    batch["descriptor_x"].to(device).float(),
                )
                query_np = query.detach().cpu().numpy()
                y_np = batch["y_query"].numpy()
                valid = batch["valid_query_mask"].numpy().astype(bool)
                for i, key in enumerate(["LAT", "APD80", "APD90"]):
                    field_true[key].append(y_np[:, :, i] * stats.field_std[i] + stats.field_mean[i])
                    field_pred[key].append(query_np[:, :, i] * stats.field_std[i] + stats.field_mean[i])
                patch_true.append(y_np[:, :, 3])
                patch_prob.append(1.0 / (1.0 + np.exp(-query_np[:, :, 3])))
                valid_all.append(valid)
                y_present.extend(batch["y_param"][:, 0].numpy().tolist())
                p_present.extend((1.0 / (1.0 + np.exp(-params.detach().cpu().numpy()[:, 0]))).tolist())
                y_condition.extend(batch["condition_label"].numpy().tolist())
                p_condition.extend(np.argmax(cond_logits.detach().cpu().numpy(), axis=1).tolist())
        valid_cat = np.concatenate(valid_all, axis=0).astype(bool)
        patch_true_cat = np.concatenate(patch_true, axis=0)
        patch_prob_cat = np.concatenate(patch_prob, axis=0)
        out.append({"model": name, "split": split, "target": "patch_mask", "metric_scope": "full_grid_training_eval", **dice_metric_np(patch_true_cat, patch_prob_cat, valid_cat)})
        for key in ["LAT", "APD80", "APD90"]:
            true = np.concatenate(field_true[key], axis=0)[valid_cat]
            pred = np.concatenate(field_pred[key], axis=0)[valid_cat]
            out.append({"model": name, "split": split, "target": key, "metric_scope": "full_grid_training_eval", **regression_metrics(true, pred)})
        present_bin = (np.asarray(p_present) >= 0.5).astype(int)
        present_row = {
            "model": name,
            "split": split,
            "target": "patch_present",
            "metric_scope": "full_grid_training_eval",
            "accuracy": float(accuracy_score(y_present, present_bin)),
            "balanced_accuracy": float(balanced_accuracy_score(y_present, present_bin)),
            "macro_f1": float(f1_score(y_present, present_bin, average="macro")),
        }
        try:
            present_row["auroc"] = float(roc_auc_score(y_present, p_present)) if len(np.unique(y_present)) > 1 else math.nan
        except Exception:
            present_row["auroc"] = math.nan
        out.append(present_row)
        out.append(
            {
                "model": name,
                "split": split,
                "target": "condition",
                "metric_scope": "full_grid_training_eval",
                "accuracy": float(accuracy_score(y_condition, p_condition)),
                "balanced_accuracy": float(balanced_accuracy_score(y_condition, p_condition)),
                "macro_f1": float(f1_score(y_condition, p_condition, average="macro")),
            }
        )
    return pd.DataFrame(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", choices=list(DEEPNET_CONFIGS), default=list(DEEPNET_CONFIGS))
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--k-queries", type=int, default=256)
    parser.add_argument("--basis-dim", type=int, default=64)
    parser.add_argument("--branch-latent", type=int, default=96)
    parser.add_argument("--trunk-hidden", type=int, default=96)
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
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    save_stats(STATS_JSON, stats)
    device = torch.device("cuda")
    all_history = []
    all_eval = []
    checkpoints = {}
    for name in args.configs:
        model, history, ckpt = train_one_config(name, DEEPNET_CONFIGS[name], rows, desc, stats, args, device)
        all_history.extend(history)
        eval_df = evaluate_full_grid(model, name, rows, desc, stats, args, device)
        all_eval.append(eval_df)
        checkpoints[name] = str(ckpt)
    hist_df = pd.DataFrame(all_history)
    eval_df = pd.concat(all_eval, ignore_index=True)
    hist_df.to_csv(TRAIN_LOG, index=False)
    eval_df.to_csv(TRAIN_EVAL, index=False)
    payload = {
        "architecture": "plain_deeponet_cnn_branch_trunk_coordinate",
        "configs": {k: DEEPNET_CONFIGS[k] for k in args.configs},
        "checkpoints": checkpoints,
        "training_log": str(TRAIN_LOG),
        "training_eval": str(TRAIN_EVAL),
        "cuda": {"torch_version": torch.__version__, "torch_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)},
        "leakage_guardrail": "Inputs are B_meas, optional B_meas-derived descriptors, and query coordinates only; target fields and patch params are labels.",
    }
    write_json(SUMMARY_JSON, payload)
    print(json.dumps({"training_log": str(TRAIN_LOG), "training_eval": str(TRAIN_EVAL), "checkpoints": checkpoints}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
