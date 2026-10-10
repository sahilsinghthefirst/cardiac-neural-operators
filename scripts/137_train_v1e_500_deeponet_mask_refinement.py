                      
                                                    

from __future__ import annotations

import argparse
import importlib.util
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

from v1e_500_common import CHECKPOINT_DIR, NEURAL_DIR, SEED, require_cuda, write_json
from v1e_deeponet_data_common import PARAM_KEYS, compute_stats, load_descriptors, save_stats, torch
from v1e_deeponet_mask_refine_data_common import make_mask_refine_loader, merge_rows_with_refine_targets


TRAIN_SPEC = importlib.util.spec_from_file_location("_deeponet_train", PROJECT_ROOT / "scripts" / "127_train_v1e_500_deeponet_field_parametric.py")
if TRAIN_SPEC is None or TRAIN_SPEC.loader is None:
    raise ImportError("Could not load DeepONet training module")
DEEPO_TRAIN = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(DEEPO_TRAIN)


TRAIN_LOG = NEURAL_DIR / "mask_refine_training_log.csv"
TRAIN_SUMMARY = NEURAL_DIR / "mask_refine_training_summary.json"
STATS_JSON = NEURAL_DIR / "v1e_500_mask_refine_stats.json"


CONFIGS = {
    "deeponet_cnn_branch_fourier_trunk_mask_refine": {
        "descriptors": False,
        "fourier": True,
        "center_weight": 1.4,
        "patch_weight": 1.3,
        "distance_weight": 0.30,
        "field_weight": 0.12,
        "condition_weighted": False,
        "init_checkpoint": "deeponet_deeponet_cnn_branch_fourier_trunk.pt",
    },
    "deeponet_cnn_branch_plus_descriptors_fourier_trunk_mask_refine": {
        "descriptors": True,
        "fourier": True,
        "center_weight": 1.4,
        "patch_weight": 1.3,
        "distance_weight": 0.30,
        "field_weight": 0.12,
        "condition_weighted": False,
        "init_checkpoint": "deeponet_deeponet_cnn_branch_plus_descriptors_fourier_trunk.pt",
    },
    "deeponet_cnn_branch_plus_descriptors_center_heavy": {
        "descriptors": True,
        "fourier": True,
        "center_weight": 2.2,
        "patch_weight": 1.4,
        "distance_weight": 0.45,
        "field_weight": 0.08,
        "condition_weighted": False,
        "init_checkpoint": "deeponet_deeponet_cnn_branch_plus_descriptors_fourier_trunk.pt",
    },
    "deeponet_cnn_branch_plus_descriptors_conduction_weighted": {
        "descriptors": True,
        "fourier": True,
        "center_weight": 1.8,
        "patch_weight": 1.7,
        "distance_weight": 0.40,
        "field_weight": 0.08,
        "condition_weighted": True,
        "init_checkpoint": "deeponet_deeponet_cnn_branch_plus_descriptors_fourier_trunk.pt",
    },
}


def seed_all() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def regression_r2(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() < 2 or np.allclose(y[mask], y[mask][0]):
        return math.nan
    ss_res = float(np.sum((y[mask] - p[mask]) ** 2))
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan


def patch_metrics_np(true, pred, valid) -> dict[str, float]:
    t = (true > 0.5) & valid.astype(bool)
    p = (pred > 0.5) & valid.astype(bool)
    tp = float(np.sum(t & p))
    fp = float(np.sum(~t & p & valid))
    fn = float(np.sum(t & ~p))
    return {
        "dice": (2.0 * tp) / max(2.0 * tp + fp + fn, 1.0),
        "iou": tp / max(tp + fp + fn, 1.0),
        "precision": tp / max(tp + fp, 1.0),
        "recall": tp / max(tp + fn, 1.0),
    }


def scalar_delta(apd90: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> float:
    v = valid.astype(bool) & np.isfinite(apd90)
    inside = (mask > 0.5) & v
    outside = (mask <= 0.5) & v
    return float(np.mean(apd90[inside]) - np.mean(apd90[outside])) if inside.any() and outside.any() else math.nan


class MaskRefineDeepONet(torch.nn.Module):
                                                                                     

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
        self.branch = DEEPO_TRAIN.CNNBranchEncoder(branch_in_channels, branch_latent, dropout)
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
        self.coeff_head = torch.nn.Linear(branch_latent, 5 * basis_dim)
        self.output_bias = torch.nn.Linear(branch_latent, 5)
        self.trunk = DEEPO_TRAIN.TrunkEncoder(basis_dim, trunk_hidden, dropout, fourier_trunk)
        self.param_head = torch.nn.Sequential(torch.nn.Linear(branch_latent, branch_latent), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(branch_latent, len(PARAM_KEYS)))
        self.area_head = torch.nn.Linear(branch_latent, 3)
        self.basis_dim = basis_dim

    def encode_branch(self, branch_x, descriptor_x=None):
        h = self.branch(branch_x)
        if self.desc_encoder is not None and descriptor_x is not None and descriptor_x.shape[1] > 0:
            h = torch.cat([h, self.desc_encoder(descriptor_x)], dim=1)
        return self.branch_project(h)

    def forward(self, branch_x, trunk_xy, descriptor_x=None):
        branch_h = self.encode_branch(branch_x, descriptor_x)
        coeff = self.coeff_head(branch_h).reshape(branch_x.shape[0], 5, self.basis_dim)
        bias = self.output_bias(branch_h)[:, None, :]
        trunk = self.trunk(trunk_xy)
        query_out = torch.einsum("boc,bkc->bko", coeff, trunk) / math.sqrt(float(self.basis_dim))
        query_out = query_out + bias
        return query_out, self.param_head(branch_h), self.area_head(branch_h)


def copy_query_channel(new_state, old_state, new_channel: int, old_channel: int, basis_dim: int) -> None:
    old_start = old_channel * basis_dim
    new_start = new_channel * basis_dim
    new_state["coeff_head.weight"][new_start : new_start + basis_dim] = old_state["coeff_head.weight"][old_start : old_start + basis_dim]
    new_state["coeff_head.bias"][new_start : new_start + basis_dim] = old_state["coeff_head.bias"][old_start : old_start + basis_dim]
    new_state["output_bias.weight"][new_channel] = old_state["output_bias.weight"][old_channel]
    new_state["output_bias.bias"][new_channel] = old_state["output_bias.bias"][old_channel]


def initialize_from_plain_deeponet(model: MaskRefineDeepONet, config: dict, basis_dim: int) -> str | None:
    ckpt_path = CHECKPOINT_DIR / str(config["init_checkpoint"])
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    old = ckpt["model_state_dict"]
    new = model.state_dict()
    copied = []
    for key, value in old.items():
        if key in new and tuple(new[key].shape) == tuple(value.shape):
            new[key] = value
            copied.append(key)
    for new_channel, old_channel in [(0, 0), (1, 2), (2, 3), (3, 3)]:
        copy_query_channel(new, old, new_channel, old_channel, basis_dim)
    new["coeff_head.weight"][4 * basis_dim : 5 * basis_dim].zero_()
    new["coeff_head.bias"][4 * basis_dim : 5 * basis_dim].zero_()
    new["output_bias.weight"][4].zero_()
    new["output_bias.bias"][4].zero_()
    model.load_state_dict(new)
    return str(ckpt_path)


def dice_loss_logits(logits, target, valid, weights=None, eps: float = 1e-6):
    prob = torch.sigmoid(logits) * valid
    target = target * valid
    if weights is not None:
        prob = prob * weights
        target = target * weights
    inter = torch.sum(prob * target, dim=1)
    denom = torch.sum(prob, dim=1) + torch.sum(target, dim=1)
    return torch.mean(1.0 - (2.0 * inter + eps) / (denom + eps))


def weighted_mean(loss, weight):
    return torch.sum(loss * weight) / (torch.sum(weight) + 1e-6)


def condition_sample_weight(batch, config, device):
    labels = batch["condition_label"].to(device)
    w = torch.ones(labels.shape[0], dtype=torch.float32, device=device)
    if config.get("condition_weighted", False):
        w = torch.where(labels == 2, torch.full_like(w, 2.0), w)
        w = torch.where(labels == 3, torch.full_like(w, 1.25), w)
    return w[:, None]


def multitask_loss(query_out, param_pred, area_logits, batch, stats, config):
    device = query_out.device
    valid = batch["valid_query_mask"].to(device).float()
    sample_w = condition_sample_weight(batch, config, device)
    present = batch["y_param"].to(device).float()[:, 0]
    valid_w = valid * sample_w

    y_field = batch["y_field"].to(device).float()
    field_loss = torch.nn.functional.smooth_l1_loss(query_out[:, :, :2], y_field, reduction="none", beta=0.75).mean(dim=2)
    loss = float(config["field_weight"]) * weighted_mean(field_loss, valid_w)

    patch_target = batch["patch_query"].to(device).float()
    boundary = batch["boundary_weight_query"].to(device).float()
    neg_pressure = torch.where(present[:, None] < 0.5, torch.full_like(valid, 2.0), torch.ones_like(valid))
    patch_weight = valid_w * boundary * neg_pressure
    patch_bce = torch.nn.functional.binary_cross_entropy_with_logits(query_out[:, :, 2], patch_target, reduction="none")
    patch_loss = weighted_mean(patch_bce, patch_weight) + dice_loss_logits(query_out[:, :, 2], patch_target, valid, sample_w)
    loss = loss + float(config["patch_weight"]) * patch_loss

    center_target = batch["center_heatmap_query"].to(device).float()
    center_prob = torch.sigmoid(query_out[:, :, 3])
    center_reg = torch.nn.functional.smooth_l1_loss(center_prob, center_target, reduction="none", beta=0.15)
    center_bce = torch.nn.functional.binary_cross_entropy_with_logits(query_out[:, :, 3], center_target, reduction="none")
    center_weight = valid_w * torch.where(present[:, None] < 0.5, torch.full_like(valid, 1.5), 1.0 + 4.0 * center_target)
    center_loss = weighted_mean(center_reg + 0.25 * center_bce, center_weight)
    loss = loss + float(config["center_weight"]) * center_loss

    distance_target = batch["distance_query"].to(device).float()
    dist_loss = torch.nn.functional.smooth_l1_loss(query_out[:, :, 4], distance_target, reduction="none", beta=0.05)
    loss = loss + float(config["distance_weight"]) * weighted_mean(dist_loss, valid_w * (0.5 + 0.5 * boundary))

    y_param = batch["y_param"].to(device).float()
    p_mask = batch["param_mask"].to(device).bool()
    present_loss = torch.nn.functional.binary_cross_entropy_with_logits(param_pred[:, 0], y_param[:, 0], reduction="none")
    loss = loss + 0.4 * torch.mean(present_loss * torch.where(y_param[:, 0] < 0.5, 2.0, 1.0))
    non = p_mask[:, 1:].any(dim=1)
    if torch.any(non):
        non_w = condition_sample_weight(batch, config, device).squeeze(1)[non]
        center_reg = torch.nn.functional.smooth_l1_loss(param_pred[non, 1:3], y_param[non, 1:3], reduction="none", beta=0.35).mean(dim=1)
        size_reg = torch.nn.functional.smooth_l1_loss(param_pred[non, 3:5], y_param[non, 3:5], reduction="none", beta=0.4).mean(dim=1)
        loss = loss + 1.2 * torch.mean(center_reg * non_w)
        loss = loss + 0.35 * torch.mean(size_reg * non_w)
        loss = loss + 0.25 * torch.nn.functional.cross_entropy(area_logits[non], batch["area_bin"].to(device)[non], reduction="mean")
    return loss


def evaluate_full_split(model, rows, desc, stats, args, device, split: str) -> dict:
    sub = rows[rows["split"].eq(split)].reset_index(drop=True)
    loader = make_mask_refine_loader(sub, stats, batch_size=args.eval_batch_size, shuffle=False, mode="full", k_queries=1024, input_norm=args.input_norm, descriptors=desc, seed=SEED + 303)
    patch_true = []
    patch_pred = []
    valid_all = []
    true_delta = []
    pred_delta = []
    hfp = []
    center_err = []
    conduction_true = []
    conduction_pred = []
    conduction_valid = []
    with torch.no_grad():
        offset = 0
        for batch in loader:
            query_out, params, _area = model(
                batch["branch_x"].to(device).float(),
                batch["trunk_xy"].to(device).float(),
                batch["descriptor_x"].to(device).float(),
            )
            pred = query_out.detach().cpu().numpy()
            params_np = params.detach().cpu().numpy()
            valid_np = batch["valid_query_mask"].numpy().astype(bool)
            patch_np = batch["patch_query"].numpy()
            prob = sigmoid_np(pred[:, :, 2])
            apd90_pred = pred[:, :, 1] * stats.field_std[2] + stats.field_mean[2]
            patch_true.append(patch_np)
            patch_pred.append(prob)
            valid_all.append(valid_np)
            y_param = batch["y_param"].numpy()
            p_mask = batch["param_mask"].numpy().astype(bool)
            param_unscaled = params_np.copy()
            param_unscaled[:, 1:] = param_unscaled[:, 1:] * stats.param_std[1:] + stats.param_mean[1:]
            for b in range(pred.shape[0]):
                row = sub.iloc[offset + b]
                mask = (prob[b].reshape(32, 32) >= 0.5).astype(np.float32)
                valid = valid_np[b].reshape(32, 32)
                true_patch = patch_np[b].reshape(32, 32)
                pred_apd = apd90_pred[b].reshape(32, 32)
                true_delta.append(float(batch["apd90_patch_delta"].numpy()[b]))
                pred_delta.append(scalar_delta(pred_apd, mask, valid))
                if float(y_param[b, 0]) < 0.5:
                    hfp.append(float(np.mean(mask[valid > 0])))
                if p_mask[b, 1] and p_mask[b, 2]:
                    tx = y_param[b, 1] * stats.param_std[1] + stats.param_mean[1]
                    ty = y_param[b, 2] * stats.param_std[2] + stats.param_mean[2]
                    center_err.append(float(math.sqrt((param_unscaled[b, 1] - tx) ** 2 + (param_unscaled[b, 2] - ty) ** 2)))
                if row["condition"] == "conduction_slowing_perturbation":
                    conduction_true.append(true_patch)
                    conduction_pred.append(prob[b].reshape(32, 32))
                    conduction_valid.append(valid)
            offset += pred.shape[0]
    valid_cat = np.concatenate(valid_all, axis=0).astype(bool)
    true_cat = np.concatenate(patch_true, axis=0)
    pred_cat = np.concatenate(patch_pred, axis=0)
    pm = patch_metrics_np(true_cat, pred_cat, valid_cat)
    cond_pm = patch_metrics_np(np.stack(conduction_true), np.stack(conduction_pred), np.stack(conduction_valid).astype(bool)) if conduction_true else {"dice": math.nan}
    present_mask = np.isfinite(true_delta) & np.isfinite(pred_delta)
    return {
        "split": split,
        "patch_dice": pm["dice"],
        "patch_iou": pm["iou"],
        "patch_precision": pm["precision"],
        "patch_recall": pm["recall"],
        "center_mae": float(np.mean(center_err)) if center_err else math.nan,
        "APD90_delta_R2_pred_mask_pred_apd": regression_r2(np.asarray(true_delta)[present_mask], np.asarray(pred_delta)[present_mask]) if np.any(present_mask) else math.nan,
        "homogeneous_false_positive_area": float(np.mean(hfp)) if hfp else 0.0,
        "conduction_dice": cond_pm.get("dice", math.nan),
    }


def train_one(name, config, rows, desc, stats, args, device):
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    train_loader = make_mask_refine_loader(train_rows, stats, batch_size=args.batch_size, shuffle=True, mode="random", k_queries=args.k_queries, input_norm=args.input_norm, descriptors=desc, seed=SEED)
    first = next(iter(make_mask_refine_loader(train_rows.head(1), stats, batch_size=1, shuffle=False, mode="random", descriptors=desc)))
    branch_in = int(first["branch_x"].shape[1] * first["branch_x"].shape[2])
    desc_dim = int(first["descriptor_x"].shape[1])
    model = MaskRefineDeepONet(
        branch_in,
        desc_dim,
        basis_dim=args.basis_dim,
        branch_latent=args.branch_latent,
        trunk_hidden=args.trunk_hidden,
        dropout=args.dropout,
        use_descriptors=config["descriptors"],
        fourier_trunk=config["fourier"],
    )
    initialized_from = initialize_from_plain_deeponet(model, config, args.basis_dim)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = float("inf")
    best_state = None
    bad = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            out = model(
                batch["branch_x"].to(device).float(),
                batch["trunk_xy"].to(device).float(),
                batch["descriptor_x"].to(device).float(),
            )
            loss = multitask_loss(*out, batch, stats, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val = evaluate_full_split(model, rows, desc, stats, args, device, "val")
        delta = val["APD90_delta_R2_pred_mask_pred_apd"]
        delta_term = delta if np.isfinite(delta) else -0.5
        score = (
            (val["center_mae"] if np.isfinite(val["center_mae"]) else 1.0)
            + 0.45 * (1.0 - val["patch_dice"])
            - 0.18 * delta_term
            + 0.8 * val["homogeneous_false_positive_area"]
            + 0.15 * (1.0 - (val["conduction_dice"] if np.isfinite(val["conduction_dice"]) else 0.0))
        )
        rec = {
            "model": name,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_score": score,
            **{f"val_{k}": v for k, v in val.items() if k != "split"},
        }
        history.append(rec)
        print(
            f"{name} epoch {epoch}: loss={rec['train_loss']:.4f} val_score={score:.4f} "
            f"dice={val['patch_dice']:.3f} center={val['center_mae']:.3f} delta={delta_term:.3f} cond={val['conduction_dice']:.3f}",
            flush=True,
        )
        if score < best_score - 1e-5:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINT_DIR / f"deeponet_mask_refine_{name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": name,
            "config": config,
            "args": vars(args),
            "initialized_from": initialized_from,
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
            "leakage_guardrail": "Inputs are B_meas, optional B_meas-derived descriptors, and query coordinates only; target maps/params are labels.",
            "condition_weighting_note": "Condition labels are used only for supervised loss weighting when enabled; they are not model inputs.",
        },
        ckpt,
    )
    return history, ckpt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--k-queries", type=int, default=384)
    parser.add_argument("--basis-dim", type=int, default=64)
    parser.add_argument("--branch-latent", type=int, default=96)
    parser.add_argument("--trunk-hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-norm", choices=["per_sample", "train_global"], default="per_sample")
    args = parser.parse_args()
    seed_all()
    require_cuda(torch)
    rows = merge_rows_with_refine_targets()
    desc = load_descriptors()
    stats = compute_stats(rows, desc)
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    save_stats(STATS_JSON, stats)
    device = torch.device("cuda")
    all_history = []
    checkpoints = {}
    for name in args.configs:
        history, ckpt = train_one(name, CONFIGS[name], rows, desc, stats, args, device)
        all_history.extend(history)
        checkpoints[name] = str(ckpt)
    hist = pd.DataFrame(all_history)
    hist.to_csv(TRAIN_LOG, index=False)
    payload = {
        "training_log": str(TRAIN_LOG),
        "checkpoints": checkpoints,
        "configs": {k: CONFIGS[k] for k in args.configs},
        "stats": str(STATS_JSON),
        "cuda": {"torch_version": torch.__version__, "torch_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)},
        "leakage_guardrail": "Inputs are B_meas, optional B_meas-derived descriptors, and query coordinates only; center/distance/mask/field/param targets are labels.",
        "condition_weighting_note": "Condition labels are used only for supervised loss weighting in the conduction_weighted config and reporting, never as model inputs.",
    }
    write_json(TRAIN_SUMMARY, payload)
    print(json.dumps({"training_log": str(TRAIN_LOG), "checkpoints": checkpoints}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
