                      
                                                          

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import pandas as pd

from v1e_500_common import CHECKPOINT_DIR, NEURAL_DIR, SEED, require_cuda, write_json
from v1e_deeponet_data_common import PARAM_KEYS, compute_stats, load_descriptors, load_index, make_loader, save_stats, torch


SPEC = importlib.util.spec_from_file_location("_deeponet_train", PROJECT_ROOT / "scripts" / "127_train_v1e_500_deeponet_field_parametric.py")
if SPEC is None or SPEC.loader is None:
    raise ImportError("Could not load DeepONet training helpers")
DEEPO_TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEEPO_TRAIN)


TRAIN_LOG = NEURAL_DIR / "descriptor_deeponet_training_log.csv"
TRAIN_EVAL = NEURAL_DIR / "descriptor_deeponet_training_eval.csv"
SUMMARY_JSON = NEURAL_DIR / "descriptor_deeponet_training_summary.json"
STATS_JSON = NEURAL_DIR / "v1e_500_descriptor_deeponet_stats.json"


class DescriptorDeepONetFieldParametric(torch.nn.Module):
    def __init__(self, desc_dim: int, *, basis_dim: int, branch_latent: int, trunk_hidden: int, dropout: float, fourier_trunk: bool) -> None:
        super().__init__()
        self.desc_encoder = torch.nn.Sequential(
            torch.nn.Linear(desc_dim, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, branch_latent),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(branch_latent, branch_latent),
            torch.nn.GELU(),
        )
        self.coeff_head = torch.nn.Linear(branch_latent, 4 * basis_dim)
        self.output_bias = torch.nn.Linear(branch_latent, 4)
        self.trunk = DEEPO_TRAIN.TrunkEncoder(basis_dim, trunk_hidden, dropout, fourier_trunk)
        self.param_head = torch.nn.Sequential(torch.nn.Linear(branch_latent, branch_latent), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(branch_latent, len(PARAM_KEYS)))
        self.area_head = torch.nn.Linear(branch_latent, 3)
        self.condition_head = torch.nn.Linear(branch_latent, 4)
        self.basis_dim = basis_dim

    def forward(self, branch_x, trunk_xy, descriptor_x=None):
        if descriptor_x is None or descriptor_x.shape[1] == 0:
            raise ValueError("descriptor-only DeepONet requires descriptor_x")
        branch_h = self.desc_encoder(descriptor_x)
        coeff = self.coeff_head(branch_h).reshape(descriptor_x.shape[0], 4, self.basis_dim)
        bias = self.output_bias(branch_h)[:, None, :]
        trunk = self.trunk(trunk_xy)
        query_out = torch.einsum("boc,bkc->bko", coeff, trunk) / math.sqrt(float(self.basis_dim))
        query_out = query_out + bias
        return query_out, self.param_head(branch_h), self.area_head(branch_h), self.condition_head(branch_h)


def train_descriptor_model(rows, desc, stats, args, device):
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
        seed=SEED + 211,
    )
    first = next(iter(make_loader(train_rows.head(1), stats, batch_size=1, shuffle=False, descriptors=desc)))
    desc_dim = int(first["descriptor_x"].shape[1])
    if desc_dim <= 0:
        raise SystemExit("Descriptor-only DeepONet cannot train because descriptor_dim is zero")
    model = DescriptorDeepONetFieldParametric(
        desc_dim,
        basis_dim=args.basis_dim,
        branch_latent=args.branch_latent,
        trunk_hidden=args.trunk_hidden,
        dropout=args.dropout,
        fourier_trunk=args.fourier_trunk,
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
    name = "descriptor_deeponet"
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
            loss = DEEPO_TRAIN.multitask_loss(*pred, batch, stats, weights)
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
                val_losses.append(float(DEEPO_TRAIN.multitask_loss(*pred, batch, stats, weights).detach().cpu()))
                prob = torch.sigmoid(pred[0][:, :, 3]).detach().cpu().numpy()
                dice_vals.append(DEEPO_TRAIN.dice_metric_np(batch["y_query"][:, :, 3].numpy(), prob, batch["valid_query_mask"].numpy().astype(bool))["dice"])
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
    ckpt = CHECKPOINT_DIR / "descriptor_deeponet_descriptor_deeponet.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": name,
            "config": {"descriptors": True, "fourier": args.fourier_trunk, "descriptor_only": True},
            "args": vars(args),
            "weights": weights,
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
            "leakage_guardrail": "Inputs are B_meas-derived descriptors and query coordinates only; target fields and patch params are labels.",
        },
        ckpt,
    )
    return model, history, ckpt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--k-queries", type=int, default=256)
    parser.add_argument("--basis-dim", type=int, default=64)
    parser.add_argument("--branch-latent", type=int, default=96)
    parser.add_argument("--trunk-hidden", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-norm", choices=["per_sample", "train_global"], default="per_sample")
    parser.add_argument("--fourier-trunk", action="store_true", default=True)
    args = parser.parse_args()
    DEEPO_TRAIN.seed_all()
    require_cuda(torch)
    rows = load_index()
    desc = load_descriptors()
    stats = compute_stats(rows, desc)
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    save_stats(STATS_JSON, stats)
    device = torch.device("cuda")
    model, history, ckpt = train_descriptor_model(rows, desc, stats, args, device)
    eval_df = DEEPO_TRAIN.evaluate_full_grid(model, "descriptor_deeponet", rows, desc, stats, args, device)
    pd.DataFrame(history).to_csv(TRAIN_LOG, index=False)
    eval_df.to_csv(TRAIN_EVAL, index=False)
    payload = {
        "architecture": "descriptor_only_deeponet",
        "checkpoint": str(ckpt),
        "training_log": str(TRAIN_LOG),
        "training_eval": str(TRAIN_EVAL),
        "cuda": {"torch_version": torch.__version__, "torch_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)},
        "leakage_guardrail": "Inputs are B_meas-derived descriptors and query coordinates only; target fields and patch params are labels.",
    }
    write_json(SUMMARY_JSON, payload)
    print(json.dumps({"training_log": str(TRAIN_LOG), "training_eval": str(TRAIN_EVAL), "checkpoint": str(ckpt)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
