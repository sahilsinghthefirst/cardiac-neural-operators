                      
                                                                                           

from __future__ import annotations

import importlib.util
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v1e_500_common import CHECKPOINT_DIR, NEURAL_DIR, ROOT, SEED, require_cuda, write_json
from v1e_deeponet_data_common import compute_stats, load_descriptors, torch
from v1e_deeponet_mask_refine_data_common import make_mask_refine_loader, merge_rows_with_refine_targets
from v1f_metric_realignment_common import (
    V1F_FIG_DIR,
    V1F_OUTPUT_DIR,
    anomaly_delta,
    compute_physiology,
    ensure_dirs,
    load_index,
    patch_delta,
    regression_summary,
)

TRAIN_SPEC = importlib.util.spec_from_file_location("_mask_refine_train", PROJECT_ROOT / "scripts" / "137_train_v1e_500_deeponet_mask_refinement.py")
if TRAIN_SPEC is None or TRAIN_SPEC.loader is None:
    raise ImportError("Could not load mask-refinement training module")
MASK_TRAIN = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(MASK_TRAIN)

OUT_CSV = V1F_OUTPUT_DIR / "oracle_apd_lat_decomposition.csv"
ROW_CSV = V1F_OUTPUT_DIR / "oracle_apd_lat_decomposition_rows.csv"
SUMMARY_JSON = V1F_OUTPUT_DIR / "oracle_apd_lat_decomposition_summary.json"


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    pos = x >= 0
    out = np.empty_like(x, dtype=np.float64)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out.astype(np.float32)


def build_model(ckpt: dict):
    args = ckpt.get("args", {})
    config = ckpt.get("config", {})
    model = MASK_TRAIN.MaskRefineDeepONet(
        int(ckpt["branch_in_channels"]),
        int(ckpt["descriptor_dim"]),
        basis_dim=int(ckpt.get("basis_dim", args.get("basis_dim", 64))),
        branch_latent=int(ckpt.get("branch_latent", args.get("branch_latent", 96))),
        trunk_hidden=int(ckpt.get("trunk_hidden", args.get("trunk_hidden", 96))),
        dropout=float(ckpt.get("dropout", args.get("dropout", 0.08))),
        use_descriptors=bool(config.get("descriptors", False)),
        fourier_trunk=bool(config.get("fourier", True)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def unscale_params(params_np: np.ndarray, stats) -> np.ndarray:
    out = params_np.copy()
    out[:, 1:] = out[:, 1:] * stats.param_std[1:] + stats.param_mean[1:]
    return out


def center_constrained_mask(prob: np.ndarray, row: dict, threshold: float = 0.65, multiplier: float = 1.2) -> np.ndarray:
    h, w = prob.shape
    if row["pred_patch_present_prob"] < 0.5:
        return np.zeros_like(prob, dtype=np.float32)
    cx = float(row["pred_center_x_norm"])
    cy = float(row["pred_center_y_norm"])
    radius = row.get("pred_radius_param", math.nan)
    area = row.get("pred_patch_area_param", math.nan)
    area_radius = math.sqrt(max(float(area), 0.0) / math.pi) if np.isfinite(area) else math.nan
    if np.isfinite(radius) and np.isfinite(area_radius):
        r = 0.5 * (float(radius) + area_radius)
    elif np.isfinite(radius):
        r = float(radius)
    else:
        r = area_radius
    if not np.isfinite(r) or not np.isfinite(cx) or not np.isfinite(cy):
        return (prob >= threshold).astype(np.float32)
    yy, xx = np.indices(prob.shape)
    x = xx / max(w - 1, 1)
    y = yy / max(h - 1, 1)
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return ((prob >= threshold) & (dist <= multiplier * r)).astype(np.float32)


def collect_model_rows(args) -> pd.DataFrame:
    require_cuda(torch)
    device = torch.device("cuda")
    rows = merge_rows_with_refine_targets(load_index())
    desc = load_descriptors()
    stats = compute_stats(rows, desc)
    out_rows = []
    paths = sorted(CHECKPOINT_DIR.glob("deeponet_mask_refine_*.pt"))
    for path in paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model_name = str(ckpt.get("model_name", path.stem.replace("deeponet_mask_refine_", "")))
        model = build_model(ckpt).to(device)
        for split in ["val", "test"]:
            sub = rows[rows["split"].eq(split)].reset_index(drop=True)
            loader = make_mask_refine_loader(
                sub,
                stats,
                batch_size=args.batch_size,
                shuffle=False,
                mode="full",
                k_queries=1024,
                input_norm=args.input_norm,
                descriptors=desc,
                seed=SEED + 910,
            )
            offset = 0
            with torch.no_grad():
                for batch in loader:
                    query, params, _area = model(
                        batch["branch_x"].to(device).float(),
                        batch["trunk_xy"].to(device).float(),
                        batch["descriptor_x"].to(device).float(),
                    )
                    q = query.detach().cpu().numpy()
                    p = params.detach().cpu().numpy()
                    p_un = unscale_params(p, stats)
                    present_prob = sigmoid_np(p[:, 0])
                    pred_lat = q[:, :, 0] * stats.field_std[0] + stats.field_mean[0]
                    pred_apd90 = q[:, :, 1] * stats.field_std[2] + stats.field_mean[2]
                    patch_prob = sigmoid_np(q[:, :, 2])
                    for b in range(q.shape[0]):
                        src = sub.iloc[offset + b]
                        phys = compute_physiology(src)
                        valid = phys["valid"].astype(bool)
                        true_mask = phys["patch"]
                        true_lat = phys["LAT"]
                        true_apd90 = phys["APD90"]
                        row_pred = {
                            "pred_patch_present_prob": float(present_prob[b]),
                            "pred_center_x_norm": float(p_un[b, 1]),
                            "pred_center_y_norm": float(p_un[b, 2]),
                            "pred_patch_area_param": float(math.exp(p_un[b, 3])) if np.isfinite(p_un[b, 3]) else math.nan,
                            "pred_radius_param": float(math.exp(p_un[b, 4])) if np.isfinite(p_un[b, 4]) else math.nan,
                        }
                        prob_map = patch_prob[b].reshape(32, 32)
                        pred_mask_bal = center_constrained_mask(prob_map, row_pred)
                        pred_mask_05 = (prob_map >= 0.5).astype(np.float32)
                        pa = pred_apd90[b].reshape(32, 32)
                        pl = pred_lat[b].reshape(32, 32)
                        base = {
                            "model": model_name,
                            "split": split,
                            "row_id": src["row_id"],
                            "base_sample_id": src["base_sample_id"],
                            "condition": src["condition"],
                            "combined_variant": src["combined_variant"],
                            "true_APD90_patch_delta": phys["scalars"]["APD90_patch_delta"],
                            "true_LAT_patch_delay": phys["scalars"]["LAT_patch_delay"],
                            "true_APD_anomaly_delta": anomaly_delta(true_apd90, phys["apd_anomaly"], valid),
                            "true_LAT_anomaly_delay": anomaly_delta(true_lat, phys["lat_anomaly"], valid),
                            "pred_true_mask_APD90_delta": patch_delta(pa, true_mask, valid),
                            "pred_true_mask_LAT_delay": patch_delta(pl, true_mask, valid),
                            "pred_true_APD_anomaly_APD90_delta": anomaly_delta(pa, phys["apd_anomaly"], valid),
                            "pred_true_LAT_anomaly_LAT_delay": anomaly_delta(pl, phys["lat_anomaly"], valid),
                            "pred_balanced_mask_APD90_delta": patch_delta(pa, pred_mask_bal, valid),
                            "pred_balanced_mask_LAT_delay": patch_delta(pl, pred_mask_bal, valid),
                            "pred_threshold_0p5_mask_APD90_delta": patch_delta(pa, pred_mask_05, valid),
                            "pred_threshold_0p5_mask_LAT_delay": patch_delta(pl, pred_mask_05, valid),
                            "balanced_mask_area_fraction": float(np.mean(pred_mask_bal[valid])) if np.any(valid) else math.nan,
                            "threshold_0p5_mask_area_fraction": float(np.mean(pred_mask_05[valid])) if np.any(valid) else math.nan,
                        }
                        out_rows.append(base)
                    offset += q.shape[0]
        print(f"collected APD/LAT oracle rows for {model_name}", flush=True)
    return pd.DataFrame(out_rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    method_specs = [
        ("true_mask_pred_apd90", "true_APD90_patch_delta", "pred_true_mask_APD90_delta", "APD90_patch_delta"),
        ("true_mask_pred_lat", "true_LAT_patch_delay", "pred_true_mask_LAT_delay", "LAT_patch_delay"),
        ("true_apd_anomaly_map_pred_apd90", "true_APD_anomaly_delta", "pred_true_APD_anomaly_APD90_delta", "APD_anomaly_delta"),
        ("true_lat_anomaly_map_pred_lat", "true_LAT_anomaly_delay", "pred_true_LAT_anomaly_LAT_delay", "LAT_anomaly_delay"),
        ("predicted_balanced_mask_pred_apd90", "true_APD90_patch_delta", "pred_balanced_mask_APD90_delta", "APD90_patch_delta"),
        ("predicted_balanced_mask_pred_lat", "true_LAT_patch_delay", "pred_balanced_mask_LAT_delay", "LAT_patch_delay"),
        ("predicted_threshold_0p5_mask_pred_apd90", "true_APD90_patch_delta", "pred_threshold_0p5_mask_APD90_delta", "APD90_patch_delta"),
        ("predicted_threshold_0p5_mask_pred_lat", "true_LAT_patch_delay", "pred_threshold_0p5_mask_LAT_delay", "LAT_patch_delay"),
    ]
    rows = []
    for (model, split), sub0 in df.groupby(["model", "split"]):
        for scope, sub in [("overall", sub0)] + [(f"condition:{c}", g) for c, g in sub0.groupby("condition")]:
            for method, true_col, pred_col, target in method_specs:
                metrics = regression_summary(sub[true_col], sub[pred_col])
                metrics.update(
                    {
                        "model": model,
                        "split": split,
                        "scope": scope,
                        "condition": scope.split(":", 1)[1] if scope.startswith("condition:") else "",
                        "method": method,
                        "target_scalar": target,
                        "true_col": true_col,
                        "pred_col": pred_col,
                    }
                )
                rows.append(metrics)
    return pd.DataFrame(rows)


def scatter_figure(df: pd.DataFrame, model: str, target: str, path: Path) -> None:
    if target == "APD90":
        pairs = [
            ("true mask", "true_APD90_patch_delta", "pred_true_mask_APD90_delta"),
            ("pred mask", "true_APD90_patch_delta", "pred_balanced_mask_APD90_delta"),
        ]
        title = "Oracle APD90 Delta Recovery"
    else:
        pairs = [
            ("true mask", "true_LAT_patch_delay", "pred_true_mask_LAT_delay"),
            ("pred mask", "true_LAT_patch_delay", "pred_balanced_mask_LAT_delay"),
        ]
        title = "Oracle LAT Delay Recovery"
    sub = df[df["model"].eq(model) & df["split"].eq("test")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), dpi=160)
    for ax, (label, true_col, pred_col) in zip(axes, pairs):
        for cond, group in sub.groupby("condition"):
            ax.scatter(group[true_col], group[pred_col], s=14, alpha=0.55, label=cond.replace("_", " "))
        vals = pd.concat([sub[true_col], sub[pred_col]]).to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = float(np.min(vals)), float(np.max(vals))
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, alpha=0.6)
        ax.set_title(label)
        ax.set_xlabel("true scalar")
        ax.set_ylabel("predicted scalar")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def conduction_failure_figure(df: pd.DataFrame, model: str) -> None:
    sub = df[df["model"].eq(model) & df["split"].eq("test") & df["condition"].eq("conduction_slowing_perturbation")].copy()
    if sub.empty:
        return
    sub["lat_abs_error"] = np.abs(sub["true_LAT_patch_delay"] - sub["pred_balanced_mask_LAT_delay"])
    worst = sub.sort_values("lat_abs_error", ascending=False).head(6)
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=160)
    ax.scatter(sub["true_LAT_patch_delay"], sub["pred_balanced_mask_LAT_delay"], s=16, alpha=0.4)
    ax.scatter(worst["true_LAT_patch_delay"], worst["pred_balanced_mask_LAT_delay"], s=38, color="#ef4444", label="largest errors")
    vals = pd.concat([sub["true_LAT_patch_delay"], sub["pred_balanced_mask_LAT_delay"]]).to_numpy(float)
    vals = vals[np.isfinite(vals)]
    if vals.size:
        lo, hi = float(np.min(vals)), float(np.max(vals))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, alpha=0.6)
    ax.set_title("Conduction-only LAT Delay Failure Examples")
    ax.set_xlabel("true LAT patch delay")
    ax.set_ylabel("predicted-mask predicted-LAT delay")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(V1F_FIG_DIR / "conduction_only_failure_examples.png")
    plt.close(fig)


def parse_args():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--input-norm", choices=["per_sample", "train_global", "none"], default="per_sample")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    row_df = collect_model_rows(args)
    row_df.to_csv(ROW_CSV, index=False)
    agg = aggregate(row_df)
    agg.to_csv(OUT_CSV, index=False)
    model = "deeponet_cnn_branch_fourier_trunk_mask_refine"
    if model not in set(row_df["model"]):
        model = sorted(row_df["model"].unique())[0]
    scatter_figure(row_df, model, "APD90", V1F_FIG_DIR / "oracle_apd90_delta_recovery_scatter.png")
    scatter_figure(row_df, model, "LAT", V1F_FIG_DIR / "oracle_lat_delay_recovery_scatter.png")
    conduction_failure_figure(row_df, model)
    best_test = {}
    test = agg[agg["split"].eq("test") & agg["scope"].eq("overall")]
    for target, sub in test.groupby("target_scalar"):
        best_test[target] = sub.sort_values("r2", ascending=False).head(5).to_dict(orient="records")
    write_json(
        SUMMARY_JSON,
        {
            "oracle_csv": OUT_CSV,
            "row_csv": ROW_CSV,
            "best_test_by_target": best_test,
            "selected_postprocess_mask": "center_constrained_probability|0.65|mean|1.2|True",
            "guardrails": {
                "true_maps_used_as_oracles_only": True,
                "condition_labels_reporting_only": True,
                "existing_checkpoints_only": True,
            },
        },
    )
    print(f"wrote {OUT_CSV}")
    print(f"wrote {ROW_CSV}")
    print(f"wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
