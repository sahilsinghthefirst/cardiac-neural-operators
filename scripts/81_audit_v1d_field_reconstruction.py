                      
                                                                            

                                                                          
                                                                        
                                                                            
                            
   

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fno_field_data_common import load_index, load_stats, make_loader, torch
from scripts_78_bridge import FNOFieldReconstructor
from v1d_field_common import CHECKPOINT_DIR, NEURAL_DIR, NEURAL_FIG_DIR, ROOT, require_cuda, scalar_targets_from_fields, write_json


STATS_JSON = NEURAL_DIR / "fno_field_train_stats.json"
CHECKPOINT = CHECKPOINT_DIR / "fno_field_reconstructor.pt"
AUDIT_CSV = NEURAL_DIR / "field_reconstruction_audit.csv"
AUDIT_JSON = NEURAL_DIR / "field_reconstruction_audit.json"
DECOMP_CSV = NEURAL_DIR / "patch_delta_decomposition.csv"
MASK_SUMMARY_CSV = NEURAL_DIR / "patch_mask_error_summary.csv"
TARGETS = ["LAT", "APD80", "APD90"]


def finite_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {"mae": math.nan, "rmse": math.nan, "r2": math.nan, "n": 0}
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if y_true.size > 1 and not np.allclose(y_true, y_true[0]) else math.nan,
        "n": int(y_true.size),
    }


def mask_metrics(true: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    t = (true > 0.5) & valid
    p = (pred > 0.5) & valid
    tp = float(np.sum(t & p))
    fp = float(np.sum(~t & p & valid))
    fn = float(np.sum(t & ~p))
    dice = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
    iou = tp / max(tp + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    true_area = float(np.mean(t[valid])) if np.any(valid) else math.nan
    pred_area = float(np.mean(p[valid])) if np.any(valid) else math.nan
    centroid_error = math.nan
    if np.any(t) and np.any(p):
        ty, tx = np.argwhere(t).mean(axis=0)
        py, px = np.argwhere(p).mean(axis=0)
        centroid_error = float(np.sqrt((tx - px) ** 2 + (ty - py) ** 2))
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "true_area_fraction": true_area,
        "pred_area_fraction": pred_area,
        "area_bias": pred_area - true_area if np.isfinite(pred_area) and np.isfinite(true_area) else math.nan,
        "centroid_error_grid_cells": centroid_error,
    }


def boundary_mask(patch: np.ndarray, valid: np.ndarray) -> np.ndarray:
    patch = (patch > 0.5) & valid
    if not np.any(patch):
        return np.zeros_like(patch, dtype=bool)
    try:
        from scipy.ndimage import binary_dilation, binary_erosion

        return (binary_dilation(patch) ^ binary_erosion(patch)) & valid
    except Exception:
        out = np.zeros_like(patch, dtype=bool)
        for y, x in np.argwhere(patch):
            y0, y1 = max(0, y - 1), min(patch.shape[0], y + 2)
            x0, x1 = max(0, x - 1), min(patch.shape[1], x + 2)
            if np.any(~patch[y0:y1, x0:x1]):
                out[y, x] = True
        return out & valid


def load_model(device: torch.device):
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    ck_args = ckpt.get("args", {})
    model = FNOFieldReconstructor(
        int(ckpt["in_channels"]),
        int(ck_args.get("width", 48)),
        int(ck_args.get("modes", 6)),
        int(ck_args.get("layers", 4)),
        float(ck_args.get("dropout", 0.08)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    args = SimpleNamespace(batch_size=int(ck_args.get("batch_size", 12)), input_norm=ck_args.get("input_norm", "per_sample"))
    return model, args, ck_args


def append_pixels(buckets: dict, target: str, true: np.ndarray, pred: np.ndarray, masks: dict[str, np.ndarray], condition: str, variant: str) -> None:
    for region, region_mask in masks.items():
        if not np.any(region_mask):
            continue
        vals_t = true[region_mask].astype(np.float64)
        vals_p = pred[region_mask].astype(np.float64)
        for group_type, group_value in [("all", "all"), ("condition", condition), ("measurement_variant", variant)]:
            key = (target, region, group_type, group_value)
            buckets[key][0].append(vals_t)
            buckets[key][1].append(vals_p)


def run_inference() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    require_cuda(torch)
    device = torch.device("cuda")
    rows = load_index()
    test_rows = rows[rows["split"].eq("test")].reset_index(drop=True)
    stats = load_stats(STATS_JSON)
    model, args, ck_args = load_model(device)
    loader = make_loader(test_rows, stats, batch_size=args.batch_size, shuffle=False, input_norm=args.input_norm)

    buckets: dict[tuple[str, str, str, str], list[list[np.ndarray]]] = defaultdict(lambda: [[], []])
    sample_records: list[dict] = []
    decomp_records: list[dict] = []
    examples: list[dict] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["x"].to(device).float()).cpu().numpy()
            y = batch["y"].numpy()
            valid = batch["valid_mask"].numpy().astype(bool)
            patch_true = (y[:, 3] > 0.5)
            patch_prob = 1.0 / (1.0 + np.exp(-pred[:, 3]))
            patch_pred = patch_prob > 0.5

            for b in range(pred.shape[0]):
                src = test_rows.iloc[offset + b]
                condition = str(src["condition"])
                variant = str(src["measurement_variant"])
                v = valid[b]
                true_fields = {
                    name: y[b, i] * stats.target_std[i] + stats.target_mean[i] for i, name in enumerate(TARGETS)
                }
                pred_fields = {
                    name: pred[b, i] * stats.target_std[i] + stats.target_mean[i] for i, name in enumerate(TARGETS)
                }
                pt = patch_true[b] & v
                pp = patch_pred[b] & v
                masks = {
                    "global": v,
                    "inside_patch": pt,
                    "outside_patch": v & ~pt,
                    "patch_boundary": boundary_mask(pt, v),
                }
                rec = {
                    "field_sample_id": src["field_sample_id"],
                    "base_sample_id": src["base_sample_id"],
                    "condition": condition,
                    "measurement_variant": variant,
                }
                for name in TARGETS:
                    append_pixels(buckets, name, true_fields[name], pred_fields[name], masks, condition, variant)
                    for region in ["global", "inside_patch", "outside_patch", "patch_boundary"]:
                        rm = masks[region]
                        metrics = finite_regression(true_fields[name][rm], pred_fields[name][rm])
                        rec[f"{name}_{region}_mae"] = metrics["mae"]
                        rec[f"{name}_{region}_rmse"] = metrics["rmse"]
                        rec[f"{name}_{region}_r2"] = metrics["r2"]
                mm = mask_metrics(patch_true[b], patch_prob[b], v)
                rec.update({f"patch_{k}": val for k, val in mm.items()})

                true_scalar = scalar_targets_from_fields(true_fields["LAT"], true_fields["APD90"], patch_true[b], v)
                pred_lat_scalar = scalar_targets_from_fields(pred_fields["LAT"], true_fields["APD90"], patch_true[b], v)
                pred_apd_true_mask = scalar_targets_from_fields(true_fields["LAT"], pred_fields["APD90"], patch_true[b], v)
                true_apd_pred_mask = scalar_targets_from_fields(true_fields["LAT"], true_fields["APD90"], patch_pred[b], v)
                pred_both = scalar_targets_from_fields(pred_fields["LAT"], pred_fields["APD90"], patch_pred[b], v)
                decomp = {
                    "field_sample_id": src["field_sample_id"],
                    "base_sample_id": src["base_sample_id"],
                    "condition": condition,
                    "measurement_variant": variant,
                    "true_LAT_range": true_scalar["LAT_range"],
                    "pred_LAT_range": pred_lat_scalar["LAT_range"],
                    "true_APD90_patch_delta": true_scalar["APD90_patch_delta"],
                    "pred_APD90_delta_true_mask_pred_apd": pred_apd_true_mask["APD90_patch_delta"],
                    "pred_APD90_delta_pred_mask_true_apd": true_apd_pred_mask["APD90_patch_delta"],
                    "pred_APD90_delta_pred_mask_pred_apd": pred_both["APD90_patch_delta"],
                    "true_patch_area_fraction": true_scalar["patch_area_fraction"],
                    "pred_patch_area_fraction": pred_both["patch_area_fraction"],
                    "patch_dice": mm["dice"],
                    "patch_centroid_error_grid_cells": mm["centroid_error_grid_cells"],
                }
                rec.update(decomp)
                sample_records.append(rec)
                decomp_records.append(decomp)
                examples.append(
                    {
                        "score": mm["dice"],
                        "sample": src["field_sample_id"],
                        "condition": condition,
                        "variant": variant,
                        "true_apd90": true_fields["APD90"].copy(),
                        "pred_apd90": pred_fields["APD90"].copy(),
                        "true_patch": patch_true[b].copy(),
                        "pred_patch": patch_prob[b].copy(),
                    }
                )
            offset += pred.shape[0]

    audit_rows = []
    for (target, region, group_type, group_value), (chunks_t, chunks_p) in buckets.items():
        yt = np.concatenate(chunks_t) if chunks_t else np.array([])
        yp = np.concatenate(chunks_p) if chunks_p else np.array([])
        audit_rows.append(
            {
                "split": "test",
                "target": target,
                "region": region,
                "group_type": group_type,
                "group_value": group_value,
                **finite_regression(yt, yp),
            }
        )
    audit_df = pd.DataFrame(audit_rows).sort_values(["target", "region", "group_type", "group_value"]).reset_index(drop=True)
    decomp_df = pd.DataFrame(decomp_records)
    sample_df = pd.DataFrame(sample_records)

    mask_rows = []
    for group_type, col in [("all", None), ("condition", "condition"), ("measurement_variant", "measurement_variant")]:
        groups = [("all", sample_df)] if col is None else list(sample_df.groupby(col))
        for group_value, sub in groups:
            mask_rows.append(
                {
                    "split": "test",
                    "group_type": group_type,
                    "group_value": group_value,
                    "n": int(len(sub)),
                    "dice_mean": float(sub["patch_dice"].mean()),
                    "dice_median": float(sub["patch_dice"].median()),
                    "iou_mean": float(sub["patch_iou"].mean()),
                    "precision_mean": float(sub["patch_precision"].mean()),
                    "recall_mean": float(sub["patch_recall"].mean()),
                    "area_bias_mean": float(sub["patch_area_bias"].mean()),
                    "centroid_error_mean": float(sub["patch_centroid_error_grid_cells"].mean(skipna=True)),
                }
            )
    mask_df = pd.DataFrame(mask_rows)
    return audit_df, decomp_df, mask_df, {"samples": sample_df, "examples": examples, "checkpoint_args": ck_args}


def summarize_decomposition(decomp_df: pd.DataFrame) -> dict:
    mappings = {
        "LAT_range_pred_LAT": ("true_LAT_range", "pred_LAT_range"),
        "APD90_delta_true_mask_pred_apd": ("true_APD90_patch_delta", "pred_APD90_delta_true_mask_pred_apd"),
        "APD90_delta_pred_mask_true_apd": ("true_APD90_patch_delta", "pred_APD90_delta_pred_mask_true_apd"),
        "APD90_delta_pred_mask_pred_apd": ("true_APD90_patch_delta", "pred_APD90_delta_pred_mask_pred_apd"),
        "patch_area_pred_mask": ("true_patch_area_fraction", "pred_patch_area_fraction"),
    }
    out = {}
    for name, (truth, pred) in mappings.items():
        mask = np.isfinite(decomp_df[truth]) & np.isfinite(decomp_df[pred])
        out[name] = finite_regression(decomp_df.loc[mask, truth].to_numpy(), decomp_df.loc[mask, pred].to_numpy())
    return out


def plot_outputs(audit_df: pd.DataFrame, decomp_df: pd.DataFrame, mask_df: pd.DataFrame, extra: dict) -> None:
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    sample_df = extra["samples"]

    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [
        sample_df["APD90_inside_patch_mae"].dropna(),
        sample_df["APD90_outside_patch_mae"].dropna(),
        sample_df["APD90_patch_boundary_mae"].dropna(),
    ]
    ax.boxplot(vals, labels=["inside", "outside", "boundary"], showfliers=False)
    ax.set_ylabel("APD90 MAE (ms)")
    ax.set_title("APD90 Error by Patch Region")
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "audit_apd90_error_inside_vs_outside_patch.png", dpi=180)
    plt.close(fig)

    for group_type, fname, title in [
        ("condition", "audit_patch_mask_dice_by_condition.png", "Patch Dice by Condition"),
        ("measurement_variant", "audit_patch_mask_dice_by_variant.png", "Patch Dice by Measurement Variant"),
    ]:
        sub = mask_df[mask_df["group_type"].eq(group_type)].sort_values("dice_mean")
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(sub["group_value"].astype(str), sub["dice_mean"], color="#4c78a8")
        ax.set_ylabel("mean Dice")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(NEURAL_FIG_DIR / fname, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    summary = summarize_decomposition(decomp_df)
    labels = list(summary)
    vals = [summary[k]["r2"] for k in labels]
    ax.bar(range(len(labels)), vals, color="#f58518")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_ylabel("R2")
    ax.set_title("Patch Scalar Decomposition")
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "audit_patch_delta_decomposition.png", dpi=180)
    plt.close(fig)

    examples = sorted(extra["examples"], key=lambda x: (np.nan_to_num(x["score"], nan=-1.0)))
    chosen = examples[:2] + examples[-2:]
    fig, axes = plt.subplots(len(chosen), 4, figsize=(10, 2.5 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]
    for r, ex in enumerate(chosen):
        for c, (key, title) in enumerate(
            [
                ("true_apd90", "true APD90"),
                ("pred_apd90", "pred APD90"),
                ("true_patch", "true patch"),
                ("pred_patch", "pred patch prob"),
            ]
        ):
            im = axes[r, c].imshow(ex[key], origin="lower", cmap="viridis" if "apd" in key else "magma")
            axes[r, c].set_title(f"{title}\n{ex['sample']} Dice={ex['score']:.2f}", fontsize=8)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            fig.colorbar(im, ax=axes[r, c], fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "audit_example_good_bad_field_maps.png", dpi=180)
    plt.close(fig)


def main() -> int:
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    audit_df, decomp_df, mask_df, extra = run_inference()
    audit_df.to_csv(AUDIT_CSV, index=False)
    decomp_df.to_csv(DECOMP_CSV, index=False)
    mask_df.to_csv(MASK_SUMMARY_CSV, index=False)
    decomp_summary = summarize_decomposition(decomp_df)
    plot_outputs(audit_df, decomp_df, mask_df, extra)
    payload = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "stats": str(STATS_JSON.relative_to(ROOT)),
        "test_rows": int(len(extra["samples"])),
        "checkpoint_args": extra["checkpoint_args"],
        "field_metric_rows": audit_df.replace({np.nan: None}).to_dict(orient="records"),
        "patch_delta_decomposition": decomp_summary,
        "mask_summary": mask_df.replace({np.nan: None}).to_dict(orient="records"),
        "figures": [
            "figures/neural_v1d/audit_apd90_error_inside_vs_outside_patch.png",
            "figures/neural_v1d/audit_patch_mask_dice_by_condition.png",
            "figures/neural_v1d/audit_patch_mask_dice_by_variant.png",
            "figures/neural_v1d/audit_patch_delta_decomposition.png",
            "figures/neural_v1d/audit_example_good_bad_field_maps.png",
        ],
        "notes": "LAT/APD/patch fields are evaluation targets only; they are not used as model inputs.",
    }
    write_json(AUDIT_JSON, payload)
    print(json.dumps({"audit_csv": str(AUDIT_CSV), "decomposition": decomp_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
