                      
                                                                      

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v1e_direct_common import FIG_DIR, NEURAL_DIR, PILOT_DIR, ROOT, ensure_dirs, write_json


OUT_CSV = NEURAL_DIR / "direct_signal_audit.csv"
OUT_CORR = NEURAL_DIR / "direct_signal_audit_correlations.csv"
OUT_REPORT = NEURAL_DIR / "direct_signal_audit_report.json"
FIG_OUT = ROOT / "figures" / "neural_v1e"


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return math.nan
    xv = x[mask].to_numpy(float)
    yv = y[mask].to_numpy(float)
    if np.allclose(xv, xv[0]) or np.allclose(yv, yv[0]):
        return math.nan
    return float(np.corrcoef(xv, yv)[0, 1])


def signal_descriptors(b: np.ndarray, mask: np.ndarray) -> dict[str, float]:
                                                                        
    valid = mask[None, :, :, :].repeat(b.shape[0], axis=0)
    filled = np.where(valid, b, 0.0).astype(np.float64)
    vals = filled[valid]
    primary = filled[:, :, :, 0]
    rms_t = np.sqrt(np.mean(primary * primary, axis=(1, 2)))
    energy = np.mean(primary * primary, axis=0)
    total_energy = float(np.sum(energy))
    g = primary.shape[1]
    yy, xx = np.indices((g, g))
    if total_energy > 0:
        cx = float(np.sum(xx * energy) / total_energy / max(g - 1, 1))
        cy = float(np.sum(yy * energy) / total_energy / max(g - 1, 1))
        p = (energy.ravel() / total_energy).clip(1e-30, None)
        entropy = float(-np.sum(p * np.log(p)) / np.log(p.size))
        concentration = float(np.max(p))
    else:
        cx = cy = entropy = concentration = math.nan
    left = energy[:, : g // 2].sum()
    right = energy[:, g // 2 :].sum()
    bottom = energy[: g // 2, :].sum()
    top = energy[g // 2 :, :].sum()
    dx = float((right - left) / (right + left + 1e-30))
    dy = float((top - bottom) / (top + bottom + 1e-30))
    peak_idx = int(np.argmax(rms_t)) if len(rms_t) else 0
    early = rms_t[: min(25, len(rms_t))]
    late = rms_t[min(150, len(rms_t)) : min(280, len(rms_t))]
    grad_y, grad_x = np.gradient(np.mean(primary, axis=0))
    return {
        "global_rms": float(np.sqrt(np.mean(vals * vals))) if vals.size else math.nan,
        "max_abs_B": float(np.max(np.abs(vals))) if vals.size else math.nan,
        "activation_window_rms_proxy": float(np.mean(early)) if len(early) else math.nan,
        "late_window_rms_proxy": float(np.mean(late)) if len(late) else math.nan,
        "time_peak_rms_index": peak_idx,
        "time_peak_rms_value": float(rms_t[peak_idx]) if len(rms_t) else math.nan,
        "energy_center_x_norm": cx,
        "energy_center_y_norm": cy,
        "spatial_entropy": entropy,
        "spatial_concentration": concentration,
        "left_right_asymmetry": dx,
        "top_bottom_asymmetry": dy,
        "mean_abs_grad_x": float(np.mean(np.abs(grad_x))),
        "mean_abs_grad_y": float(np.mean(np.abs(grad_y))),
    }


def make_figures(df: pd.DataFrame) -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    non_hom = df[df["patch_present"].eq(1)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(non_hom["patch_center_x_norm"], non_hom["energy_center_x_norm"], s=12, alpha=0.7)
    axes[0].set_xlabel("patch center x")
    axes[0].set_ylabel("magnetic energy center x")
    axes[1].scatter(non_hom["patch_center_y_norm"], non_hom["energy_center_y_norm"], s=12, alpha=0.7)
    axes[1].set_xlabel("patch center y")
    axes[1].set_ylabel("magnetic energy center y")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "direct_signal_vs_patch_center.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for variant, sub in non_hom.groupby("measurement_variant"):
        ax.scatter(sub["patch_area_fraction"], sub["global_rms"], s=10, alpha=0.6, label=variant)
    ax.set_xlabel("patch area fraction")
    ax.set_ylabel("global RMS")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "direct_signal_vs_patch_area.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    df.boxplot(column="global_rms", by="condition", ax=ax, rot=25)
    fig.suptitle("")
    ax.set_title("Direct Signal RMS By Condition")
    fig.tight_layout()
    fig.savefig(FIG_OUT / "direct_signal_by_condition.png", dpi=180)
    plt.close(fig)

    examples = df.groupby("condition", group_keys=False).head(1)
    fig, axes = plt.subplots(1, len(examples), figsize=(3 * len(examples), 3))
    axes = np.atleast_1d(axes)
    for ax, (_, row) in zip(axes, examples.iterrows()):
        b = np.load(ROOT / row["B_meas_file"])
        energy = np.mean(b[:, :, :, 0] ** 2, axis=0)
        ax.imshow(energy, origin="lower", cmap="magma")
        ax.scatter([row["energy_center_x_norm"] * (energy.shape[1] - 1)], [row["energy_center_y_norm"] * (energy.shape[0] - 1)], c="cyan", s=30)
        if row["patch_present"] == 1:
            ax.scatter([row["patch_center_x_norm"] * (energy.shape[1] - 1)], [row["patch_center_y_norm"] * (energy.shape[0] - 1)], c="white", s=25, marker="x")
        ax.set_title(row["condition"], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG_OUT / "direct_energy_centroid_examples.png", dpi=180)
    plt.close(fig)


def main() -> int:
    ensure_dirs()
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    mag = pd.read_csv(PILOT_DIR / "direct_magnetic_manifest.csv")
    targets = pd.read_csv(PILOT_DIR / "target_manifest.csv")
    ready = pd.read_csv(PILOT_DIR / "fno_ready" / "sample_index.csv")
    split_cols = ready[["row_id", "split"]].drop_duplicates()
    rows = mag.merge(targets, on=["base_sample_id", "condition"], how="left").merge(split_cols, on="row_id", how="left")
    out_rows = []
    for _, row in rows.iterrows():
        b = np.load(ROOT / row["B_meas_file"]).astype(np.float32)
        mask = np.load(ROOT / row["sensor_mask_file"]).astype(bool)
        desc = signal_descriptors(b, mask)
        out_rows.append(
            {
                "row_id": row["row_id"],
                "base_sample_id": row["base_sample_id"],
                "condition": row["condition"],
                "measurement_variant": row["measurement_variant"],
                "split": row.get("split", ""),
                "patch_present": row["patch_present"],
                "patch_center_x_norm": row["patch_center_x_norm"],
                "patch_center_y_norm": row["patch_center_y_norm"],
                "patch_area_fraction": row["patch_area_fraction"],
                "patch_radius_equiv_norm": row["patch_radius_equiv_norm"],
                "LAT_range": row["LAT_range"],
                "APD90_patch_delta": row["APD90_patch_delta"],
                **desc,
                "B_meas_file": row["B_meas_file"],
            }
        )
    df = pd.DataFrame(out_rows)
    df.to_csv(OUT_CSV, index=False)
    feature_cols = [
        "global_rms",
        "max_abs_B",
        "activation_window_rms_proxy",
        "late_window_rms_proxy",
        "time_peak_rms_index",
        "energy_center_x_norm",
        "energy_center_y_norm",
        "spatial_entropy",
        "spatial_concentration",
        "left_right_asymmetry",
        "top_bottom_asymmetry",
        "mean_abs_grad_x",
        "mean_abs_grad_y",
    ]
    target_cols = ["patch_present", "patch_center_x_norm", "patch_center_y_norm", "patch_area_fraction", "patch_radius_equiv_norm", "LAT_range", "APD90_patch_delta"]
    corr_rows = []
    for variant, sub in df.groupby("measurement_variant"):
        for feat in feature_cols:
            for target in target_cols:
                corr_rows.append({"measurement_variant": variant, "feature": feat, "target": target, "pearson_r": safe_corr(sub[feat], sub[target])})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUT_CORR, index=False)
    make_figures(df)
    best = corr[np.isfinite(corr["pearson_r"])].copy()
    best["abs_r"] = best["pearson_r"].abs()
    top = best.sort_values("abs_r", ascending=False).head(20).drop(columns=["abs_r"]).to_dict(orient="records")
    report = {
        "rows": int(len(df)),
        "base_samples": int(df["base_sample_id"].nunique()),
        "variant_counts": df["measurement_variant"].value_counts().to_dict(),
        "top_correlations": top,
        "interpretation": "Simple descriptors are diagnostic only and are not model inputs unless explicitly used by the tabular baseline.",
        "passed": bool(len(df) > 0 and len(top) > 0),
    }
    write_json(OUT_REPORT, report)
    print(json.dumps({"rows": len(df), "top_correlations": top[:5]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
