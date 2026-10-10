                      
                                                                               

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUG_READY = ROOT / "outputs" / "dataset_v1b_augmented_ready"
FIG_DIR = ROOT / "figures" / "dataset_v1b_augmented"
VARIANTS = [
    "clean",
    "noise_low",
    "noise_medium",
    "noise_high",
    "dropout_10pct",
    "dropout_25pct",
    "jitter_1mm",
    "jitter_3mm",
    "height_30mm",
    "height_70mm",
    "combined_medium",
]


def read_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def fval(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan


def by_aug(rows: list[dict]) -> dict[str, dict]:
    return {r["augmented_sample_id"]: r for r in rows}


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or np.std(xs) == 0 or np.std(ys) == 0:
        return math.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def plot_feature_shift(features: list[dict]) -> None:
    clean_by_base = {r["base_sample_id"]: fval(r, "full_RMS_mean") for r in features if r["variant"] == "clean"}
    data = []
    labels = []
    for variant in VARIANTS:
        vals = []
        for row in features:
            if row["variant"] != variant:
                continue
            base = clean_by_base.get(row["base_sample_id"])
            if base and np.isfinite(base):
                vals.append((fval(row, "full_RMS_mean") - base) / base)
        data.append(vals)
        labels.append(variant.replace("_", "\n"))
    plt.figure(figsize=(12.5, 4.8))
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.ylabel("Relative shift in full RMS mean vs clean")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_shift_by_variant.png", dpi=180)
    plt.close()


def plot_rms_noise(features: list[dict]) -> None:
    variants = ["clean", "noise_low", "noise_medium", "noise_high", "combined_medium"]
    data = [[fval(r, "full_RMS_mean") for r in features if r["variant"] == v] for v in variants]
    plt.figure(figsize=(7.5, 4.8))
    plt.boxplot(data, tick_labels=[v.replace("_", "\n") for v in variants], showmeans=True)
    plt.ylabel("Full-window RMS mean")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rms_noise_robustness.png", dpi=180)
    plt.close()


def plot_mi_di(features: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharex=True)
    for ax, key, title in [(axes[0], "full_MI_mean", "MI"), (axes[1], "full_DI_mean", "DI")]:
        data = [[fval(r, key) for r in features if r["variant"] == v] for v in VARIANTS]
        ax.boxplot(data, tick_labels=[v.replace("_", "\n") for v in VARIANTS], showmeans=True)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mi_di_by_variant.png", dpi=180)
    plt.close(fig)


def plot_corr_by_variant(features: list[dict], targets: dict[str, dict], target_key: str, feature_key: str, path: Path, ylabel: str) -> None:
    vals = []
    for variant in VARIANTS:
        xs = []
        ys = []
        for row in features:
            if row["variant"] != variant:
                continue
            target = targets.get(row["augmented_sample_id"], {})
            x = fval(target, target_key)
            y = fval(row, feature_key)
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x)
                ys.append(y)
        vals.append(pearson(xs, ys))
    plt.figure(figsize=(10.5, 4.8))
    plt.bar(range(len(VARIANTS)), vals)
    plt.xticks(range(len(VARIANTS)), [v.replace("_", "\n") for v in VARIANTS], fontsize=7)
    plt.ylabel(ylabel)
    plt.ylim(-1.05, 1.05)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    features = read_rows(AUG_READY / "features_mig_deployable_fixed_augmented.csv")
    targets = by_aug(read_rows(AUG_READY / "targets_ep_augmented.csv"))
    plot_feature_shift(features)
    plot_rms_noise(features)
    plot_mi_di(features)
    plot_corr_by_variant(features, targets, "y_lat_range", "early_activation_fixed_RMS_mean", FIG_DIR / "lat_range_correlation_by_variant.png", "corr(LAT range, early RMS mean)")
    plot_corr_by_variant(features, targets, "y_apd90_patch_delta", "mid_recovery_fixed_RMS_max", FIG_DIR / "apd_delta_correlation_by_variant.png", "corr(APD90 delta, mid RMS max)")
    print(f"Wrote figures to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
