                      
                                                             

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
READY_BASE = ROOT / "outputs" / "dataset_v1b_ready"
FIG_DIR = ROOT / "figures" / "dataset_v1b_ready"
CONDITION_ORDER = [
    "homogeneous",
    "apd_repolarization_perturbation",
    "conduction_slowing_perturbation",
    "mixed_apd_conduction_perturbation",
]


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def fval(row: dict, key: str) -> float:
    try:
        value = float(row.get(key, "nan"))
    except ValueError:
        value = math.nan
    return value


def label(condition: str) -> str:
    return condition.replace("_", "\n")


def by_sample(rows: list[dict]) -> dict[str, dict]:
    return {r["sample_id"]: r for r in rows}


def plot_windowed_by_condition(features: list[dict]) -> None:
    windows = ["activation_like", "post_activation_like", "repolarization_like", "st_t_like"]
    fig, axes = plt.subplots(1, len(windows), figsize=(16.0, 4.4), sharey=True)
    for ax, window in zip(axes, windows):
        data = []
        conditions = []
        for cond in CONDITION_ORDER:
            vals = [fval(r, f"{window}_RMS_mean") for r in features if r["condition"] == cond]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                data.append(vals)
                conditions.append(cond)
        ax.boxplot(data, tick_labels=[label(c) for c in conditions], showmeans=True)
        ax.set_title(window.replace("_", " "))
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("Window RMS mean")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "windowed_mig_by_condition.png", dpi=180)
    plt.close(fig)


def scatter_joined(features: list[dict], targets: dict[str, dict], x_key: str, y_key: str, path: Path, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(6.4, 5.0))
    for cond in CONDITION_ORDER:
        xs = []
        ys = []
        for feat in features:
            if feat["condition"] != cond:
                continue
            target = targets.get(feat["sample_id"], {})
            x = fval(target, x_key)
            y = fval(feat, y_key)
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x)
                ys.append(y)
        if xs:
            plt.scatter(xs, ys, label=cond.replace("_", " "), alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_corr(features: list[dict], targets: dict[str, dict]) -> None:
    keys = [
        ("target_APD90_delta", "APD90_patch_delta", "target"),
        ("target_LAT_range", "LAT_range", "target"),
        ("target_patch_fraction", "patch_area_fraction", "target"),
        ("full_MI_mean", "full_MI_mean", "feature"),
        ("full_RMS_mean", "full_RMS_mean", "feature"),
        ("activation_RMS", "activation_like_RMS_mean", "feature"),
        ("repol_RMS", "repolarization_like_RMS_mean", "feature"),
        ("st_t_RMS", "st_t_like_RMS_mean", "feature"),
        ("st_t_MI", "st_t_like_MI_mean", "feature"),
    ]
    rows = []
    for feat in features:
        target = targets.get(feat["sample_id"], {})
        values = []
        for _label, key, source in keys:
            values.append(fval(target if source == "target" else feat, key))
        rows.append(values)
    data = np.asarray(rows, dtype=float)
    valid_cols = np.any(np.isfinite(data), axis=0)
    data = data[:, valid_cols]
    labels = [k[0] for k, keep in zip(keys, valid_cols) if keep]
    col_means = np.nanmean(data, axis=0)
    inds = np.where(~np.isfinite(data))
    data[inds] = np.take(col_means, inds[1])
    corr = np.corrcoef(data, rowvar=False)
    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_correlation_heatmap.png", dpi=180)
    plt.close(fig)


def plot_ep_targets(targets: list[dict]) -> None:
    keys = [("APD90_patch_delta", "APD90 patch delta"), ("LAT_range", "LAT range"), ("patch_area_fraction", "Patch fraction")]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))
    for ax, (key, title) in zip(axes, keys):
        data = []
        conditions = []
        for cond in CONDITION_ORDER:
            vals = [fval(r, key) for r in targets if r["condition"] == cond]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                data.append(vals)
                conditions.append(cond)
        ax.boxplot(data, tick_labels=[label(c) for c in conditions], showmeans=True)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ep_targets_by_condition.png", dpi=180)
    plt.close(fig)


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    features = read_csv(READY_BASE / "features_mig_windowed.csv")
    targets_rows = read_csv(READY_BASE / "targets_ep.csv")
    targets = by_sample(targets_rows)
    plot_windowed_by_condition(features)
    scatter_joined(features, targets, "APD90_patch_delta", "repolarization_like_RMS_mean", FIG_DIR / "mig_vs_apd90_patch_delta.png", "APD90 patch delta (ms)", "Repolarization-like RMS mean")
    scatter_joined(features, targets, "LAT_range", "activation_like_RMS_mean", FIG_DIR / "mig_vs_lat_range.png", "LAT range (ms)", "Activation-like RMS mean")
    plot_corr(features, targets)
    plot_ep_targets(targets_rows)
    print(f"Wrote figures to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
