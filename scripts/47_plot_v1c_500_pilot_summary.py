                      
                                                        

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bocf_v1c_common import FULL_MODE, PILOT_MODE, mode_paths, read_csv_rows


def floats(rows: list[dict], key: str) -> np.ndarray:
    vals = []
    for row in rows:
        try:
            vals.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            pass
    return np.asarray(vals, dtype=float)


def by_condition_box(rows: list[dict], key: str, title: str, ylabel: str, path: Path) -> None:
    conditions = sorted({r["condition"] for r in rows})
    data = []
    labels = []
    for condition in conditions:
        vals = floats([r for r in rows if r["condition"] == condition], key)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            data.append(vals)
            labels.append(condition)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if data:
        ax.boxplot(data, labels=[c.replace("_", "\n") for c in labels], showfliers=False)
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def hist_by_condition(rows: list[dict], key: str, title: str, xlabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for condition in sorted({r["condition"] for r in rows}):
        vals = floats([r for r in rows if r["condition"] == condition], key)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ax.hist(vals, bins=24, alpha=0.45, label=condition)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def scatter_patch(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for condition in sorted({r["condition"] for r in rows if r["condition"] != "homogeneous"}):
        sub = [r for r in rows if r["condition"] == condition]
        x = floats(sub, "patch_center_x") / 1000.0
        y = floats(sub, "patch_center_y") / 1000.0
        ax.scatter(x, y, s=12, alpha=0.7, label=condition)
    ax.set_title("Patch locations")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def augmentation_shift(rows: list[dict], path: Path) -> None:
    clean = {r["base_sample_id"]: float(r["full_RMS_mean"]) for r in rows if r.get("variant") == "clean"}
    variants = sorted({r["variant"] for r in rows})
    data = []
    labels = []
    for variant in variants:
        vals = []
        for row in rows:
            if row["variant"] != variant or row["base_sample_id"] not in clean:
                continue
            base = clean[row["base_sample_id"]]
            if base != 0:
                vals.append((float(row["full_RMS_mean"]) - base) / abs(base))
        if vals:
            data.append(vals)
            labels.append(variant)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if data:
        ax.boxplot(data, labels=[l.replace("_", "\n") for l in labels], showfliers=False)
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Relative full-window RMS feature shift by variant")
    ax.set_ylabel("relative shift")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def runtime_storage(paths: dict[str, Path], path: Path) -> None:
    labels = []
    runtime = []
    storage = []
    for name, base in [("EP", paths["ep"]), ("Magnetic", paths["mag"]), ("Augmented", paths["aug"])]:
        summary = base / "run_summary.json"
        if not summary.exists():
            continue
        data = json.loads(summary.read_text())
        labels.append(name)
        runtime.append(float(data.get("runtime_seconds") or data.get("total_runtime_seconds") or 0.0) / 3600.0)
        store = data.get("storage", {})
        storage.append(float(store.get("storage_gb", data.get("storage_used_gb", 0.0))))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(labels, runtime, color="#4477aa")
    axes[0].set_title("Runtime")
    axes[0].set_ylabel("hours")
    axes[1].bar(labels, storage, color="#66aa77")
    axes[1].set_title("Storage")
    axes[1].set_ylabel("GB")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[PILOT_MODE, FULL_MODE], default=PILOT_MODE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = mode_paths(args.mode)
    fig_dir = paths["figures"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    targets = read_csv_rows(paths["ready"] / "targets_ep.csv")
    index = read_csv_rows(paths["ready"] / "sample_index.csv")
    features = read_csv_rows(paths["ready"] / "features_mig_deployable_fixed.csv")
    aug_features = read_csv_rows(paths["aug_ready"] / "features_mig_deployable_fixed_augmented.csv")
    by_condition_box(targets, "APD90_mean", "APD90 distribution by condition", "APD90 mean (ms)", fig_dir / "apd_distribution_by_condition.png")
    by_condition_box(targets, "LAT_range", "LAT range by condition", "LAT range (ms)", fig_dir / "lat_range_by_condition.png")
    scatter_patch(index, fig_dir / "patch_locations.png")
    hist_by_condition(index, "patch_radius_mm", "Patch radius distribution", "radius (mm)", fig_dir / "patch_radius_distribution.png")
    hist_by_condition(index, "apd_scale", "tau_si scale distribution", "tau_si scale", fig_dir / "tau_si_scale_distribution.png")
    hist_by_condition(index, "conductivity_scale", "Conductivity scale distribution", "regional scale", fig_dir / "conductivity_scale_distribution.png")
    by_condition_box(features, "full_RMS_mean", "Bz/MIG RMS magnitude by condition", "full RMS mean", fig_dir / "bz_magnitude_by_condition.png")
    by_condition_box(features, "early_activation_fixed_RMS_auc", "Deployable MIG feature distributions", "early RMS AUC", fig_dir / "mig_feature_distributions.png")
    augmentation_shift(aug_features, fig_dir / "augmentation_feature_shift_by_variant.png")
    runtime_storage(paths, fig_dir / "runtime_storage_summary.png")
    print(f"Wrote v1c summary figures to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
