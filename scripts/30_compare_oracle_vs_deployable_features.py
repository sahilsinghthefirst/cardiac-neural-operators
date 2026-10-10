                      
                                                                             

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
TARGETS = ["y_apd90_patch_delta", "y_lat_range", "y_condition"]
FEATURE_FILES = {
    "oracle": READY_BASE / "features_mig_oracle_windowed.csv",
    "deployable_fixed": READY_BASE / "features_mig_deployable_fixed.csv",
    "deployable_magnetic": READY_BASE / "features_mig_deployable_magnetic_windows.csv",
}


def read_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def fval(row: dict, key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan


def numeric_feature_keys(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    excluded = {"sample_id", "condition"}
    return [k for k in rows[0] if k not in excluded and not k.endswith("_start_ms") and not k.endswith("_end_ms") and not k.endswith("_n_timepoints")]


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return math.nan
    if np.std(xs) == 0 or np.std(ys) == 0:
        return math.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def correlations(feature_set: str, feature_rows: list[dict], targets_by_id: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for feature in numeric_feature_keys(feature_rows):
        for target in TARGETS:
            xs: list[float] = []
            ys: list[float] = []
            for row in feature_rows:
                target_row = targets_by_id.get(row["sample_id"], {})
                x = fval(row, feature)
                y = fval(target_row, target)
                if np.isfinite(x) and np.isfinite(y):
                    xs.append(x)
                    ys.append(y)
            r = pearson(xs, ys)
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature": feature,
                    "target": target,
                    "n": len(xs),
                    "pearson_r": r,
                    "abs_pearson_r": abs(r) if np.isfinite(r) else math.nan,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["feature_set", "feature", "target", "n", "pearson_r", "abs_pearson_r"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def best_by_target(rows: list[dict]) -> dict[tuple[str, str], dict]:
    best: dict[tuple[str, str], dict] = {}
    for row in rows:
        val = fval(row, "abs_pearson_r")
        if not np.isfinite(val):
            continue
        key = (row["feature_set"], row["target"])
        if key not in best or val > fval(best[key], "abs_pearson_r"):
            best[key] = row
    return best


def plot_heatmap(rows: list[dict]) -> None:
    best = best_by_target(rows)
    feature_sets = [k for k, p in FEATURE_FILES.items() if p.exists()]
    matrix = np.full((len(feature_sets), len(TARGETS)), np.nan)
    labels = [["" for _ in TARGETS] for _ in feature_sets]
    for i, fs in enumerate(feature_sets):
        for j, target in enumerate(TARGETS):
            row = best.get((fs, target))
            if row:
                matrix[i, j] = fval(row, "pearson_r")
                labels[i][j] = row["feature"]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(TARGETS)), TARGETS, rotation=20, ha="right")
    ax.set_yticks(range(len(feature_sets)), feature_sets)
    for i in range(len(feature_sets)):
        for j in range(len(TARGETS)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="best Pearson r")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "oracle_vs_deployable_correlation_heatmap.png", dpi=180)
    plt.close(fig)


def scatter(features: list[dict], targets: dict[str, dict], feature: str, target: str, path: Path, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(6.4, 5.0))
    for condition in sorted({r["condition"] for r in features}):
        xs = []
        ys = []
        for row in features:
            if row["condition"] != condition:
                continue
            x = fval(targets.get(row["sample_id"], {}), target)
            y = fval(row, feature)
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x)
                ys.append(y)
        if xs:
            plt.scatter(xs, ys, label=condition.replace("_", " "), alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    targets_rows = read_rows(READY_BASE / "targets_ep.csv")
    targets_by_id = {r["sample_id"]: r for r in targets_rows}
    all_corrs: list[dict] = []
    loaded: dict[str, list[dict]] = {}
    for feature_set, path in FEATURE_FILES.items():
        rows = read_rows(path)
        if not rows:
            continue
        loaded[feature_set] = rows
        all_corrs.extend(correlations(feature_set, rows, targets_by_id))
    write_csv(READY_BASE / "oracle_vs_deployable_feature_correlations.csv", all_corrs)
    plot_heatmap(all_corrs)
    fixed = loaded.get("deployable_fixed", [])
    if fixed:
        scatter(
            fixed,
            targets_by_id,
            "early_activation_fixed_RMS_mean",
            "y_lat_range",
            FIG_DIR / "deployable_mig_vs_lat_range.png",
            "LAT range (ms)",
            "Fixed early activation RMS mean",
        )
        scatter(
            fixed,
            targets_by_id,
            "late_repolarization_fixed_RMS_mean",
            "y_apd90_patch_delta",
            FIG_DIR / "deployable_mig_vs_apd90_delta.png",
            "APD90 patch delta (ms)",
            "Fixed late repolarization RMS mean",
        )
    print(f"Wrote {READY_BASE / 'oracle_vs_deployable_feature_correlations.csv'} with {len(all_corrs)} rows")
    return 0 if all_corrs else 1


if __name__ == "__main__":
    raise SystemExit(main())
