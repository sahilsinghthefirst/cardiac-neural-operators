                      
                                                                        

from __future__ import annotations

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from magnetic_common import ROOT


MAG_DATASET = ROOT / "outputs" / "magnetic_dataset_v1b"
CONDITION_ORDER = [
    "homogeneous",
    "apd_repolarization_perturbation",
    "conduction_slowing_perturbation",
    "mixed_apd_conduction_perturbation",
]


def load_rows() -> list[dict]:
    rows = []
    for row in csv.DictReader((MAG_DATASET / "manifest.csv").open()):
        if str(row.get("success")).lower() != "true":
            continue
        out_dir = ROOT / row["magnetic_output_dir"]
        summary = json.loads((out_dir / "mig_summary.json").read_text())
        b_z = np.load(out_dir / "Bz.npy")
        row["summary"] = summary
        row["mean_abs_Bz"] = float(np.mean(np.abs(b_z)))
        row["max_abs_Bz"] = float(np.max(np.abs(b_z)))
        rows.append(row)
    return rows


def condition_labels(conditions: list[str]) -> list[str]:
    return [c.replace("_", "\n") for c in conditions]


def plot_example_bz(rows: list[dict], path) -> None:
    chosen = []
    for condition in CONDITION_ORDER:
        row = next((r for r in rows if r["condition"] == condition), None)
        if row:
            chosen.append(row)
    fig, axes = plt.subplots(1, len(chosen), figsize=(4.2 * len(chosen), 4.0), squeeze=False)
    for ax, row in zip(axes[0], chosen):
        out_dir = ROOT / row["magnetic_output_dir"]
        b_z = np.load(out_dir / "Bz.npy")
        sensors = np.load(out_dir / "sensor_positions.npy")
        tidx = int(np.argmax(np.sqrt(np.mean(b_z * b_z, axis=1))))
        sc = ax.scatter(sensors[:, 0] * 1000.0, sensors[:, 1] * 1000.0, c=b_z[tidx], s=90, cmap="coolwarm")
        ax.set_aspect("equal")
        ax.set_title(row["condition"].replace("_", " "), fontsize=9)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_box(rows: list[dict], key: str, ylabel: str, path) -> None:
    conditions = [c for c in CONDITION_ORDER if any(r["condition"] == c for r in rows)]
    data = []
    for cond in conditions:
        vals = []
        for row in rows:
            if row["condition"] != cond:
                continue
            if key in row:
                vals.append(float(row[key]))
            else:
                vals.append(float(row["summary"][key]))
        data.append(vals)
    plt.figure(figsize=(10.5, 4.8))
    plt.boxplot(data, tick_labels=condition_labels(conditions), showmeans=True)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_mig_feature_distributions(rows: list[dict], path) -> None:
    keys = [("mean_MI", "Mean MI"), ("mean_DI", "Mean DI"), ("mean_RMS", "Mean RMS")]
    conditions = [c for c in CONDITION_ORDER if any(r["condition"] == c for r in rows)]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6))
    for ax, (key, title) in zip(axes, keys):
        data = [[float(r["summary"][key]) for r in rows if r["condition"] == cond] for cond in conditions]
        ax.boxplot(data, tick_labels=condition_labels(conditions), showmeans=True)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    rows = load_rows()
    magnetic_fig_dir = ROOT / "figures" / "magnetic_dataset_v1b"
    mig_fig_dir = ROOT / "figures" / "mig_dataset_v1b"
    magnetic_fig_dir.mkdir(parents=True, exist_ok=True)
    mig_fig_dir.mkdir(parents=True, exist_ok=True)
    plot_example_bz(rows, magnetic_fig_dir / "example_Bz_maps_by_condition.png")
    plot_box(rows, "mean_abs_Bz", "Mean |Bz| proxy", magnetic_fig_dir / "bz_magnitude_by_condition.png")
    plot_box(rows, "mean_MI", "Mean MI", mig_fig_dir / "mi_by_condition.png")
    plot_box(rows, "mean_DI", "Mean DI", mig_fig_dir / "di_by_condition.png")
    plot_box(rows, "mean_RMS", "Mean RMS", mig_fig_dir / "rms_by_condition.png")
    plot_mig_feature_distributions(rows, mig_fig_dir / "mig_feature_distributions_by_condition.png")
    print(f"Wrote figures to {magnetic_fig_dir} and {mig_fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
