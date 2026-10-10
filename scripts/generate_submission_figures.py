                      
                                                                  

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.family": "cmr10",
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def r2(true: np.ndarray, pred: np.ndarray) -> float:
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    denominator = np.sum((true - np.mean(true)) ** 2)
    return float(1.0 - np.sum((true - pred) ** 2) / denominator)


def dice(true: np.ndarray, pred: np.ndarray) -> float:
    tp = np.sum(true & pred)
    denominator = np.sum(true) + np.sum(pred)
    return float(2 * tp / denominator) if denominator else 1.0


def read_audit(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {(row["scope"], row["model"]): row for row in csv.DictReader(handle)}


def headline_figure(audit_path: Path, output_dir: Path) -> None:
    rows = read_audit(audit_path)
    scopes = ("overall", "conduction_slowing_perturbation")
    scope_labels = ("All held-out", "Conduction slowing")
    models = ("FNO", "DeepONet")
    colors = ("#31688e", "#35b779")
    x = np.arange(len(scopes))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(6.55, 2.05))
    for model_index, (model, color) in enumerate(zip(models, colors)):
        offset = (model_index - 0.5) * width
        lat = [float(rows[(scope, model)]["LAT_global_R2"]) for scope in scopes]
        dice_values = [
            float(rows[(scope, model)]["patch_Dice_nonhomogeneous_micro"])
            for scope in scopes
        ]
        axes[0].bar(x + offset, lat, width, label=model, color=color)
        axes[1].bar(x + offset, dice_values, width, label=model, color=color)
        for axis_index, (axis, values) in enumerate(zip(axes, (lat, dice_values))):
            for xpos, value in zip(x + offset, values):
                if axis_index == 0:
                    axis.text(xpos, value - 0.035, f"{value:.3f}", ha="center", va="top", fontsize=7, color="white")
                else:
                    axis.text(xpos, value + 0.018, f"{value:.3f}", ha="center", va="bottom", fontsize=7)

    axes[0].set_title("Global LAT reconstruction")
    axes[0].set_ylabel("Pooled R2")
    axes[0].set_ylim(0, 1.08)
    axes[1].set_title("Abnormal-region localization")
    axes[1].set_ylabel("Micro Dice")
    axes[1].set_ylim(0, 1.08)
    for axis in axes:
        axis.set_xticks(x, scope_labels)
        axis.grid(axis="y", linewidth=0.4, alpha=0.35)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncols=2, fontsize=7, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(w_pad=1.4)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"headline_same_checkpoint.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def spatial_figure(analysis_dir: Path, output_dir: Path) -> str:
    with (analysis_dir / "paired_test_cohort.csv").open(newline="", encoding="utf-8") as handle:
        cohort = list(csv.DictReader(handle))
    with np.load(analysis_dir / "paired_test_cohort_maps.npz") as archive:
        maps = {name: archive[name] for name in archive.files}

    candidates = [
        index
        for index, row in enumerate(cohort)
        if row["condition"] == "conduction_slowing_perturbation"
    ]
    scores = []
    per_case = {}
    for index in candidates:
        valid = maps["valid_mask"][index].astype(bool)
        truth = (maps["true_patch"][index] > 0.5) & valid
        model_metrics = {}
        for model in ("fno", "deeponet"):
            lat_score = r2(
                maps["true_lat"][index][valid],
                maps[f"{model}_field_pred_lat"][index][valid],
            )
            predicted = (maps[f"{model}_field_pred_patch_prob"][index] >= 0.5) & valid
            model_metrics[model] = (lat_score, dice(truth, predicted))
        score = min(value[0] for value in model_metrics.values()) - np.mean(
            [value[1] for value in model_metrics.values()]
        )
        scores.append((float(score), index))
        per_case[index] = model_metrics
    _, selected = max(scores)

    valid = maps["valid_mask"][selected].astype(bool)
    truth = (maps["true_patch"][selected] > 0.5) & valid
    lat_arrays = (
        maps["true_lat"][selected],
        maps["fno_field_pred_lat"][selected],
        maps["deeponet_field_pred_lat"][selected],
    )
    probability_arrays = (
        truth.astype(float),
        maps["fno_field_pred_patch_prob"][selected],
        maps["deeponet_field_pred_patch_prob"][selected],
    )
    titles = (
        "True LAT",
        f"FNO LAT  R2={per_case[selected]['fno'][0]:.3f}",
        f"DeepONet LAT  R2={per_case[selected]['deeponet'][0]:.3f}",
        "True abnormal region",
        f"FNO probability  Dice={per_case[selected]['fno'][1]:.3f}",
        f"DeepONet probability  Dice={per_case[selected]['deeponet'][1]:.3f}",
    )

    finite_lat = np.concatenate([array[valid] for array in lat_arrays])
    lat_min, lat_max = np.percentile(finite_lat, [1, 99])
    fig = plt.figure(figsize=(6.55, 3.25), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=(1, 1, 1, 0.055))
    axes = np.asarray([[fig.add_subplot(grid[row, column]) for column in range(3)] for row in range(2)])
    color_axes = (fig.add_subplot(grid[0, 3]), fig.add_subplot(grid[1, 3]))
    for column, array in enumerate(lat_arrays):
        display = np.where(valid, array, np.nan)
        image = axes[0, column].imshow(display, cmap="turbo", vmin=lat_min, vmax=lat_max, origin="lower")
        axes[0, column].contour(truth.astype(float), levels=[0.5], colors="black", linewidths=0.8, origin="lower")
        axes[0, column].set_title(titles[column], fontsize=8)
    colorbar = fig.colorbar(image, cax=color_axes[0])
    colorbar.set_label("LAT in simulation time units", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)

    for column, array in enumerate(probability_arrays):
        display = np.where(valid, array, np.nan)
        image = axes[1, column].imshow(display, cmap="magma", vmin=0, vmax=1, origin="lower")
        axes[1, column].contour(truth.astype(float), levels=[0.5], colors="cyan", linewidths=0.8, origin="lower")
        if column > 0 and np.nanmin(display) < 0.5 < np.nanmax(display):
            axes[1, column].contour(display, levels=[0.5], colors="white", linestyles="--", linewidths=0.8, origin="lower")
        axes[1, column].set_title(titles[column + 3], fontsize=8)
    colorbar = fig.colorbar(image, cax=color_axes[1])
    colorbar.set_label("Region probability", fontsize=7)
    colorbar.ax.tick_params(labelsize=6)

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"same_checkpoint_spatial_example.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return cohort[selected]["base_sample_id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = args.results_dir.resolve() / "01_locked_analysis" / "icbinb_required_analysis"
    headline_figure(analysis / "same_checkpoint_audit.csv", output_dir)
    selected = spatial_figure(analysis, output_dir)
    print(f"Generated figures; deterministic spatial example: {selected}")


if __name__ == "__main__":
    main()
