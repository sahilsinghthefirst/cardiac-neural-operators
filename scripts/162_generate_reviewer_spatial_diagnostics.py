                      
                                                                                  
 
                                                                                
                                                                              
                                                                            
   

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "outputs" / "icbinb_required_analysis"
PAPER_DIR = ROOT / "outputs" / "icbinb_paper_evidence_bundle"
MAPS_PATH = ANALYSIS_DIR / "paired_test_cohort_maps.npz"
COHORT_PATH = ANALYSIS_DIR / "paired_test_cohort.csv"
TABLE1_PATH = PAPER_DIR / "table1_overall_model_comparison.csv"
OUTPUT_DIR = ROOT / "outputs" / "icbinb_reviewer_spatial_diagnostics"
FIGURE_DIR = ROOT / "figures" / "icbinb_reviewer_spatial_diagnostics"

FIXED_THRESHOLD = 0.5
THRESHOLDS = np.round(np.linspace(0.05, 0.95, 37), 3)
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260902

MODEL_SPECS = {
    "fno": {
        "label": "FNO",
        "color": "#4472C4",
        "field_lat": "fno_field_pred_lat",
        "field_apd90": "fno_field_pred_apd90",
        "patch_prob": "fno_localization_pred_patch_prob",
        "saved_mask": "fno_localization_pred_patch_mask",
        "center_x": "fno_localization__pred_center_x_norm",
        "center_y": "fno_localization__pred_center_y_norm",
    },
    "deeponet": {
        "label": "DeepONet",
        "color": "#ED7D31",
        "field_lat": "deeponet_field_pred_lat",
        "field_apd90": "deeponet_field_pred_apd90",
        "patch_prob": "deeponet_localization_pred_patch_prob",
        "saved_mask": "deeponet_localization_pred_patch_mask",
        "center_x": "deeponet_localization__pred_center_x_norm",
        "center_y": "deeponet_localization__pred_center_y_norm",
    },
}

CONDITION_LABELS = {
    "apd_repolarization_perturbation": "APD perturbation",
    "conduction_slowing_perturbation": "Conduction slowing",
    "homogeneous": "Homogeneous",
    "mixed_apd_conduction_perturbation": "Mixed APD/conduction",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value: {value}")
    return result


def r2(true: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> float:
    y = np.asarray(true[valid], dtype=np.float64)
    p = np.asarray(pred[valid], dtype=np.float64)
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    if denominator <= 0:
        return float("nan")
    return 1.0 - float(np.sum((y - p) ** 2)) / denominator


def dice_counts(true: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> tuple[int, int, int, float]:
    y = np.asarray(true[valid] >= 0.5, dtype=bool)
    p = np.asarray(pred[valid], dtype=bool)
    tp = int(np.sum(y & p))
    fp = int(np.sum(~y & p))
    fn = int(np.sum(y & ~p))
    denominator = 2 * tp + fp + fn
    value = 1.0 if denominator == 0 else 2.0 * tp / denominator
    return tp, fp, fn, float(value)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def bootstrap_threshold_metrics(
    true_patch: np.ndarray,
    pred_prob: np.ndarray,
    valid: np.ndarray,
    nonhom_indices: np.ndarray,
    homogeneous_indices: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    tp_values: list[int] = []
    fp_values: list[int] = []
    fn_values: list[int] = []
    dice_values: list[float] = []
    area_values: list[float] = []
    nondegenerate_values: list[float] = []

    for index in nonhom_indices:
        predicted = pred_prob[index] >= threshold
        tp, fp, fn, dice = dice_counts(true_patch[index], predicted, valid[index])
        valid_count = int(np.sum(valid[index]))
        positive_count = int(np.sum(predicted & valid[index]))
        tp_values.append(tp)
        fp_values.append(fp)
        fn_values.append(fn)
        dice_values.append(dice)
        area_values.append(positive_count / valid_count)
        nondegenerate_values.append(float(0 < positive_count < valid_count))

    homogeneous_fp = []
    for index in homogeneous_indices:
        predicted = pred_prob[index] >= threshold
        homogeneous_fp.append(float(np.sum(predicted & valid[index]) / np.sum(valid[index])))

    tp_array = np.asarray(tp_values, dtype=np.int64)
    fp_array = np.asarray(fp_values, dtype=np.int64)
    fn_array = np.asarray(fn_values, dtype=np.int64)
    dice_array = np.asarray(dice_values, dtype=np.float64)
    area_array = np.asarray(area_values, dtype=np.float64)
    nondegenerate_array = np.asarray(nondegenerate_values, dtype=np.float64)
    homogeneous_fp_array = np.asarray(homogeneous_fp, dtype=np.float64)

    micro_denominator = 2 * np.sum(tp_array) + np.sum(fp_array) + np.sum(fn_array)
    micro_dice = 2 * np.sum(tp_array) / micro_denominator

    nonhom_draws = rng.integers(0, len(nonhom_indices), size=(BOOTSTRAP_REPLICATES, len(nonhom_indices)))
    sampled_tp = np.sum(tp_array[nonhom_draws], axis=1)
    sampled_fp = np.sum(fp_array[nonhom_draws], axis=1)
    sampled_fn = np.sum(fn_array[nonhom_draws], axis=1)
    sampled_denominator = 2 * sampled_tp + sampled_fp + sampled_fn
    bootstrap_micro = np.divide(
        2 * sampled_tp,
        sampled_denominator,
        out=np.ones_like(sampled_denominator, dtype=np.float64),
        where=sampled_denominator > 0,
    )
    bootstrap_macro = np.mean(dice_array[nonhom_draws], axis=1)
    bootstrap_area = np.mean(area_array[nonhom_draws], axis=1)
    bootstrap_nondegenerate = np.mean(nondegenerate_array[nonhom_draws], axis=1)

    homogeneous_draws = rng.integers(
        0,
        len(homogeneous_indices),
        size=(BOOTSTRAP_REPLICATES, len(homogeneous_indices)),
    )
    bootstrap_fp = np.mean(homogeneous_fp_array[homogeneous_draws], axis=1)

    micro_lower, micro_upper = percentile_interval(bootstrap_micro)
    macro_lower, macro_upper = percentile_interval(bootstrap_macro)
    fp_lower, fp_upper = percentile_interval(bootstrap_fp)
    area_lower, area_upper = percentile_interval(bootstrap_area)
    nondegenerate_lower, nondegenerate_upper = percentile_interval(bootstrap_nondegenerate)

    return {
        "threshold": float(threshold),
        "nonhomogeneous_base_cases": int(len(nonhom_indices)),
        "homogeneous_base_cases": int(len(homogeneous_indices)),
        "micro_dice": float(micro_dice),
        "micro_dice_ci_lower": micro_lower,
        "micro_dice_ci_upper": micro_upper,
        "mean_case_dice": float(np.mean(dice_array)),
        "mean_case_dice_ci_lower": macro_lower,
        "mean_case_dice_ci_upper": macro_upper,
        "homogeneous_false_positive_area": float(np.mean(homogeneous_fp_array)),
        "homogeneous_false_positive_area_ci_lower": fp_lower,
        "homogeneous_false_positive_area_ci_upper": fp_upper,
        "nondegenerate_support_rate": float(np.mean(nondegenerate_array)),
        "nondegenerate_support_rate_ci_lower": nondegenerate_lower,
        "nondegenerate_support_rate_ci_upper": nondegenerate_upper,
        "mean_predicted_area_nonhomogeneous": float(np.mean(area_array)),
        "mean_predicted_area_nonhomogeneous_ci_lower": area_lower,
        "mean_predicted_area_nonhomogeneous_ci_upper": area_upper,
    }


def metric_lookup(table1: list[dict[str, str]], metric: str) -> dict[str, str]:
    matches = [row for row in table1 if row["metric"] == metric]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Table 1 row for {metric}, found {len(matches)}")
    return matches[0]


def save_figure(fig: plt.Figure, stem: str) -> None:
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    png_path = FIGURE_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)


def spatial_examples_figure(
    maps: dict[str, np.ndarray],
    metrics: list[dict[str, object]],
    selections: list[dict[str, object]],
) -> None:
    metric_by_index = {int(row["cohort_index"]): row for row in metrics}
    selected_indices = [int(row["cohort_index"]) for row in selections]
    valid = maps["valid_mask"]

    lat_values = []
    apd_values = []
    for index in selected_indices:
        for key in ("true_lat", "fno_field_pred_lat", "deeponet_field_pred_lat"):
            lat_values.append(maps[key][index][valid[index]])
        for key in ("true_apd90", "fno_field_pred_apd90", "deeponet_field_pred_apd90"):
            apd_values.append(maps[key][index][valid[index]])
    lat_values_array = np.concatenate(lat_values)
    apd_values_array = np.concatenate(apd_values)
    lat_limits = (float(np.min(lat_values_array)), float(np.max(lat_values_array)))
    apd_limits = (float(np.min(apd_values_array)), float(np.max(apd_values_array)))

    fig = plt.figure(figsize=(14.6, 12.4))
    outer = fig.add_gridspec(2, 2, left=0.055, right=0.975, bottom=0.145, top=0.885, wspace=0.13, hspace=0.25)
    row_labels = ("LAT (ms)", "APD90 (ms)", "Patch mask")
    column_labels = ("Ground truth", "FNO", "DeepONet")
    role_labels = {
        "common_success": "Common localization success",
        "model_disagreement": "Largest model disagreement",
        "conduction_failure": "High-field / weak-mask failure",
        "homogeneous_false_positive": "Homogeneous false positive",
    }
    panel_letters = "ABCD"
    all_lat_axes = []
    all_apd_axes = []
    all_mask_axes = []

    for panel_number, selection in enumerate(selections):
        index = int(selection["cohort_index"])
        metric = metric_by_index[index]
        grid = outer[panel_number // 2, panel_number % 2].subgridspec(3, 3, wspace=0.04, hspace=0.06)
        axes = np.empty((3, 3), dtype=object)
        for row_index in range(3):
            for column_index in range(3):
                axes[row_index, column_index] = fig.add_subplot(grid[row_index, column_index])

        condition = CONDITION_LABELS[str(metric["condition"])]
        if float(metric["true_patch_present"]) >= 0.5:
            details = (
                f"FNO/Deep Dice {float(metric['fno_dice']):.2f}/{float(metric['deeponet_dice']):.2f}; "
                f"LAT R$^2$ {float(metric['fno_lat_r2_case']):.2f}/{float(metric['deeponet_lat_r2_case']):.2f}"
            )
        else:
            details = (
                f"FNO/Deep FP area {float(metric['fno_false_positive_area']):.3f}/"
                f"{float(metric['deeponet_false_positive_area']):.3f}"
            )
        block_bounds = outer[panel_number // 2, panel_number % 2].get_position(fig)
        fig.text(
            (block_bounds.x0 + block_bounds.x1) / 2,
            block_bounds.y1 + 0.017,
            f"{panel_letters[panel_number]}  {role_labels[str(selection['selection_role'])]}\n{condition}; {details}",
            fontsize=10.5,
            ha="center",
            va="bottom",
        )

        lat_maps = (
            maps["true_lat"][index],
            maps["fno_field_pred_lat"][index],
            maps["deeponet_field_pred_lat"][index],
        )
        apd_maps = (
            maps["true_apd90"][index],
            maps["fno_field_pred_apd90"][index],
            maps["deeponet_field_pred_apd90"][index],
        )
        patch_maps = (
            maps["true_patch"][index],
            maps["fno_localization_pred_patch_prob"][index],
            maps["deeponet_localization_pred_patch_prob"][index],
        )

        for column_index, data in enumerate(lat_maps):
            axes[0, column_index].imshow(
                np.ma.masked_where(~valid[index], data),
                origin="lower",
                extent=(0, 10, 0, 10),
                cmap="viridis",
                vmin=lat_limits[0],
                vmax=lat_limits[1],
                interpolation="nearest",
            )
            all_lat_axes.append(axes[0, column_index])
        for column_index, data in enumerate(apd_maps):
            axes[1, column_index].imshow(
                np.ma.masked_where(~valid[index], data),
                origin="lower",
                extent=(0, 10, 0, 10),
                cmap="plasma",
                vmin=apd_limits[0],
                vmax=apd_limits[1],
                interpolation="nearest",
            )
            all_apd_axes.append(axes[1, column_index])
        for column_index, data in enumerate(patch_maps):
            axes[2, column_index].imshow(
                np.ma.masked_where(~valid[index], data),
                origin="lower",
                extent=(0, 10, 0, 10),
                cmap="Blues",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            if column_index > 0:
                true_patch = maps["true_patch"][index]
                predicted_probability = patch_maps[column_index]
                if np.min(true_patch) < 0.5 < np.max(true_patch):
                    axes[2, column_index].contour(
                        true_patch,
                        levels=[0.5],
                        colors="black",
                        linewidths=1.1,
                        origin="lower",
                        extent=(0, 10, 0, 10),
                    )
                if np.min(predicted_probability) < FIXED_THRESHOLD < np.max(predicted_probability):
                    axes[2, column_index].contour(
                        predicted_probability,
                        levels=[FIXED_THRESHOLD],
                        colors="white",
                        linewidths=1.1,
                        linestyles="--",
                        origin="lower",
                        extent=(0, 10, 0, 10),
                    )
            all_mask_axes.append(axes[2, column_index])

        for row_index in range(3):
            for column_index in range(3):
                axis = axes[row_index, column_index]
                if row_index == 0:
                    axis.set_title(column_labels[column_index], fontsize=9.5, pad=2)
                if column_index == 0:
                    axis.set_ylabel(row_labels[row_index], fontsize=9)
                else:
                    axis.set_yticklabels([])
                if row_index == 2:
                    axis.set_xlabel("x (mm)", fontsize=8.5)
                else:
                    axis.set_xticklabels([])
                axis.tick_params(labelsize=7, length=2.5)
                axis.set_xlim(0, 10)
                axis.set_ylim(0, 10)

    lat_norm = mpl.colors.Normalize(vmin=lat_limits[0], vmax=lat_limits[1])
    apd_norm = mpl.colors.Normalize(vmin=apd_limits[0], vmax=apd_limits[1])
    probability_norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cax_lat = fig.add_axes((0.07, 0.052, 0.25, 0.014))
    cax_apd = fig.add_axes((0.375, 0.052, 0.25, 0.014))
    cax_mask = fig.add_axes((0.68, 0.052, 0.25, 0.014))
    fig.colorbar(mpl.cm.ScalarMappable(norm=lat_norm, cmap="viridis"), cax=cax_lat, orientation="horizontal").set_label("LAT (ms)", fontsize=9)
    fig.colorbar(mpl.cm.ScalarMappable(norm=apd_norm, cmap="plasma"), cax=cax_apd, orientation="horizontal").set_label("APD90 (ms)", fontsize=9)
    fig.colorbar(mpl.cm.ScalarMappable(norm=probability_norm, cmap="Blues"), cax=cax_mask, orientation="horizontal").set_label("Patch probability", fontsize=9)
    for colorbar_axis in (cax_lat, cax_apd, cax_mask):
        colorbar_axis.tick_params(labelsize=7)
        colorbar_axis.xaxis.set_label_position("top")

    fig.suptitle("Held-out spatial examples: accurate fields can coexist with weak localization", fontsize=16, y=0.985)
    fig.text(
        0.5,
        0.008,
        "Mask panels: black solid = true support; white dashed = predicted 0.5 contour. Field and localization maps use separately validation-selected checkpoints.",
        ha="center",
        fontsize=9,
    )
    save_figure(fig, "reviewer_spatial_examples")


def casewise_metrics_figure(metrics: list[dict[str, object]]) -> None:
    nonhomogeneous = [row for row in metrics if float(row["true_patch_present"]) >= 0.5]
    homogeneous = [row for row in metrics if float(row["true_patch_present"]) < 0.5]

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.7))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.91, wspace=0.26, hspace=0.30)

    condition_markers = {
        "apd_repolarization_perturbation": "o",
        "conduction_slowing_perturbation": "X",
        "mixed_apd_conduction_perturbation": "s",
    }
    for model, spec in MODEL_SPECS.items():
        for condition, marker in condition_markers.items():
            selected = [row for row in nonhomogeneous if row["condition"] == condition]
            axes[0, 0].scatter(
                [float(row[f"{model}_lat_r2_case"]) for row in selected],
                [float(row[f"{model}_dice"]) for row in selected],
                s=31,
                marker=marker,
                facecolor=spec["color"],
                edgecolor="black",
                linewidth=0.35,
                alpha=0.78,
            )
            axes[0, 1].scatter(
                [float(row[f"{model}_apd90_r2_case"]) for row in selected],
                [float(row[f"{model}_dice"]) for row in selected],
                s=31,
                marker=marker,
                facecolor=spec["color"],
                edgecolor="black",
                linewidth=0.35,
                alpha=0.78,
            )
        axes[1, 0].scatter(
            [float(row[f"{model}_center_error"]) for row in nonhomogeneous],
            [float(row[f"{model}_dice"]) for row in nonhomogeneous],
            s=31,
            marker="o" if model == "fno" else "^",
            facecolor=spec["color"],
            edgecolor="black",
            linewidth=0.35,
            alpha=0.78,
            label=spec["label"],
        )

    for axis, xlabel in (
        (axes[0, 0], "Per-case LAT $R^2$"),
        (axes[0, 1], "Per-case APD90 $R^2$"),
    ):
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Patch Dice at threshold 0.5")
        axis.axhline(0.5, color="0.55", linewidth=0.8, linestyle=":")
        axis.grid(alpha=0.20, linewidth=0.6)
    axes[0, 0].set_title("A  Dice versus LAT field accuracy")
    axes[0, 1].set_title("B  Dice versus APD90 field accuracy")

    axes[1, 0].set_title("C  Center error versus Dice")
    axes[1, 0].set_xlabel("Patch-center error (normalized sheet units)")
    axes[1, 0].set_ylabel("Patch Dice at threshold 0.5")
    axes[1, 0].grid(alpha=0.20, linewidth=0.6)

    homogeneous_sorted = sorted(homogeneous, key=lambda row: str(row["base_sample_id"]))
    x_positions = np.arange(1, len(homogeneous_sorted) + 1)
    fno_values = np.asarray([float(row["fno_false_positive_area"]) for row in homogeneous_sorted])
    deep_values = np.asarray([float(row["deeponet_false_positive_area"]) for row in homogeneous_sorted])
    for x, fno_value, deep_value in zip(x_positions, fno_values, deep_values):
        axes[1, 1].plot([x, x], [fno_value, deep_value], color="0.72", linewidth=0.8, zorder=1)
    axes[1, 1].scatter(x_positions, fno_values, s=30, color=MODEL_SPECS["fno"]["color"], marker="o", edgecolor="black", linewidth=0.35, label="FNO", zorder=2)
    axes[1, 1].scatter(x_positions, deep_values, s=34, color=MODEL_SPECS["deeponet"]["color"], marker="^", edgecolor="black", linewidth=0.35, label="DeepONet", zorder=2)
    axes[1, 1].set_title("D  False-positive area in homogeneous cases")
    axes[1, 1].set_xlabel("Held-out homogeneous case (n = 12)")
    axes[1, 1].set_ylabel("Predicted positive area fraction")
    axes[1, 1].set_xticks(x_positions)
    axes[1, 1].set_xticklabels([str(index) for index in x_positions], fontsize=8)
    axes[1, 1].grid(axis="y", alpha=0.20, linewidth=0.6)

    model_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=MODEL_SPECS["fno"]["color"], markeredgecolor="black", markersize=6, label="FNO"),
        Line2D([0], [0], marker="^", linestyle="none", markerfacecolor=MODEL_SPECS["deeponet"]["color"], markeredgecolor="black", markersize=6, label="DeepONet"),
    ]
    condition_handles = [
        Line2D([0], [0], marker=marker, linestyle="none", color="0.25", markersize=6, label=CONDITION_LABELS[condition])
        for condition, marker in condition_markers.items()
    ]
    fig.legend(
        handles=model_handles + condition_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.963),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("Case-level diagnostics reveal localization failures hidden by field scores", fontsize=15, y=0.995)
    fig.text(
        0.5,
        0.018,
        "Per-case R2 is diagnostic and is not the pooled-pixel headline R2. Every point is one independent held-out base case.",
        ha="center",
        fontsize=9,
    )
    save_figure(fig, "reviewer_casewise_metrics")


def threshold_sweep_figure(rows: list[dict[str, object]], mean_true_area: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.3), sharex=True)
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.10, top=0.91, wspace=0.27, hspace=0.28)

    panels = (
        ("micro_dice", "Non-homogeneous micro-Dice", "A  Localization overlap"),
        ("homogeneous_false_positive_area", "Mean false-positive area", "B  Homogeneous false positives"),
        ("nondegenerate_support_rate", "Nondegenerate support rate", "C  Support availability"),
        ("mean_predicted_area_nonhomogeneous", "Mean predicted area fraction", "D  Predicted support size"),
    )
    for axis, (metric, ylabel, title) in zip(axes.ravel(), panels):
        for model, spec in MODEL_SPECS.items():
            selected = sorted((row for row in rows if row["model"] == model), key=lambda row: float(row["threshold"]))
            thresholds = np.asarray([float(row["threshold"]) for row in selected])
            estimates = np.asarray([float(row[metric]) for row in selected])
            lower = np.asarray([float(row[f"{metric}_ci_lower"]) for row in selected])
            upper = np.asarray([float(row[f"{metric}_ci_upper"]) for row in selected])
            axis.plot(thresholds, estimates, color=spec["color"], linewidth=1.8, label=spec["label"])
            axis.fill_between(thresholds, lower, upper, color=spec["color"], alpha=0.16, linewidth=0)
        axis.axvline(FIXED_THRESHOLD, color="0.25", linestyle="--", linewidth=1.0)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.20, linewidth=0.6)
        axis.set_xlim(THRESHOLDS[0], THRESHOLDS[-1])
    axes[1, 0].set_xlabel("Mask-probability threshold")
    axes[1, 1].set_xlabel("Mask-probability threshold")
    axes[1, 1].axhline(mean_true_area, color="0.25", linestyle=":", linewidth=1.0, label="Mean true area")
    axes[0, 0].legend(loc="best", frameon=False)
    axes[1, 1].legend(loc="best", frameon=False)
    fig.suptitle("Descriptive held-out threshold sensitivity", fontsize=15, y=0.985)
    fig.text(
        0.5,
        0.022,
        "Bands are 95% base-sample bootstrap intervals (5,000 replicates). The sweep is descriptive and was not used to recalibrate the fixed 0.5 extractor.",
        ha="center",
        fontsize=9,
    )
    save_figure(fig, "reviewer_threshold_sweep")


def main() -> None:
    if OUTPUT_DIR.exists() or FIGURE_DIR.exists():
        raise SystemExit(
            f"Refusing to overwrite existing reviewer outputs: {OUTPUT_DIR} or {FIGURE_DIR}"
        )
    OUTPUT_DIR.mkdir(parents=True)
    FIGURE_DIR.mkdir(parents=True)
    configure_matplotlib()

    cohort = read_csv(COHORT_PATH)
    table1 = read_csv(TABLE1_PATH)
    if len(cohort) != 75:
        raise RuntimeError(f"Expected 75 paired held-out rows, found {len(cohort)}")
    if [int(row["cohort_index"]) for row in cohort] != list(range(75)):
        raise RuntimeError("Cohort rows are not ordered by cohort_index 0..74")

    with np.load(MAPS_PATH, allow_pickle=False) as archive:
        maps = {key: np.asarray(archive[key]) for key in archive.files}
    for key, value in maps.items():
        if value.shape[0] != 75:
            raise RuntimeError(f"Map array {key} has unexpected first dimension {value.shape}")

    valid = maps["valid_mask"].astype(bool)
    true_patch = maps["true_patch"]
    metrics: list[dict[str, object]] = []
    for index, cohort_row in enumerate(cohort):
        metric_row: dict[str, object] = {
            "cohort_index": index,
            "base_sample_id": cohort_row["base_sample_id"],
            "condition": cohort_row["condition"],
            "combined_variant": cohort_row["combined_variant"],
            "true_patch_present": finite_float(cohort_row["true_patch_present"]),
            "true_patch_area_fraction": finite_float(cohort_row["true_patch_area_fraction"]),
        }
        for model, spec in MODEL_SPECS.items():
            predicted_mask = maps[spec["patch_prob"]][index] >= FIXED_THRESHOLD
            if not np.array_equal(predicted_mask, maps[spec["saved_mask"]][index]):
                raise RuntimeError(f"Saved {model} mask differs from probability >= 0.5 at index {index}")
            metric_row[f"{model}_lat_r2_case"] = r2(
                maps["true_lat"][index], maps[spec["field_lat"]][index], valid[index]
            )
            metric_row[f"{model}_apd90_r2_case"] = r2(
                maps["true_apd90"][index], maps[spec["field_apd90"]][index], valid[index]
            )
            if metric_row["true_patch_present"] >= 0.5:
                _, _, _, case_dice = dice_counts(
                    true_patch[index], predicted_mask, valid[index]
                )
                center_dx = finite_float(cohort_row[spec["center_x"]]) - finite_float(
                    cohort_row["true_center_x_norm"]
                )
                center_dy = finite_float(cohort_row[spec["center_y"]]) - finite_float(
                    cohort_row["true_center_y_norm"]
                )
                metric_row[f"{model}_dice"] = case_dice
                metric_row[f"{model}_center_error"] = math.hypot(center_dx, center_dy)
                metric_row[f"{model}_false_positive_area"] = ""
            else:
                metric_row[f"{model}_dice"] = ""
                metric_row[f"{model}_center_error"] = ""
                metric_row[f"{model}_false_positive_area"] = float(
                    np.sum(predicted_mask & valid[index]) / np.sum(valid[index])
                )
        metrics.append(metric_row)

    metric_fields = [
        "cohort_index",
        "base_sample_id",
        "condition",
        "combined_variant",
        "true_patch_present",
        "true_patch_area_fraction",
        "fno_lat_r2_case",
        "fno_apd90_r2_case",
        "fno_dice",
        "fno_center_error",
        "fno_false_positive_area",
        "deeponet_lat_r2_case",
        "deeponet_apd90_r2_case",
        "deeponet_dice",
        "deeponet_center_error",
        "deeponet_false_positive_area",
    ]
    write_csv(OUTPUT_DIR / "reviewer_case_metrics.csv", metric_fields, metrics)

    nonhomogeneous = [row for row in metrics if float(row["true_patch_present"]) >= 0.5]
    homogeneous = [row for row in metrics if float(row["true_patch_present"]) < 0.5]
    common_success = max(
        nonhomogeneous,
        key=lambda row: min(float(row["fno_dice"]), float(row["deeponet_dice"])),
    )
    disagreement_pool = [row for row in nonhomogeneous if row is not common_success]
    model_disagreement = max(
        disagreement_pool,
        key=lambda row: abs(float(row["fno_dice"]) - float(row["deeponet_dice"])),
    )
    conduction_pool = [
        row
        for row in nonhomogeneous
        if row["condition"] == "conduction_slowing_perturbation"
        and row is not common_success
        and row is not model_disagreement
    ]
    conduction_failure = max(
        conduction_pool,
        key=lambda row: (
            float(row["fno_lat_r2_case"])
            + float(row["fno_apd90_r2_case"])
            + float(row["deeponet_lat_r2_case"])
            + float(row["deeponet_apd90_r2_case"])
        )
        / 4
        - (float(row["fno_dice"]) + float(row["deeponet_dice"])) / 2,
    )
    homogeneous_false_positive = max(
        homogeneous,
        key=lambda row: max(
            float(row["fno_false_positive_area"]),
            float(row["deeponet_false_positive_area"]),
        ),
    )

    selection_specs = (
        (
            "common_success",
            common_success,
            "Maximum of the lower FNO/DeepONet case Dice among all 63 non-homogeneous cases.",
        ),
        (
            "model_disagreement",
            model_disagreement,
            "Largest absolute FNO-versus-DeepONet case-Dice difference after excluding the common-success case.",
        ),
        (
            "conduction_failure",
            conduction_failure,
            "Maximum mean of four per-case field R2 values minus mean model Dice among remaining conduction-only cases.",
        ),
        (
            "homogeneous_false_positive",
            homogeneous_false_positive,
            "Largest model false-positive area among the 12 homogeneous cases.",
        ),
    )
    selections: list[dict[str, object]] = []
    for role, row, rule in selection_specs:
        selections.append(
            {
                "selection_role": role,
                "selection_rule": rule,
                **{field: row.get(field, "") for field in metric_fields},
            }
        )
    write_csv(
        OUTPUT_DIR / "reviewer_case_selections.csv",
        ["selection_role", "selection_rule"] + metric_fields,
        selections,
    )

    nonhomogeneous_indices = np.asarray(
        [index for index, row in enumerate(cohort) if finite_float(row["true_patch_present"]) >= 0.5],
        dtype=np.int64,
    )
    homogeneous_indices = np.asarray(
        [index for index, row in enumerate(cohort) if finite_float(row["true_patch_present"]) < 0.5],
        dtype=np.int64,
    )
    if len(nonhomogeneous_indices) != 63 or len(homogeneous_indices) != 12:
        raise RuntimeError("Unexpected non-homogeneous/homogeneous case counts")

    threshold_rows: list[dict[str, object]] = []
    for model_index, (model, spec) in enumerate(MODEL_SPECS.items()):
        for threshold_index, threshold in enumerate(THRESHOLDS):
            rng = np.random.default_rng(
                BOOTSTRAP_SEED + model_index * 1000 + threshold_index
            )
            result = bootstrap_threshold_metrics(
                true_patch,
                maps[spec["patch_prob"]],
                valid,
                nonhomogeneous_indices,
                homogeneous_indices,
                float(threshold),
                rng,
            )
            threshold_rows.append(
                {
                    "model": model,
                    "model_label": spec["label"],
                    "fixed_legacy_threshold": FIXED_THRESHOLD,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_unit": "base_sample_id",
                    "descriptive_test_sweep_not_calibration": True,
                    **result,
                }
            )
    threshold_fields = list(threshold_rows[0].keys())
    write_csv(OUTPUT_DIR / "reviewer_threshold_sweep.csv", threshold_fields, threshold_rows)

                                                                            
    lat_row = metric_lookup(table1, "LAT_global_R2")
    apd_row = metric_lookup(table1, "APD90_global_R2")
    dice_row = metric_lookup(table1, "patch_Dice_nonhomogeneous_micro")
    center_row = metric_lookup(table1, "patch_center_error_mean")
    fp_row = metric_lookup(table1, "homogeneous_false_positive_area_mean")
    checks: dict[str, dict[str, float]] = {}
    for model, spec in MODEL_SPECS.items():
        pooled_lat = r2(maps["true_lat"], maps[spec["field_lat"]], valid)
        pooled_apd = r2(maps["true_apd90"], maps[spec["field_apd90"]], valid)
        predicted_masks = maps[spec["patch_prob"]] >= FIXED_THRESHOLD
        pooled_tp = pooled_fp = pooled_fn = 0
        for index in nonhomogeneous_indices:
            tp, fp, fn, _ = dice_counts(true_patch[index], predicted_masks[index], valid[index])
            pooled_tp += tp
            pooled_fp += fp
            pooled_fn += fn
        pooled_dice = 2 * pooled_tp / (2 * pooled_tp + pooled_fp + pooled_fn)
        mean_center = float(
            np.mean([float(row[f"{model}_center_error"]) for row in nonhomogeneous])
        )
        mean_fp = float(
            np.mean([float(row[f"{model}_false_positive_area"]) for row in homogeneous])
        )
        expected = {
            "LAT_global_R2": finite_float(lat_row[f"{model}_estimate"]),
            "APD90_global_R2": finite_float(apd_row[f"{model}_estimate"]),
            "patch_Dice_nonhomogeneous_micro": finite_float(dice_row[f"{model}_estimate"]),
            "patch_center_error_mean": finite_float(center_row[f"{model}_estimate"]),
            "homogeneous_false_positive_area_mean": finite_float(fp_row[f"{model}_estimate"]),
        }
        observed = {
            "LAT_global_R2": pooled_lat,
            "APD90_global_R2": pooled_apd,
            "patch_Dice_nonhomogeneous_micro": pooled_dice,
            "patch_center_error_mean": mean_center,
            "homogeneous_false_positive_area_mean": mean_fp,
        }
        for metric, value in observed.items():
            if not np.isclose(value, expected[metric], rtol=0, atol=3e-7):
                raise RuntimeError(
                    f"Locked metric mismatch for {model} {metric}: observed={value}, expected={expected[metric]}"
                )
        checks[model] = observed

    fixed_rows = {
        row["model"]: row
        for row in threshold_rows
        if np.isclose(float(row["threshold"]), FIXED_THRESHOLD)
    }
    for model in MODEL_SPECS:
        if not np.isclose(
            float(fixed_rows[model]["micro_dice"]),
            checks[model]["patch_Dice_nonhomogeneous_micro"],
            rtol=0,
            atol=1e-12,
        ):
            raise RuntimeError(f"Threshold sweep fixed-point Dice mismatch for {model}")
        if not np.isclose(
            float(fixed_rows[model]["homogeneous_false_positive_area"]),
            checks[model]["homogeneous_false_positive_area_mean"],
            rtol=0,
            atol=1e-12,
        ):
            raise RuntimeError(f"Threshold sweep fixed-point FP mismatch for {model}")

    spatial_examples_figure(maps, metrics, selections)
    casewise_metrics_figure(metrics)
    mean_true_area = float(
        np.mean([float(row["true_patch_area_fraction"]) for row in nonhomogeneous])
    )
    threshold_sweep_figure(threshold_rows, mean_true_area)

    captions = """# Reviewer-requested diagnostic captions

## Spatial examples

Held-out spatial examples selected by deterministic diagnostic rules rather than manual visual preference. Each case shows ground-truth, FNO, and DeepONet LAT, APD90, and patch maps. Black solid contours mark the true support and white dashed contours mark the predicted 0.5 support. Field and localization maps use separately validation-selected checkpoints.

## Case-level metrics

Case-level relationships between patch Dice and per-case LAT/APD90 R2, center error, and homogeneous false-positive area. Each point is one independent held-out base case. Per-case R2 is diagnostic and is not the pooled-pixel headline R2.

## Threshold sweep

Descriptive held-out sensitivity of mask overlap, homogeneous false-positive area, support availability, and predicted support size to thresholds from 0.05 to 0.95. Bands are 95% base-sample bootstrap intervals from 5,000 replicates. The sweep was not used to recalibrate the fixed 0.5 extractor or select a test-optimal threshold.
"""
    (OUTPUT_DIR / "CAPTIONS.md").write_text(captions, encoding="utf-8")

    source_paths = [MAPS_PATH, COHORT_PATH, TABLE1_PATH]
    source_manifest = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in source_paths
    ]
    (OUTPUT_DIR / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": "1.0.0",
        "analysis_label": "reviewer-requested spatial and threshold diagnostics",
        "created_from_saved_predictions_only": True,
        "training_performed": False,
        "inference_performed": False,
        "simulation_performed": False,
        "test_base_cases": 75,
        "nonhomogeneous_base_cases": 63,
        "homogeneous_base_cases": 12,
        "fixed_threshold": FIXED_THRESHOLD,
        "threshold_sweep": {
            "minimum": float(THRESHOLDS[0]),
            "maximum": float(THRESHOLDS[-1]),
            "count": int(len(THRESHOLDS)),
            "descriptive_test_sweep_not_calibration": True,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_unit": "base_sample_id",
            "bootstrap_seed_base": BOOTSTRAP_SEED,
        },
        "spatial_case_selection": [
            {
                "selection_role": row["selection_role"],
                "selection_rule": row["selection_rule"],
                "cohort_index": row["cohort_index"],
                "base_sample_id": row["base_sample_id"],
                "condition": row["condition"],
            }
            for row in selections
        ],
        "locked_metric_reproduction": checks,
        "claim_boundaries": [
            "controlled synthetic fixed-sheet benchmark",
            "not patient-specific",
            "not clinically validated",
            "not real-SQUID validated",
            "not physics-informed training",
        ],
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    output_files = sorted(path for path in OUTPUT_DIR.rglob("*") if path.is_file())
    figure_files = sorted(path for path in FIGURE_DIR.rglob("*") if path.is_file())
    validation = {
        "passed": True,
        "source_hashes_recorded": True,
        "locked_metric_reproduction_passed": True,
        "fixed_threshold_sweep_parity_passed": True,
        "case_metric_rows": len(metrics),
        "selected_spatial_cases": len(selections),
        "threshold_rows": len(threshold_rows),
        "pdf_count": sum(path.suffix == ".pdf" for path in figure_files),
        "png_count": sum(path.suffix == ".png" for path in figure_files),
        "output_files_before_validation": [str(path.relative_to(ROOT)) for path in output_files],
        "figure_files": [str(path.relative_to(ROOT)) for path in figure_files],
    }
    (OUTPUT_DIR / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({"output_dir": str(OUTPUT_DIR), "figure_dir": str(FIGURE_DIR), **validation}, indent=2))


if __name__ == "__main__":
    main()
