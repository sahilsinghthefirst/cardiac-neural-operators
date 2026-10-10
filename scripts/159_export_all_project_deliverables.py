                      
                                                                             

                                                                            
                                                                              
                                                                            
                                                                            
                                                       

                                                                               
   

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


NEW_ROOT = Path(__file__).resolve().parents[1]
OLD_WINDOWS_ROOT = NEW_ROOT.parent / "bpc_fno_v2"
OLD_WSL_ROOT = NEW_ROOT.parent / "bpc_fno_v2"
DEST = Path(
    str(Path(__file__).resolve().parents[1] / "exports") + "/"
    "BPC_FNO_DeepONet_All_Deliverables_2026-09-01"
)

MANIFEST_DIR = Path("00_README_AND_MANIFESTS")
OLD_DEST = Path("01_OLDER_BPC_FNO_VOLUMETRIC")
NEW_DEST = Path("02_NEWER_FNO_DEEPONET_BENCHMARK")

COPY_CHUNK_BYTES = 8 * 1024 * 1024

OLD_EVALUATIONS = [
    (
        "4k baseline",
        OLD_WSL_ROOT / "results/monodomain_4k_20260405/evaluation_results.json",
    ),
    (
        "Recovery 1 - best loss",
        OLD_WSL_ROOT
        / "results/architecture_a_recovery_exp1_20260406/best_loss/evaluation_results.json",
    ),
    (
        "Recovery 1 - best recon",
        OLD_WSL_ROOT
        / "results/architecture_a_recovery_exp1_20260406/best_recon/evaluation_results.json",
    ),
    (
        "Recovery 2 - best loss",
        OLD_WSL_ROOT
        / "results/architecture_a_recovery_exp2_20260408/best_loss/evaluation_results.json",
    ),
    (
        "Recovery 2 - best recon",
        OLD_WSL_ROOT
        / "results/architecture_a_recovery_exp2_20260408/best_recon/evaluation_results.json",
    ),
]

OLD_SELECTED_CHECKPOINTS = [
    OLD_WSL_ROOT / "checkpoints/monodomain_4k_20260405/phase2_best.pt",
    OLD_WSL_ROOT
    / "checkpoints/architecture_a_recovery_exp1_20260406/phase2_best_loss.pt",
    OLD_WSL_ROOT
    / "checkpoints/architecture_a_recovery_exp1_20260406/phase2_best_recon.pt",
    OLD_WSL_ROOT
    / "checkpoints/architecture_a_recovery_exp2_20260408/phase2_best_loss.pt",
    OLD_WSL_ROOT
    / "checkpoints/architecture_a_recovery_exp2_20260408/phase2_best_recon.pt",
]

HISTORICAL_RESULT_DIRS = [
    "baselines_v1c",
    "neural_v1c",
    "neural_v1d",
    "neural_v1e",
    "neural_v1e_500",
    "v1f_metric_realignment",
]

RAW_EXCLUSIONS = [
    (
        "newer raw EP simulations",
        "outputs/bocf_dataset_v1c_5000 and other bocf_dataset directories",
        "Raw simulation tensors are not result deliverables; validated summaries and manifests are included.",
    ),
    (
        "newer raw magnetic tensors",
        "outputs/magnetic_dataset_* and outputs/magnetic_v0",
        "Raw magnetic movies are multi-gigabyte training inputs; result tables and current cohort arrays are included.",
    ),
    (
        "newer model-ready raw datasets",
        "outputs/dataset_v1* except the locked split/index/quality artifacts",
        "Raw model inputs are excluded; the exact current split, index, and quality reports are included through the locked inventory.",
    ),
    (
        "sensor-standoff raw reprojections",
        "outputs/icbinb_sensor_height_ood/inputs",
        "The 750 per-case NPY inputs are excluded; manifest, parity, predictions, statistics, bootstrap arrays, and validations are included.",
    ),
    (
        "early row-level prediction dumps",
        "outputs/baselines_v1c/classical_predictions.csv and historical predictions larger than 25 MiB",
        "These are non-current row dumps; their aggregate standing result CSV/JSON files are included.",
    ),
    (
        "non-selected checkpoints",
        "older and newer training checkpoints not used by a copied evaluation or the locked current contract",
        "Selected evaluated/locked checkpoints are copied; every checkpoint is indexed separately.",
    ),
    (
        "legacy raster/vector plots",
        "source PNG/PDF/SVG result figures",
        "Original visual files use mixed typography. They are indexed but replaced by seven Computer Modern figures generated from saved results.",
    ),
    (
        "known superseded diagnostic",
        "docs/archive_V1E_500_PARAMETRIC_DIAGNOSTIC_REPORT_initial_eval_merge_bug.md",
        "Explicitly archived because of an initial evaluation-merge bug; not a standing result.",
    ),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_sources() -> None:
    required = [
        OLD_WINDOWS_ROOT / "AGENTS.md",
        OLD_WSL_ROOT / "results",
        NEW_ROOT / "outputs/icbinb_required_analysis/artifact_inventory.json",
        NEW_ROOT / "outputs/icbinb_paper_evidence_bundle/validation_report.json",
    ] + [path for _, path in OLD_EVALUATIONS] + OLD_SELECTED_CHECKPOINTS
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Required export sources are missing: {missing}")
    if DEST.exists():
        raise SystemExit(f"Refusing to overwrite existing Downloads bundle: {DEST}")


def configure_computer_modern() -> dict[str, str]:
    plt.rcParams.update(
        {
            "font.family": "cmr10",
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 13,
            "axes.unicode_minus": False,
            "axes.formatter.use_mathtext": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    resolved = font_manager.findfont(
        font_manager.FontProperties(family="cmr10"), fallback_to_default=False
    )
    return {
        "text_font_family": "Computer Modern Roman",
        "matplotlib_family": "cmr10",
        "resolved_font_file": resolved,
        "mathtext_fontset": "cm",
        "pdf_fonttype": "42",
        "png_dpi": "300",
    }


def copy_streaming(
    source: Path,
    destination_relative: Path,
    *,
    classification: str,
    source_repository: str,
    records: list[dict[str, Any]],
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = DEST / destination_relative
    if destination.exists():
        raise RuntimeError(f"Duplicate destination: {destination_relative}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        for chunk in iter(lambda: reader.read(COPY_CHUNK_BYTES), b""):
            writer.write(chunk)
            digest.update(chunk)
            byte_count += len(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    shutil.copystat(source, destination)
    if destination.stat().st_size != byte_count:
        raise RuntimeError(f"Copy size mismatch: {destination_relative}")
    records.append(
        {
            "destination_path": destination_relative.as_posix(),
            "source_path": str(source),
            "source_repository": source_repository,
            "classification": classification,
            "bytes": byte_count,
            "sha256": digest.hexdigest(),
            "artifact_kind": "copied",
        }
    )


def record_generated(
    path: Path,
    *,
    source_description: str,
    records: list[dict[str, Any]],
) -> None:
    records.append(
        {
            "destination_path": path.relative_to(DEST).as_posix(),
            "source_path": source_description,
            "source_repository": "generated_from_saved_results",
            "classification": "computer_modern_figure",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "artifact_kind": "generated",
        }
    )


def copy_tree_filtered(
    source_root: Path,
    destination_root: Path,
    *,
    include,
    classification: str,
    source_repository: str,
    records: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
) -> None:
    for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
        accepted, reason = include(source)
        if accepted:
            copy_streaming(
                source,
                destination_root / source.relative_to(source_root),
                classification=classification,
                source_repository=source_repository,
                records=records,
            )
        else:
            excluded.append(
                {
                    "source_path": str(source),
                    "bytes": source.stat().st_size,
                    "reason": reason,
                }
            )


def save_figure_pair(
    fig: plt.Figure,
    stem: Path,
    *,
    source_description: str,
    records: list[dict[str, Any]],
) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight", facecolor="white", dpi=300)
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    record_generated(png, source_description=source_description, records=records)
    record_generated(pdf, source_description=source_description, records=records)


def old_evaluation_frame() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    payloads = []
    rows = []
    for label, path in OLD_EVALUATIONS:
        data = json_load(path)
        payloads.append(data)
        rows.append(
            {
                "run": label,
                "checkpoint_epoch": data["checkpoint_epoch"],
                "forward_l2": data["forward_l2_mean"],
                "forward_l2_std": data["forward_l2_std"],
                "reconstruction_r": data["reconstruction_r_mean"],
                "reconstruction_r_std": data["reconstruction_r_std"],
                "physics_residual": data["physics_residual_mean"],
                "physics_residual_std": data["physics_residual_std"],
                "uq_coverage_95": data["uq_coverage_95"],
                "uq_coverage_95_std": data["uq_coverage_95_std"],
                "n_test_samples": data["n_test_samples"],
            }
        )
    return pd.DataFrame(rows), payloads


def figure_old_run_comparison(
    frame: pd.DataFrame, output_dir: Path, records: list[dict[str, Any]]
) -> None:
    specifications = [
        ("forward_l2", "forward_l2_std", "Forward relative $L^2$", 0.05, "lower"),
        ("reconstruction_r", "reconstruction_r_std", "Reconstruction correlation", 0.50, "higher"),
        ("physics_residual", "physics_residual_std", "Physics residual", 0.02, "lower"),
        ("uq_coverage_95", "uq_coverage_95_std", "95% UQ coverage", 0.60, "higher"),
    ]
    colors = ["#4472C4", "#70AD47", "#A5A5A5", "#ED7D31", "#5B9BD5"]
    labels = ["4k base", "R1 loss", "R1 recon", "R2 loss", "R2 recon"]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.4), constrained_layout=True)
    for ax, (metric, error, title, threshold, direction) in zip(axes.ravel(), specifications):
        x = np.arange(len(frame))
        values = frame[metric].to_numpy(float)
        errors = frame[error].to_numpy(float)
        ax.bar(x, values, yerr=errors, capsize=3, color=colors, edgecolor="black", linewidth=0.45)
        ax.axhline(threshold, color="#C00000", linestyle="--", linewidth=1.1, label=f"Gate = {threshold:g}")
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.23)
        ax.legend(loc="best")
        ax.text(
            0.99,
            0.02,
            f"{direction} is better",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#555555",
        )
    fig.suptitle("Older BPC-FNO: all evaluated inverse-system checkpoints")
    save_figure_pair(
        fig,
        output_dir / "older_bpc_fno_run_comparison",
        source_description="five older evaluation_results.json files",
        records=records,
    )


def figure_old_forward_distributions(
    payloads: list[dict[str, Any]], output_dir: Path, records: list[dict[str, Any]]
) -> None:
    labels = [label for label, _ in OLD_EVALUATIONS]
    colors = ["#4472C4", "#70AD47", "#A5A5A5", "#ED7D31", "#5B9BD5"]
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    for label, color, data in zip(labels, colors, payloads):
        values = np.sort(np.asarray(data["per_sample_forward_l2"], dtype=float))
        probabilities = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, probabilities, linewidth=1.7, color=color, label=label)
    ax.axvline(0.05, color="#C00000", linestyle="--", linewidth=1.1, label="Forward gate = 0.05")
    ax.set_xlabel("Per-case relative $L^2$ forward error")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_title("Older BPC-FNO: forward-error distributions across all evaluated runs")
    ax.grid(alpha=0.23)
    ax.legend(loc="lower right", frameon=True)
    save_figure_pair(
        fig,
        output_dir / "older_bpc_fno_forward_error_distributions",
        source_description="per_sample_forward_l2 arrays from five older evaluation JSON files",
        records=records,
    )


def figure_old_uq_calibration(
    payloads: list[dict[str, Any]], output_dir: Path, records: list[dict[str, Any]]
) -> None:
    targets = np.asarray([0.50, 0.80, 0.90, 0.95, 0.99])
    labels = [label for label, _ in OLD_EVALUATIONS]
    colors = ["#4472C4", "#70AD47", "#A5A5A5", "#ED7D31", "#5B9BD5"]
    fig, ax = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="#C00000", linestyle="--", linewidth=1.1, label="Ideal")
    for label, color, data in zip(labels, colors, payloads):
        levels = np.asarray(data["uq_coverage_levels"], dtype=float)
        ax.plot(targets, levels, marker="o", markersize=4, linewidth=1.5, color=color, label=label)
    ax.set_xlim(0.46, 1.01)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Target coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Older BPC-FNO: uncertainty calibration across all evaluated runs")
    ax.grid(alpha=0.23)
    ax.legend(loc="upper left", frameon=True)
    save_figure_pair(
        fig,
        output_dir / "older_bpc_fno_uq_calibration",
        source_description="uq_coverage_levels arrays from five older evaluation JSON files",
        records=records,
    )


def metric_panel_figure(
    table: pd.DataFrame,
    *,
    title: str,
    stem: Path,
    records: list[dict[str, Any]],
    source_description: str,
) -> None:
    label_map = {
        "LAT_global_R2": "LAT global $R^2$",
        "APD90_global_R2": "APD90 global $R^2$",
        "patch_Dice_nonhomogeneous_micro": "Non-homogeneous patch Dice",
        "patch_center_error_mean": "Patch-center error",
        "homogeneous_false_positive_area_mean": "Homogeneous false-positive area",
        "APD90_patch_delta_deployed_R2": "Deployed APD90 patch-delta $R^2$",
    }
    metrics = table["metric"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0), constrained_layout=True)
    colors = ["#4472C4", "#ED7D31"]
    for ax, metric in zip(axes.ravel(), metrics):
        row = table[table["metric"].eq(metric)].iloc[0]
        values = np.asarray([row["fno_estimate"], row["deeponet_estimate"]], dtype=float)
        lower = np.asarray([row["fno_ci_lower"], row["deeponet_ci_lower"]], dtype=float)
        upper = np.asarray([row["fno_ci_upper"], row["deeponet_ci_upper"]], dtype=float)
        yerr = np.vstack([values - lower, upper - values])
        ax.bar([0, 1], values, yerr=yerr, capsize=4, color=colors, edgecolor="black", linewidth=0.45)
        ax.set_xticks([0, 1], ["FNO", "DeepONet"])
        ax.set_title(label_map[metric])
        ax.axhline(0.0, color="#555555", linewidth=0.7)
        ax.grid(axis="y", alpha=0.23)
        ax.text(
            0.5,
            0.02,
            f"n = {int(row['population_base_cases'])} bases",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#555555",
        )
    for ax in axes.ravel()[len(metrics) :]:
        ax.axis("off")
    fig.suptitle(title, y=1.035)
    save_figure_pair(
        fig,
        stem,
        source_description=source_description,
        records=records,
    )


def figure_current_v1f(
    decomposition: pd.DataFrame, output_dir: Path, records: list[dict[str, Any]]
) -> None:
    ids = [
        "magnetic_conduction_identifiability",
        "oracle_true_patch_predicted_LAT",
        "oracle_true_LAT_anomaly_predicted_LAT",
        "deployed_support_predicted_LAT",
    ]
    rows = {
        row["stage_id"]: row for row in decomposition.to_dict(orient="records")
    }
    selected = [rows[item] for item in ids]
    support = rows["deployed_support_nondegenerate_rate"]
    headings = ["Magnetic signal", "Predicted LAT", "Predicted LAT", "Deployed support"]
    subtitles = [
        "Descriptor-only RF\n75 held-out bases",
        "True patch support\n18 conduction bases",
        "True LAT-anomaly support\n18 conduction bases",
        f"Predicted support\n{int(support['numerator'])}/{int(support['denominator'])} nondegenerate",
    ]
    fills = ["#DCEBFA", "#DDF3E4", "#DDF3E4", "#FBE0DE"]
    edges = ["#3978B8", "#2D8A4B", "#2D8A4B", "#C8443A"]
    fig, ax = plt.subplots(figsize=(11.2, 3.45), dpi=220)
    ax.set_xlim(-0.65, 3.65)
    ax.set_ylim(-0.95, 1.05)
    ax.axis("off")
    for index in range(3):
        ax.add_patch(
            FancyArrowPatch(
                (index + 0.43, 0.18),
                (index + 0.72, 0.18),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.5,
                color="#6B7280",
            )
        )
    for index, (row, heading, subtitle, fill, edge) in enumerate(
        zip(selected, headings, subtitles, fills, edges)
    ):
        ax.add_patch(
            FancyBboxPatch(
                (index - 0.43, -0.42),
                0.86,
                1.16,
                boxstyle="round,pad=0.025,rounding_size=0.045",
                linewidth=1.6,
                edgecolor=edge,
                facecolor=fill,
            )
        )
        ax.text(index, 0.59, heading, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(index, 0.27, subtitle, ha="center", va="center", fontsize=9, linespacing=1.25, color="#374151")
        metric = "BA" if index == 0 else "$R^2$"
        estimate = float(row["estimate"])
        if abs(estimate) < 0.0005:
            estimate = 0.0
        lower = float(row["ci_lower_2p5"])
        upper = float(row["ci_upper_97p5"])
        if abs(upper) < 0.0005:
            upper = 0.0
        ax.text(index, -0.07, f"{metric} = {estimate:.3f}", ha="center", va="center", fontsize=13.5, fontweight="bold", color=edge)
        ax.text(index, -0.29, f"95% CI [{lower:.3f}, {upper:.3f}]", ha="center", va="center", fontsize=8.2, color="#4B5563")
    ax.text(1.5, 0.93, "Conduction information survives field reconstruction but collapses at deployed localization", ha="center", va="center", fontsize=13.2, fontweight="bold")
    ax.text(1.5, -0.76, "BA = balanced accuracy. Primary combined_bz_axial variant; grouped test bases. Oracle supports are diagnostics only.", ha="center", va="center", fontsize=8.2, color="#4B5563")
    save_figure_pair(
        fig,
        output_dir / "current_v1f_failure_decomposition",
        source_description="module3_failure_decomposition.csv",
        records=records,
    )


def figure_current_standoff(
    intervals: pd.DataFrame, output_dir: Path, records: list[dict[str, Any]]
) -> None:
    specs = [
        ("LAT_global_R2", "LAT global $R^2$"),
        ("APD90_global_R2", "APD90 global $R^2$"),
        ("patch_Dice_nonhomogeneous_micro", "Non-homogeneous patch Dice"),
        ("homogeneous_false_positive_area_mean", "Homogeneous false-positive area"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    for ax, (metric, title) in zip(axes.ravel(), specs):
        for model, color, label in [("fno", "#4472C4", "FNO"), ("deeponet", "#ED7D31", "DeepONet")]:
            rows = intervals[
                intervals["scope"].eq("overall")
                & intervals["metric"].eq(metric)
                & intervals["model_family"].eq(model)
            ].sort_values("height_mm")
            heights = rows["height_mm"].to_numpy(float)
            values = rows["estimate"].to_numpy(float)
            lower = rows["ci_lower"].to_numpy(float)
            upper = rows["ci_upper"].to_numpy(float)
            ax.errorbar(
                heights,
                values,
                yerr=np.vstack([values - lower, upper - values]),
                marker="o",
                markersize=6,
                linewidth=1.8,
                capsize=3,
                color=color,
                label=label,
            )
        ax.axvline(50, color="#777777", linestyle="--", linewidth=1.0)
        ax.set_xticks([35, 50, 65])
        ax.set_xlabel("Sensor-tissue standoff (mm)")
        ax.set_title(title)
        ax.grid(alpha=0.23)
    axes[0, 0].legend(loc="best")
    fig.suptitle("Zero-shot sensitivity to synthetic sensor-tissue standoff")
    save_figure_pair(
        fig,
        output_dir / "current_sensor_standoff_sensitivity",
        source_description="module4_bootstrap_intervals.csv",
        records=records,
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scan_legacy_figures() -> list[dict[str, Any]]:
    rows = []
    roots = [
        (OLD_WSL_ROOT / "results", "older_bpc_fno"),
        (OLD_WINDOWS_ROOT / "data/processed", "older_bpc_fno_preprocessing"),
        (NEW_ROOT / "figures", "newer_project"),
        (NEW_ROOT / "outputs/icbinb_sensor_height_ood", "newer_module4"),
        (NEW_ROOT / "outputs/icbinb_paper_evidence_bundle", "newer_module6"),
    ]
    seen: set[str] = set()
    for root, group in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".png", ".pdf", ".svg"}:
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            replacement = "numerical evidence copied; no one-to-one plot replacement"
            if group.startswith("older_bpc_fno"):
                replacement = "three consolidated older_bpc_fno Computer Modern figure pairs"
            elif "module3_failure_decomposition" in path.name or "figure1_failure" in path.name:
                replacement = "current_v1f_failure_decomposition"
            elif "sensor_height" in path.name or "figure2_sensor" in path.name:
                replacement = "current_sensor_standoff_sensitivity"
            elif "neural_v1e_500" in path.as_posix():
                replacement = "current_overall_model_comparison and current_conduction_failure"
            rows.append(
                {
                    "source_path": str(path),
                    "group": group,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "copied": False,
                    "reason": "Mixed legacy typography; omitted to satisfy Computer Modern-only visual export.",
                    "replacement": replacement,
                }
            )
    return rows


def checkpoint_index() -> list[dict[str, Any]]:
    selected = {str(path.resolve()) for path in OLD_SELECTED_CHECKPOINTS}
    inventory = json_load(NEW_ROOT / "outputs/icbinb_required_analysis/artifact_inventory.json")
    locked_new = {
        str((NEW_ROOT / row["path"]).resolve())
        for row in inventory["artifacts"]
        if row["category"] == "checkpoint" and row.get("exists")
    }
    selected |= locked_new
    roots = [
        (OLD_WSL_ROOT / "checkpoints", "older_bpc_fno"),
        (NEW_ROOT / "outputs/neural_v1c/checkpoints", "newer_v1c"),
        (NEW_ROOT / "outputs/neural_v1d/checkpoints", "newer_v1d"),
        (NEW_ROOT / "outputs/neural_v1e/checkpoints", "newer_v1e"),
        (NEW_ROOT / "outputs/neural_v1e_500/checkpoints", "newer_v1e_500"),
    ]
    rows = []
    for root, group in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.pt")):
            is_selected = str(path.resolve()) in selected
            rows.append(
                {
                    "source_path": str(path),
                    "group": group,
                    "bytes": path.stat().st_size,
                    "included": is_selected,
                    "selection_reason": (
                        "Evaluated older checkpoint or contract-locked current checkpoint"
                        if is_selected
                        else "Not selected for a standing copied evaluation"
                    ),
                }
            )
    return rows


def build_lineage(old_frame: pd.DataFrame) -> str:
    old_rows = []
    for row in old_frame.to_dict(orient="records"):
        old_rows.append(
            "| {run} | {forward_l2:.4f} | {reconstruction_r:.4f} | "
            "{physics_residual:.4f} | {uq_coverage_95:.4f} |".format(**row)
        )
    overall = pd.read_csv(
        NEW_ROOT
        / "outputs/icbinb_paper_evidence_bundle/table1_overall_model_comparison.csv"
    )
    newer_rows = []
    for row in overall.to_dict(orient="records"):
        newer_rows.append(
            f"| {row['display_label']} | {row['fno_estimate']:.4f} | "
            f"{row['deeponet_estimate']:.4f} | {int(row['population_base_cases'])} |"
        )
    return """# Result lineage and status

## Interpretation boundary

The older volumetric BPC-FNO system and the newer two-dimensional benchmark
use different simulators, targets, geometries, measurements, and evaluation
populations. Their metrics are preserved in separate sections and must not be
presented as a controlled head-to-head comparison.

## Older volumetric BPC-FNO lineage

These evaluations remain valid as historical failure-analysis results. They
showed a functioning forward path but an inverse reconstruction and UQ system
that did not pass the research gates.

| Evaluated run | Forward L2 | Reconstruction correlation | Physics residual | 95% UQ coverage |
|---|---:|---:|---:|---:|
""" + "\n".join(old_rows) + """

The package includes all five evaluation JSON files, their training/evaluation
logs, the five exact evaluated checkpoints, configs, result reports, and three
consolidated Computer Modern figures.

## Newer benchmark lineage

1. V1C: classical, CNN, MLP, and early FNO architecture baselines.
2. V1D: FNO field reconstruction and anomaly-localization diagnostics.
3. V1E: direct magnetic inputs and parametric localization.
4. V1E-500: grouped FNO, CNN, DeepONet, calibration, and refinement studies.
5. V1F: conduction-metric realignment and oracle/deployed-support diagnosis.
6. ICBINB Modules 1-6: locked contract, paired cohort, definition lock,
   grouped bootstrap, failure decomposition, and validated paper exports.

The current confirmatory table is:

| Metric | FNO | DeepONet | Independent base cases |
|---|---:|---:|---:|
""" + "\n".join(newer_rows) + """

Historical test-best family maxima remain useful as development history but
are not the confirmatory architecture comparison. The current tables use
validation-selected endpoint representatives and grouped base-sample
uncertainty. The explicitly archived initial V1E-500 merge-bug diagnostic is
not copied as a standing result.
"""


def build_readme(file_count: int, total_bytes: int) -> str:
    return f"""# BPC-FNO and DeepONet complete result deliverables

Generated: {datetime.now(timezone.utc).isoformat()}

This Windows-side package contains the standing result evidence from the
older volumetric BPC-FNO project and the newer FNO/DeepONet synthetic magnetic
inverse benchmark, from the earliest retained evaluations through ICBINB
Modules 1-6.

## What is included

- {file_count} copied or generated deliverable files ({total_bytes / 2**30:.2f} GiB).
- All five evaluated older BPC-FNO result JSON files and their exact evaluated checkpoints.
- Older result reports, configs, and training/evaluation logs.
- Historical newer-project aggregate results from V1C through V1F.
- Every artifact in the current 88-item evaluation inventory, except legacy figure files replaced by Computer Modern exports.
- All Module 1-6 tables, maps, bootstrap arrays, validations, manifests, and paper-export files.
- Seven PNG/PDF figure pairs rebuilt from saved data using Computer Modern Roman (`cmr10`) and Computer Modern math.
- Reproducibility scripts and result documentation.

## What is intentionally excluded

Raw multi-gigabyte EP/magnetic training datasets, raw standoff reprojection
inputs, non-selected checkpoints, very large non-current row-level prediction
dumps, and original mixed-font plot files are excluded. See
`EXCLUDED_ARTIFACTS.csv`, `CHECKPOINT_INDEX.csv`, and
`LEGACY_FIGURE_INDEX.csv` for explicit accounting.

## Scientific boundaries

The newer system is a controlled synthetic benchmark on a fixed 2D sheet. It
is not patient-specific, clinically validated, real-SQUID validated, or
physics-informed training. The older and newer repository metrics are not a
controlled comparison.

Start with `00_README_AND_MANIFESTS/RESULT_LINEAGE.md`, then use
`00_README_AND_MANIFESTS/FILE_MANIFEST.csv` for exact paths, sizes,
classifications, and SHA-256 hashes.
"""


def main() -> int:
    ensure_sources()
    font_provenance = configure_computer_modern()
    records: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    DEST.mkdir(parents=True, exist_ok=False)
    try:
                                                         
        for subdir in [
            Path("md_files/00_project_overview"),
            Path("md_files/03_handoffs_and_agents"),
            Path("md_files/04_results_and_reports"),
        ]:
            source_root = OLD_WINDOWS_ROOT / subdir
            if source_root.exists():
                copy_tree_filtered(
                    source_root,
                    OLD_DEST / "01_reports" / subdir.name,
                    include=lambda path: (path.suffix.lower() == ".md", "non-report file"),
                    classification="historical_standing_report",
                    source_repository="bpc_fno_v2_windows",
                    records=records,
                    excluded=excluded,
                )
        copy_tree_filtered(
            OLD_WSL_ROOT / "results",
            OLD_DEST / "02_evaluation_results",
            include=lambda path: (
                path.suffix.lower() in {".json", ".log", ".txt"},
                "legacy visual replaced by Computer Modern consolidated figure",
            ),
            classification="historical_standing_result",
            source_repository="bpc_fno_v2_wsl",
            records=records,
            excluded=excluded,
        )
        copy_tree_filtered(
            OLD_WSL_ROOT / "checkpoints",
            OLD_DEST / "03_checkpoint_logs",
            include=lambda path: (
                path.suffix.lower() == ".log",
                "checkpoint binary handled by selected-checkpoint policy",
            ),
            classification="historical_training_log",
            source_repository="bpc_fno_v2_wsl",
            records=records,
            excluded=excluded,
        )
        for checkpoint in OLD_SELECTED_CHECKPOINTS:
            copy_streaming(
                checkpoint,
                OLD_DEST / "04_selected_evaluated_checkpoints" / checkpoint.parent.name / checkpoint.name,
                classification="historical_evaluated_checkpoint",
                source_repository="bpc_fno_v2_wsl",
                records=records,
            )
        for source in sorted((OLD_WINDOWS_ROOT / "configs").glob("*.yaml")):
            copy_streaming(
                source,
                OLD_DEST / "05_configs_and_reproducibility/configs" / source.name,
                classification="historical_reproducibility",
                source_repository="bpc_fno_v2_windows",
                records=records,
            )
        old_script_names = [
            "evaluate.py",
            "train_forward.py",
            "train_joint.py",
            "train_joint_recovery.py",
            "compare_architecture_a_research_runs.py",
        ]
        for name in old_script_names:
            source = OLD_WINDOWS_ROOT / "scripts" / name
            if source.is_file():
                copy_streaming(
                    source,
                    OLD_DEST / "05_configs_and_reproducibility/scripts" / name,
                    classification="historical_reproducibility",
                    source_repository="bpc_fno_v2_windows",
                    records=records,
                )

        old_frame, old_payloads = old_evaluation_frame()
        old_figure_dir = DEST / OLD_DEST / "06_figures_computer_modern"
        figure_old_run_comparison(old_frame, old_figure_dir, records)
        figure_old_forward_distributions(old_payloads, old_figure_dir, records)
        figure_old_uq_calibration(old_payloads, old_figure_dir, records)

                                                             
        for dirname in HISTORICAL_RESULT_DIRS:
            source_root = NEW_ROOT / "outputs" / dirname
            destination_root = NEW_DEST / "01_historical_standing_results" / dirname

            def include_historical(path: Path) -> tuple[bool, str]:
                suffix = path.suffix.lower()
                if suffix in {".png", ".pdf", ".svg", ".pt"}:
                    return False, "visual replaced or checkpoint handled by current locked inventory"
                if path.stat().st_size > 25 * 1024 * 1024:
                    return False, "non-current row-level artifact larger than 25 MiB"
                return suffix in {".csv", ".json", ".npz", ".log", ".txt", ".md"}, "non-result extension"

            copy_tree_filtered(
                source_root,
                destination_root,
                include=include_historical,
                classification="newer_historical_standing_result",
                source_repository="mig-pino-v1",
                records=records,
                excluded=excluded,
            )

                                                                              
        inventory_path = NEW_ROOT / "outputs/icbinb_required_analysis/artifact_inventory.json"
        inventory = json_load(inventory_path)
        for artifact in inventory["artifacts"]:
            if not artifact.get("exists"):
                continue
            source = NEW_ROOT / artifact["path"]
            if source.suffix.lower() in {".png", ".pdf", ".svg"}:
                excluded.append(
                    {
                        "source_path": str(source),
                        "bytes": source.stat().st_size,
                        "reason": "Current visual replaced by Computer Modern export.",
                    }
                )
                continue
            copy_streaming(
                source,
                NEW_DEST / "02_current_locked_evidence" / Path(artifact["path"]),
                classification="current_contract_locked_evidence",
                source_repository="mig-pino-v1",
                records=records,
            )
        for source in [
            NEW_ROOT / "outputs/icbinb_required_analysis/evaluation_contract.json",
            NEW_ROOT / "outputs/icbinb_required_analysis/artifact_inventory.json",
        ]:
            copy_streaming(
                source,
                NEW_DEST
                / "02_current_locked_evidence"
                / source.relative_to(NEW_ROOT),
                classification="current_contract_locked_evidence",
                source_repository="mig-pino-v1",
                records=records,
            )

                                                                                       
        for source in sorted((NEW_ROOT / "docs").glob("*.md")):
            upper = source.name.upper()
            if (
                source.name.startswith("archive_")
                or "ORIENTATION" in upper
                or "PLAN" in upper
                or upper == "CODEX_NOTES.MD"
            ):
                excluded.append(
                    {
                        "source_path": str(source),
                        "bytes": source.stat().st_size,
                        "reason": "Planning/orientation/superseded documentation, not a standing result deliverable.",
                    }
                )
                continue
            copy_streaming(
                source,
                NEW_DEST / "03_result_documentation" / source.name,
                classification="newer_result_documentation",
                source_repository="mig-pino-v1",
                records=records,
            )

                                                                                   
        for source in sorted((NEW_ROOT / "scripts").glob("*.py")):
            copy_streaming(
                source,
                NEW_DEST / "04_reproducibility_scripts" / source.name,
                classification="newer_reproducibility",
                source_repository="mig-pino-v1",
                records=records,
            )

        current_figure_dir = DEST / NEW_DEST / "05_figures_computer_modern"
        overall = pd.read_csv(
            NEW_ROOT
            / "outputs/icbinb_paper_evidence_bundle/table1_overall_model_comparison.csv"
        )
        conduction = pd.read_csv(
            NEW_ROOT
            / "outputs/icbinb_paper_evidence_bundle/table2_conduction_failure.csv"
        )
        metric_panel_figure(
            overall,
            title="Grouped held-out comparison: global fields versus localization",
            stem=current_figure_dir / "current_overall_model_comparison",
            records=records,
            source_description="table1_overall_model_comparison.csv",
        )
        metric_panel_figure(
            conduction,
            title="Conduction-slowing subset: accurate fields, weak localization",
            stem=current_figure_dir / "current_conduction_failure",
            records=records,
            source_description="table2_conduction_failure.csv",
        )
        figure_current_v1f(
            pd.read_csv(
                NEW_ROOT
                / "outputs/icbinb_required_analysis/module3_failure_decomposition.csv"
            ),
            current_figure_dir,
            records,
        )
        figure_current_standoff(
            pd.read_csv(
                NEW_ROOT
                / "outputs/icbinb_sensor_height_ood/module4_bootstrap_intervals.csv"
            ),
            current_figure_dir,
            records,
        )

                               
        legacy_figures = scan_legacy_figures()
        write_csv(
            DEST / MANIFEST_DIR / "LEGACY_FIGURE_INDEX.csv",
            legacy_figures,
            ["source_path", "group", "bytes", "sha256", "copied", "reason", "replacement"],
        )
        checkpoint_rows = checkpoint_index()
        write_csv(
            DEST / MANIFEST_DIR / "CHECKPOINT_INDEX.csv",
            checkpoint_rows,
            ["source_path", "group", "bytes", "included", "selection_reason"],
        )
        exclusion_rows = [
            {"category": category, "source_path": path, "bytes": "", "reason": reason}
            for category, path, reason in RAW_EXCLUSIONS
        ] + [
            {
                "category": "individual_file",
                "source_path": row["source_path"],
                "bytes": row["bytes"],
                "reason": row["reason"],
            }
            for row in excluded
        ]
        write_csv(
            DEST / MANIFEST_DIR / "EXCLUDED_ARTIFACTS.csv",
            exclusion_rows,
            ["category", "source_path", "bytes", "reason"],
        )
        font_path = DEST / MANIFEST_DIR / "FONT_PROVENANCE.json"
        font_path.write_text(
            json.dumps(font_provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        record_generated(
            font_path,
            source_description="Matplotlib runtime font resolution",
            records=records,
        )
        lineage_path = DEST / MANIFEST_DIR / "RESULT_LINEAGE.md"
        lineage_path.write_text(build_lineage(old_frame), encoding="utf-8")
        record_generated(
            lineage_path,
            source_description="saved older evaluation JSON and current Module 6 table",
            records=records,
        )

                                                                           
        for path, description in [
            (DEST / MANIFEST_DIR / "LEGACY_FIGURE_INDEX.csv", "legacy visual inventory"),
            (DEST / MANIFEST_DIR / "CHECKPOINT_INDEX.csv", "checkpoint inventory"),
            (DEST / MANIFEST_DIR / "EXCLUDED_ARTIFACTS.csv", "explicit exclusion inventory"),
        ]:
            record_generated(path, source_description=description, records=records)

        total_before_readme = sum(int(row["bytes"]) for row in records)
        readme_path = DEST / "README.md"
        readme_path.write_text(
            build_readme(len(records) + 2, total_before_readme), encoding="utf-8"
        )
        record_generated(
            readme_path,
            source_description="bundle scope and usage guide",
            records=records,
        )

        manifest_path = DEST / MANIFEST_DIR / "FILE_MANIFEST.csv"
        write_csv(
            manifest_path,
            sorted(records, key=lambda row: row["destination_path"]),
            [
                "destination_path",
                "source_path",
                "source_repository",
                "classification",
                "bytes",
                "sha256",
                "artifact_kind",
            ],
        )
        manifest_hash = sha256_file(manifest_path)
        validation = {
            "schema_version": "1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "passed": True,
            "destination": str(DEST),
            "manifest_sha256": manifest_hash,
            "manifest_records": len(records),
            "total_manifest_bytes": sum(int(row["bytes"]) for row in records),
            "computer_modern_pdf_count": 7,
            "computer_modern_png_count": 7,
            "font_provenance": font_provenance,
            "source_artifacts_modified": False,
            "training_performed": False,
            "inference_performed": False,
            "simulation_performed": False,
            "raw_training_data_copied": False,
        }
        validation_path = DEST / MANIFEST_DIR / "EXPORT_VALIDATION.json"
        validation_path.write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
                                                                                     
        raise

    print(
        json.dumps(
            {
                "passed": True,
                "destination": str(DEST),
                "manifest_records": len(records),
                "total_gib": sum(int(row["bytes"]) for row in records) / 2**30,
                "computer_modern_pdf_count": 7,
                "computer_modern_png_count": 7,
                "legacy_figures_indexed": len(legacy_figures),
                "checkpoints_indexed": len(checkpoint_rows),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
