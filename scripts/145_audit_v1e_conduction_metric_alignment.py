from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v1f_metric_realignment_common import (
    CONDITION_ORDER,
    CONDITION_SHORT,
    V1F_FIG_DIR,
    V1F_OUTPUT_DIR,
    compute_physiology,
    condition_summary,
    ensure_dirs,
    load_index,
    unique_base_rows,
)
from v1e_500_common import write_json


OUT_CSV = V1F_OUTPUT_DIR / "conduction_metric_alignment.csv"
SUMMARY_JSON = V1F_OUTPUT_DIR / "conduction_metric_alignment_summary.json"


def boxplot_by_condition(df: pd.DataFrame, col: str, title: str, ylabel: str, path: Path) -> None:
    data = []
    labels = []
    for cond in CONDITION_ORDER:
        vals = pd.to_numeric(df.loc[df["condition"].eq(cond), col], errors="coerce").to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            data.append(vals)
            labels.append(CONDITION_SHORT[cond])
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.scatter(
        np.concatenate([np.full(len(d), i + 1) for i, d in enumerate(data)]),
        np.concatenate(data),
        s=8,
        alpha=0.25,
        color="#444444",
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def example_panel(df: pd.DataFrame, maps: dict[str, dict], anomaly_key: str, title: str, path: Path) -> None:
    chosen = []
    for cond in CONDITION_ORDER[1:]:
        sub = df[df["condition"].eq(cond)]
        if sub.empty:
            continue
        score_col = "APD_anomaly_mean_abs_patch" if anomaly_key == "apd_anomaly" else "LAT_anomaly_mean_abs_patch"
        sub = sub.sort_values(score_col, ascending=False)
        chosen.append(sub.iloc[0]["base_sample_id"])
    if not chosen:
        return
    fig, axes = plt.subplots(len(chosen), 3, figsize=(8.5, 2.9 * len(chosen)), dpi=160)
    if len(chosen) == 1:
        axes = axes[None, :]
    for r, base in enumerate(chosen):
        item = maps[base]
        patch = item["patch"]
        anomaly = item[anomaly_key]
        field = item["APD90"] if anomaly_key == "apd_anomaly" else item["LAT"]
        cond = df.loc[df["base_sample_id"].eq(base), "condition"].iloc[0]
        axes[r, 0].imshow(patch, cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_title(f"{CONDITION_SHORT[cond]} patch")
        im1 = axes[r, 1].imshow(anomaly, cmap="coolwarm")
        axes[r, 1].set_title("anomaly")
        fig.colorbar(im1, ax=axes[r, 1], fraction=0.046)
        im2 = axes[r, 2].imshow(field, cmap="viridis")
        axes[r, 2].set_title("field")
        fig.colorbar(im2, ax=axes[r, 2], fraction=0.046)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    rows = unique_base_rows(load_index())
    records = []
    maps = {}
    for row in rows.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        phys = compute_physiology(row_s)
        rec = dict(phys["scalars"])
        rec["valid_fraction"] = float(np.mean(phys["valid"] > 0.5))
        records.append(rec)
        maps[rec["base_sample_id"]] = phys

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    summary = condition_summary(
        df,
        [
            "APD90_patch_delta",
            "LAT_patch_delay",
            "activation_duration",
            "conduction_velocity_proxy_global",
            "conduction_severity_scalar",
            "APD_anomaly_mean_abs_patch",
            "LAT_anomaly_mean_abs_patch",
        ],
    )
    summary_path = V1F_OUTPUT_DIR / "conduction_metric_alignment_summary.csv"
    summary.to_csv(summary_path, index=False)

    boxplot_by_condition(
        df,
        "APD90_patch_delta",
        "APD90 Patch Delta By Condition",
        "APD90 inside - outside patch (ms)",
        V1F_FIG_DIR / "apd90_delta_by_condition.png",
    )
    boxplot_by_condition(
        df,
        "LAT_patch_delay",
        "LAT Patch Delay By Condition",
        "LAT inside - outside patch (ms)",
        V1F_FIG_DIR / "lat_delay_by_condition.png",
    )
    boxplot_by_condition(
        df,
        "activation_duration",
        "Activation Duration By Condition",
        "LAT p95 - p5 (ms)",
        V1F_FIG_DIR / "activation_duration_by_condition.png",
    )
    boxplot_by_condition(
        df,
        "conduction_severity_scalar",
        "Local Conduction-Slowing Proxy By Condition",
        "Patch median LAT gradient - background",
        V1F_FIG_DIR / "conduction_velocity_proxy_by_condition.png",
    )
    example_panel(
        df,
        maps,
        "apd_anomaly",
        "True Patch Versus APD Anomaly Examples",
        V1F_FIG_DIR / "true_patch_vs_apd_anomaly_examples.png",
    )
    example_panel(
        df,
        maps,
        "lat_anomaly",
        "True Patch Versus LAT Anomaly Examples",
        V1F_FIG_DIR / "true_patch_vs_lat_anomaly_examples.png",
    )

    by_condition = {r["condition"]: r for r in summary.to_dict(orient="records")}
    cond = by_condition.get("conduction_slowing_perturbation", {})
    apd = by_condition.get("apd_repolarization_perturbation", {})
    homog = by_condition.get("homogeneous", {})
    weak_apd = (
        abs(cond.get("APD90_patch_delta_median", math.nan)) < 0.5 * abs(apd.get("APD90_patch_delta_median", math.nan))
        if np.isfinite(cond.get("APD90_patch_delta_median", math.nan)) and np.isfinite(apd.get("APD90_patch_delta_median", math.nan))
        else False
    )
    meaningful_lat = (
        abs(cond.get("LAT_patch_delay_median", math.nan)) > max(1.0, 2.0 * abs(homog.get("LAT_patch_delay_median", 0.0) or 0.0))
        if np.isfinite(cond.get("LAT_patch_delay_median", math.nan))
        else False
    )
    write_json(
        SUMMARY_JSON,
        {
            "rows": int(len(df)),
            "base_sample_count": int(df["base_sample_id"].nunique()),
            "condition_summary_csv": summary_path,
            "condition_summary": by_condition,
            "conduction_has_weak_apd90_contrast_relative_to_apd_rows": bool(weak_apd),
            "conduction_has_meaningful_lat_delay_contrast": bool(meaningful_lat),
            "figures": sorted(str(p) for p in V1F_FIG_DIR.glob("*.png")),
            "guardrails": {
                "true_labels_used_for_audit_targets_only": True,
                "condition_labels_reporting_only": True,
                "grouped_split_preserved": True,
            },
        },
    )
    print(f"wrote {OUT_CSV}")
    print(f"wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
