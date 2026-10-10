                      
                                                                              

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if (PROJECT_ROOT / ".codex_deps" / "sklearn").exists():
    sys.path.append(str(PROJECT_ROOT / ".codex_deps" / "sklearn"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v1d_anomaly_common import binary_from_soft, dice_iou_precision_recall, make_anomaly_targets, oriented_soft_from_residual, robust_plane_residual
from v1d_field_common import NEURAL_DIR, NEURAL_FIG_DIR, ROOT, scalar_targets_from_fields, write_json
from fno_field_data_common import load_index


OUT_CSV = NEURAL_DIR / "patch_target_audit.csv"
OUT_JSON = NEURAL_DIR / "patch_target_audit.json"


def load_base_rows() -> pd.DataFrame:
    rows = load_index()
    return rows.sort_values(["base_sample_id", "measurement_variant"]).drop_duplicates("base_sample_id").reset_index(drop=True)


def main() -> int:
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_base_rows()
    audit_rows = []
    examples = {"apd": [], "lat": []}
    for row in rows.itertuples(index=False):
        valid = np.load(ROOT / row.valid_mask_grid_file).astype(bool)
        patch = np.load(ROOT / row.patch_mask_grid_file).astype(np.float32) > 0.5
        lat = np.load(ROOT / row.LAT_grid_file).astype(np.float32)
        apd90 = np.load(ROOT / row.APD90_grid_file).astype(np.float32)
        apd_resid = robust_plane_residual(apd90, valid)
        lat_resid = robust_plane_residual(lat, valid)
        apd_soft = oriented_soft_from_residual(apd_resid, valid, patch.astype(np.float32))
        lat_soft = oriented_soft_from_residual(lat_resid, valid, patch.astype(np.float32))
        patch_area = float(np.mean(patch[valid])) if np.any(valid) else np.nan
        apd_binary = binary_from_soft(apd_soft, valid, patch_area)
        lat_binary = binary_from_soft(lat_soft, valid, patch_area)
        anomaly = make_anomaly_targets(lat, apd90, patch.astype(np.float32), valid, row.condition)
        combined_binary = anomaly["anomaly_mask_binary"] > 0.5
        apd_metrics = dice_iou_precision_recall(patch, apd_binary, valid)
        lat_metrics = dice_iou_precision_recall(patch, lat_binary, valid)
        combined_metrics = dice_iou_precision_recall(patch, combined_binary, valid)
        scalars = scalar_targets_from_fields(lat, apd90, patch, valid)
        audit_rows.append(
            {
                "base_sample_id": row.base_sample_id,
                "condition": row.condition,
                "patch_area_fraction": patch_area,
                "patch_radius_mm": getattr(row, "patch_radius_mm", np.nan),
                "apd_scale": getattr(row, "apd_scale", np.nan),
                "conductivity_scale": getattr(row, "conductivity_scale", np.nan),
                "LAT_range": scalars["LAT_range"],
                "APD90_patch_delta": scalars["APD90_patch_delta"],
                "apd_anomaly_patch_dice": apd_metrics["dice"],
                "lat_anomaly_patch_dice": lat_metrics["dice"],
                "combined_anomaly_patch_dice": combined_metrics["dice"],
                "apd_anomaly_inside_mean": float(np.mean(apd_soft[patch & valid])) if np.any(patch & valid) else np.nan,
                "apd_anomaly_outside_mean": float(np.mean(apd_soft[(~patch) & valid])) if np.any((~patch) & valid) else np.nan,
                "lat_anomaly_inside_mean": float(np.mean(lat_soft[patch & valid])) if np.any(patch & valid) else np.nan,
                "lat_anomaly_outside_mean": float(np.mean(lat_soft[(~patch) & valid])) if np.any((~patch) & valid) else np.nan,
            }
        )
        if row.condition in {"apd_repolarization_perturbation", "mixed_apd_conduction_perturbation"} and len(examples["apd"]) < 4:
            examples["apd"].append((row.base_sample_id, patch, apd90, apd_soft, combined_binary))
        if row.condition in {"conduction_slowing_perturbation", "mixed_apd_conduction_perturbation"} and len(examples["lat"]) < 4:
            examples["lat"].append((row.base_sample_id, patch, lat, lat_soft, combined_binary))
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT_CSV, index=False)

    summary = {
        "n_base_samples": int(len(audit)),
        "by_condition": audit.groupby("condition")[
            [
                "patch_area_fraction",
                "apd_anomaly_patch_dice",
                "lat_anomaly_patch_dice",
                "combined_anomaly_patch_dice",
                "apd_anomaly_inside_mean",
                "apd_anomaly_outside_mean",
                "lat_anomaly_inside_mean",
                "lat_anomaly_outside_mean",
            ]
        ]
        .mean(numeric_only=True)
        .reset_index()
        .replace({np.nan: None})
        .to_dict(orient="records"),
        "interpretation": "Anomaly observability is condition-dependent; APD residuals align best with APD/mixed patches, while LAT residuals are needed for conduction/mixed patches.",
    }
    write_json(OUT_JSON, summary)

    fig, ax = plt.subplots(figsize=(8, 4))
    for cond, sub in audit.groupby("condition"):
        ax.hist(sub["patch_area_fraction"], alpha=0.55, bins=20, label=cond)
    ax.set_xlabel("geometric patch area fraction")
    ax.set_ylabel("base samples")
    ax.set_title("Patch Target Area Distribution")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "patch_target_area_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    cond_summary = audit.groupby("condition")[["apd_anomaly_patch_dice", "lat_anomaly_patch_dice", "combined_anomaly_patch_dice"]].mean()
    cond_summary.plot(kind="bar", ax=ax)
    ax.set_ylabel("Dice versus geometric patch")
    ax.set_title("Patch Observability by Condition")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "patch_observability_by_condition.png", dpi=180)
    plt.close(fig)

    for key, fname, field_title in [
        ("apd", "patch_target_vs_apd_anomaly_examples.png", "APD90"),
        ("lat", "patch_target_vs_lat_anomaly_examples.png", "LAT"),
    ]:
        ex = examples[key]
        fig, axes = plt.subplots(len(ex), 4, figsize=(10, 2.4 * max(1, len(ex))))
        if len(ex) == 1:
            axes = axes[None, :]
        for i, (sid, patch, field, soft, binary) in enumerate(ex):
            for j, (arr, title, cmap) in enumerate(
                [
                    (field, field_title, "viridis"),
                    (patch, "geometric patch", "magma"),
                    (soft, "soft anomaly", "magma"),
                    (binary, "binary anomaly", "magma"),
                ]
            ):
                im = axes[i, j].imshow(arr, origin="lower", cmap=cmap)
                axes[i, j].set_title(f"{sid}\n{title}", fontsize=8)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.02)
        fig.tight_layout()
        fig.savefig(NEURAL_FIG_DIR / fname, dpi=180)
        plt.close(fig)

    print(json.dumps({"audit_csv": str(OUT_CSV), "summary": summary["by_condition"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
