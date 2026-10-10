                      
                                                                         

from __future__ import annotations

import json
import math
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

from v1d_field_common import NEURAL_DIR, NEURAL_FIG_DIR, ROOT, write_json


OUT_CSV = NEURAL_DIR / "hybrid_localization_comparison.csv"
OUT_JSON = NEURAL_DIR / "hybrid_localization_comparison.json"


def get_all(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[(df["split"].eq("test")) & (df["group_type"].eq("all")) & (df["group_value"].eq("all"))].copy()


def metric_row(label: str, path: Path, config_filter: str | None = None) -> list[dict]:
    df = get_all(path)
    if len(df) == 0:
        return []
    if config_filter and "config" in df.columns:
        df = df[df["config"].eq(config_filter)]
    rows = []
    for row in df.to_dict(orient="records"):
        rows.append(
            {
                "label": label,
                "model": row.get("model", label),
                "config": row.get("config", config_filter or "all"),
                "geometric_dice": row.get("dice_mean", np.nan),
                "iou": row.get("iou_mean", np.nan),
                "precision": row.get("precision_mean", np.nan),
                "recall": row.get("recall_mean", np.nan),
                "homogeneous_false_positive_area": row.get("homogeneous_false_positive_area", np.nan),
                "patch_area_fraction_r2": row.get("patch_area_fraction_r2", row.get("patch_area_fraction_r2", np.nan)),
                "APD90_delta_pred_mask_pred_apd_r2": row.get("APD90_delta_pred_mask_pred_apd_r2", row.get("APD90_patch_delta_pred_mask_pred_apd_r2", np.nan)),
                "APD90_delta_pred_mask_true_apd_r2": row.get("APD90_delta_pred_mask_true_apd_r2", row.get("APD90_patch_delta_pred_mask_true_apd_r2", np.nan)),
                "APD90_delta_true_mask_pred_apd_r2": row.get("APD90_delta_true_mask_pred_apd_r2", np.nan),
                "soft_mask_APD90_delta_r2": row.get("APD90_delta_soft_mask_r2", np.nan),
            }
        )
    return rows


def main() -> int:
    rows = []
    rows.append(
        {
            "label": "original_fno_field",
            "model": "fno_field_reconstructor",
            "config": "original",
            "geometric_dice": 0.4492212768074837,
            "homogeneous_false_positive_area": 0.0510602678571428,
            "patch_area_fraction_r2": -0.11976964746354879,
            "APD90_delta_pred_mask_pred_apd_r2": 0.07789895193313057,
            "APD90_delta_true_mask_pred_apd_r2": 0.8731308445301711,
        }
    )
    rows.append(
        {
            "label": "patchaware_fno",
            "model": "patchaware_fno",
            "config": "patch_mask_apd_plus_scalar_aux",
            "geometric_dice": 0.4440440864773209,
            "patch_area_fraction_r2": -0.10382517036987693,
            "APD90_delta_pred_mask_pred_apd_r2": 0.1756849286354094,
        }
    )
    rows.extend(metric_row("anomaly_fno", NEURAL_DIR / "fno_anomaly_localizer_results.csv"))
    rows.extend(metric_row("anomaly_cnn", NEURAL_DIR / "cnn_anomaly_localizer_results.csv"))
    rows.extend(metric_row("hybrid_fno", NEURAL_DIR / "fno_hybrid_localizer_results.csv"))
    rows.extend(metric_row("hybrid_cnn", NEURAL_DIR / "cnn_hybrid_localizer_results.csv"))
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT_CSV, index=False)
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    sub = comp[comp["label"].str.contains("hybrid|original|patchaware", regex=True)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, title in [
        (axes[0], "geometric_dice", "Geometric Patch Dice"),
        (axes[1], "homogeneous_false_positive_area", "Homogeneous FP Area"),
        (axes[2], "APD90_delta_pred_mask_pred_apd_r2", "APD90 Patch-Delta R2"),
    ]:
        ax.bar(sub["label"] + "\n" + sub["config"].fillna(""), sub[col], color="#4c78a8")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "hybrid_localization_comparison.png", dpi=180)
    plt.close(fig)
    best_hybrid = comp[comp["label"].eq("hybrid_fno")].sort_values("geometric_dice", ascending=False).head(1).to_dict(orient="records")
    payload = {
        "comparison_rows": comp.replace({np.nan: None}).to_dict(orient="records"),
        "best_hybrid_fno_by_dice": best_hybrid,
        "notes": "Hybrid metrics use geometric patch mask as the primary localization target. APD/LAT residual labels are auxiliary only.",
    }
    write_json(OUT_JSON, payload)
    print(json.dumps({"comparison_csv": str(OUT_CSV), "best_hybrid_fno": best_hybrid}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
