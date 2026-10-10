                      
                                                                                  

from __future__ import annotations

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

from v1e_500_common import write_json
from v1f_metric_realignment_common import (
    V1F_FIG_DIR,
    V1F_OUTPUT_DIR,
    V1F_TARGET_DIR,
    build_multichannel_targets_for_row,
    ensure_dirs,
    load_index,
    unique_base_rows,
)


OUT_MANIFEST = V1F_OUTPUT_DIR / "v1f_multichannel_target_manifest.csv"
QUALITY_JSON = V1F_OUTPUT_DIR / "v1f_multichannel_target_quality_report.json"


def plot_target_examples(manifest: pd.DataFrame) -> None:
    nonhom = manifest[manifest["patch_present"].eq(1)].head(4)
    if nonhom.empty:
        return
    fig, axes = plt.subplots(len(nonhom), 5, figsize=(12, 2.6 * len(nonhom)), dpi=160)
    if len(nonhom) == 1:
        axes = axes[None, :]
    cols = [
        ("generic_patch_mask_file", "patch", "gray"),
        ("apd_anomaly_target_file", "APD anomaly", "coolwarm"),
        ("lat_anomaly_target_file", "LAT anomaly", "coolwarm"),
        ("center_heatmap_file", "center heat", "magma"),
        ("local_conduction_slowing_proxy_file", "cond proxy", "coolwarm"),
    ]
    for r, row in enumerate(nonhom.itertuples(index=False)):
        for c, (col, title, cmap) in enumerate(cols):
            arr = np.load(PROJECT_ROOT / getattr(row, col))
            im = axes[r, c].imshow(arr, cmap=cmap)
            axes[r, c].set_title(title)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if c != 0:
                fig.colorbar(im, ax=axes[r, c], fraction=0.046)
        axes[r, 0].set_ylabel(getattr(row, "condition").replace("_", "\n"), fontsize=7)
    fig.tight_layout()
    fig.savefig(V1F_FIG_DIR / "v1f_multichannel_target_examples.png")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    rows = load_index()
    base_rows = unique_base_rows(rows)
    by_base = {}
    for row in base_rows.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        by_base[row_s["base_sample_id"]] = build_multichannel_targets_for_row(row_s)

    manifest_rows = []
    for row in rows.itertuples(index=False):
        row_d = row._asdict()
        base_manifest = dict(by_base[row_d["base_sample_id"]])
        rec = {
            "row_id": row_d["row_id"],
            "base_sample_id": row_d["base_sample_id"],
            "split": row_d["split"],
            "condition": row_d["condition"],
            "combined_variant": row_d["combined_variant"],
        }
        rec.update(base_manifest)
        manifest_rows.append(rec)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_MANIFEST, index=False)
    plot_target_examples(pd.DataFrame(by_base.values()))

    checks = {
        "rows": int(len(manifest)),
        "base_sample_count": int(manifest["base_sample_id"].nunique()),
        "target_dir": V1F_TARGET_DIR,
        "all_files_exist": True,
        "all_maps_finite": True,
        "nonhomogeneous_heatmaps_positive": True,
        "homogeneous_heatmaps_zero": True,
        "group_leakage_count": 0,
        "notes": [],
    }
    file_cols = [c for c in manifest.columns if c.endswith("_file")]
    for rec in manifest.drop_duplicates("base_sample_id").to_dict(orient="records"):
        for col in file_cols:
            path = PROJECT_ROOT / rec[col]
            if not path.exists():
                checks["all_files_exist"] = False
                checks["notes"].append(f"missing {path}")
                continue
            if path.suffix == ".npy":
                arr = np.load(path)
                if arr.shape != (32, 32) or not np.all(np.isfinite(arr)):
                    checks["all_maps_finite"] = False
                    checks["notes"].append(f"bad map {path}: shape={arr.shape}")
        heat = np.load(PROJECT_ROOT / rec["center_heatmap_file"])
        if rec["condition"] == "homogeneous":
            if float(np.max(np.abs(heat))) > 1e-7:
                checks["homogeneous_heatmaps_zero"] = False
        else:
            if float(np.sum(heat)) <= 0.0:
                checks["nonhomogeneous_heatmaps_positive"] = False
    split_nunique = manifest.groupby("base_sample_id")["split"].nunique()
    checks["group_leakage_count"] = int((split_nunique > 1).sum())
    checks["passed"] = bool(
        checks["all_files_exist"]
        and checks["all_maps_finite"]
        and checks["nonhomogeneous_heatmaps_positive"]
        and checks["homogeneous_heatmaps_zero"]
        and checks["group_leakage_count"] == 0
    )
    write_json(QUALITY_JSON, checks)
    print(f"wrote {OUT_MANIFEST}")
    print(f"wrote {QUALITY_JSON}")


if __name__ == "__main__":
    main()
