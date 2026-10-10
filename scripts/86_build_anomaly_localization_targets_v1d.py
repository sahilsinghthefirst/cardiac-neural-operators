                      
                                                                          

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if (PROJECT_ROOT / ".codex_deps" / "sklearn").exists():
    sys.path.append(str(PROJECT_ROOT / ".codex_deps" / "sklearn"))

import numpy as np
import pandas as pd

from fno_field_data_common import load_index
from v1d_anomaly_common import ANOMALY_DIR, ANOMALY_MANIFEST, ANOMALY_QUALITY, dice_iou_precision_recall, make_anomaly_targets, robust_plane_residual
from v1d_field_common import ROOT, rel, scalar_targets_from_fields, write_json


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ANOMALY_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_index().sort_values(["base_sample_id", "measurement_variant"]).drop_duplicates("base_sample_id").reset_index(drop=True)
    manifest = []
    quality_rows = []
    for row in rows.itertuples(index=False):
        out_dir = ANOMALY_DIR / row.base_sample_id
        out_dir.mkdir(parents=True, exist_ok=True)
        valid = np.load(ROOT / row.valid_mask_grid_file).astype(bool)
        patch = np.load(ROOT / row.patch_mask_grid_file).astype(np.float32)
        lat = np.load(ROOT / row.LAT_grid_file).astype(np.float32)
        apd90 = np.load(ROOT / row.APD90_grid_file).astype(np.float32)
        targets = make_anomaly_targets(lat, apd90, patch, valid, row.condition)
        for name, arr in targets.items():
            np.save(out_dir / f"{name}.npy", arr.astype(np.float32))
        patch_bool = patch > 0.5
        anomaly_binary = targets["anomaly_mask_binary"] > 0.5
        metrics = dice_iou_precision_recall(patch_bool, anomaly_binary, valid)
        scalars_geo = scalar_targets_from_fields(lat, apd90, patch_bool, valid)
        scalars_anom = scalar_targets_from_fields(lat, apd90, anomaly_binary, valid)
        lat_resid = robust_plane_residual(lat, valid)
        apd_resid = robust_plane_residual(apd90, valid)
        meta = {
            "base_sample_id": row.base_sample_id,
            "condition": row.condition,
            "derivation": "condition-aware robust plane residual labels; APD residual for APD/mixed, positive LAT delay residual for conduction/mixed, zeros for homogeneous",
            "target_fields_are_labels_only": True,
            "geometric_patch_area_fraction": scalars_geo["patch_area_fraction"],
            "anomaly_area_fraction": scalars_anom["patch_area_fraction"],
            "anomaly_vs_geometric_patch": metrics,
            "lat_residual_min_max": [float(np.nanmin(lat_resid[valid])), float(np.nanmax(lat_resid[valid]))],
            "apd90_residual_min_max": [float(np.nanmin(apd_resid[valid])), float(np.nanmax(apd_resid[valid]))],
        }
        write_json(out_dir / "anomaly_target_metadata.json", meta)
        manifest.append(
            {
                "base_sample_id": row.base_sample_id,
                "condition": row.condition,
                "anomaly_target_dir": rel(out_dir),
                "apd_anomaly_soft_file": rel(out_dir / "apd_anomaly_soft.npy"),
                "lat_anomaly_soft_file": rel(out_dir / "lat_anomaly_soft.npy"),
                "combined_anomaly_soft_file": rel(out_dir / "combined_anomaly_soft.npy"),
                "anomaly_mask_binary_file": rel(out_dir / "anomaly_mask_binary.npy"),
                "anomaly_target_metadata_file": rel(out_dir / "anomaly_target_metadata.json"),
                "success": True,
                "notes": "",
            }
        )
        quality_rows.append(
            {
                "base_sample_id": row.base_sample_id,
                "condition": row.condition,
                "geometric_patch_area_fraction": scalars_geo["patch_area_fraction"],
                "anomaly_area_fraction": scalars_anom["patch_area_fraction"],
                "anomaly_geometric_dice": metrics["dice"],
                "anomaly_geometric_iou": metrics["iou"],
                "combined_soft_mean": float(np.mean(targets["combined_anomaly_soft"][valid])) if np.any(valid) else np.nan,
                "combined_soft_max": float(np.max(targets["combined_anomaly_soft"][valid])) if np.any(valid) else np.nan,
            }
        )
    write_csv(ANOMALY_MANIFEST, manifest)
    qdf = pd.DataFrame(quality_rows)
    report = {
        "selected_base_samples": int(len(rows)),
        "successful_targets": int(len(manifest)),
        "condition_counts": qdf["condition"].value_counts().to_dict(),
        "quality_by_condition": qdf.groupby("condition").mean(numeric_only=True).reset_index().replace({np.nan: None}).to_dict(orient="records"),
        "notes": "Anomaly maps are supervised labels derived from EP target fields. They are not model inputs.",
    }
    write_json(ANOMALY_QUALITY, report)
    print(json.dumps({"manifest": str(ANOMALY_MANIFEST), "report": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
