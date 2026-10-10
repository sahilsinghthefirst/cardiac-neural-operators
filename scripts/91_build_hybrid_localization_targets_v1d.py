                      
                                         

                                                        
                                                
   

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
from v1d_anomaly_common import make_anomaly_targets
from v1d_field_common import ROOT, rel, write_json
from v1d_hybrid_common import HYBRID_DIR, HYBRID_MANIFEST, HYBRID_QUALITY, binary_metrics


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
    HYBRID_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_index().sort_values(["base_sample_id", "measurement_variant"]).drop_duplicates("base_sample_id").reset_index(drop=True)
    manifest = []
    qrows = []
    for row in rows.itertuples(index=False):
        out_dir = HYBRID_DIR / row.base_sample_id
        out_dir.mkdir(parents=True, exist_ok=True)
        valid = np.load(ROOT / row.valid_mask_grid_file).astype(bool)
        geom = np.load(ROOT / row.patch_mask_grid_file).astype(np.float32)
        lat = np.load(ROOT / row.LAT_grid_file).astype(np.float32)
        apd90 = np.load(ROOT / row.APD90_grid_file).astype(np.float32)
        targets = make_anomaly_targets(lat, apd90, geom, valid, row.condition)
        if row.condition == "homogeneous":
            geom[:] = 0.0
            for key in targets:
                targets[key][:] = 0.0
            negative = valid.astype(np.float32)
        else:
            negative = np.zeros_like(geom, dtype=np.float32)
        np.save(out_dir / "geometric_patch_mask.npy", geom.astype(np.float32))
        np.save(out_dir / "apd_anomaly_soft.npy", targets["apd_anomaly_soft"].astype(np.float32))
        np.save(out_dir / "lat_anomaly_soft.npy", targets["lat_anomaly_soft"].astype(np.float32))
        np.save(out_dir / "combined_anomaly_soft.npy", targets["combined_anomaly_soft"].astype(np.float32))
        np.save(out_dir / "homogeneous_negative_mask.npy", negative.astype(np.float32))
        aux_mask = targets["anomaly_mask_binary"] > 0.5
        metrics = binary_metrics(geom > 0.5, aux_mask, valid)
        meta = {
            "base_sample_id": row.base_sample_id,
            "condition": row.condition,
            "primary_target": "geometric_patch_mask",
            "auxiliary_targets": ["apd_anomaly_soft", "lat_anomaly_soft", "combined_anomaly_soft"],
            "homogeneous_negative_mask_is_loss_only": True,
            "target_fields_are_labels_only": True,
            "condition_observability_note": {
                "apd_repolarization_perturbation": "APD residuals are auxiliary observability labels.",
                "conduction_slowing_perturbation": "LAT residuals are auxiliary observability labels.",
                "mixed_apd_conduction_perturbation": "Both APD and LAT residuals can guide localization.",
                "homogeneous": "All positive targets are zero and the valid tissue is a negative calibration mask.",
            },
            "auxiliary_binary_vs_geometric": metrics,
        }
        write_json(out_dir / "hybrid_target_metadata.json", meta)
        manifest.append(
            {
                "base_sample_id": row.base_sample_id,
                "condition": row.condition,
                "hybrid_target_dir": rel(out_dir),
                "geometric_patch_mask_file": rel(out_dir / "geometric_patch_mask.npy"),
                "apd_anomaly_soft_file_hybrid": rel(out_dir / "apd_anomaly_soft.npy"),
                "lat_anomaly_soft_file_hybrid": rel(out_dir / "lat_anomaly_soft.npy"),
                "combined_anomaly_soft_file_hybrid": rel(out_dir / "combined_anomaly_soft.npy"),
                "homogeneous_negative_mask_file": rel(out_dir / "homogeneous_negative_mask.npy"),
                "hybrid_target_metadata_file": rel(out_dir / "hybrid_target_metadata.json"),
                "success": True,
                "notes": "",
            }
        )
        qrows.append(
            {
                "base_sample_id": row.base_sample_id,
                "condition": row.condition,
                "geometric_area_fraction": float(np.mean((geom > 0.5)[valid])) if np.any(valid) else np.nan,
                "aux_binary_geometric_dice": metrics["dice"],
                "apd_aux_inside_mean": float(np.mean(targets["apd_anomaly_soft"][(geom > 0.5) & valid])) if np.any((geom > 0.5) & valid) else np.nan,
                "apd_aux_outside_mean": float(np.mean(targets["apd_anomaly_soft"][(geom <= 0.5) & valid])) if np.any((geom <= 0.5) & valid) else np.nan,
                "lat_aux_inside_mean": float(np.mean(targets["lat_anomaly_soft"][(geom > 0.5) & valid])) if np.any((geom > 0.5) & valid) else np.nan,
                "lat_aux_outside_mean": float(np.mean(targets["lat_anomaly_soft"][(geom <= 0.5) & valid])) if np.any((geom <= 0.5) & valid) else np.nan,
                "homogeneous_negative_area": float(np.mean(negative[valid])) if np.any(valid) else np.nan,
            }
        )
    write_csv(HYBRID_MANIFEST, manifest)
    qdf = pd.DataFrame(qrows)
    report = {
        "selected_base_samples": int(len(rows)),
        "successful_targets": int(len(manifest)),
        "condition_counts": qdf["condition"].value_counts().to_dict(),
        "quality_by_condition": qdf.groupby("condition").mean(numeric_only=True).reset_index().replace({np.nan: None}).to_dict(orient="records"),
        "notes": "Geometric patch is the primary localization label; APD/LAT residual maps are auxiliary labels only and are never model inputs.",
    }
    write_json(HYBRID_QUALITY, report)
    print(json.dumps({"manifest": str(HYBRID_MANIFEST), "report": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
