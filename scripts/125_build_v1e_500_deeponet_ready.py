                      
                                                                 
 
                                                                            
                                                                      
   

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from v1e_500_common import PILOT_DIR, ROOT, no_group_leakage, rel, write_json


DEEPO_DIR = PILOT_DIR / "deeponet_ready"
SAMPLE_INDEX = DEEPO_DIR / "sample_index.csv"
QUERY_POINTS = DEEPO_DIR / "query_points_32x32.npy"
SPLIT_JSON = DEEPO_DIR / "splits" / "grouped_train_val_test.json"
SPLIT_CSV = DEEPO_DIR / "splits" / "grouped_train_val_test.csv"
REPORT = DEEPO_DIR / "dataset_quality_report.json"
DESC_CSV = DEEPO_DIR / "direct_descriptors.csv"
DESC_JSON = DEEPO_DIR / "direct_descriptors.json"

COMBINED_MANIFEST = PILOT_DIR / "combined_input_manifest.csv"
TARGET_MANIFEST = PILOT_DIR / "target_manifest.csv"
PARAM_SAMPLE_INDEX = PILOT_DIR / "parametric_ready" / "sample_index.csv"
PARAM_SPLIT_JSON = PILOT_DIR / "parametric_ready" / "splits" / "grouped_train_val_test.json"
FIELD_DESC_CSV = PILOT_DIR / "field_ready" / "direct_descriptors.csv"
FIELD_DESC_JSON = PILOT_DIR / "field_ready" / "direct_descriptors.json"


def channel_desc(primary: np.ndarray, mask2: np.ndarray, prefix: str) -> dict[str, float]:
    names = [
        "rms",
        "max_abs",
        "early_rms",
        "late_rms",
        "peak_time",
        "energy_cx",
        "energy_cy",
        "entropy",
        "concentration",
        "lr_asym",
        "tb_asym",
        "mean_abs_grad",
    ]
    if not np.any(mask2):
        return {f"{prefix}_{name}": 0.0 for name in names}
    vals = primary[:, mask2]
    rms_t = np.sqrt(np.mean(vals * vals, axis=1)) if vals.size else np.zeros(primary.shape[0])
    energy = np.mean(primary * primary, axis=0)
    total = float(np.sum(energy[mask2]))
    grid = primary.shape[1]
    yy, xx = np.indices((grid, grid))
    if total > 0:
        cx = float(np.sum(xx[mask2] * energy[mask2]) / total / max(grid - 1, 1))
        cy = float(np.sum(yy[mask2] * energy[mask2]) / total / max(grid - 1, 1))
        p = (energy[mask2].ravel() / total).clip(1e-30, None)
        entropy = float(-np.sum(p * np.log(p)) / np.log(max(p.size, 2)))
        concentration = float(np.max(p))
    else:
        cx = cy = entropy = concentration = 0.0
    left = energy[:, : grid // 2][mask2[:, : grid // 2]].sum()
    right = energy[:, grid // 2 :][mask2[:, grid // 2 :]].sum()
    bottom = energy[: grid // 2, :][mask2[: grid // 2, :]].sum()
    top = energy[grid // 2 :, :][mask2[grid // 2 :, :]].sum()
    gy, gx = np.gradient(np.mean(primary, axis=0))
    return {
        f"{prefix}_rms": float(np.sqrt(np.mean(vals * vals))) if vals.size else 0.0,
        f"{prefix}_max_abs": float(np.max(np.abs(vals))) if vals.size else 0.0,
        f"{prefix}_early_rms": float(np.mean(rms_t[:25])) if len(rms_t) else 0.0,
        f"{prefix}_late_rms": float(np.mean(rms_t[150:280])) if len(rms_t) > 150 else 0.0,
        f"{prefix}_peak_time": float(np.argmax(rms_t)) if len(rms_t) else 0.0,
        f"{prefix}_energy_cx": cx,
        f"{prefix}_energy_cy": cy,
        f"{prefix}_entropy": entropy,
        f"{prefix}_concentration": concentration,
        f"{prefix}_lr_asym": float((right - left) / (right + left + 1e-30)),
        f"{prefix}_tb_asym": float((top - bottom) / (top + bottom + 1e-30)),
        f"{prefix}_mean_abs_grad": float(np.mean(np.abs(gx[mask2])) + np.mean(np.abs(gy[mask2]))),
    }


def descriptor_row(row) -> dict[str, float | str]:
    b = np.load(ROOT / row.B_meas_file).astype(np.float64)
    mask = np.load(ROOT / row.sensor_mask_file).astype(bool)
    valid = mask[None, :, :, :].repeat(b.shape[0], axis=0)
    vals = b[valid]
    out: dict[str, float | str] = {
        "row_id": row.row_id,
        "base_sample_id": row.base_sample_id,
        "condition": row.condition,
        "combined_variant": row.combined_variant,
        "split": row.split,
        "global_rms": float(np.sqrt(np.mean(vals * vals))) if vals.size else math.nan,
        "max_abs_B": float(np.max(np.abs(vals))) if vals.size else math.nan,
    }
    for channel in range(b.shape[-1]):
        primary = np.where(mask[:, :, channel][None, :, :], b[:, :, :, channel], 0.0)
        out.update(channel_desc(primary, mask[:, :, channel], f"ch{channel}"))
    return out


def build_or_reuse_descriptors(rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    DEEPO_DIR.mkdir(parents=True, exist_ok=True)
    if FIELD_DESC_CSV.exists():
        desc = pd.read_csv(FIELD_DESC_CSV)
        desc = desc[desc["row_id"].isin(set(rows["row_id"]))].copy()
        desc = rows[["row_id"]].merge(desc, on="row_id", how="left", validate="one_to_one")
        source = rel(FIELD_DESC_CSV)
        reused = True
    else:
        desc = pd.DataFrame([descriptor_row(row) for row in rows.itertuples(index=False)])
        source = "recomputed from B_meas"
        reused = False
    if "combined_variant" not in desc.columns and "measurement_variant" in desc.columns:
        desc = desc.rename(columns={"measurement_variant": "combined_variant"})
    desc.to_csv(DESC_CSV, index=False)
    feature_cols = [c for c in desc.columns if c not in {"row_id", "base_sample_id", "condition", "combined_variant", "split"}]
    finite = bool(np.isfinite(desc[feature_cols].to_numpy(float)).all()) if feature_cols else False
    report = {
        "row_count": int(len(desc)),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "source": source,
        "reused_existing_descriptor_csv": reused,
        "source_descriptor_json": rel(FIELD_DESC_JSON) if FIELD_DESC_JSON.exists() else None,
        "windows": {
            "activation_window_rms_proxy": "fixed samples 0:25, not LAT-defined",
            "late_window_rms_proxy": "fixed samples 150:280, not RT/APD-defined",
        },
        "leakage_guardrail": "Descriptors use only B_meas and fixed time windows, not LAT/APD/patch targets.",
        "passed": bool(len(desc) == len(rows) and desc["row_id"].is_unique and finite),
    }
    write_json(DESC_JSON, report)
    return desc, report


def make_query_points(grid_size: int = 32) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)


def load_split_payload(param_ready: pd.DataFrame) -> dict:
    if PARAM_SPLIT_JSON.exists():
        payload = json.loads(PARAM_SPLIT_JSON.read_text())
    else:
        base_split = dict(param_ready[["base_sample_id", "split"]].drop_duplicates().to_numpy())
        payload = {"base_split": base_split, "strategy": "reconstructed from parametric_ready/sample_index.csv"}
    if "base_split" not in payload:
        payload["base_split"] = dict(param_ready[["base_sample_id", "split"]].drop_duplicates().to_numpy())
    return payload


def main() -> int:
    required_inputs = [COMBINED_MANIFEST, TARGET_MANIFEST, PARAM_SAMPLE_INDEX, PARAM_SPLIT_JSON]
    missing_inputs = [rel(path) for path in required_inputs if not path.exists()]
    if missing_inputs:
        raise SystemExit(f"Missing authoritative v1e 500 dataset paths: {missing_inputs}")

    DEEPO_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.read_csv(COMBINED_MANIFEST)
    targets = pd.read_csv(TARGET_MANIFEST)
    param_ready = pd.read_csv(PARAM_SAMPLE_INDEX)
    split_payload_src = load_split_payload(param_ready)
    split_map = param_ready[["base_sample_id", "split"]].drop_duplicates()

    rows = combined.merge(targets, on=["base_sample_id", "condition"], how="inner", validate="many_to_one")
    rows = rows.merge(split_map, on="base_sample_id", how="left", validate="many_to_one")
    rows = rows.rename(columns={"measurement_variant": "combined_variant"})
    rows["descriptor_row_id"] = rows["row_id"]
    rows["descriptor_file"] = rel(DESC_CSV)
    rows = rows.sort_values(["base_sample_id", "combined_variant"]).reset_index(drop=True)

    keep_first = [
        "row_id",
        "base_sample_id",
        "split",
        "condition",
        "combined_variant",
        "B_meas_file",
        "coordinate_channels_file",
        "sensor_mask_file",
        "LAT_grid_file",
        "APD80_grid_file",
        "APD90_grid_file",
        "patch_mask_grid_file",
        "valid_mask_grid_file",
        "patch_param_targets_file",
        "descriptor_row_id",
        "descriptor_file",
    ]
    remaining = [c for c in rows.columns if c not in keep_first]
    rows = rows[keep_first + remaining]

    query = make_query_points(32)
    np.save(QUERY_POINTS, query)
    desc, desc_report = build_or_reuse_descriptors(rows)
    rows.to_csv(SAMPLE_INDEX, index=False)

    split_rows = rows[["row_id", "base_sample_id", "condition", "combined_variant", "split"]].copy()
    split_rows = split_rows.rename(columns={"combined_variant": "measurement_variant"})
    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    split_rows.to_csv(SPLIT_CSV, index=False)
    split_payload = {
        "strategy": "preserved v1e 500 parametric grouped split by base_sample_id",
        "source_split_json": rel(PARAM_SPLIT_JSON),
        "source_strategy": split_payload_src.get("strategy"),
        "base_sample_count": int(rows["base_sample_id"].nunique()),
        "row_count": int(len(rows)),
        "base_counts_by_split": rows.drop_duplicates("base_sample_id")["split"].value_counts().to_dict(),
        "row_counts_by_split": rows["split"].value_counts().to_dict(),
        "base_split": dict(rows[["base_sample_id", "split"]].drop_duplicates().to_numpy()),
    }
    SPLIT_JSON.write_text(json.dumps(split_payload, indent=2, sort_keys=True) + "\n")

    required_file_cols = [
        "B_meas_file",
        "coordinate_channels_file",
        "sensor_mask_file",
        "LAT_grid_file",
        "APD80_grid_file",
        "APD90_grid_file",
        "patch_mask_grid_file",
        "valid_mask_grid_file",
        "patch_param_targets_file",
    ]
    missing = []
    shape_errors = []
    for row in rows.itertuples(index=False):
        for col in required_file_cols:
            value = getattr(row, col, "")
            if not value or not (ROOT / value).exists():
                missing.append(f"{row.row_id}:{col}")
        try:
            b = np.load(ROOT / row.B_meas_file)
            coords = np.load(ROOT / row.coordinate_channels_file)
            sensor_mask = np.load(ROOT / row.sensor_mask_file)
            lat = np.load(ROOT / row.LAT_grid_file)
            apd80 = np.load(ROOT / row.APD80_grid_file)
            apd90 = np.load(ROOT / row.APD90_grid_file)
            patch = np.load(ROOT / row.patch_mask_grid_file)
            valid = np.load(ROOT / row.valid_mask_grid_file)
            if b.shape != (301, 32, 32, 4):
                shape_errors.append(f"{row.row_id}:B_meas_shape={b.shape}")
            if coords.shape[:2] != (32, 32) or coords.shape[-1] < 2:
                shape_errors.append(f"{row.row_id}:coordinate_shape={coords.shape}")
            if sensor_mask.shape != (32, 32, 4):
                shape_errors.append(f"{row.row_id}:sensor_mask_shape={sensor_mask.shape}")
            if not (lat.shape == apd80.shape == apd90.shape == patch.shape == valid.shape == (32, 32)):
                shape_errors.append(f"{row.row_id}:target_shapes")
        except Exception as exc:
            shape_errors.append(f"{row.row_id}:{exc!r}")

    leakage = no_group_leakage(split_rows.to_dict(orient="records"))
    descriptor_missing = sorted(set(rows["row_id"]) - set(desc["row_id"]))
    descriptor_extra = sorted(set(desc["row_id"]) - set(rows["row_id"]))
    query_ok = bool(query.shape == (1024, 2) and np.all(np.isfinite(query)) and np.all(query >= 0.0) and np.all(query <= 1.0))
    expected_rows = 1500
    expected_bases = 500
    report = {
        "row_count": int(len(rows)),
        "expected_row_count": expected_rows,
        "base_sample_count": int(rows["base_sample_id"].nunique()),
        "expected_base_sample_count": expected_bases,
        "variant_counts": rows["combined_variant"].value_counts().to_dict(),
        "condition_counts_base": rows.drop_duplicates("base_sample_id")["condition"].value_counts().to_dict(),
        "split_counts_base": split_payload["base_counts_by_split"],
        "split_counts_rows": split_payload["row_counts_by_split"],
        "query_points_file": rel(QUERY_POINTS),
        "query_points_shape": list(query.shape),
        "query_points_normalized_xy_32x32": query_ok,
        "missing_file_count": len(missing),
        "missing_files": missing[:50],
        "shape_error_count": len(shape_errors),
        "shape_errors": shape_errors[:50],
        "group_leakage": leakage,
        "descriptor_rows": int(len(desc)),
        "descriptor_feature_count": int(desc_report["feature_count"]),
        "descriptor_missing_count": len(descriptor_missing),
        "descriptor_missing_rows": descriptor_missing[:50],
        "descriptor_extra_count": len(descriptor_extra),
        "descriptor_extra_rows": descriptor_extra[:50],
        "descriptor_report": rel(DESC_JSON),
        "leakage_guardrail": "DeepONet inputs are B_meas, optional B_meas-derived descriptors, and query coordinates only. Target fields and patch params are labels.",
        "passed": bool(
            len(rows) == expected_rows
            and rows["base_sample_id"].nunique() == expected_bases
            and query_ok
            and not missing
            and not shape_errors
            and not leakage
            and not descriptor_missing
            and not descriptor_extra
            and desc_report["passed"]
        ),
    }
    write_json(REPORT, report)
    print(
        json.dumps(
            {
                "row_count": report["row_count"],
                "base_sample_count": report["base_sample_count"],
                "split_counts_base": report["split_counts_base"],
                "descriptor_feature_count": report["descriptor_feature_count"],
                "passed": report["passed"],
            },
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
