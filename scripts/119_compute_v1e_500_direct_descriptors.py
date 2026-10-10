                      
                                                                            

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from v1e_500_common import PILOT_DIR, ROOT, write_json


FIELD_READY = PILOT_DIR / "field_ready"
OUT_CSV = FIELD_READY / "direct_descriptors.csv"
OUT_JSON = FIELD_READY / "direct_descriptors.json"


def channel_desc(primary: np.ndarray, mask2: np.ndarray, prefix: str) -> dict[str, float]:
    if not np.any(mask2):
        return {f"{prefix}_{name}": 0.0 for name in ["rms", "max_abs", "early_rms", "late_rms", "peak_time", "energy_cx", "energy_cy", "entropy", "concentration", "lr_asym", "tb_asym", "mean_abs_grad"]}
    vals = primary[:, mask2]
    rms_t = np.sqrt(np.mean(vals * vals, axis=1)) if vals.size else np.zeros(primary.shape[0])
    energy = np.mean(primary * primary, axis=0)
    total = float(np.sum(energy[mask2]))
    g = primary.shape[1]
    yy, xx = np.indices((g, g))
    if total > 0:
        cx = float(np.sum(xx[mask2] * energy[mask2]) / total / max(g - 1, 1))
        cy = float(np.sum(yy[mask2] * energy[mask2]) / total / max(g - 1, 1))
        p = (energy[mask2].ravel() / total).clip(1e-30, None)
        entropy = float(-np.sum(p * np.log(p)) / np.log(max(p.size, 2)))
        concentration = float(np.max(p))
    else:
        cx = cy = entropy = concentration = 0.0
    left = energy[:, : g // 2][mask2[:, : g // 2]].sum()
    right = energy[:, g // 2 :][mask2[:, g // 2 :]].sum()
    bottom = energy[: g // 2, :][mask2[: g // 2, :]].sum()
    top = energy[g // 2 :, :][mask2[g // 2 :, :]].sum()
    gy, gx = np.gradient(np.mean(primary, axis=0))
    return {
        f"{prefix}_rms": float(np.sqrt(np.mean(vals * vals))) if vals.size else 0.0,
        f"{prefix}_max_abs": float(np.max(np.abs(vals))) if vals.size else 0.0,
        f"{prefix}_early_rms": float(np.mean(rms_t[:25])),
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


def descriptor_row(row) -> dict:
    b = np.load(ROOT / row.B_meas_file).astype(np.float64)
    mask = np.load(ROOT / row.sensor_mask_file).astype(bool)
    valid = mask[None, :, :, :].repeat(b.shape[0], axis=0)
    vals = b[valid]
    out = {
        "row_id": row.row_id,
        "base_sample_id": row.base_sample_id,
        "condition": row.condition,
        "combined_variant": row.combined_variant,
        "split": row.split,
        "global_rms": float(np.sqrt(np.mean(vals * vals))) if vals.size else math.nan,
        "max_abs_B": float(np.max(np.abs(vals))) if vals.size else math.nan,
    }
    for c in range(b.shape[-1]):
        out.update(channel_desc(np.where(mask[:, :, c][None, :, :], b[:, :, :, c], 0.0), mask[:, :, c], f"ch{c}"))
    return out


def main() -> int:
    rows = pd.read_csv(FIELD_READY / "sample_index.csv")
    desc = pd.DataFrame([descriptor_row(row) for row in rows.itertuples(index=False)])
    desc.to_csv(OUT_CSV, index=False)
    feature_cols = [c for c in desc.columns if c not in {"row_id", "base_sample_id", "condition", "combined_variant", "split"}]
    finite = np.isfinite(desc[feature_cols].to_numpy(float)).all()
    report = {
        "row_count": int(len(desc)),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "windows": {
            "activation_window_rms_proxy": "fixed samples 0:25, not LAT-defined",
            "late_window_rms_proxy": "fixed samples 150:280, not RT/APD-defined",
        },
        "passed": bool(len(desc) == 1500 and finite),
        "leakage_guardrail": "Descriptors use only B_meas and fixed time windows, not LAT/APD/patch targets.",
    }
    write_json(OUT_JSON, report)
    print(json.dumps({"row_count": report["row_count"], "feature_count": report["feature_count"], "passed": report["passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
