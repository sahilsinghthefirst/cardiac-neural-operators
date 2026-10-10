                      
                                                                              

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from v1e_direct_common import COMBINED_INPUT_DIR, PILOT_DIR, ROOT, ensure_dirs, rel, write_json


OUT_MANIFEST = PILOT_DIR / "combined_input_manifest.csv"
OUT_REPORT = PILOT_DIR / "combined_input_quality_report.json"

COMBINED = {
    "combined_bz_axial": ["direct_bz_50mm", "direct_axial_grad_30mm_baseline"],
    "combined_bz_axial_noise": ["direct_bz_50mm", "direct_axial_grad_30mm_baseline", "direct_bz_noise_medium"],
    "combined_all_available": ["direct_bz_50mm", "direct_axial_grad_30mm_baseline", "direct_bz_noise_medium", "direct_bz_masked_noise"],
}
CHANNEL_LAYOUT = ["direct_bz_50mm", "direct_axial_grad_30mm_baseline", "direct_bz_noise_medium", "direct_bz_masked_noise"]


def load_variant(group: pd.DataFrame, variant: str):
    row = group[group["measurement_variant"].eq(variant)]
    if row.empty:
        raise ValueError(f"Missing variant {variant}")
    row = row.iloc[0]
    b = np.load(ROOT / row["B_meas_file"]).astype(np.float32)
    mask = np.load(ROOT / row["sensor_mask_file"]).astype(bool)
    pos = np.load(ROOT / row["sensor_positions_file"])
    coords = np.load(ROOT / row["coordinate_channels_file"]).astype(np.float32)
    return row, b[:, :, :, 0], mask[:, :, 0], pos, coords


def main() -> int:
    ensure_dirs()
    direct = pd.read_csv(PILOT_DIR / "direct_magnetic_manifest.csv")
    out_rows = []
    failures = []
    for sample_id, group in direct.groupby("base_sample_id"):
        condition = group["condition"].iloc[0]
        for combined_variant, variants in COMBINED.items():
            try:
                arrays_by_variant = {}
                masks_by_variant = {}
                rows = []
                pos = None
                coords = None
                for variant in variants:
                    row, b, mask, pos_i, coords_i = load_variant(group, variant)
                    arrays_by_variant[variant] = b
                    masks_by_variant[variant] = mask
                    rows.append(row)
                    pos = pos_i if pos is None else pos
                    coords = coords_i if coords is None else coords
                shape0 = next(iter(arrays_by_variant.values())).shape
                if any(arr.shape != shape0 for arr in arrays_by_variant.values()):
                    raise ValueError("Shape mismatch among source variants")
                arrays = []
                masks = []
                for channel in CHANNEL_LAYOUT:
                    if channel in arrays_by_variant:
                        arrays.append(arrays_by_variant[channel])
                        masks.append(masks_by_variant[channel])
                    else:
                        arrays.append(np.zeros(shape0, dtype=np.float32))
                        masks.append(np.zeros(shape0[1:], dtype=bool))
                b_meas = np.stack(arrays, axis=-1).astype(np.float32)
                sensor_mask = np.stack(masks, axis=-1).astype(bool)
                out_dir = COMBINED_INPUT_DIR / f"{sample_id}__{combined_variant}"
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / "B_meas.npy", b_meas)
                np.save(out_dir / "sensor_positions.npy", pos)
                np.save(out_dir / "sensor_mask.npy", sensor_mask)
                np.save(out_dir / "coordinate_channels.npy", coords[:, :, :3].astype(np.float32))
                np.save(out_dir / "time_ms.npy", np.load(ROOT / rows[0]["time_ms_file"]).astype(np.float32))
                metadata = {
                    "base_sample_id": sample_id,
                    "condition": condition,
                    "combined_variant": combined_variant,
                    "source_variants": variants,
                    "channel_layout": CHANNEL_LAYOUT,
                    "active_channel_definitions": variants,
                    "tensor_shape": list(b_meas.shape),
                    "leakage_guardrail": "Only magnetic measurements and coordinate channels are included; no target fields, condition, or patch params are model inputs.",
                    "scientific_note": "Combined direct SQUID-like channels are configurable approximations, not exact SQUID hardware emulation.",
                }
                write_json(out_dir / "measurement_metadata.json", metadata)
                valid = sensor_mask[None, :, :, :].repeat(b_meas.shape[0], axis=0)
                vals = b_meas[valid]
                success = bool(vals.size and np.all(np.isfinite(vals)) and not np.allclose(vals, 0.0))
                out_rows.append(
                    {
                        "row_id": f"{sample_id}__{combined_variant}",
                        "base_sample_id": sample_id,
                        "condition": condition,
                        "measurement_variant": combined_variant,
                        "magnetic_input_dir": rel(out_dir),
                        "B_meas_file": rel(out_dir / "B_meas.npy"),
                        "sensor_positions_file": rel(out_dir / "sensor_positions.npy"),
                        "sensor_mask_file": rel(out_dir / "sensor_mask.npy"),
                        "coordinate_channels_file": rel(out_dir / "coordinate_channels.npy"),
                        "time_ms_file": rel(out_dir / "time_ms.npy"),
                        "measurement_metadata_file": rel(out_dir / "measurement_metadata.json"),
                        "time_steps": int(b_meas.shape[0]),
                        "sensor_grid": int(b_meas.shape[1]),
                        "channels": int(b_meas.shape[-1]),
                        "success": success,
                        "notes": "",
                    }
                )
            except Exception as exc:
                failures.append({"base_sample_id": sample_id, "condition": condition, "combined_variant": combined_variant, "error": repr(exc)})
    df = pd.DataFrame(out_rows)
    df.to_csv(OUT_MANIFEST, index=False)
    bad = []
    for _, row in df.iterrows():
        b = np.load(ROOT / row["B_meas_file"])
        mask = np.load(ROOT / row["sensor_mask_file"]).astype(bool)
        valid = mask[None, :, :, :].repeat(b.shape[0], axis=0)
        vals = b[valid]
        if not vals.size or not np.all(np.isfinite(vals)) or np.allclose(vals, 0):
            bad.append(row["row_id"])
    report = {
        "rows_generated": int(len(df)),
        "expected_rows": int(direct["base_sample_id"].nunique() * len(COMBINED)),
        "base_samples": int(direct["base_sample_id"].nunique()),
        "combined_variants": list(COMBINED),
        "variant_counts": df["measurement_variant"].value_counts().to_dict() if len(df) else {},
        "channel_counts": df.groupby("measurement_variant")["channels"].first().to_dict() if len(df) else {},
        "failures": failures,
        "bad_rows": bad,
        "passed": bool(len(df) == direct["base_sample_id"].nunique() * len(COMBINED) and not failures and not bad),
    }
    write_json(OUT_REPORT, report)
    print(json.dumps({k: report[k] for k in ["rows_generated", "expected_rows", "variant_counts", "passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
