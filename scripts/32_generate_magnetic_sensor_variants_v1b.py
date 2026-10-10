                      
                                                                    

                                                                     
                                                       
   

from __future__ import annotations

import csv
import json
import shutil
import zlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from magnetic_common import biot_savart_projection, compute_mig_curves


ROOT = Path(__file__).resolve().parents[1]
BASE_MAG = ROOT / "outputs" / "magnetic_dataset_v1b"
AUG_MAG = ROOT / "outputs" / "magnetic_dataset_v1b_augmented"
VARIANTS = [
    "clean",
    "noise_low",
    "noise_medium",
    "noise_high",
    "dropout_10pct",
    "dropout_25pct",
    "jitter_1mm",
    "jitter_3mm",
    "height_30mm",
    "height_70mm",
    "combined_medium",
]


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_manifest() -> list[dict]:
    return list(csv.DictReader((BASE_MAG / "manifest.csv").open()))


def seed_for(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def clean_rms(b_z: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(b_z * b_z)))


def choose_dropout_mask(n_sensors: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    keep = np.ones(n_sensors, dtype=bool)
    n_drop = max(1, int(round(frac * n_sensors)))
    drop = rng.choice(n_sensors, size=min(n_drop, n_sensors - 1), replace=False)
    keep[drop] = False
    return keep


def add_noise(b_vec: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return (b_vec + rng.normal(0.0, sigma, size=b_vec.shape)).astype(np.float32)


def apply_dropout(b_vec: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = b_vec.copy()
    out[:, ~mask, :] = np.nan
    return out, out[:, :, 2]


def compute_curves(b_z: np.ndarray, mask: np.ndarray, times: np.ndarray) -> tuple[list[dict], dict]:
    valid = mask & np.all(np.isfinite(b_z), axis=0)
    if not np.any(valid):
        raise ValueError("No valid sensors remain after masking")
    rows, summary = compute_mig_curves(b_z[:, valid], times)
    summary["num_total_sensors"] = int(mask.size)
    summary["num_valid_sensors"] = int(np.sum(valid))
    summary["dropped_sensor_count"] = int(mask.size - np.sum(valid))
    summary["dropped_channels_encoded_as"] = "NaN in Bz/B_vec with sensor_mask.npy marking valid sensors"
    return rows, summary


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def save_outputs(out_dir: Path, b_vec: np.ndarray, sensors: np.ndarray, mask: np.ndarray, times: np.ndarray, metadata: dict) -> None:
    b_z = b_vec[:, :, 2]
    np.save(out_dir / "B_vec.npy", b_vec.astype(np.float32))
    np.save(out_dir / "Bz.npy", b_z.astype(np.float32))
    np.save(out_dir / "sensor_positions.npy", sensors.astype(np.float64))
    np.save(out_dir / "sensor_mask.npy", mask.astype(bool))
    np.save(out_dir / "time_ms.npy", times.astype(np.float64))
    curves, summary = compute_curves(b_z, mask, times)
    write_csv(out_dir / "mig_curves.csv", curves, ["time", "MI", "DI", "RMS"])
    write_json(out_dir / "mig_summary.json", {**summary, **metadata})
    write_json(out_dir / "augmentation_metadata.json", metadata)


def recompute_b(base_dir: Path, sensors: np.ndarray) -> np.ndarray:
    j_eq = np.load(base_dir / "J_eq_elem.npy")
    centroids = np.load(base_dir / "element_centroids.npy")
    areas = np.load(base_dir / "element_areas.npy")
    return biot_savart_projection(j_eq, centroids, areas, sensors)


def variant_data(base_dir: Path, variant: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    clean_b_vec = np.load(base_dir / "B_vec.npy")
    clean_b_z = np.load(base_dir / "Bz.npy")
    sensors = np.load(base_dir / "sensor_positions.npy")
    mask = np.ones(sensors.shape[0], dtype=bool)
    sigma_base = clean_rms(clean_b_z)
    metadata = {
        "noise_sigma": 0.0,
        "noise_fraction_of_clean_rms": 0.0,
        "dropout_fraction": 0.0,
        "jitter_std_m": 0.0,
        "height_m": float(sensors[0, 2]),
        "B_arrays_drop_encoding": "dropped sensors are NaN; sensor_mask.npy is authoritative",
    }

    if variant == "clean":
        b_vec = clean_b_vec.copy()
    elif variant.startswith("noise_"):
        frac = {"noise_low": 0.01, "noise_medium": 0.05, "noise_high": 0.10}[variant]
        metadata["noise_fraction_of_clean_rms"] = frac
        metadata["noise_sigma"] = frac * sigma_base
        b_vec = add_noise(clean_b_vec, metadata["noise_sigma"], rng)
    elif variant.startswith("dropout_"):
        frac = {"dropout_10pct": 0.10, "dropout_25pct": 0.25}[variant]
        metadata["dropout_fraction"] = frac
        mask = choose_dropout_mask(sensors.shape[0], frac, rng)
        b_vec, _b_z = apply_dropout(clean_b_vec, mask)
    elif variant.startswith("jitter_"):
        std = {"jitter_1mm": 0.001, "jitter_3mm": 0.003}[variant]
        metadata["jitter_std_m"] = std
        sensors = sensors.copy()
        sensors[:, :2] += rng.normal(0.0, std, size=sensors[:, :2].shape)
        b_vec = recompute_b(base_dir, sensors)
    elif variant.startswith("height_"):
        height = {"height_30mm": 0.03, "height_70mm": 0.07}[variant]
        metadata["height_m"] = height
        sensors = sensors.copy()
        sensors[:, 2] = height
        b_vec = recompute_b(base_dir, sensors)
    elif variant == "combined_medium":
        metadata["jitter_std_m"] = 0.001
        metadata["noise_fraction_of_clean_rms"] = 0.05
        metadata["noise_sigma"] = 0.05 * sigma_base
        metadata["dropout_fraction"] = 0.10
        sensors = sensors.copy()
        sensors[:, :2] += rng.normal(0.0, 0.001, size=sensors[:, :2].shape)
        b_vec = recompute_b(base_dir, sensors)
        b_vec = add_noise(b_vec, metadata["noise_sigma"], rng)
        mask = choose_dropout_mask(sensors.shape[0], 0.10, rng)
        b_vec, _b_z = apply_dropout(b_vec, mask)
    else:
        raise ValueError(f"Unknown variant {variant}")
    return b_vec.astype(np.float32), sensors.astype(np.float64), mask, metadata


def process_row(row: dict, variant: str) -> dict:
    base_sample_id = row["sample_id"]
    augmented_sample_id = f"{base_sample_id}__{variant}"
    base_dir = ROOT / row["magnetic_output_dir"]
    out_dir = AUG_MAG / augmented_sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed_for(augmented_sample_id))
    try:
        times = np.load(base_dir / "time_ms.npy")
        if variant == "clean":
            for name in ["B_vec.npy", "Bz.npy", "sensor_positions.npy", "time_ms.npy"]:
                shutil.copy2(base_dir / name, out_dir / name)
            b_vec = np.load(out_dir / "B_vec.npy")
            sensors = np.load(out_dir / "sensor_positions.npy")
            mask = np.ones(sensors.shape[0], dtype=bool)
            metadata_extra = {
                "noise_sigma": 0.0,
                "noise_fraction_of_clean_rms": 0.0,
                "dropout_fraction": 0.0,
                "jitter_std_m": 0.0,
                "height_m": float(sensors[0, 2]),
                "B_arrays_drop_encoding": "dropped sensors are NaN; sensor_mask.npy is authoritative",
            }
        else:
            b_vec, sensors, mask, metadata_extra = variant_data(base_dir, variant, rng)
        metadata = {
            "created_utc": now_utc(),
            "augmented_sample_id": augmented_sample_id,
            "base_sample_id": base_sample_id,
            "condition": row["condition"],
            "variant": variant,
            "source_base_magnetic_dir": row["magnetic_output_dir"],
            "scientific_note": "Sensor/noise/geometry measurement robustness variant; targets inherit base EP sample.",
            "source_definition": "Base source remains J_eq = -D_eff grad(Vm); B fields are synthetic/proxy fields.",
            "success": True,
            **metadata_extra,
        }
        save_outputs(out_dir, b_vec, sensors, mask, times, metadata)
        return manifest_row(augmented_sample_id, base_sample_id, row["condition"], variant, out_dir, True, "")
    except Exception as exc:
        metadata = {
            "created_utc": now_utc(),
            "augmented_sample_id": augmented_sample_id,
            "base_sample_id": base_sample_id,
            "condition": row["condition"],
            "variant": variant,
            "error": repr(exc),
            "success": False,
        }
        write_json(out_dir / "augmentation_metadata.json", metadata)
        return manifest_row(augmented_sample_id, base_sample_id, row["condition"], variant, out_dir, False, repr(exc))


def manifest_row(aug_id: str, base_id: str, condition: str, variant: str, out_dir: Path, success: bool, notes: str) -> dict:
    return {
        "augmented_sample_id": aug_id,
        "base_sample_id": base_id,
        "condition": condition,
        "variant": variant,
        "magnetic_output_dir": rel(out_dir),
        "Bz_file": rel(out_dir / "Bz.npy") if (out_dir / "Bz.npy").exists() else "",
        "B_vec_file": rel(out_dir / "B_vec.npy") if (out_dir / "B_vec.npy").exists() else "",
        "mig_curves_file": rel(out_dir / "mig_curves.csv") if (out_dir / "mig_curves.csv").exists() else "",
        "mig_summary_file": rel(out_dir / "mig_summary.json") if (out_dir / "mig_summary.json").exists() else "",
        "sensor_mask_file": rel(out_dir / "sensor_mask.npy") if (out_dir / "sensor_mask.npy").exists() else "",
        "success": success,
        "notes": notes,
    }


def main() -> int:
    AUG_MAG.mkdir(parents=True, exist_ok=True)
    base_rows = [r for r in read_manifest() if str(r.get("success")).lower() == "true"]
    rows = []
    for row in base_rows:
        for variant in VARIANTS:
            out = process_row(row, variant)
            rows.append(out)
            print(f"{out['augmented_sample_id']}: success={out['success']} {out['notes']}")
    fields = [
        "augmented_sample_id",
        "base_sample_id",
        "condition",
        "variant",
        "magnetic_output_dir",
        "Bz_file",
        "B_vec_file",
        "mig_curves_file",
        "mig_summary_file",
        "sensor_mask_file",
        "success",
        "notes",
    ]
    write_csv(AUG_MAG / "manifest.csv", rows, fields)
    return 0 if len(rows) == 70 * len(VARIANTS) and all(r["success"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
