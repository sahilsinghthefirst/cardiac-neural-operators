                                                                
 
                                                                   
                                                                             
                                                                                
                                                                           
                                           
   

from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import time
import zlib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

MODEL = "Bueno"
FORMULATION = "monodomain"
DEFAULT_NBEATS = 20
DEFAULT_BCL_MS = 1000.0
FINAL_BEAT_MS = 300.0

PILOT_MODE = "pilot500"
FULL_MODE = "full5000"
PILOT_COUNT = 500
FULL_COUNT = 5000

CONDITIONS = [
    "homogeneous",
    "apd_repolarization_perturbation",
    "conduction_slowing_perturbation",
    "mixed_apd_conduction_perturbation",
]

CONDITION_LABELS = {
    "homogeneous": 0,
    "apd_repolarization_perturbation": 1,
    "conduction_slowing_perturbation": 2,
    "mixed_apd_conduction_perturbation": 3,
}

COMPOSITIONS = {
    PILOT_MODE: {
        "homogeneous": 75,
        "apd_repolarization_perturbation": 125,
        "conduction_slowing_perturbation": 125,
        "mixed_apd_conduction_perturbation": 175,
    },
    FULL_MODE: {
        "homogeneous": 750,
        "apd_repolarization_perturbation": 1250,
        "conduction_slowing_perturbation": 1250,
        "mixed_apd_conduction_perturbation": 1750,
    },
}

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

FIXED_WINDOWS = {
    "full": None,
    "early_activation_fixed": (0.0, 25.0),
    "mid_recovery_fixed": (25.0, 150.0),
    "late_repolarization_fixed": (150.0, 280.0),
    "broad_st_t_fixed": (25.0, 280.0),
}

MAGNETIC_WINDOWS = [
    "activation_magnetic",
    "repolarization_magnetic",
    "broad_after_activation",
]

SIGNALS = ["MI", "DI", "RMS"]
STATS = ["mean", "max", "auc"]

EP_REQUIRED_FILES = [
    "vm.igb",
    "mesh.pts",
    "mesh.elem",
    "mesh.lon",
    "LAT.dat",
    "RT_repol_proxy.dat",
    "APD_repol_proxy.dat",
    "APD80.dat",
    "APD90.dat",
    "region_mask_nodes.csv",
    "element_region_tags.csv",
    "metadata.json",
]


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def mode_paths(mode: str) -> dict[str, Path]:
    if mode not in {PILOT_MODE, FULL_MODE}:
        raise ValueError(f"Unknown v1c mode {mode!r}")
    suffix = "500_pilot" if mode == PILOT_MODE else "5000"
    return {
        "ep": ROOT / "outputs" / f"bocf_dataset_v1c_{suffix}",
        "mag": ROOT / "outputs" / f"magnetic_dataset_v1c_{suffix}",
        "mag_debug": ROOT / "outputs" / f"magnetic_dataset_v1c_{suffix}_debug_full",
        "aug": ROOT / "outputs" / f"magnetic_dataset_v1c_{suffix}_augmented",
        "ready": ROOT / "outputs" / f"dataset_v1c_{suffix}_ready",
        "aug_ready": ROOT / "outputs" / f"dataset_v1c_{suffix}_augmented_ready",
        "figures": ROOT / "figures" / f"dataset_v1c_{suffix}",
    }


def expected_composition(mode: str, n_samples: int | None = None) -> dict[str, int]:
    base = COMPOSITIONS[mode]
    target = PILOT_COUNT if mode == PILOT_MODE else FULL_COUNT
    if n_samples is None or n_samples == target:
        return dict(base)
    scaled = {k: int(round(v * n_samples / target)) for k, v in base.items()}
    diff = n_samples - sum(scaled.values())
    order = ["mixed_apd_conduction_perturbation", "apd_repolarization_perturbation", "conduction_slowing_perturbation", "homogeneous"]
    i = 0
    while diff != 0:
        key = order[i % len(order)]
        if diff > 0:
            scaled[key] += 1
            diff -= 1
        elif scaled[key] > 0:
            scaled[key] -= 1
            diff += 1
        i += 1
    return scaled


def sample_id_for(condition: str, index: int) -> str:
    prefixes = {
        "homogeneous": "hom",
        "apd_repolarization_perturbation": "apd",
        "conduction_slowing_perturbation": "cond",
        "mixed_apd_conduction_perturbation": "mixed",
    }
    return f"v1c_{prefixes[condition]}_{index:05d}"


def seed_for_text(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def _patch_parameters(rng: random.Random, condition: str) -> dict:
    has_patch = condition != "homogeneous"
    if not has_patch:
        return {
            "patch_shape": "none",
            "patch_center_x": None,
            "patch_center_y": None,
            "patch_radius_mm": None,
            "patch_radius": None,
        }
    radius_mm = rng.uniform(0.7, 2.5)
    radius_um = radius_mm * 1000.0
    margin_um = radius_um + 500.0
    center_x = rng.uniform(margin_um, 10000.0 - margin_um)
    center_y = rng.uniform(margin_um, 10000.0 - margin_um)
    return {
        "patch_shape": "circle",
        "patch_center_x": center_x,
        "patch_center_y": center_y,
        "patch_radius_mm": radius_mm,
        "patch_radius": radius_um,
    }


def create_sample_plan(mode: str, seed: int = 20260525, n_samples: int | None = None) -> list[dict]:
    rng = random.Random(seed)
    comp = expected_composition(mode, n_samples)
    samples: list[dict] = []
    global_index = 0
    for condition in CONDITIONS:
        for local_index in range(comp[condition]):
            sample_seed = rng.randrange(1, 2**31 - 1)
            srng = random.Random(sample_seed)
            patch = _patch_parameters(srng, condition)
            has_apd = condition in {"apd_repolarization_perturbation", "mixed_apd_conduction_perturbation"}
            has_cond = condition in {"conduction_slowing_perturbation", "mixed_apd_conduction_perturbation"}
            apd_scale = srng.uniform(1.03, 1.45) if has_apd else None
            cond_scale = srng.uniform(0.30, 0.90) if has_cond else None
            sample = {
                "sample_id": sample_id_for(condition, local_index),
                "condition": condition,
                "global_index": global_index,
                "condition_index": local_index,
                "seed": sample_seed,
                "model": MODEL,
                "formulation": FORMULATION,
                "apd_parameter": "tau_si" if has_apd else None,
                "apd_scale": apd_scale,
                "conductivity_scale": cond_scale,
                "base_gil": srng.uniform(0.85, 1.15),
                "anisotropy_factor": srng.uniform(3.0, 5.0),
                "stimulus_location": srng.choice(["left", "right", "bottom", "top"]),
                "nbeats": DEFAULT_NBEATS,
                "BCL": DEFAULT_BCL_MS,
                "bcl_ms": DEFAULT_BCL_MS,
                "tend_ms": FINAL_BEAT_MS,
                **patch,
            }
            samples.append(sample)
            global_index += 1
    rng.shuffle(samples)
    return samples


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open()))


def write_csv_rows(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_clean(data), indent=2, sort_keys=True) + "\n")


def json_clean(value):
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    if isinstance(value, tuple):
        return [json_clean(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return json_clean(value.item())
    if isinstance(value, Path):
        return str(value)
    return value


def truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def recursive_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def default_workers() -> int:
    cpus = os.cpu_count() or 1
    return max(1, min(16, cpus - 2))


def save_array(path: Path, array: np.ndarray, compress: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        out = path.with_suffix(".npz")
        np.savez_compressed(out, data=array)
        return rel(out)
    np.save(path, array)
    return rel(path)


def load_array(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path) as data:
            return data["data"]
    return np.load(path)


def array_file(out_dir: Path, stem: str) -> Path | None:
    for suffix in [".npy", ".npz"]:
        p = out_dir / f"{stem}{suffix}"
        if p.exists():
            return p
    return None


def compute_bz_projection(
    j_te3: np.ndarray,
    centroids_e3: np.ndarray,
    areas_e: np.ndarray,
    sensors_s3: np.ndarray,
    mu0_over_4pi: float = 1e-7,
    eps: float = 1e-18,
    chunk_time: int = 64,
) -> np.ndarray:
                                                     
 
                                                          
                                                                      
       

    r_se3 = sensors_s3[:, None, :] - centroids_e3[None, :, :]
    r_norm = np.linalg.norm(r_se3, axis=2)
    weight = areas_e[None, :] / (r_norm**3 + eps)
    kernel_x = (r_se3[:, :, 1] * weight).astype(np.float64)
    kernel_y = (-r_se3[:, :, 0] * weight).astype(np.float64)
    out = np.empty((j_te3.shape[0], sensors_s3.shape[0]), dtype=np.float32)
    for start in range(0, j_te3.shape[0], chunk_time):
        end = min(start + chunk_time, j_te3.shape[0])
        jx = j_te3[start:end, :, 0].astype(np.float64)
        jy = j_te3[start:end, :, 1].astype(np.float64)
        out[start:end] = (mu0_over_4pi * (jx @ kernel_x.T + jy @ kernel_y.T)).astype(np.float32)
    return out


def compute_bvec_projection(
    j_te3: np.ndarray,
    centroids_e3: np.ndarray,
    areas_e: np.ndarray,
    sensors_s3: np.ndarray,
) -> np.ndarray:
    from magnetic_common import biot_savart_projection

    return biot_savart_projection(j_te3, centroids_e3, areas_e, sensors_s3)


def compute_mig_curves_from_bz(b_z: np.ndarray, times: np.ndarray, sensor_mask: np.ndarray | None = None) -> tuple[list[dict], dict]:
    from magnetic_common import compute_mig_curves

    b = np.asarray(b_z, dtype=np.float64)
    if sensor_mask is None:
        sensor_mask = np.ones(b.shape[1], dtype=bool)
    valid = sensor_mask.astype(bool) & np.all(np.isfinite(b), axis=0)
    if not np.any(valid):
        raise ValueError("No valid sensors remain for MIG curve computation")
    rows, summary = compute_mig_curves(b[:, valid], times)
    summary.update(
        {
            "num_total_sensors": int(b.shape[1]),
            "num_valid_sensors": int(np.sum(valid)),
            "dropped_sensor_count": int(b.shape[1] - np.sum(valid)),
        }
    )
    return rows, summary


def write_mig_curves(path: Path, rows: list[dict]) -> None:
    write_csv_rows(path, rows, ["time", "MI", "DI", "RMS"])


def load_curves(path: Path) -> dict[str, np.ndarray]:
    arr = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.atleast_1d(arr[name]).astype(float) for name in arr.dtype.names}


def clean_bounds(start: float, end: float, times: np.ndarray) -> tuple[float, float]:
    lo = float(np.min(times))
    hi = float(np.max(times))
    start = max(lo, start)
    end = min(hi, end)
    if end < start:
        mid = min(max((start + end) / 2.0, lo), hi)
        return mid, mid
    return start, end


def features_for_window(curves: dict[str, np.ndarray], times: np.ndarray, start: float, end: float) -> tuple[dict, int]:
    start, end = clean_bounds(start, end, times)
    mask = (times >= start) & (times <= end)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(times - ((start + end) / 2.0))))
        mask[idx] = True
    t = times[mask]
    out: dict[str, float] = {}
    for signal in SIGNALS:
        vals = curves[signal][mask]
        out[f"{signal}_mean"] = float(np.mean(vals))
        out[f"{signal}_max"] = float(np.max(vals))
        out[f"{signal}_auc"] = float(np.trapezoid(vals, t) if len(t) > 1 else 0.0)
    return out, int(np.sum(mask))


def add_window_features(out: dict, name: str, curves: dict[str, np.ndarray], times: np.ndarray, start: float, end: float) -> None:
    start, end = clean_bounds(start, end, times)
    feats, npts = features_for_window(curves, times, start, end)
    out[f"{name}_start_ms"] = start
    out[f"{name}_end_ms"] = end
    out[f"{name}_n_timepoints"] = npts
    for key, value in feats.items():
        out[f"{name}_{key}"] = value


def fixed_feature_row(sample_id: str, condition: str, curves: dict[str, np.ndarray], times: np.ndarray, id_field: str = "sample_id") -> dict:
    row = {id_field: sample_id, "condition": condition}
    for name, bounds in FIXED_WINDOWS.items():
        start, end = (float(np.min(times)), float(np.max(times))) if bounds is None else bounds
        add_window_features(row, name, curves, times, start, end)
    return row


def magnetic_window_bounds(curves: dict[str, np.ndarray], times: np.ndarray) -> tuple[dict[str, tuple[float, float]], float, float]:
    early = (times >= 0.0) & (times <= 60.0)
    if not np.any(early):
        early = np.ones_like(times, dtype=bool)
    early_idx = np.where(early)[0]
    activation_idx = int(early_idx[np.argmax(curves["RMS"][early])])
    activation_peak = float(times[activation_idx])

    late = times >= 80.0
    if not np.any(late):
        late = np.ones_like(times, dtype=bool)
    late_idx = np.where(late)[0]
    rms_late = curves["RMS"][late]
    mi_late = curves["MI"][late]
    rms_norm = rms_late / (np.max(rms_late) + 1e-30)
    mi_norm = mi_late / (np.max(mi_late) + 1e-30)
    late_peak_idx = int(late_idx[np.argmax(rms_norm + mi_norm)])
    late_peak = float(times[late_peak_idx])
    bounds = {
        "activation_magnetic": (activation_peak - 10.0, activation_peak + 10.0),
        "repolarization_magnetic": (late_peak - 30.0, late_peak + 30.0),
        "broad_after_activation": (activation_peak + 10.0, 280.0),
    }
    return bounds, activation_peak, late_peak


def magnetic_feature_row(sample_id: str, condition: str, curves: dict[str, np.ndarray], times: np.ndarray, id_field: str = "sample_id") -> dict:
    bounds, activation_peak, late_peak = magnetic_window_bounds(curves, times)
    row = {id_field: sample_id, "condition": condition, "activation_peak_ms": activation_peak, "late_peak_ms": late_peak}
    for name, (start, end) in bounds.items():
        add_window_features(row, name, curves, times, start, end)
    return row


def fixed_fieldnames(id_field: str = "sample_id", extra: Iterable[str] = ()) -> list[str]:
    fields = [id_field, *extra, "condition"]
    for window in FIXED_WINDOWS:
        fields += [f"{window}_start_ms", f"{window}_end_ms", f"{window}_n_timepoints"]
        for signal in SIGNALS:
            for stat in STATS:
                fields.append(f"{window}_{signal}_{stat}")
    return fields


def magnetic_fieldnames(id_field: str = "sample_id", extra: Iterable[str] = ()) -> list[str]:
    fields = [id_field, *extra, "condition", "activation_peak_ms", "late_peak_ms"]
    for window in MAGNETIC_WINDOWS:
        fields += [f"{window}_start_ms", f"{window}_end_ms", f"{window}_n_timepoints"]
        for signal in SIGNALS:
            for stat in STATS:
                fields.append(f"{window}_{signal}_{stat}")
    return fields


def valid_values(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr) & (arr >= 0)]


def stats(arr: np.ndarray, prefix: str) -> dict:
    vals = valid_values(arr)
    if vals.size == 0:
        return {f"{prefix}_{k}": math.nan for k in ["mean", "std", "min", "max"]}
    return {
        f"{prefix}_mean": float(np.mean(vals)),
        f"{prefix}_std": float(np.std(vals)),
        f"{prefix}_min": float(np.min(vals)),
        f"{prefix}_max": float(np.max(vals)),
    }


def load_node_mask(path: Path) -> np.ndarray:
    mask = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(mask["abnormal"]).astype(bool)


def mean_or_nan(arr: np.ndarray) -> float:
    vals = valid_values(arr)
    return float(np.mean(vals)) if vals.size else math.nan


def ep_target_row(index_row: dict) -> dict:
    condition = index_row["condition"]
    apd80 = np.loadtxt(ROOT / index_row["apd80_file"])
    apd90 = np.loadtxt(ROOT / index_row["apd90_file"])
    lat = np.loadtxt(ROOT / index_row["lat_file"])
    mask = load_node_mask(ROOT / index_row["mask_file"])
    out = {
        "sample_id": index_row["sample_id"],
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        **stats(apd80, "APD80"),
        **stats(apd90, "APD90"),
        "LAT_min": math.nan,
        "LAT_max": math.nan,
        "LAT_range": math.nan,
        "APD80_inside_patch": math.nan,
        "APD80_outside_patch": math.nan,
        "APD80_patch_delta": math.nan,
        "APD90_inside_patch": math.nan,
        "APD90_outside_patch": math.nan,
        "APD90_patch_delta": math.nan,
        "LAT_inside_patch": math.nan,
        "LAT_outside_patch": math.nan,
        "LAT_patch_delta": math.nan,
        "patch_area_fraction": math.nan,
        "notes": "",
    }
    lat_valid = valid_values(lat)
    if lat_valid.size:
        out["LAT_min"] = float(np.min(lat_valid))
        out["LAT_max"] = float(np.max(lat_valid))
        out["LAT_range"] = float(np.max(lat_valid) - np.min(lat_valid))
    if np.any(mask):
        apd80_in = mean_or_nan(apd80[mask])
        apd80_out = mean_or_nan(apd80[~mask])
        apd90_in = mean_or_nan(apd90[mask])
        apd90_out = mean_or_nan(apd90[~mask])
        lat_in = mean_or_nan(lat[mask])
        lat_out = mean_or_nan(lat[~mask])
        out.update(
            {
                "APD80_inside_patch": apd80_in,
                "APD80_outside_patch": apd80_out,
                "APD80_patch_delta": apd80_in - apd80_out,
                "APD90_inside_patch": apd90_in,
                "APD90_outside_patch": apd90_out,
                "APD90_patch_delta": apd90_in - apd90_out,
                "LAT_inside_patch": lat_in,
                "LAT_outside_patch": lat_out,
                "LAT_patch_delta": lat_in - lat_out,
                "patch_area_fraction": float(np.mean(mask)),
            }
        )
    elif condition != "homogeneous":
        out["notes"] = "non-homogeneous sample has empty patch mask"
    out["y_condition"] = out["condition_label"]
    out["y_apd90_patch_delta"] = out["APD90_patch_delta"]
    out["y_lat_range"] = out["LAT_range"]
    out["y_patch_area_fraction"] = out["patch_area_fraction"]
    return out


TARGET_FIELDS = [
    "sample_id",
    "condition",
    "condition_label",
    "APD80_mean",
    "APD80_std",
    "APD80_min",
    "APD80_max",
    "APD90_mean",
    "APD90_std",
    "APD90_min",
    "APD90_max",
    "LAT_min",
    "LAT_max",
    "LAT_range",
    "APD80_inside_patch",
    "APD80_outside_patch",
    "APD80_patch_delta",
    "APD90_inside_patch",
    "APD90_outside_patch",
    "APD90_patch_delta",
    "LAT_inside_patch",
    "LAT_outside_patch",
    "LAT_patch_delta",
    "patch_area_fraction",
    "y_condition",
    "y_apd90_patch_delta",
    "y_lat_range",
    "y_patch_area_fraction",
    "notes",
]


def grouped_splits(rows: list[dict], seed: int = 20260525) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    by_base: dict[str, dict] = {}
    for row in rows:
        by_base.setdefault(row["base_sample_id"], {"condition": row["condition"], "rows": []})["rows"].append(row)
    by_condition: dict[str, list[str]] = {}
    for base_id, data in by_base.items():
        by_condition.setdefault(data["condition"], []).append(base_id)

    base_split: dict[str, str] = {}
    for condition, base_ids in sorted(by_condition.items()):
        ids = list(base_ids)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(0.70 * n))
        n_val = max(1, int(round(0.15 * n))) if n >= 3 else 0
        if n_train + n_val >= n and n >= 3:
            n_train = n - n_val - 1
        for base_id in ids[:n_train]:
            base_split[base_id] = "train"
        for base_id in ids[n_train : n_train + n_val]:
            base_split[base_id] = "val"
        for base_id in ids[n_train + n_val :]:
            base_split[base_id] = "test"

    split_rows = []
    for row in rows:
        split_rows.append(
            {
                "augmented_sample_id": row["augmented_sample_id"],
                "base_sample_id": row["base_sample_id"],
                "condition": row["condition"],
                "variant": row["variant"],
                "split": base_split[row["base_sample_id"]],
            }
        )
    payload = {
        "created_utc": now_utc(),
        "strategy": "grouped by base_sample_id, stratified by condition with approximate 70/15/15 proportions",
        "base_sample_count": len(by_base),
        "augmented_sample_count": len(split_rows),
        "base_counts_by_split": dict(Counter(base_split.values())),
        "augmented_counts_by_split": dict(Counter(r["split"] for r in split_rows)),
        "base_split": base_split,
    }
    return split_rows, payload


def copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def estimate_scaled(value: float, observed_count: int, target_count: int = FULL_COUNT) -> float:
    if observed_count <= 0:
        return 0.0
    return float(value) * target_count / observed_count


def monotonic_storage_summary(base_dir: Path, observed_count: int) -> dict:
    size = recursive_size_bytes(base_dir)
    return {
        "storage_bytes": int(size),
        "storage_gb": float(size / 1e9),
        "observed_count": int(observed_count),
        "estimated_storage_5000_bytes": int(estimate_scaled(size, observed_count, FULL_COUNT)),
        "estimated_storage_5000_gb": float(estimate_scaled(size, observed_count, FULL_COUNT) / 1e9),
    }
