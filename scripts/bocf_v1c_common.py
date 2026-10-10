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


def default_workers() -> int:
    cpus = os.cpu_count() or 1
    return max(1, min(16, cpus - 2))


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
