                                                             

                                                        
                                                                       
                                                           
   

from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "outputs" / "dataset_v1d_fno_field_pilot"
FIELD_TARGET_DIR = PILOT_DIR / "field_targets"
MAG_INPUT_DIR = PILOT_DIR / "magnetic_inputs"
READY_DIR = PILOT_DIR / "fno_field_ready"
NEURAL_DIR = ROOT / "outputs" / "neural_v1d"
FIG_DIR = ROOT / "figures" / "dataset_v1d_fno_field_pilot"
NEURAL_FIG_DIR = ROOT / "figures" / "neural_v1d"
CHECKPOINT_DIR = NEURAL_DIR / "checkpoints"
V1C_EP_MANIFEST = ROOT / "outputs" / "bocf_dataset_v1c_5000" / "manifest.csv"
SEED = 20260525
CONDITION_COUNTS = {
    "homogeneous": 75,
    "apd_repolarization_perturbation": 125,
    "conduction_slowing_perturbation": 125,
    "mixed_apd_conduction_perturbation": 175,
}
CONDITION_LABELS = {
    "homogeneous": 0,
    "apd_repolarization_perturbation": 1,
    "conduction_slowing_perturbation": 2,
    "mixed_apd_conduction_perturbation": 3,
}
EP_FILES = [
    "vm.igb",
    "mesh.pts",
    "mesh.elem",
    "mesh.lon",
    "LAT.dat",
    "RT_repol_proxy.dat",
    "APD80.dat",
    "APD90.dat",
    "region_mask_nodes.csv",
    "element_region_tags.csv",
    "metadata.json",
]
MAG_VARIANTS = [
    "ideal_bz",
    "axial_gradiometer_approx",
    "planar_gradient_x_approx",
    "planar_gradient_y_approx",
    "combined_bz_axial",
    "noisy_bz_medium",
    "masked_noisy_bz",
]


def ensure_dirs() -> None:
    for path in [PILOT_DIR, FIELD_TARGET_DIR, MAG_INPUT_DIR, READY_DIR, NEURAL_DIR, FIG_DIR, NEURAL_FIG_DIR, CHECKPOINT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path | str) -> str:
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_absolute() else str(path)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open()))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_clean(value):
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    if isinstance(value, tuple):
        return [json_clean(v) for v in value]
    if isinstance(value, np.generic):
        return json_clean(value.item())
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_clean(payload), indent=2, sort_keys=True) + "\n")


def truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_node_map(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float32)
    return np.atleast_1d(arr).astype(np.float32)


def load_region_mask(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return np.atleast_1d(data["abnormal"]).astype(bool)


def read_pts(path: Path) -> np.ndarray:
    rows = path.read_text().splitlines()
    n = int(rows[0].strip())
    pts = np.array([[float(v) for v in line.split()[:3]] for line in rows[1 : n + 1]], dtype=np.float64)
    return pts


def target_grid_axes(pts: np.ndarray, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = pts[:, 0]
    y = pts[:, 1]
    xs = np.linspace(float(x.min()), float(x.max()), grid_size)
    ys = np.linspace(float(y.min()), float(y.max()), grid_size)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return xs, ys, xx, yy


def interpolate_to_grid(pts: np.ndarray, values: np.ndarray, grid_size: int, method: str = "linear") -> tuple[np.ndarray, np.ndarray]:
                                                     

                                                                            
                                                                                
                          
       
    _xs, _ys, xx, yy = target_grid_axes(pts, grid_size)
    xy = pts[:, :2]
    finite = np.isfinite(values)
    try:
        from scipy.interpolate import griddata

        nearest = griddata(xy[finite], values[finite], (xx, yy), method="nearest")
        if method == "nearest":
            grid = nearest
        else:
            linear = griddata(xy[finite], values[finite], (xx, yy), method="linear")
            grid = np.where(np.isfinite(linear), linear, nearest)
    except Exception:
        grid = nearest_grid(xy[finite], values[finite], xx, yy)
    valid = np.isfinite(grid)
    return grid.astype(np.float32), valid.astype(bool)


def nearest_grid(xy: np.ndarray, values: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    query = np.column_stack([xx.ravel(), yy.ravel()])
    out = np.empty(query.shape[0], dtype=np.float32)
    chunk = 2048
    for start in range(0, query.shape[0], chunk):
        end = min(start + chunk, query.shape[0])
        diff = query[start:end, None, :] - xy[None, :, :]
        idx = np.argmin(np.sum(diff * diff, axis=2), axis=1)
        out[start:end] = values[idx]
    return out.reshape(xx.shape)


def scalar_targets_from_fields(lat: np.ndarray, apd90: np.ndarray, patch: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    v = valid.astype(bool) & np.isfinite(lat) & np.isfinite(apd90)
    out = {
        "LAT_range": math.nan,
        "APD90_patch_delta": math.nan,
        "patch_area_fraction": math.nan,
    }
    if np.any(v):
        lat_vals = lat[v]
        out["LAT_range"] = float(np.max(lat_vals) - np.min(lat_vals))
    patch_bool = patch.astype(bool) & v
    out["patch_area_fraction"] = float(np.mean(patch_bool[v])) if np.any(v) else math.nan
    if np.any(patch_bool) and np.any(v & ~patch_bool):
        out["APD90_patch_delta"] = float(np.mean(apd90[patch_bool]) - np.mean(apd90[v & ~patch_bool]))
    return out


def grouped_split_rows(rows: list[dict], seed: int = SEED) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    base_by_condition: dict[str, list[str]] = {}
    for row in rows:
        base_by_condition.setdefault(row["condition"], [])
        if row["base_sample_id"] not in base_by_condition[row["condition"]]:
            base_by_condition[row["condition"]].append(row["base_sample_id"])
    base_split: dict[str, str] = {}
    for condition, base_ids in sorted(base_by_condition.items()):
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
                "field_sample_id": row["field_sample_id"],
                "base_sample_id": row["base_sample_id"],
                "condition": row["condition"],
                "measurement_variant": row["measurement_variant"],
                "split": base_split[row["base_sample_id"]],
            }
        )
    payload = {
        "strategy": "grouped by base_sample_id, stratified by condition with approximate 70/15/15 proportions",
        "base_sample_count": len(base_split),
        "row_count": len(rows),
        "base_counts_by_split": dict(Counter(base_split.values())),
        "row_counts_by_split": dict(Counter(r["split"] for r in split_rows)),
        "base_split": base_split,
    }
    return split_rows, payload


def no_group_leakage(split_rows: list[dict]) -> list[str]:
    seen: dict[str, set[str]] = {}
    for row in split_rows:
        seen.setdefault(row["base_sample_id"], set()).add(row["split"])
    return sorted(base for base, splits in seen.items() if len(splits) > 1)


def recursive_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def require_cuda(torch_module) -> None:
    if not torch_module.cuda.is_available():
        report = {
            "torch_file": str(Path(torch_module.__file__).resolve()),
            "torch_version": torch_module.__version__,
            "torch_version_cuda": torch_module.version.cuda,
            "cuda_available": False,
            "message": "CUDA unavailable; v1d neural training must not silently run on CPU.",
        }
        write_json(NEURAL_DIR / "cuda_required_diagnostic.json", report)
        raise SystemExit("CUDA unavailable; wrote outputs/neural_v1d/cuda_required_diagnostic.json")


@dataclass
class FieldNormStats:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

