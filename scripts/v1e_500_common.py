                                                                              

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "outputs" / "dataset_v1e_500_combined_pilot"
TARGET_DIR = PILOT_DIR / "targets"
DIRECT_INPUT_DIR = PILOT_DIR / "direct_inputs"
COMBINED_INPUT_DIR = PILOT_DIR / "combined_inputs"
READY_DIR = PILOT_DIR / "parametric_ready"
NEURAL_DIR = ROOT / "outputs" / "neural_v1e_500"
FIG_DIR = ROOT / "figures" / "v1e_500_combined_pilot"
NEURAL_FIG_DIR = ROOT / "figures" / "neural_v1e_500"
CHECKPOINT_DIR = NEURAL_DIR / "checkpoints"
V1C_EP_MANIFEST = ROOT / "outputs" / "bocf_dataset_v1c_5000" / "manifest.csv"
SEED = 20260525

COUNTS = {
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
EP_FILES = {
    "vm.igb": "vm_file",
    "mesh.pts": "mesh_pts_file",
    "mesh.elem": "mesh_elem_file",
    "mesh.lon": "mesh_lon_file",
    "LAT.dat": "LAT_file",
    "RT_repol_proxy.dat": "RT_repol_proxy_file",
    "APD80.dat": "APD80_file",
    "APD90.dat": "APD90_file",
    "region_mask_nodes.csv": "region_mask_nodes_file",
    "element_region_tags.csv": "element_region_tags_file",
    "metadata.json": "metadata_file",
}
DIRECT_VARIANTS = [
    "direct_bz_50mm",
    "direct_axial_grad_30mm_baseline",
    "direct_bz_noise_medium",
    "direct_bz_masked_noise",
]
CHANNEL_LAYOUT = DIRECT_VARIANTS
COMBINED_VARIANTS = {
    "combined_bz_axial": ["direct_bz_50mm", "direct_axial_grad_30mm_baseline"],
    "combined_bz_axial_noise": ["direct_bz_50mm", "direct_axial_grad_30mm_baseline", "direct_bz_noise_medium"],
    "combined_all_available": ["direct_bz_50mm", "direct_axial_grad_30mm_baseline", "direct_bz_noise_medium", "direct_bz_masked_noise"],
}


def ensure_dirs() -> None:
    for path in [PILOT_DIR, TARGET_DIR, DIRECT_INPUT_DIR, COMBINED_INPUT_DIR, READY_DIR, NEURAL_DIR, FIG_DIR, NEURAL_FIG_DIR, CHECKPOINT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path | str) -> str:
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_absolute() else str(path)


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
    return np.array([[float(v) for v in line.split()[:3]] for line in rows[1 : n + 1]], dtype=np.float64)


def target_grid_axes(pts: np.ndarray, grid_size: int):
    xs = np.linspace(float(pts[:, 0].min()), float(pts[:, 0].max()), grid_size)
    ys = np.linspace(float(pts[:, 1].min()), float(pts[:, 1].max()), grid_size)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return xs, ys, xx, yy


def nearest_grid(xy: np.ndarray, values: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    query = np.column_stack([xx.ravel(), yy.ravel()])
    out = np.empty(query.shape[0], dtype=np.float32)
    for start in range(0, query.shape[0], 2048):
        end = min(start + 2048, query.shape[0])
        diff = query[start:end, None, :] - xy[None, :, :]
        idx = np.argmin(np.sum(diff * diff, axis=2), axis=1)
        out[start:end] = values[idx]
    return out.reshape(xx.shape)


def interpolate_to_grid(pts: np.ndarray, values: np.ndarray, grid_size: int, method: str = "linear"):
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
    return grid.astype(np.float32), np.isfinite(grid)


def patch_params_from_mask(patch: np.ndarray, valid: np.ndarray, condition: str, eps: float = 1e-6) -> dict:
    valid_bool = valid.astype(bool)
    patch_bool = patch.astype(bool) & valid_bool
    area = float(np.mean(patch_bool[valid_bool])) if np.any(valid_bool) else math.nan
    if condition == "homogeneous" or not np.any(patch_bool):
        return {
            "patch_present": 0,
            "center_x_norm": math.nan,
            "center_y_norm": math.nan,
            "patch_area_fraction": 0.0,
            "equivalent_radius_norm": math.nan,
            "log_area_fraction": math.nan,
            "log_radius": math.nan,
            "area_bin": -1,
            "area_bin_label": "none",
        }
    yy, xx = np.indices(patch_bool.shape)
    xs = xx[patch_bool].astype(np.float64)
    ys = yy[patch_bool].astype(np.float64)
    radius = math.sqrt(max(area, 0.0) / math.pi)
    if area < 0.055:
        area_bin = 0
        area_label = "small"
    elif area < 0.11:
        area_bin = 1
        area_label = "medium"
    else:
        area_bin = 2
        area_label = "large"
    return {
        "patch_present": 1,
        "center_x_norm": float(np.mean(xs) / max(patch_bool.shape[1] - 1, 1)),
        "center_y_norm": float(np.mean(ys) / max(patch_bool.shape[0] - 1, 1)),
        "patch_area_fraction": area,
        "equivalent_radius_norm": float(radius),
        "log_area_fraction": float(math.log(area + eps)),
        "log_radius": float(math.log(radius + eps)),
        "area_bin": area_bin,
        "area_bin_label": area_label,
    }


def grouped_split_rows(rows: list[dict], seed: int = SEED):
    rng = random.Random(seed)
    by_cond: dict[str, list[str]] = {}
    for row in rows:
        by_cond.setdefault(row["condition"], [])
        if row["base_sample_id"] not in by_cond[row["condition"]]:
            by_cond[row["condition"]].append(row["base_sample_id"])
    base_split = {}
    for _condition, ids0 in sorted(by_cond.items()):
        ids = list(ids0)
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
    split_rows = [{"row_id": row["row_id"], "base_sample_id": row["base_sample_id"], "condition": row["condition"], "measurement_variant": row["measurement_variant"], "split": base_split[row["base_sample_id"]]} for row in rows]
    return split_rows, {
        "strategy": "grouped by base_sample_id, stratified by condition",
        "base_sample_count": len(base_split),
        "row_count": len(rows),
        "base_counts_by_split": dict(Counter(base_split.values())),
        "row_counts_by_split": dict(Counter(r["split"] for r in split_rows)),
        "base_split": base_split,
    }


def no_group_leakage(split_rows: list[dict]) -> list[str]:
    seen: dict[str, set[str]] = {}
    for row in split_rows:
        seen.setdefault(row["base_sample_id"], set()).add(row["split"])
    return sorted(k for k, v in seen.items() if len(v) > 1)


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
            "message": "CUDA unavailable; v1e 500 neural training must not silently run on CPU.",
        }
        write_json(NEURAL_DIR / "cuda_required_diagnostic.json", report)
        raise SystemExit("CUDA unavailable; wrote outputs/neural_v1e_500/cuda_required_diagnostic.json")
