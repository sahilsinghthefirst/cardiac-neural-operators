                                                         
 
                                                                           
                                                                                
                                                                               
   

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "outputs" / "dataset_v1e_direct_magnetic_pilot"
TARGET_DIR = PILOT_DIR / "targets"
MAG_INPUT_DIR = PILOT_DIR / "direct_magnetic_inputs"
COMBINED_INPUT_DIR = PILOT_DIR / "combined_inputs"
READY_DIR = PILOT_DIR / "fno_ready"
COMBINED_READY_DIR = PILOT_DIR / "combined_fno_ready"
NEURAL_DIR = ROOT / "outputs" / "neural_v1e"
FIG_DIR = ROOT / "figures" / "v1e_direct_magnetic_pilot"
NEURAL_FIG_DIR = ROOT / "figures" / "neural_v1e"
CHECKPOINT_DIR = NEURAL_DIR / "checkpoints"
V1C_EP_MANIFEST = ROOT / "outputs" / "bocf_dataset_v1c_5000" / "manifest.csv"
SEED = 20260525

DEFAULT_COUNTS = {
    "homogeneous": 15,
    "apd_repolarization_perturbation": 25,
    "conduction_slowing_perturbation": 25,
    "mixed_apd_conduction_perturbation": 35,
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
    "direct_bz_noise_medium",
    "direct_axial_grad_30mm_baseline",
    "direct_bz_masked_noise",
]


def ensure_dirs() -> None:
    for path in [PILOT_DIR, TARGET_DIR, MAG_INPUT_DIR, COMBINED_INPUT_DIR, READY_DIR, COMBINED_READY_DIR, NEURAL_DIR, FIG_DIR, NEURAL_FIG_DIR, CHECKPOINT_DIR]:
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


def truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def proportional_counts(n_samples: int) -> dict[str, int]:
    if n_samples == 100:
        return dict(DEFAULT_COUNTS)
    if n_samples == 50:
        return {
            "homogeneous": 8,
            "apd_repolarization_perturbation": 12,
            "conduction_slowing_perturbation": 12,
            "mixed_apd_conduction_perturbation": 18,
        }
    total = sum(DEFAULT_COUNTS.values())
    raw = {k: n_samples * v / total for k, v in DEFAULT_COUNTS.items()}
    counts = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = n_samples - sum(counts.values())
    for key, _ in sorted(raw.items(), key=lambda kv: kv[1] - math.floor(kv[1]), reverse=True)[:remainder]:
        counts[key] += 1
    return counts


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


def target_grid_axes(pts: np.ndarray, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = pts[:, 0]
    y = pts[:, 1]
    xs = np.linspace(float(x.min()), float(x.max()), grid_size)
    ys = np.linspace(float(y.min()), float(y.max()), grid_size)
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
    return grid.astype(np.float32), np.isfinite(grid)


def scalar_targets_from_fields(lat: np.ndarray, apd90: np.ndarray, patch: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    v = valid.astype(bool) & np.isfinite(lat) & np.isfinite(apd90)
    out = {"LAT_range": math.nan, "APD90_patch_delta": math.nan, "patch_area_fraction": math.nan}
    if np.any(v):
        lat_vals = lat[v]
        out["LAT_range"] = float(np.max(lat_vals) - np.min(lat_vals))
    patch_bool = patch.astype(bool) & v
    out["patch_area_fraction"] = float(np.mean(patch_bool[v])) if np.any(v) else math.nan
    if np.any(patch_bool) and np.any(v & ~patch_bool):
        out["APD90_patch_delta"] = float(np.mean(apd90[patch_bool]) - np.mean(apd90[v & ~patch_bool]))
    return out


def patch_params_from_mask(patch: np.ndarray, valid: np.ndarray, condition: str) -> dict[str, float | int | str | None]:
    valid_bool = valid.astype(bool)
    patch_bool = patch.astype(bool) & valid_bool
    area_fraction = float(np.mean(patch_bool[valid_bool])) if np.any(valid_bool) else math.nan
    if condition == "homogeneous" or not np.any(patch_bool):
        return {
            "patch_present": 0,
            "patch_center_x_norm": math.nan,
            "patch_center_y_norm": math.nan,
            "patch_area_fraction": 0.0,
            "patch_radius_equiv_norm": math.nan,
            "patch_bbox_xmin_norm": math.nan,
            "patch_bbox_xmax_norm": math.nan,
            "patch_bbox_ymin_norm": math.nan,
            "patch_bbox_ymax_norm": math.nan,
        }
    yy, xx = np.indices(patch_bool.shape)
    xs = xx[patch_bool].astype(np.float64)
    ys = yy[patch_bool].astype(np.float64)
    denom_x = max(patch_bool.shape[1] - 1, 1)
    denom_y = max(patch_bool.shape[0] - 1, 1)
    radius = math.sqrt(max(area_fraction, 0.0) / math.pi)
    return {
        "patch_present": 1,
        "patch_center_x_norm": float(np.mean(xs) / denom_x),
        "patch_center_y_norm": float(np.mean(ys) / denom_y),
        "patch_area_fraction": area_fraction,
        "patch_radius_equiv_norm": float(radius),
        "patch_bbox_xmin_norm": float(np.min(xs) / denom_x),
        "patch_bbox_xmax_norm": float(np.max(xs) / denom_x),
        "patch_bbox_ymin_norm": float(np.min(ys) / denom_y),
        "patch_bbox_ymax_norm": float(np.max(ys) / denom_y),
    }


def grouped_split_rows(rows: list[dict], seed: int = SEED) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    base_by_condition: dict[str, list[str]] = {}
    for row in rows:
        base_by_condition.setdefault(row["condition"], [])
        if row["base_sample_id"] not in base_by_condition[row["condition"]]:
            base_by_condition[row["condition"]].append(row["base_sample_id"])
    base_split: dict[str, str] = {}
    for _condition, base_ids in sorted(base_by_condition.items()):
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
                "row_id": row["row_id"],
                "base_sample_id": row["base_sample_id"],
                "condition": row["condition"],
                "measurement_variant": row["measurement_variant"],
                "split": base_split[row["base_sample_id"]],
            }
        )
    return split_rows, {
        "strategy": "grouped by base_sample_id, stratified by condition with approximate 70/15/15 proportions",
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
            "message": "CUDA unavailable; v1e FNO/CUDA magnetic generation must not silently run on CPU.",
        }
        write_json(NEURAL_DIR / "cuda_required_diagnostic.json", report)
        raise SystemExit("CUDA unavailable; wrote outputs/neural_v1e/cuda_required_diagnostic.json")


@dataclass
class FieldStats:
    input_mean: float
    input_std: float
    target_mean: np.ndarray
    target_std: np.ndarray


def ensure_torch():
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    from torch_import_utils import import_torch_prefer_env

    return import_torch_prefer_env(ROOT)
