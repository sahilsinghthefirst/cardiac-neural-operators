from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EP_BASE = ROOT / "outputs" / "bocf_data_v0"
MAG_BASE = ROOT / "outputs" / "magnetic_v0"
SIM_IDS = [
    "homogeneous_smoke",
    "apd_patch_smoke",
    "conduction_slow_patch_smoke",
    "ischemia_like_mixed_smoke",
]
FIGURE_NAMES = {
    "homogeneous_smoke": "homogeneous",
    "apd_patch_smoke": "apd_patch",
    "conduction_slow_patch_smoke": "conduction_slow",
    "ischemia_like_mixed_smoke": "mixed",
}


@dataclass
class MeshData:
    pts_raw: np.ndarray
    pts_m: np.ndarray
    elements: np.ndarray
    tags: np.ndarray
    element_types: list[str]
    coord_scale_to_m: float
    coord_units: str


def read_pts(path: Path, coord_scale_to_m: float = 1e-6, coord_units: str = "um") -> tuple[np.ndarray, np.ndarray]:
    rows = path.read_text().splitlines()
    n = int(rows[0].strip())
    pts = np.array([[float(v) for v in line.split()[:3]] for line in rows[1 : n + 1]], dtype=np.float64)
    return pts, pts * coord_scale_to_m


def read_elem(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = path.read_text().splitlines()
    n = int(rows[0].strip())
    elems: list[list[int]] = []
    tags: list[int] = []
    types: list[str] = []
    for line in rows[1 : n + 1]:
        parts = line.split()
        etype = parts[0]
        if etype != "Tr":
            raise ValueError(f"Only triangular Tr elements are supported in v0; found {etype!r} in {path}")
        elems.append([int(parts[1]), int(parts[2]), int(parts[3])])
        tags.append(int(parts[4]) if len(parts) > 4 else -1)
        types.append(etype)
    return np.asarray(elems, dtype=np.int64), np.asarray(tags, dtype=np.int64), types


def load_mesh(ep_dir: Path, coord_scale_to_m: float = 1e-6, coord_units: str = "um") -> MeshData:
    pts_raw, pts_m = read_pts(ep_dir / "mesh.pts", coord_scale_to_m, coord_units)
    elements, tags, element_types = read_elem(ep_dir / "mesh.elem")
    return MeshData(
        pts_raw=pts_raw,
        pts_m=pts_m,
        elements=elements,
        tags=tags,
        element_types=element_types,
        coord_scale_to_m=coord_scale_to_m,
        coord_units=coord_units,
    )


def triangle_geometry(mesh: MeshData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:


    tri = mesh.pts_m[mesh.elements]
    p0 = tri[:, 0, :]
    p1 = tri[:, 1, :]
    p2 = tri[:, 2, :]
    e1 = p1 - p0
    e2 = p2 - p0
    normal = np.cross(e1, e2)
    areas = 0.5 * np.linalg.norm(normal, axis=1)
    if np.any(areas <= 0):
        raise ValueError("Degenerate triangle with zero area detected")

    g00 = np.einsum("ij,ij->i", e1, e1)
    g01 = np.einsum("ij,ij->i", e1, e2)
    g11 = np.einsum("ij,ij->i", e2, e2)
    det = g00 * g11 - g01 * g01
    if np.any(np.abs(det) < 1e-30):
        raise ValueError("Degenerate triangle metric detected")

    inv00 = g11 / det
    inv01 = -g01 / det
    inv10 = -g01 / det
    inv11 = g00 / det
    coeff1 = e1 * inv00[:, None] + e2 * inv10[:, None]
    coeff2 = e1 * inv01[:, None] + e2 * inv11[:, None]
    coeff0 = -coeff1 - coeff2
    grad_coeffs = np.stack([coeff0, coeff1, coeff2], axis=1)
    centroids = (p0 + p1 + p2) / 3.0
    return centroids, areas, grad_coeffs


def load_vm_tn(vm_file: Path, node_count: int) -> tuple[np.ndarray, np.ndarray, dict]:

    try:
        from carputils.carpio import igb
    except ImportError:


        import sys

        root = Path(__file__).resolve().parents[1]
        candidates = sorted((root / "venv_carputils" / "lib").glob("python*/site-packages"))
        for site in candidates:
            if str(site) not in sys.path:
                sys.path.append(str(site))
        from carputils.carpio import igb

    raw, header, times = igb.read(str(vm_file))
    raw_shape = list(raw.shape)
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D Vm array from IGB, got shape {raw.shape}")
    if raw.shape[0] == node_count:
        vm_tn = raw.T
        orientation = "nodes_time_transposed_to_time_nodes"
    elif raw.shape[1] == node_count:
        vm_tn = raw
        orientation = "already_time_nodes"
    else:
        raise ValueError(f"Vm shape {raw.shape} is incompatible with node count {node_count}")
    vm_tn = np.asarray(vm_tn, dtype=np.float64)
    debug = {
        "vm_file": str(vm_file.relative_to(ROOT)),
        "raw_shape": raw_shape,
        "normalized_shape_time_nodes": list(vm_tn.shape),
        "orientation": orientation,
        "num_time_steps": int(vm_tn.shape[0]),
        "num_nodes": int(vm_tn.shape[1]),
        "min_vm": float(np.min(vm_tn)),
        "max_vm": float(np.max(vm_tn)),
        "igb_header": {str(k): _json_safe(v) for k, v in header.items()},
        "time_start": float(times[0]) if len(times) else None,
        "time_end": float(times[-1]) if len(times) else None,
        "time_count": int(len(times)),
    }
    return vm_tn, np.asarray(times, dtype=np.float64), debug


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def compute_element_gradients(vm_tn: np.ndarray, elements: np.ndarray, grad_coeffs: np.ndarray, chunk_size: int = 32) -> np.ndarray:

    out = np.empty((vm_tn.shape[0], elements.shape[0], 3), dtype=np.float32)
    n0 = elements[:, 0]
    n1 = elements[:, 1]
    n2 = elements[:, 2]
    c0 = grad_coeffs[:, 0, :]
    c1 = grad_coeffs[:, 1, :]
    c2 = grad_coeffs[:, 2, :]
    for start in range(0, vm_tn.shape[0], chunk_size):
        end = min(start + chunk_size, vm_tn.shape[0])
        chunk = vm_tn[start:end]
        grad = (
            chunk[:, n0, None] * c0[None, :, :]
            + chunk[:, n1, None] * c1[None, :, :]
            + chunk[:, n2, None] * c2[None, :, :]
        )
        out[start:end] = grad.astype(np.float32)
    return out
