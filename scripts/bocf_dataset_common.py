from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MESH = ROOT / "outputs" / "bocf_data_v0" / "homogeneous_smoke"
REFERENCE_MESH = Path(os.environ.get("OPENCARP_MESH_PREFIX", "/usr/local/lib/opencarp/share/examples/02_EP_tissue/05B_Cellular_Dynamics_Heterogeneity/TestSlabMesher2D_i"))
DATASET_BASE = ROOT / "outputs" / "bocf_dataset_v1"
MODEL = "Bueno"
FORMULATION = "monodomain"
NORMAL_TAG = 1
PATCH_TAG = 3
ALL_TAGS = [NORMAL_TAG, PATCH_TAG]
DEFAULT_APD_PARAMETER = "tau_si"


def now_utc() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def read_pts(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    n = int(lines[0])
    return np.array([[float(x) for x in line.split()[:3]] for line in lines[1 : n + 1]], dtype=float)


def read_elem(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    lines = path.read_text().splitlines()
    n = int(lines[0])
    elems: list[list[int]] = []
    tags: list[int] = []
    types: list[str] = []
    for line in lines[1 : n + 1]:
        parts = line.split()
        if parts[0] != "Tr":
            raise ValueError(f"Only Tr triangle elements are supported, found {parts[0]}")
        elems.append([int(parts[1]), int(parts[2]), int(parts[3])])
        tags.append(int(parts[4]))
        types.append(parts[0])
    return np.array(elems, dtype=int), np.array(tags, dtype=int), types


def write_elem(path: Path, elems: np.ndarray, tags: np.ndarray) -> None:
    with path.open("w") as f:
        f.write(f"{len(elems)}\n")
        for elem, tag in zip(elems, tags):
            f.write(f"Tr {elem[0]} {elem[1]} {elem[2]} {int(tag)}\n")


def copy_pts_lon(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_mesh_file(".pts"), out_dir / "mesh.pts")
    shutil.copy2(source_mesh_file(".lon"), out_dir / "mesh.lon")


def create_patch_tags(
    pts: np.ndarray,
    elems: np.ndarray,
    center_xy: tuple[float, float] | None,
    radius_um: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    tags = np.full(len(elems), NORMAL_TAG, dtype=int)
    if center_xy is None or radius_um is None or radius_um <= 0:
        return tags, np.zeros(len(pts), dtype=int)
    centroids = pts[elems, :2].mean(axis=1)
    dist = np.linalg.norm(centroids - np.array(center_xy, dtype=float)[None, :], axis=1)
    tags[dist <= radius_um] = PATCH_TAG
    node_mask = np.zeros(len(pts), dtype=int)
    for elem in elems[tags == PATCH_TAG]:
        node_mask[elem] = 1
    return tags, node_mask


def write_masks(out_dir: Path, pts: np.ndarray, elems: np.ndarray, tags: np.ndarray) -> dict[str, str]:
    node_mask = np.zeros(len(pts), dtype=int)
    for elem in elems[tags == PATCH_TAG]:
        node_mask[elem] = 1
    elem_file = out_dir / "element_region_tags.csv"
    with elem_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["element_id", "region_tag", "abnormal"])
        for i, tag in enumerate(tags):
            writer.writerow([i, int(tag), int(tag == PATCH_TAG)])
    node_file = out_dir / "region_mask_nodes.csv"
    with node_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x_um", "y_um", "z_um", "abnormal"])
        for i, (xyz, mask) in enumerate(zip(pts, node_mask)):
            writer.writerow([i, xyz[0], xyz[1], xyz[2], int(mask)])
    return {
        "element_region_tags": str(elem_file.relative_to(ROOT)),
        "region_mask_nodes": str(node_file.relative_to(ROOT)),
    }


def write_vtx(path: Path, node_ids: Iterable[int]) -> None:
    ids = sorted(int(i) for i in node_ids)
    with path.open("w") as f:
        f.write(f"{len(ids)}\n")
        f.write("extra\n")
        for node_id in ids:
            f.write(f"{node_id}\n")


def stimulus_nodes(pts: np.ndarray, location: str, band_um: float = 500.0) -> np.ndarray:
    x = pts[:, 0]
    y = pts[:, 1]
    if location == "left":
        mask = x <= x.min() + band_um
    elif location == "right":
        mask = x >= x.max() - band_um
    elif location == "bottom":
        mask = y <= y.min() + band_um
    elif location == "top":
        mask = y >= y.max() - band_um
    else:
        raise ValueError(f"Unknown stimulus location {location!r}")
    return np.where(mask)[0]


def tag_list_options(prefix: str, tags: list[int]) -> list[str | int]:
    opts: list[str | int] = [f"-{prefix}.num_IDs", len(tags)]
    for i, tag in enumerate(tags):
        opts += [f"-{prefix}.ID[{i}]", tag]
    return opts


def physics_options(tags: list[int]) -> list[str | int]:
    opts: list[str | int] = ["-num_phys_regions", 2]
    for idx, (name, ptype) in enumerate([("Intracellular domain", 0), ("Extracellular domain", 1)]):
        prefix = f"phys_region[{idx}]"
        opts += [f"-{prefix}.name", name, f"-{prefix}.ptype", ptype]
        opts += tag_list_options(prefix, tags)
    return opts


def imp_options(apd_parameter: str | None, apd_scale: float | None, patch: bool) -> list[str | int]:
    if apd_parameter is None or apd_scale is None or math.isclose(apd_scale, 1.0):
        return ["-num_imp_regions", 1, "-imp_region[0].im", MODEL] + tag_list_options("imp_region[0]", ALL_TAGS)
    if not patch:
        return [
            "-num_imp_regions",
            1,
            "-imp_region[0].im",
            MODEL,
            "-imp_region[0].im_param",
            f"{apd_parameter}*{apd_scale}",
        ] + tag_list_options("imp_region[0]", ALL_TAGS)
    return (
        ["-num_imp_regions", 2, "-imp_region[0].im", MODEL]
        + tag_list_options("imp_region[0]", [NORMAL_TAG])
        + [
            "-imp_region[1].im",
            MODEL,
            "-imp_region[1].im_param",
            f"{apd_parameter}*{apd_scale}",
        ]
        + tag_list_options("imp_region[1]", [PATCH_TAG])
    )


def gregion_options(
    conductivity_scale: float | None,
    patch: bool,
    base_gil: float,
    anisotropy_factor: float,
) -> list[str | int | float]:
    def region(idx: int, tag_list: list[int], gil: float) -> list[str | int | float]:
        return [
            f"-gregion[{idx}].g_il",
            gil,
            f"-gregion[{idx}].g_it",
            gil / anisotropy_factor,
            f"-gregion[{idx}].g_in",
            gil / anisotropy_factor,
        ] + tag_list_options(f"gregion[{idx}]", tag_list)

    if not patch or conductivity_scale is None or math.isclose(conductivity_scale, 1.0):
        return ["-num_gregions", 1] + region(0, ALL_TAGS, base_gil)
    return ["-num_gregions", 2] + region(0, [NORMAL_TAG], base_gil) + region(1, [PATCH_TAG], base_gil * conductivity_scale)


def opencarp_command(
    out_dir: Path,
    apd_parameter: str | None,
    apd_scale: float | None,
    conductivity_scale: float | None,
    stimulus_file: Path,
    patch: bool,
    base_gil: float = 1.0,
    anisotropy_factor: float = 4.0,
    tend_ms: float = 300.0,
    nbeats: int = 1,
    bcl_ms: float = 1000.0,
) -> list[str | int | float | Path]:
    cmd: list[str | int | float | Path] = [
        "openCARP",
        "-simID",
        out_dir,
        "-meshname",
        out_dir / "mesh",
        "-dt",
        20,
        "-timedt",
        25,
        "-spacedt",
        1,
        "-tend",
        tend_ms if nbeats <= 1 else (nbeats - 1) * bcl_ms + tend_ms,
        "-experiment",
        0,
        "-parab_solve",
        1,
        "-mass_lumping",
        0,
        "-bidomain",
        0,
        "-num_stim",
        1,
        "-stim[0].ptcl.start",
        0,
        "-stim[0].pulse.strength",
        60,
        "-stim[0].ptcl.duration",
        10,
        "-stim[0].ptcl.npls",
        nbeats,
        "-stim[0].crct.type",
        0,
        "-stim[0].elec.vtx_file",
        stimulus_file,
        "-num_LATs",
        2,
        "-lats[0].all",
        0,
        "-lats[0].method",
        1,
        "-lats[0].measurand",
        0,
        "-lats[0].threshold",
        -40,
        "-lats[0].ID",
        "ACTs",
        "-lats[1].all",
        0,
        "-lats[1].method",
        1,
        "-lats[1].measurand",
        0,
        "-lats[1].threshold",
        -60,
        "-lats[1].mode",
        1,
        "-lats[1].ID",
        "REPs",
    ]
    if nbeats > 1:
        cmd += ["-stim[0].ptcl.bcl", bcl_ms]
    cmd += physics_options(ALL_TAGS)
    cmd += imp_options(apd_parameter, apd_scale, patch)
    cmd += gregion_options(conductivity_scale, patch, base_gil, anisotropy_factor)
    return cmd


def run_command(cmd: list, out_dir: Path, label: str) -> subprocess.CompletedProcess:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{label}.command.txt").write_text(" ".join(str(x) for x in cmd) + "\n")
    result = subprocess.run([str(x) for x in cmd], cwd=ROOT, text=True, capture_output=True, check=False)
    (out_dir / f"{label}.stdout").write_text(result.stdout)
    (out_dir / f"{label}.stderr").write_text(result.stderr)
    (out_dir / f"{label}.returncode.txt").write_text(f"{result.returncode}\n")
    return result


def load_vm(out_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    vm_file = out_dir / "vm.igb"
    if not vm_file.exists():
        return None, None, {"vm_exists": False}
    from carputils.carpio import igb

    vm_raw, header, times = igb.read(str(vm_file))
    vm_tn = vm_raw.T if vm_raw.shape[0] == int(header.get("x", vm_raw.shape[0])) else vm_raw
    return vm_tn, np.asarray(times), {
        "vm_exists": True,
        "raw_shape": list(vm_raw.shape),
        "time_nodes_shape": list(vm_tn.shape),
        "min_vm": float(np.min(vm_tn)),
        "max_vm": float(np.max(vm_tn)),
    }


def derive_maps(out_dir: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    act = out_dir / "init_acts_ACTs-thresh.dat"
    rep = out_dir / "init_acts_REPs-thresh.dat"
    if act.exists():
        shutil.copy2(act, out_dir / "LAT.dat")
        outputs["LAT"] = str((out_dir / "LAT.dat").relative_to(ROOT))
    if rep.exists():
        shutil.copy2(rep, out_dir / "RT_repol_proxy.dat")
        outputs["RT_repol_proxy"] = str((out_dir / "RT_repol_proxy.dat").relative_to(ROOT))
    if act.exists() and rep.exists():
        acts = np.loadtxt(act)
        reps = np.loadtxt(rep)
        apd = np.where((acts >= 0) & (reps >= 0), reps - acts, -1.0)
        np.savetxt(out_dir / "APD_repol_proxy.dat", apd, fmt="%.6f")
        outputs["APD_repol_proxy"] = str((out_dir / "APD_repol_proxy.dat").relative_to(ROOT))
    vm_tn, times, _debug = load_vm(out_dir)
    if vm_tn is not None and times is not None:
        rest = np.mean(vm_tn[: min(5, vm_tn.shape[0]), :], axis=0)
        peak_idx = np.argmax(vm_tn, axis=0)
        peak = vm_tn[peak_idx, np.arange(vm_tn.shape[1])]
        apa = peak - rest
        for frac, label in [(0.80, "APD80"), (0.90, "APD90")]:
            thresh = rest + (1.0 - frac) * apa
            vals = np.full(vm_tn.shape[1], -1.0)
            for node in range(vm_tn.shape[1]):
                below = np.where(vm_tn[peak_idx[node] :, node] <= thresh[node])[0]
                if below.size:
                    vals[node] = times[peak_idx[node] + below[0]] - times[peak_idx[node]]
            np.savetxt(out_dir / f"{label}.dat", vals, fmt="%.6f")
            outputs[label] = str((out_dir / f"{label}.dat").relative_to(ROOT))
    return outputs


def summarize_outputs(out_dir: Path, pts: np.ndarray, stimulus_location: str) -> dict[str, float | int | bool | None]:
    summary: dict[str, float | int | bool | None] = {}
    lat_file = out_dir / "LAT.dat"
    for name in ["LAT", "APD80", "APD90"]:
        path = out_dir / f"{name}.dat"
        if path.exists():
            arr = np.loadtxt(path)
            valid = arr[np.isfinite(arr) & (arr >= 0)]
            summary[f"{name.lower()}_valid_count"] = int(valid.size)
            summary[f"{name.lower()}_mean"] = float(np.mean(valid)) if valid.size else None
            summary[f"{name.lower()}_std"] = float(np.std(valid)) if valid.size else None
            summary[f"{name.lower()}_min"] = float(np.min(valid)) if valid.size else None
            summary[f"{name.lower()}_max"] = float(np.max(valid)) if valid.size else None
            summary[f"{name.lower()}_range"] = float(np.max(valid) - np.min(valid)) if valid.size else None
    summary["activation_speed_proxy_m_per_s"] = None
    if lat_file.exists():
        lat = np.loadtxt(lat_file)
        speed = activation_speed_proxy(pts, lat, stimulus_location)
        summary["activation_speed_proxy_m_per_s"] = speed
    lat_valid = summary.get("lat_valid_count", 0) or 0
    apd_valid = summary.get("apd90_valid_count", 0) or 0
    summary["propagation_stable"] = bool(lat_valid > 0.9 * len(pts) and apd_valid > 0.8 * len(pts))
    return summary


def activation_speed_proxy(pts: np.ndarray, lat: np.ndarray, stimulus_location: str) -> float | None:
    x = pts[:, 0]
    y = pts[:, 1]
    if stimulus_location == "left":
        dist = x - x.min()
    elif stimulus_location == "right":
        dist = x.max() - x
    elif stimulus_location == "bottom":
        dist = y - y.min()
    elif stimulus_location == "top":
        dist = y.max() - y
    else:
        return None
    valid = np.isfinite(lat) & (lat >= 0) & (dist > 1000) & (dist < dist.max() - 500)
    if valid.sum() < 10:
        return None
    slope, _intercept = np.polyfit(dist[valid], lat[valid], deg=1)
    if slope <= 0:
        return None
    return float((1.0 / slope) * 1e-3)


def prepare_run_mesh_and_stimulus(
    out_dir: Path,
    patch_center: tuple[float, float] | None,
    patch_radius: float | None,
    stimulus_location: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path, dict[str, str]]:
    copy_pts_lon(out_dir)
    pts = read_pts(out_dir / "mesh.pts")
    elems, _old_tags, _types = read_elem(source_mesh_file(".elem"))
    tags, _node_mask = create_patch_tags(pts, elems, patch_center, patch_radius)
    write_elem(out_dir / "mesh.elem", elems, tags)
    masks = write_masks(out_dir, pts, elems, tags)
    stim_file = out_dir / f"stim_{stimulus_location}.vtx"
    write_vtx(stim_file, stimulus_nodes(pts, stimulus_location))
    return pts, elems, tags, stim_file, masks


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def run_ep_sample(
    out_dir: Path,
    sample_id: str,
    condition: str,
    patch_center: tuple[float, float] | None = None,
    patch_radius: float | None = None,
    apd_parameter: str | None = None,
    apd_scale: float | None = None,
    conductivity_scale: float | None = None,
    stimulus_location: str = "left",
    base_gil: float = 1.0,
    anisotropy_factor: float = 4.0,
    tend_ms: float = 300.0,
    nbeats: int = 1,
    seed: int | None = None,
    force: bool = False,
) -> dict:
    meta_file = out_dir / "metadata.json"
    if meta_file.exists() and not force:
        metadata = json.loads(meta_file.read_text())
        if metadata.get("success"):
            metadata["skipped_existing"] = True
            return metadata
    out_dir.mkdir(parents=True, exist_ok=True)
    patch = patch_center is not None and patch_radius is not None and patch_radius > 0
    pts, elems, tags, stim_file, masks = prepare_run_mesh_and_stimulus(out_dir, patch_center, patch_radius, stimulus_location)
    cmd = opencarp_command(
        out_dir=out_dir,
        apd_parameter=apd_parameter,
        apd_scale=apd_scale,
        conductivity_scale=conductivity_scale,
        stimulus_file=stim_file,
        patch=patch,
        base_gil=base_gil,
        anisotropy_factor=anisotropy_factor,
        tend_ms=tend_ms,
        nbeats=nbeats,
    )
    result = run_command(cmd, out_dir, "opencarp")
    derived = derive_maps(out_dir) if result.returncode == 0 else {}
    vm_tn, _times, vm_debug = load_vm(out_dir)
    summary = summarize_outputs(out_dir, pts, stimulus_location) if result.returncode == 0 else {}
    metadata = {
        "created_utc": now_utc(),
        "sample_id": sample_id,
        "condition": condition,
        "model": MODEL,
        "formulation": FORMULATION,
        "geometry": "1 cm x 1 cm openCARP triangular sheet, randomized element tag patch",
        "seed": seed,
        "apd_parameter": apd_parameter,
        "apd_scale": apd_scale,
        "conductivity_scale": conductivity_scale,
        "base_gil": base_gil,
        "anisotropy_factor": anisotropy_factor,
        "patch_center_x": patch_center[0] if patch_center else None,
        "patch_center_y": patch_center[1] if patch_center else None,
        "patch_radius": patch_radius,
        "patch_element_count": int(np.sum(tags == PATCH_TAG)),
        "stimulus_location": stimulus_location,
        "conditioning_strategy": "single-beat tissue run by default; nbeats option recorded but final-beat state conditioning is not yet used",
        "nbeats": nbeats,
        "tend_ms": tend_ms,
        "success": result.returncode == 0 and vm_tn is not None,
        "returncode": result.returncode,
        "command_used": " ".join(str(x) for x in cmd),
        "mask_files": masks,
        "derived_outputs": derived,
        "vm_debug": vm_debug,
        "summary": summary,
        "notes": "" if result.returncode == 0 else "openCARP failed; inspect opencarp stdout/stderr logs",
        "scientific_note": "Controlled synthetic EP benchmark; not clinical ischemia and not current reconstruction.",
    }
    write_json(meta_file, metadata)
    return metadata


def recursive_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def run_conditioned_ep_sample(
    out_dir: Path,
    sample_id: str,
    condition: str,
    prepacing_lats: Path,
    patch_center: tuple[float, float] | None = None,
    patch_radius: float | None = None,
    apd_parameter: str | None = None,
    apd_scale: float | None = None,
    conductivity_scale: float | None = None,
    stimulus_location: str = "left",
    base_gil: float = 1.0,
    anisotropy_factor: float = 4.0,
    tend_ms: float = 300.0,
    nbeats: int = 20,
    bcl_ms: float = 1000.0,
    seed: int | None = None,
    force: bool = False,
) -> dict:


    if nbeats <= 1:
        metadata = run_ep_sample(
            out_dir=out_dir,
            sample_id=sample_id,
            condition=condition,
            patch_center=patch_center,
            patch_radius=patch_radius,
            apd_parameter=apd_parameter,
            apd_scale=apd_scale,
            conductivity_scale=conductivity_scale,
            stimulus_location=stimulus_location,
            base_gil=base_gil,
            anisotropy_factor=anisotropy_factor,
            tend_ms=tend_ms,
            nbeats=1,
            seed=seed,
            force=force,
        )
        metadata.update(
            {
                "conditioning_used": False,
                "BCL": None,
                "bcl_ms": None,
                "prepacing_beats": 0,
                "final_beat_extraction": "single 300 ms tissue beat; no conditioning",
                "output_size_bytes": recursive_size_bytes(out_dir),
            }
        )
        write_json(out_dir / "metadata.json", metadata)
        return metadata

    meta_file = out_dir / "metadata.json"
    if meta_file.exists() and not force:
        metadata = json.loads(meta_file.read_text())
        if metadata.get("success"):
            metadata["skipped_existing"] = True
            return metadata

    if not prepacing_lats.exists():
        raise FileNotFoundError(f"Missing prepacing LAT file: {prepacing_lats}")

    out_dir.mkdir(parents=True, exist_ok=True)
    patch = patch_center is not None and patch_radius is not None and patch_radius > 0
    pts, elems, tags, stim_file, masks = prepare_run_mesh_and_stimulus(out_dir, patch_center, patch_radius, stimulus_location)

    lats = np.loadtxt(prepacing_lats)
    if lats.shape[0] != len(pts):
        raise ValueError(f"Prepacing LAT count {lats.shape[0]} does not match node count {len(pts)}")
    prepacing_lats_dst = out_dir / "prepacing_lats.dat"
    shutil.copy2(prepacing_lats, prepacing_lats_dst)

    cmd = opencarp_command(
        out_dir=out_dir,
        apd_parameter=apd_parameter,
        apd_scale=apd_scale,
        conductivity_scale=conductivity_scale,
        stimulus_file=stim_file,
        patch=patch,
        base_gil=base_gil,
        anisotropy_factor=anisotropy_factor,
        tend_ms=tend_ms,
        nbeats=1,
        bcl_ms=bcl_ms,
    )
    cmd += [
        "-prepacing_beats",
        nbeats - 1,
        "-prepacing_lats",
        prepacing_lats_dst,
        "-prepacing_bcl",
        bcl_ms,
    ]

    start = time.perf_counter()
    result = run_command(cmd, out_dir, "opencarp")
    run_seconds = time.perf_counter() - start
    derived = derive_maps(out_dir) if result.returncode == 0 else {}
    vm_tn, _times, vm_debug = load_vm(out_dir)
    summary = summarize_outputs(out_dir, pts, stimulus_location) if result.returncode == 0 else {}
    metadata = {
        "created_utc": now_utc(),
        "sample_id": sample_id,
        "condition": condition,
        "model": MODEL,
        "formulation": FORMULATION,
        "geometry": "1 cm x 1 cm openCARP triangular sheet, randomized element tag patch",
        "seed": seed,
        "apd_parameter": apd_parameter,
        "apd_scale": apd_scale,
        "conductivity_scale": conductivity_scale,
        "base_gil": base_gil,
        "anisotropy_factor": anisotropy_factor,
        "patch_center_x": patch_center[0] if patch_center else None,
        "patch_center_y": patch_center[1] if patch_center else None,
        "patch_radius": patch_radius,
        "patch_element_count": int(np.sum(tags == PATCH_TAG)),
        "stimulus_location": stimulus_location,
        "conditioning_used": True,
        "conditioning_strategy": "LAT-guided openCARP prepacing followed by one final Bueno monodomain tissue beat",
        "nbeats": nbeats,
        "BCL": bcl_ms,
        "bcl_ms": bcl_ms,
        "prepacing_beats": nbeats - 1,
        "prepacing_lats_file": str(prepacing_lats_dst.relative_to(ROOT)),
        "source_prepacing_lats": str(prepacing_lats.relative_to(ROOT)) if prepacing_lats.is_relative_to(ROOT) else str(prepacing_lats),
        "final_beat_extraction": "openCARP prepacing initializes states; vm.igb, LAT, RT, APD80, and APD90 are from the one saved final 300 ms tissue beat",
        "tend_ms": tend_ms,
        "success": result.returncode == 0 and vm_tn is not None,
        "returncode": result.returncode,
        "command_used": " ".join(str(x) for x in cmd),
        "mask_files": masks,
        "derived_outputs": derived,
        "vm_debug": vm_debug,
        "summary": summary,
        "run_seconds": run_seconds,
        "output_size_bytes": recursive_size_bytes(out_dir),
        "notes": "" if result.returncode == 0 else "openCARP failed; inspect opencarp stdout/stderr logs",
        "scientific_note": "Controlled synthetic EP benchmark; not clinical ischemia and not current reconstruction.",
    }
    write_json(meta_file, metadata)
    return metadata


def manifest_row(metadata: dict, out_dir: Path) -> dict:
    return {
        "sample_id": metadata.get("sample_id"),
        "condition": metadata.get("condition"),
        "model": metadata.get("model", MODEL),
        "formulation": metadata.get("formulation", FORMULATION),
        "geometry": metadata.get("geometry"),
        "apd_parameter": metadata.get("apd_parameter"),
        "apd_scale": metadata.get("apd_scale"),
        "conductivity_scale": metadata.get("conductivity_scale"),
        "patch_center_x": metadata.get("patch_center_x"),
        "patch_center_y": metadata.get("patch_center_y"),
        "patch_radius": metadata.get("patch_radius"),
        "stimulus_location": metadata.get("stimulus_location"),
        "success": metadata.get("success", False),
        "output_dir": str(out_dir.relative_to(ROOT)),
        "notes": metadata.get("notes", ""),
    }


def source_mesh_file(suffix: str) -> Path:
    legacy = SOURCE_MESH / f"mesh{suffix}"
    path = legacy if legacy.exists() else REFERENCE_MESH.with_suffix(suffix)
    if not path.is_file():
        raise FileNotFoundError(f"Missing reference mesh {path}; set OPENCARP_MESH_PREFIX to an installed openCARP example mesh prefix")
    return path
