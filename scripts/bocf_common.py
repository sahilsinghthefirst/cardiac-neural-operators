                                                                    

                                                                             
                                                                             
                                  
   

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / "venv_carputils" / "bin" / "python"
EXAMPLE_DIR = Path("/usr/local/lib/opencarp/share/examples/02_EP_tissue/05B_Cellular_Dynamics_Heterogeneity")
MESH_BASE = EXAMPLE_DIR / "TestSlabMesher2D_i"
STIM_VTX = EXAMPLE_DIR / "LeftSideStim"
ALL_TAGS = [0, 1, 2, 3]
ABNORMAL_TAG = 3


def run_command(cmd: list[str], log_dir: Path, name: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
                                                                
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{name}.command.txt").write_text(" ".join(map(str, cmd)) + "\n")
    result = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )
    (log_dir / f"{name}.stdout").write_text(result.stdout)
    (log_dir / f"{name}.stderr").write_text(result.stderr)
    (log_dir / f"{name}.returncode.txt").write_text(f"{result.returncode}\n")
    return result


def read_pts(path: Path) -> np.ndarray:
                                                                  
    rows = path.read_text().splitlines()
    n = int(rows[0].strip())
    pts = np.array([[float(v) for v in line.split()[:3]] for line in rows[1 : n + 1]], dtype=float)
    return pts


def read_elem(path: Path) -> tuple[np.ndarray, np.ndarray]:
                                                                           
    rows = path.read_text().splitlines()
    n = int(rows[0].strip())
    elems: list[list[int]] = []
    tags: list[int] = []
    for line in rows[1 : n + 1]:
        parts = line.split()
        etype = parts[0]
        if etype == "Tr":
            conn_count = 3
        elif etype == "Tt":
            conn_count = 4
        elif etype in {"Qd", "Hx"}:
            conn_count = 4 if etype == "Qd" else 8
        else:
            raise ValueError(f"Unsupported element type {etype!r} in {path}")
        elems.append([int(v) for v in parts[1 : 1 + conn_count]])
        tags.append(int(parts[1 + conn_count]))
    return np.array(elems, dtype=int), np.array(tags, dtype=int)


def copy_reference_mesh(out_dir: Path) -> dict[str, str]:
                                                                                  
    copied: dict[str, str] = {}
    for suffix in [".pts", ".elem", ".lon"]:
        src = MESH_BASE.with_suffix(suffix)
        dst = out_dir / f"mesh{suffix}"
        shutil.copy2(src, dst)
        copied[suffix] = str(dst.relative_to(ROOT))
    shutil.copy2(STIM_VTX.with_suffix(".vtx"), out_dir / "stim_left.vtx")
    return copied


def write_region_masks(out_dir: Path, abnormal_tags: Iterable[int] = (ABNORMAL_TAG,)) -> dict[str, str]:
                                                                                 
    pts = read_pts(out_dir / "mesh.pts")
    elems, tags = read_elem(out_dir / "mesh.elem")
    abnormal_tags = set(abnormal_tags)

    node_mask = np.zeros(len(pts), dtype=int)
    for elem, tag in zip(elems, tags):
        if int(tag) in abnormal_tags:
            node_mask[elem] = 1

    elem_tag_file = out_dir / "element_region_tags.csv"
    with elem_tag_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["element_id", "region_tag", "abnormal"])
        for i, tag in enumerate(tags):
            writer.writerow([i, int(tag), int(int(tag) in abnormal_tags)])

    node_mask_file = out_dir / "region_mask_nodes.csv"
    with node_mask_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x_um", "y_um", "z_um", "abnormal"])
        for i, (xyz, mask) in enumerate(zip(pts, node_mask)):
            writer.writerow([i, xyz[0], xyz[1], xyz[2], int(mask)])

    return {
        "element_region_tags": str(elem_tag_file.relative_to(ROOT)),
        "region_mask_file": str(node_mask_file.relative_to(ROOT)),
    }


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


def base_tissue_command(sim_dir: Path, tend_ms: float = 300.0) -> list[str | int | float]:
    return [
        "openCARP",
        "-simID",
        sim_dir,
        "-meshname",
        MESH_BASE,
        "-dt",
        20,
        "-timedt",
        25,
        "-spacedt",
        1,
        "-tend",
        tend_ms,
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
        "-stim[0].crct.type",
        0,
        "-stim[0].elec.vtx_file",
        STIM_VTX,
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
    ] + physics_options(ALL_TAGS)


def bueno_imp_region_options(region_params: dict[int, str] | None = None) -> list[str | int]:
                                                                      
    region_params = region_params or {}
    if not region_params:
        return [
            "-num_imp_regions",
            1,
            "-imp_region[0].im",
            "Bueno",
        ] + tag_list_options("imp_region[0]", ALL_TAGS)

    normal_tags = [tag for tag in ALL_TAGS if tag not in region_params]
    opts: list[str | int] = ["-num_imp_regions", len(region_params) + (1 if normal_tags else 0)]
    idx = 0
    if normal_tags:
        opts += [f"-imp_region[{idx}].im", "Bueno"]
        opts += tag_list_options(f"imp_region[{idx}]", normal_tags)
        idx += 1
    for tag, param in sorted(region_params.items()):
        opts += [f"-imp_region[{idx}].im", "Bueno", f"-imp_region[{idx}].im_param", param]
        opts += tag_list_options(f"imp_region[{idx}]", [tag])
        idx += 1
    return opts


def gregion_options(scales: dict[int, float] | None = None, base_gil: float = 1.0, anisotropy: float = 4.0) -> list[str | int | float]:
    scales = scales or {}
    if not scales:
        return [
            "-num_gregions",
            1,
            "-gregion[0].g_il",
            base_gil,
            "-gregion[0].g_it",
            base_gil / anisotropy,
            "-gregion[0].g_in",
            base_gil / anisotropy,
        ] + tag_list_options("gregion[0]", ALL_TAGS)

    normal_tags = [tag for tag in ALL_TAGS if tag not in scales]
    opts: list[str | int | float] = ["-num_gregions", len(scales) + (1 if normal_tags else 0)]
    idx = 0
    if normal_tags:
        opts += [
            f"-gregion[{idx}].g_il",
            base_gil,
            f"-gregion[{idx}].g_it",
            base_gil / anisotropy,
            f"-gregion[{idx}].g_in",
            base_gil / anisotropy,
        ] + tag_list_options(f"gregion[{idx}]", normal_tags)
        idx += 1
    for tag, scale in sorted(scales.items()):
        gil = base_gil * scale
        opts += [
            f"-gregion[{idx}].g_il",
            gil,
            f"-gregion[{idx}].g_it",
            gil / anisotropy,
            f"-gregion[{idx}].g_in",
            gil / anisotropy,
        ] + tag_list_options(f"gregion[{idx}]", [tag])
        idx += 1
    return opts


def compute_apd_maps(sim_dir: Path) -> dict[str, str]:
                                                                            
    outputs: dict[str, str] = {}
    act_file = sim_dir / "init_acts_ACTs-thresh.dat"
    rep_file = sim_dir / "init_acts_REPs-thresh.dat"
    if act_file.exists():
        shutil.copy2(act_file, sim_dir / "LAT.dat")
        outputs["lat_file"] = str((sim_dir / "LAT.dat").relative_to(ROOT))
    if rep_file.exists():
        shutil.copy2(rep_file, sim_dir / "RT_repol_proxy.dat")
        outputs["rt_file"] = str((sim_dir / "RT_repol_proxy.dat").relative_to(ROOT))
    if act_file.exists() and rep_file.exists():
        acts = np.loadtxt(act_file)
        reps = np.loadtxt(rep_file)
        apd_proxy = np.where((acts >= 0) & (reps >= 0), reps - acts, -1.0)
        out = sim_dir / "APD_repol_proxy.dat"
        np.savetxt(out, apd_proxy, fmt="%.6f")
        outputs["apd_file"] = str(out.relative_to(ROOT))

    vm_file = sim_dir / "vm.igb"
    if vm_file.exists():
        from carputils.carpio import igb

        vm, _header, times = igb.read(str(vm_file))
        rest = np.nanmean(vm[:, : min(5, vm.shape[1])], axis=1)
        peak_idx = np.nanargmax(vm, axis=1)
        peak = vm[np.arange(vm.shape[0]), peak_idx]
        apa = peak - rest
        for frac, label in [(0.80, "APD80"), (0.90, "APD90")]:
            thresh = rest + (1.0 - frac) * apa
            vals = np.full(vm.shape[0], -1.0, dtype=float)
            for node in range(vm.shape[0]):
                after = vm[node, peak_idx[node] :]
                below = np.where(after <= thresh[node])[0]
                if below.size:
                    vals[node] = times[peak_idx[node] + below[0]] - times[peak_idx[node]]
            out = sim_dir / f"{label}.dat"
            np.savetxt(out, vals, fmt="%.6f")
            outputs[f"{label.lower()}_file"] = str(out.relative_to(ROOT))
    return outputs


def make_vm_snapshot(sim_dir: Path, figure_path: Path, title: str, target_time_ms: float = 60.0) -> None:
                                                       
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from carputils.carpio import igb

    pts = read_pts(sim_dir / "mesh.pts")
    elems, _tags = read_elem(sim_dir / "mesh.elem")
    triangles = elems[:, :3]
    vm, _header, times = igb.read(str(sim_dir / "vm.igb"))
    tidx = int(np.argmin(np.abs(times - target_time_ms)))
    tri = mtri.Triangulation(pts[:, 0] / 1000.0, pts[:, 1] / 1000.0, triangles)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.2, 5.3))
    pc = plt.tripcolor(tri, vm[:, tidx], shading="gouraud", cmap="turbo")
    plt.gca().set_aspect("equal")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(f"{title} Vm at {times[tidx]:.0f} ms")
    cbar = plt.colorbar(pc)
    cbar.set_label("Vm (mV)")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close()


def make_scalar_map(sim_dir: Path, values_file: Path, figure_path: Path, title: str, cbar_label: str) -> None:
                                                
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    pts = read_pts(sim_dir / "mesh.pts")
    elems, _tags = read_elem(sim_dir / "mesh.elem")
    values = np.loadtxt(values_file)
    tri = mtri.Triangulation(pts[:, 0] / 1000.0, pts[:, 1] / 1000.0, elems[:, :3])
    masked = np.ma.masked_where(values < 0, values)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.2, 5.3))
    pc = plt.tripcolor(tri, masked, shading="gouraud", cmap="viridis")
    plt.gca().set_aspect("equal")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(title)
    cbar = plt.colorbar(pc)
    cbar.set_label(cbar_label)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close()


def write_metadata(sim_dir: Path, metadata: dict) -> None:
    metadata = {
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        **metadata,
    }
    (sim_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

