                      
                                                                      
 
                                                                                
                                                                               
                                                                        
   

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from magnetic_common import compute_element_gradients, load_mesh, load_vm_tn, triangle_geometry
from v1e_direct_common import (
    DIRECT_VARIANTS,
    FIG_DIR,
    MAG_INPUT_DIR,
    PILOT_DIR,
    ROOT,
    ensure_dirs,
    recursive_size_bytes,
    rel,
    require_cuda,
    write_json,
)
from torch_import_utils import import_torch_prefer_env


torch = import_torch_prefer_env(PROJECT_ROOT)
OUT_MANIFEST = PILOT_DIR / "direct_magnetic_manifest.csv"
OUT_REPORT = PILOT_DIR / "direct_magnetic_quality_report.json"


def sensor_grid(mesh, grid: int, height_m: float, margin_fraction: float = 0.1):
    xy_min = mesh.pts_m[:, :2].min(axis=0)
    xy_max = mesh.pts_m[:, :2].max(axis=0)
    span = xy_max - xy_min
    margin = np.where(span > 0, span * margin_fraction, 0.001)
    xs = np.linspace(xy_min[0] - margin[0], xy_max[0] + margin[0], grid)
    ys = np.linspace(xy_min[1] - margin[1], xy_max[1] + margin[1], grid)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    z_plane = float(np.mean(mesh.pts_m[:, 2]))
    sensors = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z_plane + height_m)])
    return sensors.astype(np.float32), xs.astype(np.float32), ys.astype(np.float32), float(z_plane + height_m)


def coordinate_channels(xs: np.ndarray, ys: np.ndarray, height_m: float, mask_any: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    xnorm = 2.0 * (xx - xs.min()) / max(float(xs.max() - xs.min()), 1e-12) - 1.0
    ynorm = 2.0 * (yy - ys.min()) / max(float(ys.max() - ys.min()), 1e-12) - 1.0
    hnorm = np.full_like(xnorm, height_m / 0.05, dtype=np.float32)
    return np.stack([xnorm, ynorm, hnorm, mask_any.astype(np.float32)], axis=-1).astype(np.float32)


def compute_source(ep_dir: Path):
    mesh = load_mesh(ep_dir)
    centroids, areas, grad_coeffs = triangle_geometry(mesh)
    vm_tn, times, vm_debug = load_vm_tn(ep_dir / "vm.igb", mesh.pts_raw.shape[0])
    grad = compute_element_gradients(vm_tn, mesh.elements, grad_coeffs)
    j_eq = (-1.0 * grad).astype(np.float32)
    meta = {
        "source_definition": "J_eq_elem = -D_eff * grad(Vm), D_eff=1.0",
        "source_kind": "equivalent current proxy, not intracellular current reconstruction",
        "num_elements": int(mesh.elements.shape[0]),
        "num_time_steps": int(j_eq.shape[0]),
        "num_nodes": int(mesh.pts_raw.shape[0]),
        "vm_debug": vm_debug,
    }
    return j_eq, centroids.astype(np.float32), areas.astype(np.float32), times.astype(np.float32), mesh, meta


def project_bz_cuda(
    j_te3: np.ndarray,
    centroids_e3: np.ndarray,
    areas_e: np.ndarray,
    sensors_s3: np.ndarray,
    time_chunk: int = 32,
    sensor_chunk: int = 1024,
    mu0_over_4pi: float = 1e-7,
    eps: float = 1e-18,
) -> np.ndarray:
    device = torch.device("cuda")
    j = torch.as_tensor(j_te3, dtype=torch.float32, device=device)
    centroids = torch.as_tensor(centroids_e3, dtype=torch.float32, device=device)
    areas = torch.as_tensor(areas_e, dtype=torch.float32, device=device)
    sensors = torch.as_tensor(sensors_s3, dtype=torch.float32, device=device)
    out = torch.empty((j.shape[0], sensors.shape[0]), dtype=torch.float32, device=device)
    for s0 in range(0, sensors.shape[0], sensor_chunk):
        s1 = min(s0 + sensor_chunk, sensors.shape[0])
        r = sensors[s0:s1, None, :] - centroids[None, :, :]
        rnorm = torch.linalg.norm(r, dim=2)
        weight = areas[None, :] / (rnorm**3 + eps)
        kx = (r[:, :, 1] * weight).transpose(0, 1).contiguous()
        ky = (-r[:, :, 0] * weight).transpose(0, 1).contiguous()
        for t0 in range(0, j.shape[0], time_chunk):
            t1 = min(t0 + time_chunk, j.shape[0])
            out[t0:t1, s0:s1] = mu0_over_4pi * (j[t0:t1, :, 0] @ kx + j[t0:t1, :, 1] @ ky)
        del r, rnorm, weight, kx, ky
    result = out.detach().cpu().numpy()
    del out, j, centroids, areas, sensors
    torch.cuda.empty_cache()
    return result.astype(np.float32)


def save_variant(
    sample_id: str,
    condition: str,
    variant: str,
    b_grid: np.ndarray,
    sensors: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    height_m: float,
    times: np.ndarray,
    sensor_mask: np.ndarray,
    metadata: dict,
) -> dict:
    out_dir = MAG_INPUT_DIR / f"{sample_id}__{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    b_channels = b_grid[:, :, :, None].astype(np.float32)
    mask = sensor_mask[:, :, None].astype(bool)
    coords = coordinate_channels(xs, ys, height_m, mask[:, :, 0])
    np.save(out_dir / "B_meas.npy", b_channels)
    np.save(out_dir / "sensor_positions.npy", sensors.astype(np.float32))
    np.save(out_dir / "sensor_mask.npy", mask)
    np.save(out_dir / "coordinate_channels.npy", coords)
    np.save(out_dir / "time_ms.npy", times.astype(np.float32))
    write_json(out_dir / "measurement_metadata.json", metadata)
    valid = mask[None, :, :, :].repeat(b_channels.shape[0], axis=0)
    vals = b_channels[valid]
    return {
        "row_id": f"{sample_id}__{variant}",
        "base_sample_id": sample_id,
        "condition": condition,
        "measurement_variant": variant,
        "magnetic_input_dir": rel(out_dir),
        "B_meas_file": rel(out_dir / "B_meas.npy"),
        "sensor_positions_file": rel(out_dir / "sensor_positions.npy"),
        "sensor_mask_file": rel(out_dir / "sensor_mask.npy"),
        "coordinate_channels_file": rel(out_dir / "coordinate_channels.npy"),
        "time_ms_file": rel(out_dir / "time_ms.npy"),
        "measurement_metadata_file": rel(out_dir / "measurement_metadata.json"),
        "time_steps": int(b_channels.shape[0]),
        "sensor_grid": int(b_channels.shape[1]),
        "channels": int(b_channels.shape[-1]),
        "success": bool(vals.size and np.all(np.isfinite(vals)) and not np.allclose(vals, 0.0)),
        "notes": "",
    }


def process_sample(row: pd.Series, grid: int, height_m: float, baseline_m: float, noise_seed: int, args) -> tuple[list[dict], dict]:
    sample_id = row["base_sample_id"]
    ep_dir = ROOT / row["output_dir"]
    start = time.perf_counter()
    j_eq, centroids, areas, times, mesh, source_meta = compute_source(ep_dir)
    sensors, xs, ys, z = sensor_grid(mesh, grid, height_m)
    direct_flat = project_bz_cuda(j_eq, centroids, areas, sensors, args.time_chunk, args.sensor_chunk)
    direct = direct_flat.reshape(direct_flat.shape[0], grid, grid)
    sensors_low = sensors.copy()
    sensors_high = sensors.copy()
    sensors_low[:, 2] -= baseline_m / 2.0
    sensors_high[:, 2] += baseline_m / 2.0
    low = project_bz_cuda(j_eq, centroids, areas, sensors_low, args.time_chunk, args.sensor_chunk).reshape(direct.shape)
    high = project_bz_cuda(j_eq, centroids, areas, sensors_high, args.time_chunk, args.sensor_chunk).reshape(direct.shape)
    axial = (high - low) / baseline_m
    rms = float(np.sqrt(np.mean(direct * direct)))
    rng = np.random.default_rng(noise_seed)
    noise = rng.normal(0.0, 0.05 * max(rms, 1e-30), size=direct.shape).astype(np.float32)
    mask_full = np.ones((grid, grid), dtype=bool)
    mask_drop = mask_full.copy().reshape(-1)
    drop_count = max(1, int(round(0.10 * grid * grid)))
    mask_drop[rng.choice(grid * grid, size=drop_count, replace=False)] = False
    mask_drop = mask_drop.reshape(grid, grid)
    masked = direct + noise
    masked[:, ~mask_drop] = 0.0
    rows = []
    payloads = {
        "direct_bz_50mm": (direct, mask_full, "Bz at direct 50 mm sensor plane"),
        "direct_bz_noise_medium": (direct + noise, mask_full, "Bz with Gaussian noise at 5 percent clean RMS"),
        "direct_axial_grad_30mm_baseline": (axial, mask_full, "finite-difference axial gradiometer approximation"),
        "direct_bz_masked_noise": (masked, mask_drop, "Bz with 5 percent noise and 10 percent deterministic random sensor masking"),
    }
    for variant in DIRECT_VARIANTS:
        b, mask, note = payloads[variant]
        metadata = {
            "base_sample_id": sample_id,
            "condition": row["condition"],
            "variant": variant,
            "operator_kind": "direct SQUID-like configurable approximation, not exact hardware emulation",
            "direct_projection": True,
            "source_current": "J_eq = -D_eff grad(Vm), equivalent-current proxy",
            "B_meas_shape": [int(b.shape[0]), grid, grid, 1],
            "sensor_grid": grid,
            "height_m": height_m,
            "baseline_m": baseline_m,
            "sensor_z_m": z,
            "clean_rms": rms,
            "noise_std": 0.05 * max(rms, 1e-30) if "noise" in variant else 0.0,
            "source_metadata": source_meta,
            "channel_semantics": [note],
        }
        rows.append(save_variant(sample_id, row["condition"], variant, b, sensors, xs, ys, height_m, times, mask, metadata))
    return rows, {"base_sample_id": sample_id, "runtime_seconds": time.perf_counter() - start, "clean_rms": rms}


def make_figures(df: pd.DataFrame, runtime_rows: list[dict]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    examples = df.groupby("measurement_variant", group_keys=False).head(1)
    fig, axes = plt.subplots(1, len(examples), figsize=(3.2 * len(examples), 3))
    axes = np.atleast_1d(axes)
    for ax, (_, row) in zip(axes, examples.iterrows()):
        b = np.load(ROOT / row["B_meas_file"])
        im = ax.imshow(b[min(50, b.shape[0] - 1), :, :, 0], origin="lower", cmap="coolwarm")
        ax.set_title(row["measurement_variant"], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "direct_bz_examples.png", dpi=180)
    plt.close(fig)

    rms_rows = []
    for _, row in df.iterrows():
        b = np.load(ROOT / row["B_meas_file"])
        mask = np.load(ROOT / row["sensor_mask_file"]).astype(bool)
        valid = mask[None, :, :, :].repeat(b.shape[0], axis=0)
        rms_rows.append({"variant": row["measurement_variant"], "rms": float(np.sqrt(np.mean(b[valid] ** 2)))})
    rdf = pd.DataFrame(rms_rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    rdf.boxplot(column="rms", by="variant", ax=ax, rot=20)
    fig.suptitle("")
    ax.set_title("Direct Input RMS By Variant")
    ax.set_ylabel("relative units")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "direct_input_rms_by_variant.png", dpi=180)
    plt.close(fig)

    rtdf = pd.DataFrame(runtime_rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rtdf["runtime_seconds"], bins=20)
    ax.set_title("Direct Generation Runtime Per Base Sample")
    ax.set_xlabel("seconds")
    ax.set_ylabel("samples")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "direct_generation_runtime.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-grid", type=int, default=32, choices=[16, 32, 64])
    parser.add_argument("--height-mm", type=float, default=50.0)
    parser.add_argument("--baseline-mm", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--time-chunk", type=int, default=32)
    parser.add_argument("--sensor-chunk", type=int, default=1024)
    args = parser.parse_args()
    ensure_dirs()
    require_cuda(torch)
    pilot = pd.read_csv(PILOT_DIR / "manifest.csv")
    if args.limit:
        pilot = pilot.head(args.limit)
    all_rows = []
    runtime_rows = []
    failures = []
    start = time.perf_counter()
    for i, row in pilot.iterrows():
        try:
            rows, runtime = process_sample(row, args.sensor_grid, args.height_mm / 1000.0, args.baseline_mm / 1000.0, 20260525 + i, args)
            all_rows.extend(rows)
            runtime_rows.append(runtime)
            print(f"{row['base_sample_id']}: direct magnetic {len(rows)} variants ok in {runtime['runtime_seconds']:.2f}s", flush=True)
        except Exception as exc:
            failures.append({"base_sample_id": row["base_sample_id"], "condition": row["condition"], "error": repr(exc)})
            print(f"{row['base_sample_id']}: direct magnetic failed {exc!r}", flush=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_MANIFEST, index=False)
    if len(df):
        make_figures(df, runtime_rows)
    bad_finite = []
    bad_zero = []
    for _, row in df.iterrows():
        b = np.load(ROOT / row["B_meas_file"])
        mask = np.load(ROOT / row["sensor_mask_file"]).astype(bool)
        valid = mask[None, :, :, :].repeat(b.shape[0], axis=0)
        vals = b[valid]
        if not np.all(np.isfinite(vals)):
            bad_finite.append(row["row_id"])
        if vals.size == 0 or np.allclose(vals, 0):
            bad_zero.append(row["row_id"])
    storage = recursive_size_bytes(MAG_INPUT_DIR)
    runtime = time.perf_counter() - start
    per_base_storage = storage / max(int(pilot.shape[0]), 1)
    report = {
        "sensor_grid": args.sensor_grid,
        "height_mm": args.height_mm,
        "baseline_mm": args.baseline_mm,
        "base_samples_requested": int(len(pilot)),
        "variant_count": len(DIRECT_VARIANTS),
        "rows_generated": int(len(df)),
        "expected_rows": int(len(pilot) * len(DIRECT_VARIANTS)),
        "failures": failures,
        "bad_finite": bad_finite,
        "all_zero_valid_signals": bad_zero,
        "condition_counts": df["condition"].value_counts().to_dict() if len(df) else {},
        "variant_counts": df["measurement_variant"].value_counts().to_dict() if len(df) else {},
        "runtime_seconds": runtime,
        "runtime_per_base_seconds_mean": float(np.mean([r["runtime_seconds"] for r in runtime_rows])) if runtime_rows else math.nan,
        "storage_bytes": int(storage),
        "storage_gb": float(storage / 1e9),
        "estimated_500_sample_storage_gb": float(per_base_storage * 500 / 1e9),
        "estimated_5000_sample_storage_gb": float(per_base_storage * 5000 / 1e9),
        "estimated_500_sample_runtime_hours": float(runtime / max(len(pilot), 1) * 500 / 3600.0),
        "estimated_5000_sample_runtime_hours": float(runtime / max(len(pilot), 1) * 5000 / 3600.0),
        "direct_not_upsampled": True,
        "passed": bool(len(df) >= 0.95 * len(pilot) * len(DIRECT_VARIANTS) and not failures and not bad_finite and not bad_zero),
        "scientific_note": "Direct high-resolution SQUID-like synthetic measurement operator; not exact SQUID hardware emulation.",
    }
    write_json(OUT_REPORT, report)
    print(json.dumps({k: report[k] for k in ["rows_generated", "expected_rows", "runtime_seconds", "storage_gb", "passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
