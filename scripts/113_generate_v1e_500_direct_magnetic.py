from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from magnetic_common import compute_element_gradients, load_mesh, load_vm_tn, triangle_geometry
from torch_import_utils import import_torch_prefer_env
from v1e_500_common import (
    CHANNEL_LAYOUT,
    COMBINED_INPUT_DIR,
    COMBINED_VARIANTS,
    DIRECT_INPUT_DIR,
    DIRECT_VARIANTS,
    FIG_DIR,
    PILOT_DIR,
    ROOT,
    ensure_dirs,
    recursive_size_bytes,
    rel,
    require_cuda,
    write_json,
)


torch = import_torch_prefer_env(ROOT)
COMBINED_MANIFEST = PILOT_DIR / "combined_input_manifest.csv"
REPORT_PATH = PILOT_DIR / "direct_generation_quality_report.json"


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


def coordinate_channels(xs: np.ndarray, ys: np.ndarray, height_m: float) -> np.ndarray:
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    xnorm = 2.0 * (xx - xs.min()) / max(float(xs.max() - xs.min()), 1e-12) - 1.0
    ynorm = 2.0 * (yy - ys.min()) / max(float(ys.max() - ys.min()), 1e-12) - 1.0
    hnorm = np.full_like(xnorm, height_m / 0.05, dtype=np.float32)
    return np.stack([xnorm, ynorm, hnorm], axis=-1).astype(np.float32)


def compute_source(ep_dir):
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
    time_chunk: int,
    sensor_chunk: int,
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


def save_direct(sample_id, condition, variant, b_grid, sensors, coords, times, mask, metadata):
    out_dir = DIRECT_INPUT_DIR / f"{sample_id}__{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    b = b_grid[:, :, :, None].astype(np.float32)
    mask3 = mask[:, :, None].astype(bool)
    np.save(out_dir / "B_meas.npy", b)
    np.save(out_dir / "sensor_positions.npy", sensors.astype(np.float32))
    np.save(out_dir / "sensor_mask.npy", mask3)
    np.save(out_dir / "coordinate_channels.npy", coords.astype(np.float32))
    np.save(out_dir / "time_ms.npy", times.astype(np.float32))
    write_json(out_dir / "measurement_metadata.json", metadata)
    valid = mask3[None, :, :, :].repeat(b.shape[0], axis=0)
    vals = b[valid]
    return {
        "row_id": f"{sample_id}__{variant}",
        "base_sample_id": sample_id,
        "condition": condition,
        "measurement_variant": variant,
        "input_dir": rel(out_dir),
        "B_meas_file": rel(out_dir / "B_meas.npy"),
        "sensor_positions_file": rel(out_dir / "sensor_positions.npy"),
        "sensor_mask_file": rel(out_dir / "sensor_mask.npy"),
        "coordinate_channels_file": rel(out_dir / "coordinate_channels.npy"),
        "time_ms_file": rel(out_dir / "time_ms.npy"),
        "measurement_metadata_file": rel(out_dir / "measurement_metadata.json"),
        "channels": 1,
        "success": bool(vals.size and np.all(np.isfinite(vals)) and not np.allclose(vals, 0.0)),
    }


def save_combined(sample_id, condition, variant, payloads, sensors, coords, times, metadata):
    out_dir = COMBINED_INPUT_DIR / f"{sample_id}__{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    active = set(COMBINED_VARIANTS[variant])
    template = next(iter(payloads.values()))[0]
    b = np.zeros((*template.shape, len(CHANNEL_LAYOUT)), dtype=np.float32)
    mask = np.zeros((template.shape[1], template.shape[2], len(CHANNEL_LAYOUT)), dtype=bool)
    for c, channel_name in enumerate(CHANNEL_LAYOUT):
        if channel_name in active:
            b[:, :, :, c] = payloads[channel_name][0]
            mask[:, :, c] = payloads[channel_name][1]
    np.save(out_dir / "B_meas.npy", b)
    np.save(out_dir / "sensor_positions.npy", sensors.astype(np.float32))
    np.save(out_dir / "sensor_mask.npy", mask)
    np.save(out_dir / "coordinate_channels.npy", coords.astype(np.float32))
    np.save(out_dir / "time_ms.npy", times.astype(np.float32))
    write_json(out_dir / "measurement_metadata.json", metadata)
    valid = mask[None, :, :, :].repeat(b.shape[0], axis=0)
    vals = b[valid]
    return {
        "row_id": f"{sample_id}__{variant}",
        "base_sample_id": sample_id,
        "condition": condition,
        "measurement_variant": variant,
        "combined_channel_names": "|".join(CHANNEL_LAYOUT),
        "active_channel_names": "|".join(COMBINED_VARIANTS[variant]),
        "input_dir": rel(out_dir),
        "B_meas_file": rel(out_dir / "B_meas.npy"),
        "sensor_positions_file": rel(out_dir / "sensor_positions.npy"),
        "sensor_mask_file": rel(out_dir / "sensor_mask.npy"),
        "coordinate_channels_file": rel(out_dir / "coordinate_channels.npy"),
        "time_ms_file": rel(out_dir / "time_ms.npy"),
        "measurement_metadata_file": rel(out_dir / "measurement_metadata.json"),
        "time_steps": int(b.shape[0]),
        "sensor_grid": int(b.shape[1]),
        "channels": int(b.shape[-1]),
        "success": bool(vals.size and np.all(np.isfinite(vals)) and not np.allclose(vals, 0.0)),
        "notes": "",
    }


def process_sample(row, args):
    sample_id = row["base_sample_id"]
    ep_dir = ROOT / row["output_dir"]
    start = time.perf_counter()
    j_eq, centroids, areas, times, mesh, source_meta = compute_source(ep_dir)
    sensors, xs, ys, z = sensor_grid(mesh, args.sensor_grid, args.height_m)
    coords = coordinate_channels(xs, ys, args.height_m)
    direct = project_bz_cuda(j_eq, centroids, areas, sensors, args.time_chunk, args.sensor_chunk).reshape(
        len(times), args.sensor_grid, args.sensor_grid
    )
    sensors_low = sensors.copy()
    sensors_high = sensors.copy()
    sensors_low[:, 2] -= args.baseline_m / 2.0
    sensors_high[:, 2] += args.baseline_m / 2.0
    low = project_bz_cuda(j_eq, centroids, areas, sensors_low, args.time_chunk, args.sensor_chunk).reshape(direct.shape)
    high = project_bz_cuda(j_eq, centroids, areas, sensors_high, args.time_chunk, args.sensor_chunk).reshape(direct.shape)
    axial = (high - low) / args.baseline_m
    rms = float(np.sqrt(np.mean(direct * direct)))
    rng = np.random.default_rng(args.seed + abs(hash(sample_id)) % 10_000_000)
    noise = rng.normal(0.0, 0.05 * max(rms, 1e-30), size=direct.shape).astype(np.float32)
    mask_full = np.ones((args.sensor_grid, args.sensor_grid), dtype=bool)
    mask_drop = mask_full.reshape(-1).copy()
    drop_count = max(1, int(round(0.10 * mask_drop.size)))
    mask_drop[rng.choice(mask_drop.size, size=drop_count, replace=False)] = False
    mask_drop = mask_drop.reshape(args.sensor_grid, args.sensor_grid)
    masked = direct + noise
    masked[:, ~mask_drop] = 0.0
    payloads = {
        "direct_bz_50mm": (direct, mask_full, "Bz at 50 mm sensor plane"),
        "direct_axial_grad_30mm_baseline": (axial, mask_full, "finite-difference axial gradiometer approximation"),
        "direct_bz_noise_medium": (direct + noise, mask_full, "Bz with Gaussian noise at 5 percent clean RMS"),
        "direct_bz_masked_noise": (masked, mask_drop, "Bz with 5 percent noise and 10 percent deterministic sensor masking"),
    }
    direct_rows = []
    for variant in DIRECT_VARIANTS:
        b, mask, note = payloads[variant]
        metadata = {
            "base_sample_id": sample_id,
            "condition": row["condition"],
            "variant": variant,
            "operator_kind": "direct SQUID-like configurable approximation, not exact hardware emulation",
            "direct_projection": True,
            "source_current": "J_eq = -D_eff grad(Vm), equivalent-current proxy",
            "sensor_grid": args.sensor_grid,
            "height_m": args.height_m,
            "baseline_m": args.baseline_m,
            "sensor_z_m": z,
            "clean_rms": rms,
            "channel_semantics": [note],
            "source_metadata": source_meta,
        }
        direct_rows.append(save_direct(sample_id, row["condition"], variant, b, sensors, coords, times, mask, metadata))
    combined_rows = []
    for variant in COMBINED_VARIANTS:
        metadata = {
            "base_sample_id": sample_id,
            "condition": row["condition"],
            "variant": variant,
            "operator_kind": "combined-channel direct SQUID-like configurable approximation, not exact hardware emulation",
            "direct_projection": True,
            "source_current": "J_eq = -D_eff grad(Vm), equivalent-current proxy",
            "sensor_grid": args.sensor_grid,
            "height_m": args.height_m,
            "baseline_m": args.baseline_m,
            "channel_layout": CHANNEL_LAYOUT,
            "active_channels": COMBINED_VARIANTS[variant],
            "source_metadata": source_meta,
        }
        combined_rows.append(save_combined(sample_id, row["condition"], variant, payloads, sensors, coords, times, metadata))
    return direct_rows, combined_rows, {"base_sample_id": sample_id, "runtime_seconds": time.perf_counter() - start, "clean_rms": rms}


def make_figures(combined_df: pd.DataFrame, runtime_rows: list[dict]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    examples = combined_df.groupby("measurement_variant", group_keys=False).head(1)
    fig, axes = plt.subplots(1, len(examples), figsize=(3.3 * len(examples), 3.2))
    axes = np.atleast_1d(axes)
    for ax, (_, row) in zip(axes, examples.iterrows()):
        b = np.load(ROOT / row["B_meas_file"])
        im = ax.imshow(b[min(50, b.shape[0] - 1), :, :, 0], origin="lower", cmap="coolwarm")
        ax.set_title(row["measurement_variant"], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v1e_500_direct_combined_examples.png", dpi=180)
    plt.close(fig)

    rtdf = pd.DataFrame(runtime_rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rtdf["runtime_seconds"], bins=30)
    ax.set_xlabel("seconds")
    ax.set_ylabel("base samples")
    ax.set_title("Direct 32x32 Generation Runtime")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v1e_500_direct_generation_runtime.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-grid", type=int, default=32, choices=[16, 32, 64])
    parser.add_argument("--height-m", type=float, default=0.05)
    parser.add_argument("--baseline-m", type=float, default=0.03)
    parser.add_argument("--time-chunk", type=int, default=32)
    parser.add_argument("--sensor-chunk", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260525)
    args = parser.parse_args()
    ensure_dirs()
    require_cuda(torch)
    pilot = pd.read_csv(PILOT_DIR / "manifest.csv")
    start = time.perf_counter()
    direct_rows = []
    combined_rows = []
    runtime_rows = []
    failures = []
    for i, row in pilot.iterrows():
        try:
            drows, crows, rt = process_sample(row, args)
            direct_rows.extend(drows)
            combined_rows.extend(crows)
            runtime_rows.append(rt)
            if (i + 1) % 25 == 0:
                print(f"generated {i + 1}/{len(pilot)} base samples", flush=True)
        except Exception as exc:
            failures.append({"base_sample_id": row["base_sample_id"], "condition": row["condition"], "error": repr(exc)})
            print(f"FAILED {row['base_sample_id']}: {exc!r}", flush=True)
    direct_df = pd.DataFrame(direct_rows)
    combined_df = pd.DataFrame(combined_rows)
    direct_df.to_csv(PILOT_DIR / "direct_input_manifest.csv", index=False)
    combined_df.to_csv(COMBINED_MANIFEST, index=False)
    if len(combined_df):
        make_figures(combined_df, runtime_rows)
    runtime = time.perf_counter() - start
    storage = recursive_size_bytes(DIRECT_INPUT_DIR) + recursive_size_bytes(COMBINED_INPUT_DIR)
    expected_direct = len(pilot) * len(DIRECT_VARIANTS)
    expected_combined = len(pilot) * len(COMBINED_VARIANTS)
    report = {
        "sensor_grid": args.sensor_grid,
        "base_samples": int(len(pilot)),
        "direct_rows_generated": int(len(direct_df)),
        "expected_direct_rows": int(expected_direct),
        "combined_rows_generated": int(len(combined_df)),
        "expected_combined_rows": int(expected_combined),
        "direct_success": int(direct_df["success"].sum()) if len(direct_df) else 0,
        "combined_success": int(combined_df["success"].sum()) if len(combined_df) else 0,
        "variant_counts": combined_df["measurement_variant"].value_counts().to_dict() if len(combined_df) else {},
        "condition_counts_base": pilot["condition"].value_counts().to_dict(),
        "failures": failures,
        "runtime_seconds": runtime,
        "mean_runtime_seconds_per_base_sample": float(np.mean([r["runtime_seconds"] for r in runtime_rows])) if runtime_rows else None,
        "storage_bytes": int(storage),
        "storage_gb": storage / 1e9,
        "estimated_5000_runtime_hours": runtime / max(len(pilot), 1) * 5000 / 3600,
        "estimated_5000_storage_gb": storage / max(len(pilot), 1) * 5000 / 1e9,
        "passed": bool(
            len(failures) == 0
            and len(direct_df) == expected_direct
            and len(combined_df) == expected_combined
            and int(direct_df["success"].sum()) >= int(0.98 * expected_direct)
            and int(combined_df["success"].sum()) >= int(0.98 * expected_combined)
        ),
        "scientific_note": "B fields are synthetic/proxy fields in relative units from an equivalent-current proxy, not exact SQUID emulation.",
    }
    write_json(REPORT_PATH, report)
    print(json.dumps({k: report[k] for k in ["combined_rows_generated", "runtime_seconds", "storage_gb", "passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
