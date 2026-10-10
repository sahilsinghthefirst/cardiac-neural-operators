                      
                                                                               

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v1d_field_common import (
    FIG_DIR,
    MAG_INPUT_DIR,
    MAG_VARIANTS,
    PILOT_DIR,
    ROOT,
    ensure_dirs,
    rel,
    recursive_size_bytes,
    write_json,
)


OUT_MANIFEST = PILOT_DIR / "magnetic_input_manifest.csv"
OUT_REPORT = PILOT_DIR / "magnetic_input_quality_report.json"


def sensor_grid_from_existing(base_positions: np.ndarray, grid: int, height_m: float):
    xy_min = base_positions[:, :2].min(axis=0)
    xy_max = base_positions[:, :2].max(axis=0)
    xs = np.linspace(xy_min[0], xy_max[0], grid)
    ys = np.linspace(xy_min[1], xy_max[1], grid)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    z = float(np.mean(base_positions[:, 2]) + (height_m - 0.05))
    sensors = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)])
    return sensors, xs, ys, z


def coordinate_channels(xs: np.ndarray, ys: np.ndarray, height_m: float, mask_any: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    xnorm = 2.0 * (xx - xs.min()) / max(xs.max() - xs.min(), 1e-12) - 1.0
    ynorm = 2.0 * (yy - ys.min()) / max(ys.max() - ys.min(), 1e-12) - 1.0
    hnorm = np.full_like(xnorm, height_m / 0.05, dtype=np.float64)
    return np.stack([xnorm, ynorm, hnorm, mask_any.astype(float)], axis=-1).astype(np.float32)


def upsample_bz(bz_8: np.ndarray, grid: int) -> np.ndarray:
    bz_8 = bz_8.reshape(bz_8.shape[0], 8, 8).astype(np.float32)
    if grid == 8:
        return bz_8
    try:
        from scipy.ndimage import zoom

        return zoom(bz_8, (1, grid / 8.0, grid / 8.0), order=1).astype(np.float32)
    except Exception:
        xs_old = np.linspace(0.0, 1.0, 8)
        xs_new = np.linspace(0.0, 1.0, grid)
        tmp = np.stack([np.vstack([np.interp(xs_new, xs_old, row) for row in frame]) for frame in bz_8])
        out = np.stack([np.vstack([np.interp(xs_new, xs_old, tmp[t, :, j]) for j in range(grid)]).T for t in range(tmp.shape[0])])
        return out.astype(np.float32)


def load_existing_bz(sample_id: str, variant: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if variant is None:
        base = ROOT / "outputs" / "magnetic_dataset_v1c_5000" / sample_id
    else:
        base = ROOT / "outputs" / "magnetic_dataset_v1c_5000_augmented" / f"{sample_id}__{variant}"
    bz = np.load(base / "Bz.npy").astype(np.float32)
    times = np.load(base / "time_ms.npy").astype(np.float32)
    sensors = np.load(base / "sensor_positions.npy").astype(np.float64)
    return bz, times, sensors


def save_variant(
    sample_id: str,
    condition: str,
    variant: str,
    b_channels: np.ndarray,
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
    mask_any = np.any(sensor_mask, axis=-1)
    coords = coordinate_channels(xs, ys, height_m, mask_any)
    np.save(out_dir / "B_meas.npy", b_channels.astype(np.float32))
    np.save(out_dir / "sensor_positions.npy", sensors.astype(np.float64))
    np.save(out_dir / "sensor_mask.npy", sensor_mask.astype(bool))
    np.save(out_dir / "coordinate_channels.npy", coords.astype(np.float32))
    np.save(out_dir / "time_ms.npy", times.astype(np.float32))
    write_json(out_dir / "measurement_metadata.json", metadata)
    valid = sensor_mask.astype(bool)
    finite = np.isfinite(b_channels[valid[None, :, :, :].repeat(b_channels.shape[0], axis=0)])
    return {
        "field_sample_id": f"{sample_id}__{variant}",
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
        "time_steps": b_channels.shape[0],
        "sensor_grid": b_channels.shape[1],
        "channels": b_channels.shape[-1],
        "success": bool(finite.size and np.all(finite)),
        "notes": "",
    }


def process_sample(row: pd.Series, grid: int, height_m: float, baseline_m: float, noise_seed: int) -> list[dict]:
    sample_id = row["base_sample_id"]
    clean8, times, base_sensors = load_existing_bz(sample_id)
    h30_8, _, _ = load_existing_bz(sample_id, "height_30mm")
    h70_8, _, _ = load_existing_bz(sample_id, "height_70mm")
    sensors, xs, ys, z = sensor_grid_from_existing(base_sensors, grid, height_m)
    clean = upsample_bz(clean8, grid)
    h30 = upsample_bz(h30_8, grid)
    h70 = upsample_bz(h70_8, grid)
    axial = (h70 - h30) / 0.04
    dy = float(np.mean(np.diff(ys))) if len(ys) > 1 else 1.0
    dx = float(np.mean(np.diff(xs))) if len(xs) > 1 else 1.0
    grad_y, grad_x = np.gradient(clean, dy, dx, axis=(1, 2))
    rms = float(np.sqrt(np.mean(clean * clean)))
    rng = np.random.default_rng(noise_seed)
    noise = rng.normal(0.0, 0.05 * max(rms, 1e-30), size=clean.shape).astype(np.float32)
    drop_mask = np.ones((grid, grid, 2), dtype=bool)
    flat = rng.choice(grid * grid, size=max(1, int(round(0.10 * grid * grid))), replace=False)
    drop_single = np.ones(grid * grid, dtype=bool)
    drop_single[flat] = False
    rows = []

    def two_channels(first: np.ndarray, second: np.ndarray | None = None, mask0: np.ndarray | None = None, mask1: bool = False):
        second_arr = np.zeros_like(first) if second is None else second
        b = np.stack([first, second_arr], axis=-1)
        m = np.zeros((grid, grid, 2), dtype=bool)
        m[:, :, 0] = True if mask0 is None else mask0.reshape(grid, grid)
        m[:, :, 1] = mask1
        return b, m

    variant_payloads = {
        "ideal_bz": two_channels(clean),
        "axial_gradiometer_approx": two_channels(axial),
        "planar_gradient_x_approx": two_channels(grad_x),
        "planar_gradient_y_approx": two_channels(grad_y),
        "combined_bz_axial": two_channels(clean, axial, mask1=True),
        "noisy_bz_medium": two_channels(clean + noise),
        "masked_noisy_bz": two_channels(clean + noise, mask0=drop_single),
    }
    for variant in MAG_VARIANTS:
        b, mask = variant_payloads[variant]
        if variant == "masked_noisy_bz":
            b = b.copy()
            b[:, ~mask[:, :, 0], 0] = 0.0
        metadata = {
            "base_sample_id": sample_id,
            "condition": row["condition"],
            "variant": variant,
            "operator_kind": "SQUID-like configurable approximation, not exact hardware emulation",
            "B_meas_shape": list(b.shape),
            "sensor_grid": grid,
            "height_m": height_m,
            "baseline_m": baseline_m,
            "planar_gradient_baseline_m": dx,
            "clean_rms": rms,
            "noise_std": 0.05 * max(rms, 1e-30) if "noisy" in variant else 0.0,
            "source_current": "J_eq = -D_eff grad(Vm), equivalent-current proxy",
            "source_shortcut": "Pilot uses existing v1c current-proxy Biot-Savart Bz outputs upsampled to the requested grid; axial approximation uses existing 30mm/70mm height variants.",
            "channel_semantics": ["primary measurement", "secondary measurement if enabled; otherwise zero-filled"],
        }
        rows.append(save_variant(sample_id, row["condition"], variant, b, sensors, xs, ys, height_m, times, mask, metadata))
    return rows


def make_figures(rows: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    examples = rows.groupby("measurement_variant", group_keys=False).head(1)
    n = len(examples)
    cols = 4
    rows_n = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3, rows_n * 3))
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, row) in zip(axes, examples.iterrows()):
        b = np.load(ROOT / row["B_meas_file"])
        im = ax.imshow(b[min(50, b.shape[0] - 1), :, :, 0], origin="lower", cmap="coolwarm")
        ax.set_title(row["measurement_variant"], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "example_squidlike_inputs.png", dpi=180)
    plt.close(fig)

    rms_rows = []
    for _, row in rows.iterrows():
        b = np.load(ROOT / row["B_meas_file"])
        mask = np.load(ROOT / row["sensor_mask_file"]).astype(bool)
        valid = mask[None, :, :, :].repeat(b.shape[0], axis=0)
        rms_rows.append({"variant": row["measurement_variant"], "rms": float(np.sqrt(np.mean(b[valid] ** 2)))})
    rdf = pd.DataFrame(rms_rows)
    fig, ax = plt.subplots(figsize=(10, 4))
    rdf.boxplot(column="rms", by="variant", ax=ax, rot=30)
    fig.suptitle("")
    ax.set_title("Input RMS By SQUID-like Variant")
    ax.set_ylabel("relative units")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "input_rms_by_variant.png", dpi=180)
    plt.close(fig)

    first = rows.iloc[0]
    pos = np.load(ROOT / first["sensor_positions_file"])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pos[:, 0], pos[:, 1], s=8)
    ax.set_aspect("equal")
    ax.set_title("SQUID-like Sensor Grid Geometry")
    ax.set_xlabel("x m")
    ax.set_ylabel("y m")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sensor_grid_geometry.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor-grid", type=int, default=32)
    parser.add_argument("--height-mm", type=float, default=50.0)
    parser.add_argument("--baseline-mm", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    ensure_dirs()
    pilot = pd.read_csv(PILOT_DIR / "manifest.csv")
    if args.limit:
        pilot = pilot.head(args.limit)
    all_rows = []
    failures = []
    start = time.perf_counter()
    for i, row in pilot.iterrows():
        try:
            out = process_sample(row, args.sensor_grid, args.height_mm / 1000.0, args.baseline_mm / 1000.0, noise_seed=20260525 + i)
            all_rows.extend(out)
            print(f"{row['base_sample_id']}: {len(out)} magnetic variants ok")
        except Exception as exc:
            failures.append({"base_sample_id": row["base_sample_id"], "condition": row["condition"], "error": repr(exc)})
            print(f"{row['base_sample_id']}: magnetic failed {exc!r}")
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_MANIFEST, index=False)
    if len(df):
        make_figures(df)
    bad_zero = []
    bad_finite = []
    for _, row in df.iterrows():
        b = np.load(ROOT / row["B_meas_file"])
        mask = np.load(ROOT / row["sensor_mask_file"]).astype(bool)
        valid = mask[None, :, :, :].repeat(b.shape[0], axis=0)
        vals = b[valid]
        if not np.all(np.isfinite(vals)):
            bad_finite.append(row["field_sample_id"])
        if vals.size == 0 or np.allclose(vals, 0):
            bad_zero.append(row["field_sample_id"])
    storage = recursive_size_bytes(MAG_INPUT_DIR)
    report = {
        "sensor_grid": args.sensor_grid,
        "height_mm": args.height_mm,
        "baseline_mm": args.baseline_mm,
        "base_samples_requested": int(len(pilot)),
        "variant_count": len(MAG_VARIANTS),
        "rows_generated": int(len(df)),
        "expected_rows": int(len(pilot) * len(MAG_VARIANTS)),
        "failures": failures,
        "bad_finite": bad_finite,
        "all_zero_valid_signals": bad_zero,
        "condition_counts": df["condition"].value_counts().to_dict() if len(df) else {},
        "variant_counts": df["measurement_variant"].value_counts().to_dict() if len(df) else {},
        "runtime_seconds": time.perf_counter() - start,
        "storage_bytes": int(storage),
        "storage_gb": float(storage / 1e9),
        "estimated_full_5000_storage_gb": float(storage / max(len(pilot), 1) * 5000 / 1e9),
        "passed": bool(len(df) >= 0.98 * len(pilot) * len(MAG_VARIANTS) and not failures and not bad_finite and not bad_zero),
        "scientific_note": "SQUID-like configurable measurement operators, not exact SQUID hardware emulation.",
    }
    write_json(OUT_REPORT, report)
    print(json.dumps({k: report[k] for k in ["rows_generated", "expected_rows", "runtime_seconds", "passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
