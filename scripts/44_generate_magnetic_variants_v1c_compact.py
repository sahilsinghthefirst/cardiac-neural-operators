                      
                                                                               

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from bocf_v1c_common import (
    FULL_MODE,
    PILOT_MODE,
    VARIANTS,
    array_file,
    compute_bvec_projection,
    compute_bz_projection,
    compute_mig_curves_from_bz,
    default_workers,
    load_array,
    mode_paths,
    monotonic_storage_summary,
    now_utc,
    read_csv_rows,
    rel,
    save_array,
    seed_for_text,
    truthy,
    write_csv_rows,
    write_json,
    write_mig_curves,
)
from magnetic_common import compute_element_gradients, load_mesh, load_vm_tn, triangle_geometry


MANIFEST_FIELDS = [
    "augmented_sample_id",
    "base_sample_id",
    "condition",
    "variant",
    "magnetic_output_dir",
    "Bz_file",
    "B_vec_file",
    "mig_curves_file",
    "mig_summary_file",
    "sensor_mask_file",
    "success",
    "notes",
]


def compute_source(ep_dir: Path):
    mesh = load_mesh(ep_dir)
    centroids, areas, grad_coeffs = triangle_geometry(mesh)
    vm_tn, _times, _debug = load_vm_tn(ep_dir / "vm.igb", mesh.pts_raw.shape[0])
    grad = compute_element_gradients(vm_tn, mesh.elements, grad_coeffs)
    return (-1.0 * grad).astype(np.float32), centroids, areas


def choose_dropout_mask(n_sensors: int, frac: float, rng: np.random.Generator) -> np.ndarray:
    keep = np.ones(n_sensors, dtype=bool)
    n_drop = max(1, int(round(frac * n_sensors)))
    drop = rng.choice(n_sensors, size=min(n_drop, n_sensors - 1), replace=False)
    keep[drop] = False
    return keep


def manifest_row(aug_id: str, base_id: str, condition: str, variant: str, out_dir: Path, success: bool, notes: str) -> dict:
    bz = array_file(out_dir, "Bz")
    bvec = array_file(out_dir, "B_vec")
    mask = array_file(out_dir, "sensor_mask")
    return {
        "augmented_sample_id": aug_id,
        "base_sample_id": base_id,
        "condition": condition,
        "variant": variant,
        "magnetic_output_dir": rel(out_dir),
        "Bz_file": rel(bz) if bz else "",
        "B_vec_file": rel(bvec) if bvec else "",
        "mig_curves_file": rel(out_dir / "mig_curves.csv") if (out_dir / "mig_curves.csv").exists() else "",
        "mig_summary_file": rel(out_dir / "mig_summary.json") if (out_dir / "mig_summary.json").exists() else "",
        "sensor_mask_file": rel(mask) if mask else "",
        "success": success,
        "notes": notes,
    }


def save_variant(out_dir: Path, b_z: np.ndarray, sensors: np.ndarray, mask: np.ndarray, times: np.ndarray, metadata: dict, compress: bool) -> None:
    save_array(out_dir / "Bz.npy", b_z.astype(np.float32), compress)
    save_array(out_dir / "sensor_positions.npy", sensors.astype(np.float64), compress)
    save_array(out_dir / "sensor_mask.npy", mask.astype(bool), compress)
    save_array(out_dir / "time_ms.npy", times.astype(np.float64), compress)
    curves, summary = compute_mig_curves_from_bz(b_z, times, mask)
    write_mig_curves(out_dir / "mig_curves.csv", curves)
    write_json(out_dir / "mig_summary.json", {**summary, **metadata})
    write_json(out_dir / "augmentation_metadata.json", metadata)


def variant_bz(
    variant: str,
    clean_bz: np.ndarray,
    sensors: np.ndarray,
    rng: np.random.Generator,
    source,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    out_bz = clean_bz.copy()
    out_sensors = sensors.copy()
    mask = np.ones(sensors.shape[0], dtype=bool)
    clean_rms = float(np.sqrt(np.nanmean(clean_bz * clean_bz)))
    metadata = {
        "noise_sigma": 0.0,
        "noise_fraction_of_clean_rms": 0.0,
        "dropout_fraction": 0.0,
        "jitter_std_m": 0.0,
        "height_m": float(sensors[0, 2]),
        "compact_variant_note": "v1c compact variants store Bz only; noise is applied to Bz for deployable MIG features.",
        "B_arrays_drop_encoding": "dropped sensors are NaN in Bz; sensor_mask is authoritative",
    }
    if variant == "clean":
        return out_bz, out_sensors, mask, metadata
    if variant.startswith("noise_"):
        frac = {"noise_low": 0.01, "noise_medium": 0.05, "noise_high": 0.10}[variant]
        metadata["noise_fraction_of_clean_rms"] = frac
        metadata["noise_sigma"] = frac * clean_rms
        out_bz = clean_bz + rng.normal(0.0, metadata["noise_sigma"], size=clean_bz.shape)
        return out_bz.astype(np.float32), out_sensors, mask, metadata
    if variant.startswith("dropout_"):
        frac = {"dropout_10pct": 0.10, "dropout_25pct": 0.25}[variant]
        metadata["dropout_fraction"] = frac
        mask = choose_dropout_mask(sensors.shape[0], frac, rng)
        out_bz[:, ~mask] = np.nan
        return out_bz.astype(np.float32), out_sensors, mask, metadata
    if variant.startswith("jitter_"):
        std = {"jitter_1mm": 0.001, "jitter_3mm": 0.003}[variant]
        metadata["jitter_std_m"] = std
        out_sensors[:, :2] += rng.normal(0.0, std, size=out_sensors[:, :2].shape)
        j_eq, centroids, areas = source
        out_bz = compute_bz_projection(j_eq, centroids, areas, out_sensors)
        return out_bz, out_sensors, mask, metadata
    if variant.startswith("height_"):
        height = {"height_30mm": 0.03, "height_70mm": 0.07}[variant]
        metadata["height_m"] = height
        out_sensors[:, 2] = height
        j_eq, centroids, areas = source
        out_bz = compute_bz_projection(j_eq, centroids, areas, out_sensors)
        return out_bz, out_sensors, mask, metadata
    if variant == "combined_medium":
        metadata["jitter_std_m"] = 0.001
        metadata["noise_fraction_of_clean_rms"] = 0.05
        metadata["noise_sigma"] = 0.05 * clean_rms
        metadata["dropout_fraction"] = 0.10
        out_sensors[:, :2] += rng.normal(0.0, 0.001, size=out_sensors[:, :2].shape)
        j_eq, centroids, areas = source
        out_bz = compute_bz_projection(j_eq, centroids, areas, out_sensors)
        out_bz = out_bz + rng.normal(0.0, metadata["noise_sigma"], size=out_bz.shape)
        mask = choose_dropout_mask(sensors.shape[0], 0.10, rng)
        out_bz[:, ~mask] = np.nan
        return out_bz.astype(np.float32), out_sensors, mask, metadata
    raise ValueError(f"Unknown variant {variant}")


def process_base(row: dict, aug_base: Path, force: bool, compress: bool, store_bvec: bool) -> list[dict]:
    base_id = row["sample_id"]
    condition = row["condition"]
    base_mag_dir = Path(__file__).resolve().parents[1] / row["magnetic_output_dir"]
    ep_dir = Path(__file__).resolve().parents[1] / row["ep_output_dir"]
    clean_bz = load_array(array_file(base_mag_dir, "Bz"))
    sensors = load_array(array_file(base_mag_dir, "sensor_positions"))
    times = load_array(array_file(base_mag_dir, "time_ms"))
    source = None
    rows: list[dict] = []
    for variant in VARIANTS:
        aug_id = f"{base_id}__{variant}"
        out_dir = aug_base / aug_id
        meta_file = out_dir / "augmentation_metadata.json"
        if meta_file.exists() and not force:
            try:
                meta = json.loads(meta_file.read_text())
                if meta.get("success"):
                    rows.append(manifest_row(aug_id, base_id, condition, variant, out_dir, True, ""))
                    continue
            except Exception:
                pass
        out_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        try:
            if variant in {"jitter_1mm", "jitter_3mm", "height_30mm", "height_70mm", "combined_medium"} and source is None:
                source = compute_source(ep_dir)
            rng = np.random.default_rng(seed_for_text(aug_id))
            b_z, out_sensors, mask, extra = variant_bz(variant, clean_bz, sensors, rng, source)
            metadata = {
                "created_utc": now_utc(),
                "augmented_sample_id": aug_id,
                "base_sample_id": base_id,
                "condition": condition,
                "variant": variant,
                "source_base_magnetic_dir": row["magnetic_output_dir"],
                "source_ep_dir": row["ep_output_dir"],
                "success": True,
                "runtime_seconds": time.perf_counter() - start,
                "scientific_note": "Sensor/noise/geometry measurement robustness variant; targets inherit the base EP sample.",
                "source_definition": "Base source remains J_eq = -D_eff grad(Vm); B fields are synthetic/proxy fields.",
                **extra,
            }
            save_variant(out_dir, b_z, out_sensors, mask, times, metadata, compress)
            if store_bvec:
                if source is None:
                    source = compute_source(ep_dir)
                j_eq, centroids, areas = source
                b_vec = compute_bvec_projection(j_eq, centroids, areas, out_sensors)
                b_vec[:, ~mask, :] = np.nan
                save_array(out_dir / "B_vec.npy", b_vec.astype(np.float32), compress)
            rows.append(manifest_row(aug_id, base_id, condition, variant, out_dir, True, ""))
        except Exception as exc:
            write_json(
                meta_file,
                {
                    "created_utc": now_utc(),
                    "augmented_sample_id": aug_id,
                    "base_sample_id": base_id,
                    "condition": condition,
                    "variant": variant,
                    "success": False,
                    "error": repr(exc),
                    "runtime_seconds": time.perf_counter() - start,
                },
            )
            rows.append(manifest_row(aug_id, base_id, condition, variant, out_dir, False, repr(exc)))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[PILOT_MODE, FULL_MODE], default=PILOT_MODE)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--compress-arrays", action="store_true")
    parser.add_argument("--store-bvec", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = mode_paths(args.mode)
    base_rows = [r for r in read_csv_rows(paths["mag"] / "manifest.csv") if truthy(r.get("success"))]
    if args.limit is not None:
        base_rows = base_rows[: args.limit]
    paths["aug"].mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rows: list[dict] = []
    if args.workers <= 1:
        for row in base_rows:
            out_rows = process_base(row, paths["aug"], args.force, args.compress_arrays, args.store_bvec)
            rows.extend(out_rows)
            print(f"{row['sample_id']}: {sum(1 for r in out_rows if truthy(r['success']))}/{len(out_rows)} variants")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(process_base, row, paths["aug"], args.force, args.compress_arrays, args.store_bvec) for row in base_rows]
            for fut in as_completed(futures):
                out_rows = fut.result()
                rows.extend(out_rows)
                print(f"{out_rows[0]['base_sample_id']}: {sum(1 for r in out_rows if truthy(r['success']))}/{len(out_rows)} variants")
    rows.sort(key=lambda r: r["augmented_sample_id"])
    write_csv_rows(paths["aug"] / "manifest.csv", rows, MANIFEST_FIELDS)
    success_count = sum(1 for r in rows if truthy(r.get("success")))
    summary = {
        "created_utc": now_utc(),
        "mode": args.mode,
        "base_sample_count": len(base_rows),
        "variant_count": len(VARIANTS),
        "row_count": len(rows),
        "success_count": success_count,
        "failure_count": len(rows) - success_count,
        "runtime_seconds": time.perf_counter() - start,
        "storage": monotonic_storage_summary(paths["aug"], len(rows)),
        "compact_storage": True,
        "B_vec_stored": bool(args.store_bvec),
    }
    write_json(paths["aug"] / "run_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ["success_count", "failure_count", "runtime_seconds"]}, indent=2))
    return 0 if success_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
