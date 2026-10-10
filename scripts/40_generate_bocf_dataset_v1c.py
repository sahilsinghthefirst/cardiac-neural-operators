                      
                                                       
 
                                                                            
                                                                            
                                                                             
                                                                
   

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from bocf_dataset_common import (
    DEFAULT_APD_PARAMETER,
    ROOT,
    derive_maps,
    load_vm,
    manifest_row,
    opencarp_command,
    prepare_run_mesh_and_stimulus,
    recursive_size_bytes,
    run_command,
    run_conditioned_ep_sample,
    write_json as write_ep_json,
)
from bocf_v1c_common import (
    COMPOSITIONS,
    DEFAULT_BCL_MS,
    DEFAULT_NBEATS,
    EP_REQUIRED_FILES,
    FINAL_BEAT_MS,
    FULL_MODE,
    PILOT_MODE,
    create_sample_plan,
    default_workers,
    expected_composition,
    mode_paths,
    now_utc,
    rel,
    truthy,
    write_csv_rows,
    write_json,
)


MANIFEST_FIELDS = [
    "sample_id",
    "condition",
    "model",
    "formulation",
    "geometry",
    "apd_parameter",
    "apd_scale",
    "conductivity_scale",
    "patch_shape",
    "patch_center_x",
    "patch_center_y",
    "patch_radius",
    "patch_radius_mm",
    "stimulus_location",
    "base_gil",
    "anisotropy_factor",
    "nbeats",
    "BCL",
    "conditioning_used",
    "prepacing_reference_dir",
    "success",
    "output_dir",
    "runtime_seconds",
    "output_size_bytes",
    "notes",
]


def set_spacedt(cmd: list, spacedt: float) -> list:
    out = list(cmd)
    for i, value in enumerate(out[:-1]):
        if str(value) == "-spacedt":
            out[i + 1] = spacedt
            return out
    out += ["-spacedt", spacedt]
    return out


def run_prepacing_reference(sample: dict, ref_dir: Path, force: bool) -> dict:
    lat_file = ref_dir / "LAT.dat"
    meta_file = ref_dir / "prepacing_reference_metadata.json"
    if lat_file.exists() and meta_file.exists() and not force:
        meta = json.loads(meta_file.read_text())
        if meta.get("success"):
            meta["skipped_existing"] = True
            return meta

    ref_dir.mkdir(parents=True, exist_ok=True)
    patch_center = None
    if sample["patch_center_x"] is not None:
        patch_center = (float(sample["patch_center_x"]), float(sample["patch_center_y"]))
    patch_radius = sample["patch_radius"]
    patch = patch_center is not None and patch_radius is not None and float(patch_radius) > 0
    pts, _elems, _tags, stim_file, _masks = prepare_run_mesh_and_stimulus(
        ref_dir, patch_center, patch_radius, sample["stimulus_location"]
    )
    cmd = opencarp_command(
        out_dir=ref_dir,
        apd_parameter=sample["apd_parameter"],
        apd_scale=sample["apd_scale"],
        conductivity_scale=sample["conductivity_scale"],
        stimulus_file=stim_file,
        patch=patch,
        base_gil=float(sample["base_gil"]),
        anisotropy_factor=float(sample["anisotropy_factor"]),
        tend_ms=FINAL_BEAT_MS,
        nbeats=1,
        bcl_ms=DEFAULT_BCL_MS,
    )
                                                                    
                                                                   
    cmd = set_spacedt(cmd, 300)
    start = time.perf_counter()
    result = run_command(cmd, ref_dir, "opencarp_prepacing_reference")
    runtime = time.perf_counter() - start
    derived = derive_maps(ref_dir) if result.returncode == 0 else {}
    vm_tn, _times, vm_debug = load_vm(ref_dir)
    meta = {
        "created_utc": now_utc(),
        "sample_id": sample["sample_id"],
        "condition": sample["condition"],
        "purpose": "single-beat sparse-output LAT reference for LAT-guided prepacing",
        "success": bool(result.returncode == 0 and lat_file.exists()),
        "returncode": result.returncode,
        "runtime_seconds": runtime,
        "node_count": int(len(pts)),
        "derived_outputs": derived,
        "vm_debug": vm_debug,
        "command_used": " ".join(str(x) for x in cmd),
        "notes": "" if result.returncode == 0 else "openCARP prepacing reference failed; inspect logs",
    }
    write_ep_json(meta_file, meta)
    return meta


def run_one_sample(sample: dict, ep_base: Path, force: bool) -> dict:
    sample_id = sample["sample_id"]
    out_dir = ep_base / "samples" / sample_id
    start = time.perf_counter()
    try:
        meta_file = out_dir / "metadata.json"
        if meta_file.exists() and not force:
            existing = json.loads(meta_file.read_text())
            if existing.get("success"):
                row = manifest_row(existing, out_dir)
                row.update(
                    {
                        "patch_shape": existing.get("patch_shape"),
                        "patch_radius_mm": existing.get("patch_radius_mm"),
                        "base_gil": existing.get("base_gil"),
                        "anisotropy_factor": existing.get("anisotropy_factor"),
                        "nbeats": existing.get("nbeats"),
                        "BCL": existing.get("BCL"),
                        "conditioning_used": existing.get("conditioning_used"),
                        "prepacing_reference_dir": existing.get("prepacing_reference_dir", ""),
                        "runtime_seconds": existing.get("total_runtime_seconds", existing.get("run_seconds", "")),
                        "output_size_bytes": existing.get("output_size_bytes", ""),
                    }
                )
                row["success"] = True
                return row

        ref_dir = out_dir / "_prepacing_reference"
        ref_meta = run_prepacing_reference(sample, ref_dir, force)
        if not ref_meta.get("success"):
            out_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "created_utc": now_utc(),
                **sample,
                "success": False,
                "conditioning_used": False,
                "prepacing_reference_dir": rel(ref_dir),
                "notes": "Prepacing LAT reference failed; final conditioned beat was not run.",
                "total_runtime_seconds": time.perf_counter() - start,
            }
            write_ep_json(out_dir / "metadata.json", meta)
            return failure_row(sample, out_dir, meta["notes"], time.perf_counter() - start)

        patch_center = None
        if sample["patch_center_x"] is not None:
            patch_center = (float(sample["patch_center_x"]), float(sample["patch_center_y"]))
        metadata = run_conditioned_ep_sample(
            out_dir=out_dir,
            sample_id=sample_id,
            condition=sample["condition"],
            prepacing_lats=ref_dir / "LAT.dat",
            patch_center=patch_center,
            patch_radius=sample["patch_radius"],
            apd_parameter=sample["apd_parameter"],
            apd_scale=sample["apd_scale"],
            conductivity_scale=sample["conductivity_scale"],
            stimulus_location=sample["stimulus_location"],
            base_gil=float(sample["base_gil"]),
            anisotropy_factor=float(sample["anisotropy_factor"]),
            tend_ms=FINAL_BEAT_MS,
            nbeats=DEFAULT_NBEATS,
            bcl_ms=DEFAULT_BCL_MS,
            seed=int(sample["seed"]),
            force=force,
        )
        total_runtime = time.perf_counter() - start
        metadata.update(
            {
                "v1c_dataset_version": "v1c",
                "patch_shape": sample["patch_shape"],
                "patch_radius_mm": sample["patch_radius_mm"],
                "prepacing_reference_dir": rel(ref_dir),
                "prepacing_reference_runtime_seconds": ref_meta.get("runtime_seconds"),
                "total_runtime_seconds": total_runtime,
                "compact_storage": True,
                "randomization_ranges": {
                    "patch_radius_mm": [0.7, 2.5],
                    "tau_si_scale": [1.03, 1.45],
                    "conductivity_scale": [0.30, 0.90],
                    "anisotropy_factor": [3.0, 5.0],
                    "base_gil": [0.85, 1.15],
                },
                "scientific_note": (
                    "Bueno/BOCF monodomain controlled synthetic benchmark. Mixed samples are "
                    "APD/repolarization proxy plus conduction-slowing perturbations, not clinical ischemia."
                ),
            }
        )
        metadata["output_size_bytes"] = recursive_size_bytes(out_dir)
        write_ep_json(out_dir / "metadata.json", metadata)
        row = manifest_row(metadata, out_dir)
        row.update(
            {
                "patch_shape": sample["patch_shape"],
                "patch_radius_mm": sample["patch_radius_mm"],
                "base_gil": sample["base_gil"],
                "anisotropy_factor": sample["anisotropy_factor"],
                "nbeats": DEFAULT_NBEATS,
                "BCL": DEFAULT_BCL_MS,
                "conditioning_used": metadata.get("conditioning_used"),
                "prepacing_reference_dir": rel(ref_dir),
                "runtime_seconds": total_runtime,
                "output_size_bytes": metadata["output_size_bytes"],
            }
        )
        return row
    except Exception as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "created_utc": now_utc(),
            **sample,
            "success": False,
            "error": repr(exc),
            "notes": "Exception while generating v1c EP sample; inspect metadata and logs.",
            "total_runtime_seconds": time.perf_counter() - start,
        }
        write_ep_json(out_dir / "metadata.json", meta)
        return failure_row(sample, out_dir, repr(exc), time.perf_counter() - start)


def failure_row(sample: dict, out_dir: Path, notes: str, runtime: float) -> dict:
    return {
        "sample_id": sample["sample_id"],
        "condition": sample["condition"],
        "model": sample.get("model", "Bueno"),
        "formulation": sample.get("formulation", "monodomain"),
        "geometry": "1 cm x 1 cm openCARP triangular sheet",
        "apd_parameter": sample.get("apd_parameter"),
        "apd_scale": sample.get("apd_scale"),
        "conductivity_scale": sample.get("conductivity_scale"),
        "patch_shape": sample.get("patch_shape"),
        "patch_center_x": sample.get("patch_center_x"),
        "patch_center_y": sample.get("patch_center_y"),
        "patch_radius": sample.get("patch_radius"),
        "patch_radius_mm": sample.get("patch_radius_mm"),
        "stimulus_location": sample.get("stimulus_location"),
        "base_gil": sample.get("base_gil"),
        "anisotropy_factor": sample.get("anisotropy_factor"),
        "nbeats": sample.get("nbeats", DEFAULT_NBEATS),
        "BCL": sample.get("BCL", DEFAULT_BCL_MS),
        "conditioning_used": False,
        "prepacing_reference_dir": "",
        "success": False,
        "output_dir": rel(out_dir),
        "runtime_seconds": runtime,
        "output_size_bytes": recursive_size_bytes(out_dir),
        "notes": notes,
    }


def write_run_summary(ep_base: Path, rows: list[dict], total_runtime: float, mode: str, seed: int, workers: int) -> None:
    success_rows = [r for r in rows if truthy(r.get("success"))]
    output_size = recursive_size_bytes(ep_base)
    avg_runtime = sum(float(r.get("runtime_seconds") or 0.0) for r in rows) / len(rows) if rows else 0.0
    summary = {
        "created_utc": now_utc(),
        "mode": mode,
        "seed": seed,
        "workers": workers,
        "expected_composition": expected_composition(mode, len(rows)),
        "row_count": len(rows),
        "success_count": len(success_rows),
        "failure_count": len(rows) - len(success_rows),
        "total_runtime_seconds": total_runtime,
        "average_runtime_per_sample_seconds": avg_runtime,
        "estimated_runtime_5000_seconds": avg_runtime * 5000.0 / max(workers, 1),
        "storage_used_bytes": output_size,
        "storage_used_gb": output_size / 1e9,
        "estimated_storage_5000_bytes": int(output_size * 5000 / max(len(rows), 1)),
        "estimated_storage_5000_gb": output_size * 5000 / max(len(rows), 1) / 1e9,
        "required_files": EP_REQUIRED_FILES,
        "compact_note": "Reference runs use sparse Vm output; final conditioned outputs keep standard EP files.",
    }
    write_json(ep_base / "run_summary.json", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[PILOT_MODE, FULL_MODE], default=PILOT_MODE)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--compact", action="store_true", help="Recorded for compatibility; v1c EP is compact by design.")
    parser.add_argument("--run-full-5000", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == FULL_MODE and not args.run_full_5000:
        raise SystemExit("Refusing full5000 generation without --run-full-5000")
    paths = mode_paths(args.mode)
    ep_base = paths["ep"]
    ep_base.mkdir(parents=True, exist_ok=True)
    samples = create_sample_plan(args.mode, args.seed, args.n_samples)
    write_json(
        ep_base / "sample_plan.json",
        {
            "created_utc": now_utc(),
            "mode": args.mode,
            "seed": args.seed,
            "composition": expected_composition(args.mode, len(samples)),
            "samples": samples,
        },
    )
    if args.dry_run:
        print(f"Dry run wrote sample plan for {len(samples)} samples to {ep_base / 'sample_plan.json'}")
        return 0

    start = time.perf_counter()
    rows: list[dict] = []
    if args.workers <= 1:
        for sample in samples:
            row = run_one_sample(sample, ep_base, args.force)
            rows.append(row)
            print(f"{row['sample_id']}: success={row['success']} {row.get('notes', '')}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one_sample, sample, ep_base, args.force) for sample in samples]
            for fut in as_completed(futures):
                row = fut.result()
                rows.append(row)
                print(f"{row['sample_id']}: success={row['success']} {row.get('notes', '')}")
    rows.sort(key=lambda r: r["sample_id"])
    write_csv_rows(ep_base / "manifest.csv", rows, MANIFEST_FIELDS)
    total_runtime = time.perf_counter() - start
    write_run_summary(ep_base, rows, total_runtime, args.mode, args.seed, args.workers)
    success_count = sum(1 for r in rows if truthy(r.get("success")))
    print(f"EP v1c {args.mode}: {success_count}/{len(rows)} successful in {total_runtime:.1f}s")
    return 0 if success_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
