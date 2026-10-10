from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FINAL_DIR = OUTPUTS_DIR / "icbinb_sensor_height_ood"
ANALYSIS_DIR = OUTPUTS_DIR / "icbinb_required_analysis"
COHORT_PATH = ANALYSIS_DIR / "paired_test_cohort.csv"
FIELD_INDEX_PATH = (
    OUTPUTS_DIR / "dataset_v1e_500_combined_pilot" / "field_ready" / "sample_index.csv"
)
BASE_MANIFEST_PATH = OUTPUTS_DIR / "dataset_v1e_500_combined_pilot" / "manifest.csv"
CONTRACT_PATH = ANALYSIS_DIR / "evaluation_contract.json"

HEIGHTS_MM = (35, 50, 65)
NOMINAL_HEIGHT_MM = 50
BASELINE_M = 0.03
GRID_SIZE = 32
ACTIVE_CHANNELS = ("direct_bz_50mm", "direct_axial_grad_30mm_baseline")
CHANNEL_LAYOUT = (
    "direct_bz_50mm",
    "direct_axial_grad_30mm_baseline",
    "direct_bz_noise_medium",
    "direct_bz_masked_noise",
)


def load_numeric_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = load_numeric_module(
    "_v1e_direct_generator", PROJECT_ROOT / "scripts" / "113_generate_v1e_500_direct_magnetic.py"
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def require_inputs() -> None:
    missing = [
        str(path)
        for path in (COHORT_PATH, FIELD_INDEX_PATH, BASE_MANIFEST_PATH, CONTRACT_PATH)
        if not path.is_file()
    ]
    if missing:
        raise SystemExit(f"Missing prerequisites: {missing}")
    contract = read_json(CONTRACT_PATH)
    if not contract.get("validation", {}).get("passed"):
        raise SystemExit("The ICBINB evaluation contract is not valid.")


def canonical_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = pd.read_csv(COHORT_PATH).sort_values("cohort_index").reset_index(drop=True)
    if len(cohort) != 75 or cohort["base_sample_id"].nunique() != 75:
        raise SystemExit("The canonical cohort must contain exactly 75 independent base samples.")
    if set(cohort["split"]) != {"test"} or set(cohort["combined_variant"]) != {"combined_bz_axial"}:
        raise SystemExit("The canonical cohort is not the held-out combined_bz_axial cohort.")

    index = pd.read_csv(FIELD_INDEX_PATH)
    rows = index[
        index["split"].eq("test")
        & index["combined_variant"].eq("combined_bz_axial")
        & index["base_sample_id"].isin(cohort["base_sample_id"])
    ].copy()
    rows = cohort[["cohort_index", "base_sample_id"]].merge(
        rows, on="base_sample_id", how="left", validate="one_to_one"
    ).sort_values("cohort_index").reset_index(drop=True)
    if len(rows) != 75 or rows["B_meas_file"].isna().any():
        raise SystemExit("Could not resolve all canonical cases in the field-ready index.")

    base = pd.read_csv(BASE_MANIFEST_PATH)
    base = rows[["cohort_index", "base_sample_id"]].merge(
        base, on="base_sample_id", how="left", validate="one_to_one"
    ).sort_values("cohort_index").reset_index(drop=True)
    if len(base) != 75 or base["output_dir"].isna().any():
        raise SystemExit("Could not resolve all canonical cases in the EP manifest.")
    return rows, base


def project_height(j_eq, centroids, areas, times, mesh, height_mm: int, args):
    height_m = height_mm / 1000.0
    sensors, xs, ys, sensor_z = GENERATOR.sensor_grid(mesh, GRID_SIZE, height_m)
    direct = GENERATOR.project_bz_cuda(
        j_eq, centroids, areas, sensors, args.time_chunk, args.sensor_chunk
    ).reshape(len(times), GRID_SIZE, GRID_SIZE)
    sensors_low = sensors.copy()
    sensors_high = sensors.copy()
    sensors_low[:, 2] -= BASELINE_M / 2.0
    sensors_high[:, 2] += BASELINE_M / 2.0
    low = GENERATOR.project_bz_cuda(
        j_eq, centroids, areas, sensors_low, args.time_chunk, args.sensor_chunk
    ).reshape(direct.shape)
    high = GENERATOR.project_bz_cuda(
        j_eq, centroids, areas, sensors_high, args.time_chunk, args.sensor_chunk
    ).reshape(direct.shape)
    axial = ((high - low) / BASELINE_M).astype(np.float32)
    coords = GENERATOR.coordinate_channels(xs, ys, height_m)
    return direct.astype(np.float32), axial, sensors, coords, float(sensor_z)


def build_tensor(direct: np.ndarray, axial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = (*direct.shape, len(CHANNEL_LAYOUT))
    measurement = np.zeros(shape, dtype=np.float32)
    measurement[..., 0] = direct
    measurement[..., 1] = axial
    mask = np.zeros((GRID_SIZE, GRID_SIZE, len(CHANNEL_LAYOUT)), dtype=bool)
    mask[..., 0] = True
    mask[..., 1] = True
    return measurement, mask


def compare_nominal(
    source_row: pd.Series,
    direct: np.ndarray,
    axial: np.ndarray,
    sensors: np.ndarray,
    coords: np.ndarray,
) -> dict[str, Any]:
    saved = np.load(PROJECT_ROOT / source_row["B_meas_file"], mmap_mode="r")
    saved_sensors = np.load(PROJECT_ROOT / source_row["sensor_positions_file"], mmap_mode="r")
    saved_coords = np.load(PROJECT_ROOT / source_row["coordinate_channels_file"], mmap_mode="r")
    checks = {
        "direct_bz_exact": bool(np.array_equal(direct, saved[..., 0])),
        "axial_gradient_exact": bool(np.array_equal(axial, saved[..., 1])),
        "sensor_positions_exact": bool(np.array_equal(sensors, saved_sensors)),
        "coordinate_channels_exact": bool(np.array_equal(coords, saved_coords)),
        "inactive_channels_zero": bool(np.count_nonzero(saved[..., 2:]) == 0),
    }
    return {
        **checks,
        "passed": bool(all(checks.values())),
        "direct_bz_max_abs_difference": float(np.max(np.abs(direct - saved[..., 0]))),
        "axial_gradient_max_abs_difference": float(np.max(np.abs(axial - saved[..., 1]))),
    }


def save_generated_case(
    root: Path,
    source_row: pd.Series,
    source_meta: dict[str, Any],
    height_mm: int,
    direct: np.ndarray,
    axial: np.ndarray,
    sensors: np.ndarray,
    coords: np.ndarray,
    times: np.ndarray,
    sensor_z: float,
) -> dict[str, Any]:
    row_id = f"{source_row['base_sample_id']}__combined_bz_axial__height_{height_mm:03d}mm"
    out_dir = root / "inputs" / f"height_{height_mm:03d}mm" / row_id
    published_dir = FINAL_DIR / out_dir.relative_to(root)
    out_dir.mkdir(parents=True, exist_ok=False)
    measurement, mask = build_tensor(direct, axial)
    np.save(out_dir / "B_meas.npy", measurement)
    np.save(out_dir / "sensor_positions.npy", sensors.astype(np.float32))
    np.save(out_dir / "sensor_mask.npy", mask)
    np.save(out_dir / "coordinate_channels.npy", coords.astype(np.float32))
    np.save(out_dir / "time_ms.npy", times.astype(np.float32))
    metadata = {
        "base_sample_id": str(source_row["base_sample_id"]),
        "condition": str(source_row["condition"]),
        "variant": "combined_bz_axial",
        "height_mm": height_mm,
        "height_m": height_mm / 1000.0,
        "nominal_training_height_mm": NOMINAL_HEIGHT_MM,
        "baseline_m": BASELINE_M,
        "sensor_z_m": sensor_z,
        "sensor_grid": GRID_SIZE,
        "channel_layout": list(CHANNEL_LAYOUT),
        "active_channels": list(ACTIVE_CHANNELS),
        "operator_kind": "direct synthetic Biot-Savart projection for acquisition-geometry sensitivity",
        "direct_projection": True,
        "source_current": "J_eq = -D_eff grad(Vm), equivalent-current proxy",
        "source_metadata": source_meta,
        "ep_simulation_rerun": False,
        "training_or_adaptation_performed": False,
        "claim_boundary": "Zero-shot sensor-tissue standoff stress test; not patient-specific validation.",
    }
    write_json(out_dir / "measurement_metadata.json", metadata)
    return {
        "ood_row_id": row_id,
        "height_mm": height_mm,
        "height_m": height_mm / 1000.0,
        "storage_kind": "isolated_generated",
        "B_meas_file": repo_relative(published_dir / "B_meas.npy"),
        "sensor_positions_file": repo_relative(published_dir / "sensor_positions.npy"),
        "sensor_mask_file": repo_relative(published_dir / "sensor_mask.npy"),
        "coordinate_channels_file": repo_relative(published_dir / "coordinate_channels.npy"),
        "time_ms_file": repo_relative(published_dir / "time_ms.npy"),
        "measurement_metadata_file": repo_relative(published_dir / "measurement_metadata.json"),
    }


def nominal_reference(source_row: pd.Series) -> dict[str, Any]:
    return {
        "ood_row_id": f"{source_row['base_sample_id']}__combined_bz_axial__height_050mm",
        "height_mm": NOMINAL_HEIGHT_MM,
        "height_m": NOMINAL_HEIGHT_MM / 1000.0,
        "storage_kind": "validated_existing_nominal_reference",
        "B_meas_file": str(source_row["B_meas_file"]),
        "sensor_positions_file": str(source_row["sensor_positions_file"]),
        "sensor_mask_file": str(source_row["sensor_mask_file"]),
        "coordinate_channels_file": str(source_row["coordinate_channels_file"]),
        "time_ms_file": str(source_row["time_ms_file"]),
        "measurement_metadata_file": str(source_row["measurement_metadata_file"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-chunk", type=int, default=32)
    parser.add_argument("--sensor-chunk", type=int, default=1024)
    args = parser.parse_args()
    if args.time_chunk < 1 or args.sensor_chunk < 1:
        raise SystemExit("Chunk sizes must be positive.")
    if FINAL_DIR.exists():
        raise SystemExit(f"Refusing to overwrite existing Module 4 directory: {FINAL_DIR}")
    require_inputs()
    rows, base_rows = canonical_rows()
    if not GENERATOR.torch.cuda.is_available():
        raise SystemExit("CUDA is required for the parity-identical projection path.")

    start = time.perf_counter()
    parity_records: list[dict[str, Any]] = []
    output_records: list[dict[str, Any]] = []
    runtimes: list[dict[str, Any]] = []
    temp_name = tempfile.mkdtemp(prefix=".icbinb_sensor_height_ood_", dir=OUTPUTS_DIR)
    temp_root = Path(temp_name)
    published = False
    try:
        for position, (source_row, base_row) in enumerate(
            zip(rows.to_dict("records"), base_rows.to_dict("records")), start=1
        ):
            source = pd.Series(source_row)
            base = pd.Series(base_row)
            case_start = time.perf_counter()
            ep_dir = PROJECT_ROOT / str(base["output_dir"])
            j_eq, centroids, areas, times, mesh, source_meta = GENERATOR.compute_source(ep_dir)
            generated: dict[int, tuple[Any, ...]] = {}
            for height_mm in HEIGHTS_MM:
                generated[height_mm] = project_height(
                    j_eq, centroids, areas, times, mesh, height_mm, args
                )

            nominal = generated[NOMINAL_HEIGHT_MM]
            parity = compare_nominal(source, nominal[0], nominal[1], nominal[2], nominal[3])
            parity_records.append(
                {
                    "cohort_index": int(source["cohort_index"]),
                    "base_sample_id": str(source["base_sample_id"]),
                    **parity,
                }
            )
            if not parity["passed"]:
                raise RuntimeError(
                    f"Nominal 50 mm parity failed for {source['base_sample_id']}: {parity}"
                )

            common = {
                "cohort_index": int(source["cohort_index"]),
                "source_row_id": str(source["row_id"]),
                "base_sample_id": str(source["base_sample_id"]),
                "split": "test",
                "condition": str(source["condition"]),
                "combined_variant": "combined_bz_axial",
            }
            output_records.append({**common, **nominal_reference(source)})
            for height_mm in (35, 65):
                direct, axial, sensors, coords, sensor_z = generated[height_mm]
                files = save_generated_case(
                    temp_root,
                    source,
                    source_meta,
                    height_mm,
                    direct,
                    axial,
                    sensors,
                    coords,
                    times,
                    sensor_z,
                )
                output_records.append({**common, **files})
            runtimes.append(
                {
                    "base_sample_id": str(source["base_sample_id"]),
                    "seconds": time.perf_counter() - case_start,
                }
            )
            if position % 10 == 0 or position == len(rows):
                print(f"projected and parity-checked {position}/{len(rows)} base cases", flush=True)

        manifest = pd.DataFrame(output_records).sort_values(
            ["height_mm", "cohort_index"]
        ).reset_index(drop=True)
        parity_df = pd.DataFrame(parity_records).sort_values("cohort_index")
        runtime_df = pd.DataFrame(runtimes)
        expected_pairs = {(int(h), str(b)) for h in HEIGHTS_MM for b in rows["base_sample_id"]}
        observed_pairs = set(zip(manifest["height_mm"].astype(int), manifest["base_sample_id"].astype(str)))
        errors: list[str] = []
        if expected_pairs != observed_pairs:
            errors.append("The output manifest does not contain exactly 75 cases at each height.")
        if manifest.duplicated(["height_mm", "base_sample_id"]).any():
            errors.append("Duplicate (height_mm, base_sample_id) rows were generated.")
        if not bool(parity_df["passed"].all()):
            errors.append("At least one nominal 50 mm parity check failed.")
        for record in manifest.to_dict("records"):
            for key in (
                "B_meas_file",
                "sensor_positions_file",
                "sensor_mask_file",
                "coordinate_channels_file",
                "time_ms_file",
                "measurement_metadata_file",
            ):
                path = PROJECT_ROOT / str(record[key])
                check_path = (
                    temp_root / path.relative_to(FINAL_DIR)
                    if path.is_relative_to(FINAL_DIR)
                    else path
                )
                if not check_path.is_file():
                    errors.append(f"Missing output referenced by manifest: {path}")
        manifest.to_csv(temp_root / "sensor_height_manifest.csv", index=False)
        parity_df.to_csv(temp_root / "nominal_50mm_parity.csv", index=False)
        runtime_df.to_csv(temp_root / "generation_runtime.csv", index=False)
        elapsed = time.perf_counter() - start
        report = {
            "schema_version": "1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "passed": not errors,
            "errors": errors,
            "cohort_base_cases": int(rows["base_sample_id"].nunique()),
            "manifest_rows": int(len(manifest)),
            "heights_mm": list(HEIGHTS_MM),
            "rows_per_height": {
                str(int(key)): int(value)
                for key, value in manifest["height_mm"].value_counts().sort_index().items()
            },
            "nominal_parity_cases": int(len(parity_df)),
            "nominal_bit_exact_cases": int(parity_df["passed"].sum()),
            "ep_simulations_rerun": 0,
            "biot_savart_reprojections": int(len(rows) * len(HEIGHTS_MM) * 3),
            "new_tensor_heights_mm": [35, 65],
            "nominal_tensor_policy": "Reference the validated existing 50 mm tensor; do not copy or overwrite it.",
            "runtime_seconds": elapsed,
            "mean_runtime_seconds_per_base_case": float(runtime_df["seconds"].mean()),
            "source_cohort": repo_relative(COHORT_PATH),
            "source_cohort_sha256": sha256(COHORT_PATH),
            "source_contract": repo_relative(CONTRACT_PATH),
            "source_contract_sha256": sha256(CONTRACT_PATH),
            "scientific_scope": "Synthetic held-out sensor-tissue standoff sensitivity with frozen downstream models.",
            "claim_boundary": "Not patient-specific, clinical, real-SQUID, or learned geometry conditioning.",
        }
        write_json(temp_root / "generation_validation.json", report)
        if errors:
            print(json.dumps(report, indent=2), flush=True)
            raise RuntimeError("Module 4 input validation failed; final outputs were not published.")
        temp_root.replace(FINAL_DIR)
        published = True
        print(
            json.dumps(
                {
                    "passed": True,
                    "output_dir": str(FINAL_DIR),
                    "manifest_rows": len(manifest),
                    "nominal_bit_exact_cases": int(parity_df["passed"].sum()),
                    "runtime_seconds": elapsed,
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    finally:
        if not published and temp_root.exists():
            shutil.rmtree(temp_root)


if __name__ == "__main__":
    raise SystemExit(main())
