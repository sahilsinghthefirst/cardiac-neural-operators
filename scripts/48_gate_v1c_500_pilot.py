                      
                                                        

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from bocf_v1c_common import (
    COMPOSITIONS,
    FULL_COUNT,
    PILOT_MODE,
    mode_paths,
    read_csv_rows,
    recursive_size_bytes,
    truthy,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
SUCCESS_REPORT = ROOT / "docs" / "V1C_500_PILOT_SUCCESS_REPORT.md"
DIAGNOSTIC_REPORT = ROOT / "docs" / "V1C_500_PILOT_DIAGNOSTIC_REPORT.md"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return json.loads(path.read_text())


def numeric_column(rows: list[dict], key: str) -> np.ndarray:
    vals = []
    for row in rows:
        try:
            val = float(row.get(key, ""))
            if math.isfinite(val):
                vals.append(val)
        except (TypeError, ValueError):
            pass
    return np.asarray(vals, dtype=float)


def pass_fail_row(name: str, passed: bool, detail: str) -> str:
    return f"| {name} | {'PASS' if passed else 'FAIL'} | {detail} |"


def estimate_runtime_hours(paths: dict[str, Path]) -> tuple[float, dict]:
    details = {}
    total = 0.0
    for label, base in [("ep", paths["ep"]), ("magnetic", paths["mag"]), ("augmentation", paths["aug"])]:
        summary = load_json(base / "run_summary.json")
        runtime = float(summary.get("runtime_seconds") or summary.get("total_runtime_seconds") or 0.0)
        row_count = float(summary.get("row_count") or summary.get("base_sample_count") or 500.0)
        if label == "augmentation":
            row_count = float(summary.get("base_sample_count") or 500.0)
        est = runtime * FULL_COUNT / max(row_count, 1.0)
        details[label] = {"observed_seconds": runtime, "estimated_5000_seconds": est}
        total += est
    return total / 3600.0, details


def estimate_storage_gb(paths: dict[str, Path]) -> tuple[float, dict]:
    details = {}
    total = 0.0
    for label, base in [("ep", paths["ep"]), ("magnetic", paths["mag"]), ("magnetic_debug", paths["mag_debug"]), ("augmentation", paths["aug"]), ("ready", paths["ready"]), ("aug_ready", paths["aug_ready"])]:
        size = recursive_size_bytes(base)
        if label == "magnetic_debug":
                                                                                    
            est = size
        else:
            est = size * FULL_COUNT / 500.0
        details[label] = {"observed_gb": size / 1e9, "estimated_5000_gb": est / 1e9}
        total += est
    return total / 1e9, details


def gate() -> tuple[bool, dict, list[tuple[str, bool, str]]]:
    paths = mode_paths(PILOT_MODE)
    ep_report = load_json(paths["ep"] / "dataset_quality_report.json")
    mag_report = load_json(paths["mag"] / "magnetic_quality_report.json")
    aug_report = load_json(paths["aug"] / "augmentation_quality_report.json")
    ready_report = load_json(paths["ready"] / "dataset_ready_quality_report.json")
    aug_ready_report = load_json(paths["aug_ready"] / "augmented_dataset_quality_report.json")
    ep_rows = read_csv_rows(paths["ep"] / "manifest.csv")
    mag_rows = read_csv_rows(paths["mag"] / "manifest.csv")
    aug_rows = read_csv_rows(paths["aug"] / "manifest.csv")
    targets = read_csv_rows(paths["ready"] / "targets_ep.csv")
    index = read_csv_rows(paths["ready"] / "sample_index.csv")
    fixed = read_csv_rows(paths["ready"] / "features_mig_deployable_fixed.csv")
    split_rows = read_csv_rows(paths["aug_ready"] / "splits" / "grouped_train_val_test.csv")

    gates: list[tuple[str, bool, str]] = []
    ep_success = sum(1 for r in ep_rows if truthy(r.get("success")))
    mag_success = sum(1 for r in mag_rows if truthy(r.get("success")))
    aug_success = sum(1 for r in aug_rows if truthy(r.get("success")))
    gates.append(("EP generation success >=98%", ep_success / max(len(ep_rows), 1) >= 0.98, f"{ep_success}/{len(ep_rows)}"))
    gates.append(("Magnetic generation success >=98%", mag_success / max(len(mag_rows), 1) >= 0.98, f"{mag_success}/{len(mag_rows)}"))
    gates.append(("Augmentation success >=98%", aug_success / max(len(aug_rows), 1) >= 0.98, f"{aug_success}/{len(aug_rows)}"))
    gates.append(("EP quality report passed", bool(ep_report.get("passed")), "; ".join(ep_report.get("problems", [])) or "ok"))
    gates.append(("Magnetic quality report passed", bool(mag_report.get("passed")), "; ".join(mag_report.get("problems", [])) or "ok"))
    gates.append(("Augmentation quality report passed", bool(aug_report.get("passed")), "; ".join(aug_report.get("problems", [])) or "ok"))
    gates.append(("Ready tables align", bool(ready_report.get("passed")) and bool(aug_ready_report.get("passed")), "base and augmented ready reports"))

    expected = COMPOSITIONS[PILOT_MODE]
    counts = {condition: 0 for condition in expected}
    for row in ep_rows:
        if truthy(row.get("success")):
            counts[row["condition"]] += 1
    gates.append(("Condition counts exact", counts == expected, f"{counts} expected {expected}"))

    nonhom_bad = []
    for row in index:
        if row["condition"] != "homogeneous":
            try:
                mask = np.genfromtxt(ROOT / row["mask_file"], delimiter=",", names=True)
                if not np.any(np.atleast_1d(mask["abnormal"]).astype(bool)):
                    nonhom_bad.append(row["sample_id"])
            except Exception:
                nonhom_bad.append(row["sample_id"])
    gates.append(("Non-homogeneous masks nonempty", not nonhom_bad, f"{len(nonhom_bad)} bad masks"))

    all_zero_bz = []
    for row in mag_rows:
        if not truthy(row.get("success")):
            continue
        try:
            bz = np.load(ROOT / row["Bz_file"])
            if np.allclose(bz, 0.0):
                all_zero_bz.append(row["sample_id"])
        except Exception:
            all_zero_bz.append(row["sample_id"])
    gates.append(("No all-zero Bz", not all_zero_bz, f"{len(all_zero_bz)} problematic samples"))

    finite_features = True
    for row in fixed:
        for key, value in row.items():
            if key in {"sample_id", "condition"}:
                continue
            try:
                if not math.isfinite(float(value)):
                    finite_features = False
                    break
            except (TypeError, ValueError):
                finite_features = False
                break
    gates.append(("Deployable features finite", finite_features, "fixed feature table"))

    split_by_base = {}
    for row in split_rows:
        split_by_base.setdefault(row["base_sample_id"], set()).add(row["split"])
    leaked = [base for base, splits in split_by_base.items() if len(splits) > 1]
    gates.append(("No grouped split leakage", not leaked, f"{len(leaked)} leaked base_sample_id values"))

    apd90 = numeric_column(targets, "APD90_mean")
    lat_range = numeric_column(targets, "LAT_range")
    patch_x = numeric_column([r for r in index if r["condition"] != "homogeneous"], "patch_center_x")
    patch_y = numeric_column([r for r in index if r["condition"] != "homogeneous"], "patch_center_y")
    apd_scale = numeric_column(index, "apd_scale")
    cond_scale = numeric_column(index, "conductivity_scale")
    gates.append(("APD90 not collapsed", apd90.size > 1 and np.std(apd90) > 1e-6, f"std={float(np.std(apd90)) if apd90.size else 'nan'}"))
    gates.append(("LAT range not collapsed", lat_range.size > 1 and np.std(lat_range) > 1e-6, f"std={float(np.std(lat_range)) if lat_range.size else 'nan'}"))
    gates.append(("Patch locations not collapsed", patch_x.size > 1 and patch_y.size > 1 and np.std(patch_x) > 1.0 and np.std(patch_y) > 1.0, "x/y spread"))
    gates.append(("Severity parameters not collapsed", np.std(apd_scale) > 1e-6 and np.std(cond_scale) > 1e-6, "tau_si and conductivity spread"))

    runtime_5000_hours, runtime_detail = estimate_runtime_hours(paths)
    storage_5000_gb, storage_detail = estimate_storage_gb(paths)
    runtime_warn = runtime_5000_hours > 24.0
    storage_warn = storage_5000_gb > 250.0
    gates.append(("Runtime estimate evaluated", True, f"{runtime_5000_hours:.2f} h for 5,000" + (" WARNING >24h" if runtime_warn else "")))
    gates.append(("Storage estimate evaluated", True, f"{storage_5000_gb:.2f} GB for 5,000" + (" WARNING >250GB" if storage_warn else "")))

    passed = all(passed for _name, passed, _detail in gates if not _name.endswith("evaluated"))
    payload = {
        "passed": passed,
        "ep_success": ep_success,
        "magnetic_success": mag_success,
        "augmentation_success": aug_success,
        "runtime_5000_hours_estimate": runtime_5000_hours,
        "runtime_detail": runtime_detail,
        "storage_5000_gb_estimate": storage_5000_gb,
        "storage_detail": storage_detail,
        "runtime_warning_over_24h": runtime_warn,
        "storage_warning_over_250gb": storage_warn,
        "problematic_masks": nonhom_bad[:50],
        "problematic_bz": all_zero_bz[:50],
        "split_leakage": leaked[:50],
        "counts": counts,
        "expected_counts": expected,
    }
    write_json(paths["ready"] / "v1c_500_pilot_gate.json", payload)
    return passed, payload, gates


def write_report(passed: bool, payload: dict, gates: list[tuple[str, bool, str]]) -> Path:
    rows = "\n".join(pass_fail_row(name, ok, detail.replace("\n", " ")) for name, ok, detail in gates)
    command = "venv_carputils/bin/python scripts/49_run_v1c_500_pilot_pipeline.py --mode full5000 --workers 8 --compact --seed 20260525 --run-full-5000"
    resume = command + " --skip-existing"
    if passed:
        text = f"""# V1C 500 Pilot Success Report

The v1c 500-sample pilot passed the configured data-layer quality gates. This report does not launch the 5,000-sample run and does not implement ML training.

## Final Counts

- EP successful samples: {payload['ep_success']}/500
- Magnetic successful samples: {payload['magnetic_success']}/500
- Augmented successful variants: {payload['augmentation_success']}/5500
- Estimated 5,000-sample runtime: {payload['runtime_5000_hours_estimate']:.2f} hours
- Estimated 5,000-sample storage: {payload['storage_5000_gb_estimate']:.2f} GB

## Gate Table

| Gate | Result | Detail |
|---|---:|---|
{rows}

## What Went Well

- The conditioned Bueno/BOCF monodomain EP layer produced the expected pilot composition.
- Compact magnetic-forward outputs preserved MIG-ready Bz signals without storing heavy current/vector arrays for every sample.
- Sensor/noise/geometry variants were generated as measurement robustness variants, not new EP physiology.
- Deployable feature tables use fixed-time or magnetic-signal windows only.
- Grouped splits keep every variant of a base sample in the same split.

## Important Figures To Inspect

- `figures/dataset_v1c_500_pilot/apd_distribution_by_condition.png`
- `figures/dataset_v1c_500_pilot/lat_range_by_condition.png`
- `figures/dataset_v1c_500_pilot/patch_locations.png`
- `figures/dataset_v1c_500_pilot/runtime_storage_summary.png`

## Launch Command For 5,000

```bash
{command}
```

## Resume Command For 5,000

```bash
{resume}
```

## Known Limitations

- Bueno/BOCF remains a minimal human ventricular tissue model for controlled synthetic benchmarking.
- Mixed cases are APD/repolarization proxy plus conduction-slowing perturbations, not clinical ischemia.
- J_eq is an equivalent-current proxy, not intracellular current reconstruction.
- B fields are synthetic/proxy relative fields until calibration is added.
"""
        SUCCESS_REPORT.write_text(text)
        return SUCCESS_REPORT
    failed = [f"- {name}: {detail}" for name, ok, detail in gates if not ok]
    text = f"""# V1C 500 Pilot Diagnostic Report

The v1c 500-sample pilot did not pass all configured gates. The 5,000-sample run should not be launched until these issues are fixed and the pilot is rerun.

## Failed Gates

{chr(10).join(failed) if failed else '- No explicit failed gate recorded; inspect warnings and missing reports.'}

## Gate Table

| Gate | Result | Detail |
|---|---:|---|
{rows}

## Symptoms And Likely Causes

- Failed EP gates usually indicate openCARP run failures, missing derived maps, or invalid randomized patch/mask metadata.
- Failed magnetic gates usually indicate compact Bz generation issues, missing sensor metadata, or all-zero/nonfinite projected fields.
- Failed augmented gates usually indicate sensor mask handling or transient recomputation failures for jitter/height variants.
- Failed ready-table gates usually indicate missing manifests, path mismatches, or deployable feature rows not aligning with targets.

## Problematic Samples

- Mask issues: {payload.get('problematic_masks', [])[:20]}
- Bz issues: {payload.get('problematic_bz', [])[:20]}
- Split leakage: {payload.get('split_leakage', [])[:20]}

## Recommended Fixes

1. Inspect the quality reports under `outputs/bocf_dataset_v1c_500_pilot/`, `outputs/magnetic_dataset_v1c_500_pilot/`, and `outputs/magnetic_dataset_v1c_500_pilot_augmented/`.
2. Inspect per-sample openCARP stdout/stderr logs for failed EP samples.
3. Rerun only failed or missing stages using the orchestrator with `--skip-existing`.
4. Rerun this gate before considering the 5,000-sample command.

## Safe Rerun

```bash
venv_carputils/bin/python scripts/49_run_v1c_500_pilot_pipeline.py --workers 8 --compact --seed 20260525 --skip-existing
```
"""
    DIAGNOSTIC_REPORT.write_text(text)
    return DIAGNOSTIC_REPORT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def main() -> int:
    _args = parse_args()
    passed, payload, gates = gate()
    report = write_report(passed, payload, gates)
    print(json.dumps({"passed": passed, "report": str(report), "runtime_5000_hours": payload["runtime_5000_hours_estimate"], "storage_5000_gb": payload["storage_5000_gb_estimate"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
