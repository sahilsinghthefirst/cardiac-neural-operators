                      
                                                            

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from bocf_v1c_common import FULL_MODE, mode_paths, read_csv_rows, recursive_size_bytes, truthy, write_json


ROOT = Path(__file__).resolve().parents[1]
SUCCESS_REPORT = ROOT / "docs" / "V1C_5000_SUCCESS_REPORT.md"
DIAGNOSTIC_REPORT = ROOT / "docs" / "V1C_5000_DIAGNOSTIC_REPORT.md"
EXPECTED_COUNTS = {
    "homogeneous": 750,
    "apd_repolarization_perturbation": 1250,
    "conduction_slowing_perturbation": 1250,
    "mixed_apd_conduction_perturbation": 1750,
}
EXPECTED_BASE = 5000
EXPECTED_AUG = 55000


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


def report_row(name: str, passed: bool, detail: str) -> str:
    return f"| {name} | {'PASS' if passed else 'FAIL'} | {detail} |"


def storage_summary(paths: dict[str, Path]) -> tuple[float, list[str]]:
    rows = []
    total = 0
    for label in ["ep", "mag", "mag_debug", "aug", "ready", "aug_ready"]:
        path = paths[label]
        size = recursive_size_bytes(path)
        total += size
        rows.append(f"- `{path.relative_to(ROOT)}`: {size / 1e9:.2f} GB")
    return total / 1e9, rows


def runtime_summary(paths: dict[str, Path]) -> tuple[float, list[str]]:
    rows = []
    total = 0.0
    for label, base in [("EP", paths["ep"]), ("Magnetic", paths["mag"]), ("Augmentation", paths["aug"])]:
        data = load_json(base / "run_summary.json")
        seconds = float(data.get("runtime_seconds") or data.get("total_runtime_seconds") or 0.0)
        total += seconds
        rows.append(f"- {label}: {seconds / 3600.0:.2f} h")
    pipeline = load_json(paths["ep"] / "pipeline_run_summary.json")
    pipe_seconds = float(pipeline.get("runtime_seconds") or 0.0)
    if pipe_seconds:
        rows.append(f"- Pipeline wall time: {pipe_seconds / 3600.0:.2f} h")
        total = pipe_seconds
    return total / 3600.0, rows


def gate() -> tuple[bool, dict, list[tuple[str, bool, str]]]:
    paths = mode_paths(FULL_MODE)
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
    ep_success = sum(1 for row in ep_rows if truthy(row.get("success")))
    mag_success = sum(1 for row in mag_rows if truthy(row.get("success")))
    aug_success = sum(1 for row in aug_rows if truthy(row.get("success")))
    gates.append(("EP generation success >=98%", ep_success / max(len(ep_rows), 1) >= 0.98, f"{ep_success}/{len(ep_rows)}"))
    gates.append(("Magnetic generation success >=98%", mag_success / max(len(mag_rows), 1) >= 0.98, f"{mag_success}/{len(mag_rows)}"))
    gates.append(("Augmentation success >=98%", aug_success / max(len(aug_rows), 1) >= 0.98, f"{aug_success}/{len(aug_rows)}"))
    gates.append(("EP quality report passed", bool(ep_report.get("passed")), "; ".join(ep_report.get("problems", [])) or "ok"))
    gates.append(("Magnetic quality report passed", bool(mag_report.get("passed")), "; ".join(mag_report.get("problems", [])) or "ok"))
    gates.append(("Augmentation quality report passed", bool(aug_report.get("passed")), "; ".join(aug_report.get("problems", [])) or "ok"))
    gates.append(("Ready tables align", bool(ready_report.get("passed")) and bool(aug_ready_report.get("passed")), "base and augmented ready reports"))

    counts = Counter(row["condition"] for row in ep_rows if truthy(row.get("success")))
    gates.append(("Expected full condition counts", dict(counts) == EXPECTED_COUNTS, f"{dict(counts)} expected {EXPECTED_COUNTS}"))
    gates.append(("Base sample count", len(ep_rows) == EXPECTED_BASE, f"{len(ep_rows)}/{EXPECTED_BASE}"))
    gates.append(("Augmented row count", len(aug_rows) == EXPECTED_AUG, f"{len(aug_rows)}/{EXPECTED_AUG}"))

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
    gates.append(("Grouped train/val/test split exists", bool(split_rows), f"{len(split_rows)} split rows"))
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

    total_storage_gb, storage_rows = storage_summary(paths)
    total_runtime_h, runtime_rows = runtime_summary(paths)
    split_counts = Counter(row["split"] for row in split_rows)
    payload = {
        "passed": all(ok for _name, ok, _detail in gates),
        "ep_success": ep_success,
        "magnetic_success": mag_success,
        "augmentation_success": aug_success,
        "base_sample_count": len(ep_rows),
        "augmented_row_count": len(aug_rows),
        "condition_counts": dict(counts),
        "split_counts": dict(split_counts),
        "storage_gb": total_storage_gb,
        "storage_rows": storage_rows,
        "runtime_hours": total_runtime_h,
        "runtime_rows": runtime_rows,
        "problematic_masks": nonhom_bad[:100],
        "problematic_bz": all_zero_bz[:100],
        "split_leakage": leaked[:100],
    }
    write_json(paths["ready"] / "v1c_5000_gate.json", payload)
    return payload["passed"], payload, gates


def write_report(passed: bool, payload: dict, gates: list[tuple[str, bool, str]]) -> Path:
    gate_table = "\n".join(report_row(name, ok, detail.replace("\n", " ")) for name, ok, detail in gates)
    storage = "\n".join(payload["storage_rows"])
    runtime = "\n".join(payload["runtime_rows"])
    paths = mode_paths(FULL_MODE)
    if passed:
        text = f"""# V1C 5000 Success Report

The full v1c 5,000-sample data-layer run passed the configured quality gates. This report does not implement ML training.

## Final Counts

- EP successful samples: {payload['ep_success']}/{payload['base_sample_count']}
- Magnetic successful samples: {payload['magnetic_success']}/{payload['base_sample_count']}
- Augmented successful variants: {payload['augmentation_success']}/{payload['augmented_row_count']}
- Base sample count: {payload['base_sample_count']}
- Augmented row count: {payload['augmented_row_count']}
- Total storage used: {payload['storage_gb']:.2f} GB
- Runtime: {payload['runtime_hours']:.2f} hours

## Condition Counts

```json
{json.dumps(payload['condition_counts'], indent=2, sort_keys=True)}
```

## Grouped Split Counts

```json
{json.dumps(payload['split_counts'], indent=2, sort_keys=True)}
```

## Gate Table

| Gate | Result | Detail |
|---|---:|---|
{gate_table}

## Storage

{storage}

## Runtime

{runtime}

## Manifests And Ready Tables

- EP manifest: `{(paths['ep'] / 'manifest.csv').relative_to(ROOT)}`
- Magnetic manifest: `{(paths['mag'] / 'manifest.csv').relative_to(ROOT)}`
- Augmented manifest: `{(paths['aug'] / 'manifest.csv').relative_to(ROOT)}`
- Base sample index: `{(paths['ready'] / 'sample_index.csv').relative_to(ROOT)}`
- EP targets: `{(paths['ready'] / 'targets_ep.csv').relative_to(ROOT)}`
- Base deployable fixed features: `{(paths['ready'] / 'features_mig_deployable_fixed.csv').relative_to(ROOT)}`
- Augmented sample index: `{(paths['aug_ready'] / 'sample_index_augmented.csv').relative_to(ROOT)}`
- Augmented deployable fixed features: `{(paths['aug_ready'] / 'features_mig_deployable_fixed_augmented.csv').relative_to(ROOT)}`
- Grouped splits: `{(paths['aug_ready'] / 'splits' / 'grouped_train_val_test.csv').relative_to(ROOT)}`

## Important Figures

- `figures/dataset_v1c_5000/apd_distribution_by_condition.png`
- `figures/dataset_v1c_5000/lat_range_by_condition.png`
- `figures/dataset_v1c_5000/patch_locations.png`
- `figures/dataset_v1c_5000/runtime_storage_summary.png`

## Warnings

- B fields remain synthetic/proxy fields in relative units.
- `J_eq` remains an equivalent-current proxy, not intracellular current reconstruction.
- Mixed samples remain synthetic APD/repolarization proxy plus conduction-slowing perturbations, not clinical ischemia.

## Next Recommended Step

Run classical deployable MIG baselines using grouped splits, starting with logistic/ridge/random forest models.
"""
        SUCCESS_REPORT.write_text(text)
        return SUCCESS_REPORT

    failed = "\n".join(f"- {name}: {detail}" for name, ok, detail in gates if not ok)
    text = f"""# V1C 5000 Diagnostic Report

The full v1c 5,000-sample data-layer run did not pass all configured gates. Do not proceed to ML until the issues below are resolved or explicitly accepted.

## Failed Gates

{failed if failed else '- No failed gate details available; inspect missing reports and logs.'}

## Gate Table

| Gate | Result | Detail |
|---|---:|---|
{gate_table}

## Symptoms

- EP successful samples: {payload['ep_success']}/{payload['base_sample_count']}
- Magnetic successful samples: {payload['magnetic_success']}/{payload['base_sample_count']}
- Augmented successful variants: {payload['augmentation_success']}/{payload['augmented_row_count']}
- Problematic masks: {payload.get('problematic_masks', [])[:20]}
- Problematic Bz samples: {payload.get('problematic_bz', [])[:20]}
- Split leakage: {payload.get('split_leakage', [])[:20]}

## Relevant Log Paths

- Pipeline logs: `{(paths['ep'] / 'pipeline_logs').relative_to(ROOT)}`
- EP sample logs: `{(paths['ep'] / 'samples').relative_to(ROOT)}`
- EP quality report: `{(paths['ep'] / 'dataset_quality_report.json').relative_to(ROOT)}`
- Magnetic quality report: `{(paths['mag'] / 'magnetic_quality_report.json').relative_to(ROOT)}`
- Augmentation quality report: `{(paths['aug'] / 'augmentation_quality_report.json').relative_to(ROOT)}`

## Resume Safety

Resume is safe if failures are missing/partial outputs rather than invalid science settings:

```bash
venv_carputils/bin/python scripts/49_run_v1c_500_pilot_pipeline.py --mode full5000 --workers 8 --compact --seed 20260525 --run-full-5000 --skip-existing
```

## Recommended Fixes

1. Inspect the failed gate details and quality reports.
2. Inspect per-sample openCARP stdout/stderr for failed EP samples.
3. Resume with `--skip-existing` after fixing the root cause.
4. Rerun this gate before any ML work.
"""
    DIAGNOSTIC_REPORT.write_text(text)
    return DIAGNOSTIC_REPORT


def main() -> int:
    passed, payload, gates = gate()
    report = write_report(passed, payload, gates)
    print(json.dumps({"passed": passed, "report": str(report), "storage_gb": payload["storage_gb"], "runtime_hours": payload["runtime_hours"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
