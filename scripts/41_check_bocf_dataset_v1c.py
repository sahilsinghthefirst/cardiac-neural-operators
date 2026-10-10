                      
                                                          

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from bocf_v1c_common import (
    CONDITIONS,
    EP_REQUIRED_FILES,
    FULL_MODE,
    PILOT_MODE,
    expected_composition,
    load_node_mask,
    mode_paths,
    read_csv_rows,
    recursive_size_bytes,
    rel,
    truthy,
    valid_values,
    write_csv_rows,
    write_json,
)


def finite_map(path: Path) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size == 0:
        return False, "missing_or_empty"
    try:
        arr = np.loadtxt(path)
        vals = valid_values(np.asarray(arr))
        return bool(vals.size and np.all(np.isfinite(vals))), ""
    except Exception as exc:
        return False, repr(exc)


def check_row(row: dict) -> dict:
    out_dir = Path(__file__).resolve().parents[1] / row["output_dir"]
    problems: list[str] = []
    if not truthy(row.get("success")):
        problems.append("manifest_row_unsuccessful")
    for name in EP_REQUIRED_FILES:
        path = out_dir / name
        if not path.exists() or path.stat().st_size == 0:
            problems.append(f"missing_or_empty:{name}")
    for name in ["APD80.dat", "APD90.dat", "LAT.dat"]:
        ok, msg = finite_map(out_dir / name)
        if not ok:
            problems.append(f"nonfinite_or_missing:{name}:{msg}")
    try:
        mask = load_node_mask(out_dir / "region_mask_nodes.csv")
        has_patch = bool(np.any(mask))
        if row["condition"] == "homogeneous" and has_patch:
            problems.append("homogeneous_mask_not_empty")
        if row["condition"] != "homogeneous" and not has_patch:
            problems.append("nonhomogeneous_mask_empty")
    except Exception as exc:
        problems.append(f"mask_load_failed:{exc!r}")
    try:
        if row["condition"] in {"apd_repolarization_perturbation", "mixed_apd_conduction_perturbation"}:
            scale = float(row["apd_scale"])
            if not (1.03 <= scale <= 1.45):
                problems.append("apd_scale_out_of_range")
        if row["condition"] in {"conduction_slowing_perturbation", "mixed_apd_conduction_perturbation"}:
            scale = float(row["conductivity_scale"])
            if not (0.30 <= scale <= 0.90):
                problems.append("conductivity_scale_out_of_range")
        if row["condition"] != "homogeneous":
            radius = float(row["patch_radius_mm"])
            if not (0.7 <= radius <= 2.5):
                problems.append("patch_radius_out_of_range")
        aniso = float(row["anisotropy_factor"])
        if not (3.0 <= aniso <= 5.0):
            problems.append("anisotropy_factor_out_of_range")
        gil = float(row["base_gil"])
        if not (0.85 <= gil <= 1.15):
            problems.append("base_gil_out_of_range")
    except Exception as exc:
        problems.append(f"metadata_range_check_failed:{exc!r}")
    return {
        "sample_id": row["sample_id"],
        "condition": row["condition"],
        "success": not problems,
        "problem_count": len(problems),
        "problems": ";".join(problems),
        "output_dir": row["output_dir"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[PILOT_MODE, FULL_MODE], default=PILOT_MODE)
    parser.add_argument("--expected-count", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = mode_paths(args.mode)
    ep_base = paths["ep"]
    manifest = ep_base / "manifest.csv"
    rows = read_csv_rows(manifest)
    report_rows = [check_row(row) for row in rows]
    counts = Counter(row["condition"] for row in rows if truthy(row.get("success")))
    expected_count = args.expected_count or sum(expected_composition(args.mode).values())
    expected_counts = expected_composition(args.mode, expected_count)
    duplicate_ids = [sid for sid, count in Counter(row["sample_id"] for row in rows).items() if count > 1]
    rounded_patch = Counter(
        (
            row["condition"],
            row.get("patch_center_x", "")[:6],
            row.get("patch_center_y", "")[:6],
            row.get("patch_radius_mm", "")[:5],
        )
        for row in rows
        if row["condition"] != "homogeneous"
    )
    duplicate_patch_metadata = sum(1 for c in rounded_patch.values() if c > 1)
    passed_rows = sum(1 for r in report_rows if r["success"])
    problems = []
    if len(rows) != expected_count:
        problems.append(f"row_count {len(rows)} != expected {expected_count}")
    for condition in CONDITIONS:
        if counts.get(condition, 0) != expected_counts.get(condition, 0):
            problems.append(f"{condition} count {counts.get(condition, 0)} != expected {expected_counts.get(condition, 0)}")
    if duplicate_ids:
        problems.append(f"duplicate sample IDs: {duplicate_ids[:10]}")
    hidden_failures = [r["sample_id"] for r in rows if not truthy(r.get("success"))]
    if hidden_failures:
        problems.append(f"failed manifest rows: {hidden_failures[:20]}")
    failed_checks = [r for r in report_rows if not r["success"]]
    if failed_checks:
        problems.append(f"{len(failed_checks)} samples failed file/map/mask checks")
    payload = {
        "mode": args.mode,
        "manifest": rel(manifest),
        "expected_count": expected_count,
        "row_count": len(rows),
        "passed_sample_checks": passed_rows,
        "failed_sample_checks": len(report_rows) - passed_rows,
        "expected_condition_counts": expected_counts,
        "successful_condition_counts": dict(counts),
        "duplicate_sample_ids": duplicate_ids,
        "duplicate_patch_metadata_count_rounded": duplicate_patch_metadata,
        "storage_bytes": recursive_size_bytes(ep_base),
        "storage_gb": recursive_size_bytes(ep_base) / 1e9,
        "passed": not problems,
        "problems": problems,
    }
    write_csv_rows(ep_base / "dataset_quality_report.csv", report_rows)
    write_json(ep_base / "dataset_quality_report.json", payload)
    print(json.dumps({k: payload[k] for k in ["passed", "row_count", "passed_sample_checks", "problems"]}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
