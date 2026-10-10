                      
                                                                    

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from bocf_dataset_common import DATASET_BASE, ROOT, write_csv, write_json


EXPECTED_COUNT = 70


def check_sample(row: dict) -> dict:
    out_dir = ROOT / row["output_dir"]
    result = {
        "sample_id": row["sample_id"],
        "condition": row["condition"],
        "success": row["success"] == "True" or row["success"] is True,
        "passed": True,
        "notes": "",
    }
    problems: list[str] = []
    if not out_dir.exists():
        problems.append("missing output directory")
    meta_file = out_dir / "metadata.json"
    if result["success"] and not meta_file.exists():
        problems.append("missing metadata")
    core = ["vm.igb", "mesh.pts", "mesh.elem", "mesh.lon", "region_mask_nodes.csv", "element_region_tags.csv"]
    for name in core:
        if result["success"] and not (out_dir / name).exists():
            problems.append(f"missing {name}")
    for name in ["LAT.dat", "APD80.dat", "APD90.dat"]:
        path = out_dir / name
        if result["success"] and not path.exists():
            problems.append(f"missing {name}")
        elif path.exists():
            vals = np.loadtxt(path)
            valid = vals[vals >= 0]
            if not np.all(np.isfinite(vals)):
                problems.append(f"{name} has NaN/Inf")
            if valid.size == 0:
                problems.append(f"{name} has no valid values")
    mask_file = out_dir / "region_mask_nodes.csv"
    if mask_file.exists():
        mask = np.genfromtxt(mask_file, delimiter=",", names=True)
        abnormal_sum = int(np.sum(mask["abnormal"])) if mask.size else 0
        if row["condition"] == "homogeneous" and abnormal_sum != 0:
            problems.append("homogeneous sample has nonempty abnormal mask")
        if row["condition"] != "homogeneous" and abnormal_sum == 0:
            problems.append("patch sample has empty abnormal mask")
    result["passed"] = not problems
    result["notes"] = "; ".join(problems)
    return result


def main() -> int:
    manifest = DATASET_BASE / "manifest.csv"
    rows = list(csv.DictReader(manifest.open())) if manifest.exists() else []
    report_rows = [check_sample(row) for row in rows]
    passed = len(rows) == EXPECTED_COUNT and all(r["passed"] for r in report_rows)
    summary = {
        "expected_count": EXPECTED_COUNT,
        "observed_count": len(rows),
        "successful_manifest_rows": sum(1 for r in rows if r.get("success") == "True"),
        "passed_rows": sum(1 for r in report_rows if r["passed"]),
        "passed": passed,
        "notes": "" if passed else "See dataset_quality_report.csv for per-sample issues.",
    }
    write_json(DATASET_BASE / "dataset_quality_report.json", summary)
    write_csv(DATASET_BASE / "dataset_quality_report.csv", report_rows, ["sample_id", "condition", "success", "passed", "notes"])
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

