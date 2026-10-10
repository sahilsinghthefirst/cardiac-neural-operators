                      
                                                               

from __future__ import annotations

import csv
import json

import numpy as np

from bocf_dataset_common import ROOT, write_csv, write_json


DATASET_BASE = ROOT / "outputs" / "bocf_dataset_v1b_conditioned"
EXPECTED_COUNT = 70


def truthy(value) -> bool:
    return value is True or str(value).lower() == "true"


def check_sample(row: dict) -> dict:
    out_dir = ROOT / row["output_dir"]
    result = {
        "sample_id": row["sample_id"],
        "condition": row["condition"],
        "success": truthy(row.get("success")),
        "passed": True,
        "notes": "",
    }
    problems: list[str] = []
    if not out_dir.exists():
        problems.append("missing output directory")
    meta_file = out_dir / "metadata.json"
    metadata = {}
    if result["success"] and not meta_file.exists():
        problems.append("missing metadata")
    elif meta_file.exists():
        metadata = json.loads(meta_file.read_text())
    for key in ["nbeats", "BCL", "final_beat_extraction", "conditioning_used"]:
        if result["success"] and key not in metadata:
            problems.append(f"metadata missing {key}")
    core = [
        "vm.igb",
        "mesh.pts",
        "mesh.elem",
        "mesh.lon",
        "LAT.dat",
        "RT_repol_proxy.dat",
        "APD_repol_proxy.dat",
        "APD80.dat",
        "APD90.dat",
        "region_mask_nodes.csv",
        "element_region_tags.csv",
        "metadata.json",
        "opencarp.command.txt",
        "opencarp.stdout",
        "opencarp.stderr",
        "opencarp.returncode.txt",
    ]
    for name in core:
        path = out_dir / name
        if result["success"] and not path.exists():
            problems.append(f"missing {name}")
        elif path.exists() and path.stat().st_size == 0 and name not in {"opencarp.stderr"}:
            problems.append(f"empty {name}")
    for name in ["LAT.dat", "RT_repol_proxy.dat", "APD_repol_proxy.dat", "APD80.dat", "APD90.dat"]:
        path = out_dir / name
        if path.exists():
            vals = np.loadtxt(path)
            if not np.all(np.isfinite(vals)):
                problems.append(f"{name} has NaN/Inf")
            valid = vals[vals >= 0]
            if result["success"] and valid.size == 0:
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
        "successful_manifest_rows": sum(1 for r in rows if truthy(r.get("success"))),
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
