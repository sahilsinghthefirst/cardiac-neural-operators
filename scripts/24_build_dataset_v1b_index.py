                      
                                                                              

                                                                             
                                      
   

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EP_BASE = ROOT / "outputs" / "bocf_dataset_v1b_conditioned"
MAG_BASE = ROOT / "outputs" / "magnetic_dataset_v1b"
READY_BASE = ROOT / "outputs" / "dataset_v1b_ready"


FIELDS = [
    "sample_id",
    "condition",
    "ep_output_dir",
    "magnetic_output_dir",
    "vm_file",
    "lat_file",
    "rt_file",
    "apd80_file",
    "apd90_file",
    "mask_file",
    "Bz_file",
    "B_vec_file",
    "mig_curves_file",
    "mig_summary_file",
    "apd_parameter",
    "apd_scale",
    "conductivity_scale",
    "patch_center_x",
    "patch_center_y",
    "patch_radius",
    "nbeats",
    "BCL",
    "conditioning_used",
    "ep_success",
    "magnetic_success",
    "success",
    "notes",
]


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def truthy(value) -> bool:
    return value is True or str(value).lower() == "true"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict]) -> None:
    payload = {
        "created_utc": now_utc(),
        "source_ep_manifest": rel(EP_BASE / "manifest.csv"),
        "source_magnetic_manifest": rel(MAG_BASE / "manifest.csv"),
        "row_count": len(rows),
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    ep_rows = read_csv(EP_BASE / "manifest.csv")
    mag_rows = {row["sample_id"]: row for row in read_csv(MAG_BASE / "manifest.csv")}
    rows: list[dict] = []
    for ep_row in ep_rows:
        sample_id = ep_row["sample_id"]
        ep_dir = ROOT / ep_row["output_dir"]
        mag_row = mag_rows.get(sample_id, {})
        mag_dir = ROOT / mag_row.get("magnetic_output_dir", f"outputs/magnetic_dataset_v1b/{sample_id}")
        ep_meta = read_json(ep_dir / "metadata.json")
        mag_meta = read_json(mag_dir / "magnetic_metadata.json")
        ep_success = truthy(ep_row.get("success")) and bool(ep_meta.get("success", True))
        magnetic_success = truthy(mag_row.get("success")) and bool(mag_meta.get("success", True))
        notes = "; ".join(x for x in [ep_row.get("notes", ""), mag_row.get("notes", "")] if x)
        rows.append(
            {
                "sample_id": sample_id,
                "condition": ep_row.get("condition", ep_meta.get("condition", "")),
                "ep_output_dir": rel(ep_dir),
                "magnetic_output_dir": rel(mag_dir),
                "vm_file": rel(ep_dir / "vm.igb"),
                "lat_file": rel(ep_dir / "LAT.dat"),
                "rt_file": rel(ep_dir / "RT_repol_proxy.dat") if (ep_dir / "RT_repol_proxy.dat").exists() else "",
                "apd80_file": rel(ep_dir / "APD80.dat"),
                "apd90_file": rel(ep_dir / "APD90.dat"),
                "mask_file": rel(ep_dir / "region_mask_nodes.csv"),
                "Bz_file": mag_row.get("Bz_file", rel(mag_dir / "Bz.npy")),
                "B_vec_file": mag_row.get("B_vec_file", rel(mag_dir / "B_vec.npy")),
                "mig_curves_file": mag_row.get("mig_curves_file", rel(mag_dir / "mig_curves.csv")),
                "mig_summary_file": mag_row.get("mig_summary_file", rel(mag_dir / "mig_summary.json")),
                "apd_parameter": ep_meta.get("apd_parameter"),
                "apd_scale": ep_meta.get("apd_scale"),
                "conductivity_scale": ep_meta.get("conductivity_scale"),
                "patch_center_x": ep_meta.get("patch_center_x"),
                "patch_center_y": ep_meta.get("patch_center_y"),
                "patch_radius": ep_meta.get("patch_radius"),
                "nbeats": ep_meta.get("nbeats", ep_row.get("nbeats")),
                "BCL": ep_meta.get("BCL", ep_row.get("BCL")),
                "conditioning_used": ep_meta.get("conditioning_used"),
                "ep_success": ep_success,
                "magnetic_success": magnetic_success,
                "success": ep_success and magnetic_success,
                "notes": notes,
            }
        )
    write_csv(READY_BASE / "sample_index.csv", rows)
    write_json(READY_BASE / "sample_index.json", rows)
    print(f"Wrote {READY_BASE / 'sample_index.csv'} with {len(rows)} rows")
    return 0 if len(rows) == 70 and all(truthy(r["success"]) for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
