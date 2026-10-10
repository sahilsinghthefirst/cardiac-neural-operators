                      
                                                           

from __future__ import annotations

import csv
import json
from pathlib import Path

from bocf_common import ROOT


BASE = ROOT / "outputs" / "bocf_data_v0"
MANIFEST = BASE / "manifest.csv"


def rel(path: Path | None) -> str:
    if path and path.exists():
        return str(path.relative_to(ROOT))
    return ""


def row_for(out_dir: Path) -> dict[str, str | bool]:
    meta_file = out_dir / "metadata.json"
    metadata = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    return {
        "sim_id": metadata.get("sim_id", out_dir.name),
        "condition": metadata.get("condition", ""),
        "model": metadata.get("model", ""),
        "formulation": metadata.get("formulation", ""),
        "output_dir": str(out_dir.relative_to(ROOT)),
        "vm_file": rel(out_dir / "vm.igb"),
        "mesh_file": rel(out_dir / "mesh.pts"),
        "lat_file": rel(out_dir / "LAT.dat"),
        "apd_file": rel(out_dir / "APD90.dat") or rel(out_dir / "APD_repol_proxy.dat"),
        "region_mask_file": rel(out_dir / "region_mask_nodes.csv"),
        "metadata_file": rel(meta_file),
        "success": bool(metadata.get("success", False)),
        "notes": "" if metadata.get("success", False) else "Missing or failed metadata/output",
    }


def main() -> int:
    BASE.mkdir(parents=True, exist_ok=True)
    rows = [row_for(path) for path in sorted(BASE.iterdir()) if path.is_dir()]
    fields = [
        "sim_id",
        "condition",
        "model",
        "formulation",
        "output_dir",
        "vm_file",
        "mesh_file",
        "lat_file",
        "apd_file",
        "region_mask_file",
        "metadata_file",
        "success",
        "notes",
    ]
    with MANIFEST.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

