                      
                                                    

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from bocf_common import ROOT, read_pts


BASE = ROOT / "outputs" / "bocf_data_v0"
EXPECTED = [
    "homogeneous_smoke",
    "apd_patch_smoke",
    "conduction_slow_patch_smoke",
    "ischemia_like_mixed_smoke",
]


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def check_node_file(path: Path, node_count: int, errors: list[str]) -> None:
    if not path.exists():
        return
    require(path.stat().st_size > 0, f"{path} is empty", errors)
    values = np.loadtxt(path)
    require(values.shape[0] == node_count, f"{path} has {values.shape[0]} rows, expected {node_count}", errors)


def main() -> int:
    errors: list[str] = []
    for name in EXPECTED:
        out_dir = BASE / name
        require(out_dir.exists(), f"Missing expected output directory {out_dir}", errors)
        if not out_dir.exists():
            continue
        metadata_file = out_dir / "metadata.json"
        require(metadata_file.exists(), f"Missing metadata.json in {out_dir}", errors)
        metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
        require(bool(metadata.get("success")), f"{out_dir} metadata does not report success", errors)
        vm_file = out_dir / "vm.igb"
        require(vm_file.exists() and vm_file.stat().st_size > 0, f"Missing/empty Vm output {vm_file}", errors)
        mesh_file = out_dir / "mesh.pts"
        require(mesh_file.exists() and mesh_file.stat().st_size > 0, f"Missing/empty mesh file {mesh_file}", errors)
        if mesh_file.exists():
            node_count = len(read_pts(mesh_file))
            for fname in ["LAT.dat", "RT_repol_proxy.dat", "APD_repol_proxy.dat", "APD80.dat", "APD90.dat"]:
                check_node_file(out_dir / fname, node_count, errors)
        for path in out_dir.iterdir():
            if path.is_file():
                require(path.stat().st_size > 0 or path.name == "petsc_err_log.txt", f"Empty file {path}", errors)

    manifest = BASE / "manifest.csv"
    require(manifest.exists() and manifest.stat().st_size > 0, f"Missing/empty manifest {manifest}", errors)
    if manifest.exists():
        with manifest.open() as f:
            rows = list(csv.DictReader(f))
        ids = {row["sim_id"] for row in rows}
        for name in EXPECTED:
            require(name in ids, f"Manifest missing {name}", errors)

    old_ttp = ROOT / "outputs" / "bench_ttp_flags"
    require(old_ttp.exists(), "Old tenTusscher output directory outputs/bench_ttp_flags is missing", errors)
    if old_ttp.exists():
        require(any(old_ttp.iterdir()), "Old tenTusscher output directory outputs/bench_ttp_flags is empty", errors)

    if errors:
        print("BOCF output checks failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("BOCF output checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
