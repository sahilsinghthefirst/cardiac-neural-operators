                      
                                                                     

from __future__ import annotations

from pathlib import Path

from bocf_common import (
    ROOT,
    base_tissue_command,
    bueno_imp_region_options,
    compute_apd_maps,
    copy_reference_mesh,
    gregion_options,
    make_vm_snapshot,
    run_command,
    write_metadata,
    write_region_masks,
)


OUT_DIR = ROOT / "outputs" / "bocf_data_v0" / "homogeneous_smoke"
FIGURE = ROOT / "figures" / "ep_examples" / "bueno_homogeneous_vm_snapshot.png"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh_files = copy_reference_mesh(OUT_DIR)
    mask_files = write_region_masks(OUT_DIR, abnormal_tags=())
    cmd = base_tissue_command(OUT_DIR) + bueno_imp_region_options() + gregion_options()
    result = run_command([str(x) for x in cmd], OUT_DIR, "opencarp_homogeneous", cwd=ROOT)
    derived = {}
    if result.returncode == 0:
        derived = compute_apd_maps(OUT_DIR)
        make_vm_snapshot(OUT_DIR, FIGURE, "Bueno homogeneous")

    write_metadata(
        OUT_DIR,
        {
            "sim_id": "homogeneous_smoke",
            "condition": "normal homogeneous tissue",
            "model": "Bueno",
            "formulation": "monodomain",
            "geometry": "official openCARP 1 cm x 1 cm tagged 2D sheet",
            "perturbation_type": "none",
            "patch_location": None,
            "patch_size": None,
            "command_used": " ".join(map(str, cmd)),
            "returncode": result.returncode,
            "success": result.returncode == 0,
            "mesh_files": mesh_files,
            "mask_files": mask_files,
            "output_files_generated": derived,
        },
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

