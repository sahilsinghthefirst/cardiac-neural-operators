                      
                                                                               

from __future__ import annotations

from bocf_common import (
    ABNORMAL_TAG,
    ROOT,
    base_tissue_command,
    bueno_imp_region_options,
    compute_apd_maps,
    copy_reference_mesh,
    gregion_options,
    make_scalar_map,
    make_vm_snapshot,
    run_command,
    write_metadata,
    write_region_masks,
)


CASES = [
    {
        "sim_id": "apd_patch_smoke",
        "condition": "APD/repolarization heterogeneity region",
        "perturbation_type": "repolarization proxy / APD heterogeneity",
        "imp_params": {ABNORMAL_TAG: "tau_si*1.25"},
        "g_scales": {},
        "parameter_modified": "tau_si*1.25",
        "conductivity_scale": None,
        "vm_figure": ROOT / "figures" / "ep_examples" / "bueno_apd_patch_vm_snapshot.png",
        "map_figure": ROOT / "figures" / "ep_examples" / "bueno_apd_map_patch.png",
        "map_file": "APD90.dat",
        "map_title": "Bueno APD90 map with APD heterogeneity region",
        "map_label": "APD90 (ms)",
    },
    {
        "sim_id": "conduction_slow_patch_smoke",
        "condition": "conduction-slowing region",
        "perturbation_type": "conduction-slowing perturbation",
        "imp_params": {},
        "g_scales": {ABNORMAL_TAG: 0.35},
        "parameter_modified": None,
        "conductivity_scale": 0.35,
        "vm_figure": ROOT / "figures" / "ep_examples" / "bueno_conduction_patch_vm_snapshot.png",
        "map_figure": ROOT / "figures" / "ep_examples" / "bueno_lat_map_conduction_patch.png",
        "map_file": "LAT.dat",
        "map_title": "Bueno LAT map with conduction-slowing region",
        "map_label": "LAT (ms)",
    },
    {
        "sim_id": "ischemia_like_mixed_smoke",
        "condition": "ischemia-like mixed perturbation",
        "perturbation_type": "APD/repolarization proxy + conduction-slowing synthetic benchmark",
        "imp_params": {ABNORMAL_TAG: "tau_si*1.25"},
        "g_scales": {ABNORMAL_TAG: 0.35},
        "parameter_modified": "tau_si*1.25",
        "conductivity_scale": 0.35,
        "vm_figure": ROOT / "figures" / "ep_examples" / "bueno_mixed_vm_snapshot.png",
        "map_figure": ROOT / "figures" / "ep_examples" / "bueno_mixed_apd_map.png",
        "map_file": "APD90.dat",
        "map_title": "Bueno APD90 map with ischemia-like mixed perturbation",
        "map_label": "APD90 (ms)",
        "extra_maps": [
            {
                "map_figure": ROOT / "figures" / "ep_examples" / "bueno_mixed_lat_map.png",
                "map_file": "LAT.dat",
                "map_title": "Bueno LAT map with ischemia-like mixed perturbation",
                "map_label": "LAT (ms)",
            }
        ],
    },
]


def run_case(case: dict) -> int:
    out_dir = ROOT / "outputs" / "bocf_data_v0" / case["sim_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_files = copy_reference_mesh(out_dir)
    mask_files = write_region_masks(out_dir, abnormal_tags=(ABNORMAL_TAG,))
    cmd = (
        base_tissue_command(out_dir)
        + bueno_imp_region_options(case["imp_params"])
        + gregion_options(case["g_scales"])
    )
    result = run_command([str(x) for x in cmd], out_dir, f"opencarp_{case['sim_id']}", cwd=ROOT)
    derived = {}
    if result.returncode == 0:
        derived = compute_apd_maps(out_dir)
        make_vm_snapshot(out_dir, case["vm_figure"], f"Bueno {case['sim_id'].replace('_', ' ')}")
        scalar = out_dir / case["map_file"]
        if scalar.exists():
            make_scalar_map(out_dir, scalar, case["map_figure"], case["map_title"], case["map_label"])
        for map_spec in case.get("extra_maps", []):
            scalar = out_dir / map_spec["map_file"]
            if scalar.exists():
                make_scalar_map(
                    out_dir,
                    scalar,
                    map_spec["map_figure"],
                    map_spec["map_title"],
                    map_spec["map_label"],
                )

    write_metadata(
        out_dir,
        {
            "sim_id": case["sim_id"],
            "condition": case["condition"],
            "model": "Bueno",
            "formulation": "monodomain",
            "geometry": "official openCARP 1 cm x 1 cm tagged 2D sheet",
            "perturbation_type": case["perturbation_type"],
            "patch_location": "top tagged stripe of official sheet mesh, region tag 3",
            "patch_size": "approximately 10 mm x 2.5 mm stripe in 10 mm x 10 mm sheet",
            "parameter_modified": case["parameter_modified"],
            "conductivity_scale": case["conductivity_scale"],
            "command_used": " ".join(map(str, cmd)),
            "returncode": result.returncode,
            "success": result.returncode == 0,
            "mesh_files": mesh_files,
            "mask_files": mask_files,
            "output_files_generated": derived,
            "scientific_note": (
                "Controlled EP perturbation for synthetic benchmarking only. This is not a "
                "clinically validated ischemia simulator and does not claim intracellular "
                "current reconstruction."
            ),
        },
    )
    return result.returncode


def main() -> int:
    codes = [run_case(case) for case in CASES]
    return 0 if all(code == 0 for code in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
