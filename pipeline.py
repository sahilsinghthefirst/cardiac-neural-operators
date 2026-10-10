import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = {
    'simulate': ('scripts/40_generate_bocf_dataset_v1c.py', '--mode', 'full5000', '--run-full-5000'),
    'select-cases': ('scripts/111_build_v1e_500_combined_pilot_manifest.py',),
    'targets': ('scripts/112_build_v1e_500_parametric_targets.py',),
    'magnetic': ('scripts/113_generate_v1e_500_direct_magnetic.py',),
    'splits': ('scripts/114_build_v1e_500_parametric_ready.py',),
    'field-inputs': ('scripts/118_build_v1e_500_field_ready_dataset.py',),
    'descriptors': ('scripts/119_compute_v1e_500_direct_descriptors.py',),
    'deeponet-inputs': ('scripts/125_build_v1e_500_deeponet_ready.py',),
    'train-fno': ('scripts/121_train_v1e_500_fno_field_parametric_multitask.py',),
    'train-deeponet': ('scripts/127_train_v1e_500_deeponet_field_parametric.py',),
    'evaluate-deeponet': ('scripts/129_evaluate_v1e_500_deeponet.py',),
    'cache-deeponet': ('scripts/132_calibrate_v1e_500_deeponet_masks.py',),
    'refine-targets': ('scripts/135_build_v1e_500_center_mask_targets.py',),
    'train-refinement': ('scripts/137_train_v1e_500_deeponet_mask_refinement.py',),
    'evaluate-refinement': ('scripts/138_evaluate_v1e_500_deeponet_mask_refinement.py',),
    'conduction': ('scripts/145_audit_v1e_conduction_metric_alignment.py',),
    'identifiability': ('scripts/147_audit_v1f_magnetic_identifiability.py',),
    'oracle': ('scripts/148_oracle_decompose_v1f_apd_lat_scalar_recovery.py',),
    'contract': ('scripts/150_build_icbinb_evaluation_contract.py',),
    'cache-fno': ('scripts/151_export_icbinb_fno_full_grid_cache.py',),
    'cohort': ('scripts/152_build_icbinb_paired_test_cohort.py',),
    'bootstrap': ('scripts/153_run_icbinb_module2_paired_bootstrap.py',),
    'decompose': ('scripts/154_build_icbinb_module3_failure_decomposition.py',),
    'generate-heights': ('scripts/155_generate_icbinb_sensor_height_ood.py',),
    'evaluate-heights': ('scripts/156_run_icbinb_module4_sensor_height_ood.py',),
    'controls': ('supplemental/revision_controls.py', '--root', str(ROOT), '--out', str(ROOT / 'outputs/additional_controls'), '--coordinates'),
    'verify': ('supplemental/verify_controls.py', '--canonical', str(ROOT / 'supplemental/paired_test_cohort_maps.npz')),
}
GROUPS = {
    'prepare': ('select-cases', 'targets', 'magnetic', 'splits', 'field-inputs', 'descriptors', 'deeponet-inputs'),
    'train': ('train-fno', 'train-deeponet'),
    'refinement': ('evaluate-deeponet', 'cache-deeponet', 'refine-targets', 'train-refinement', 'evaluate-refinement', 'conduction', 'identifiability', 'oracle'),
    'evaluate': ('contract', 'cache-fno', 'contract', 'cohort', 'contract', 'bootstrap'),
    'height': ('contract', 'generate-heights', 'evaluate-heights', 'contract'),
    'localization': ('decompose', 'contract'),
}
GROUPS['all'] = ('simulate', *GROUPS['prepare'], *GROUPS['train'], *GROUPS['refinement'], *GROUPS['evaluate'], *GROUPS['height'], *GROUPS['localization'], 'controls')


def main():
    parser = argparse.ArgumentParser(
        description='Run the cardiac reconstruction benchmark stages in dependency order.',
        epilog='Use Linux with openCARP for simulation, carputils for simulation and magnetic/height generation, and CUDA PyTorch for training and magnetic projection. Set OPENCARP_MESH_PREFIX if the example mesh is outside the standard installation. Run groups in order: simulate, prepare, train, refinement, evaluate, height, localization, controls. Refinement also creates the DeepONet caches required by evaluate. All runs that sequence, regenerating 5,000 simulations and selecting the 500-case benchmark. The contract enforces the recorded validation winners and refinement threshold; retraining can change these selections and halt evaluation. Evaluation stages require fresh output directories and refuse to overwrite existing results. Verify checks bundled control maps with NumPy and pandas only. Inspect commands with --dry-run. Pass stage options after an individual step name. Historical noisy inputs use Python hash-based seeds and are not guaranteed to reproduce bit for bit.',
    )
    parser.add_argument('stage', choices=[*STEPS, *GROUPS])
    parser.add_argument('--dry-run', action='store_true')
    args, extra = parser.parse_known_args()
    if extra and args.stage in GROUPS:
        parser.error('Extra options can only be passed to an individual step.')
    for stage in GROUPS.get(args.stage, (args.stage,)):
        script, *options = STEPS[stage]
        command = [sys.executable, str(ROOT / script), *options, *extra]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
