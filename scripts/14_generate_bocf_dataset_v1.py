                      
                                                              

from __future__ import annotations

import argparse
import random

from bocf_dataset_common import (
    DATASET_BASE,
    DEFAULT_APD_PARAMETER,
    ROOT,
    manifest_row,
    read_pts,
    run_ep_sample,
    write_csv,
)


CONDITIONS = [
    ("homogeneous", 10),
    ("apd_repolarization_perturbation", 20),
    ("conduction_slowing_perturbation", 20),
    ("mixed_apd_conduction_perturbation", 20),
]
STIM_LOCS = ["left", "bottom", "right", "top"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--apd-parameter", default=DEFAULT_APD_PARAMETER)
    return parser.parse_args()


def random_patch(rng: random.Random, pts) -> tuple[tuple[float, float], float]:
    radius = rng.uniform(900.0, 2200.0)
    xmin, ymin = pts[:, :2].min(axis=0)
    xmax, ymax = pts[:, :2].max(axis=0)
    margin = radius + 500.0
    cx = rng.uniform(xmin + margin, xmax - margin)
    cy = rng.uniform(ymin + margin, ymax - margin)
    return (cx, cy), radius


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    samples_dir = DATASET_BASE / "samples"
    source_pts = read_pts(ROOT / "outputs" / "bocf_data_v0" / "homogeneous_smoke" / "mesh.pts")
    rows = []
    index = 0
    for condition, count in CONDITIONS:
        for _ in range(count):
            index += 1
            sample_id = f"bocf_v1_{index:04d}"
            stim = rng.choice(STIM_LOCS)
            patch_center = None
            patch_radius = None
            apd_scale = None
            cond_scale = None
            apd_parameter = None
            if condition != "homogeneous":
                patch_center, patch_radius = random_patch(rng, source_pts)
            if condition in {"apd_repolarization_perturbation", "mixed_apd_conduction_perturbation"}:
                apd_parameter = args.apd_parameter
                apd_scale = rng.uniform(1.10, 1.40)
            if condition in {"conduction_slowing_perturbation", "mixed_apd_conduction_perturbation"}:
                cond_scale = rng.uniform(0.35, 0.80)
            metadata = run_ep_sample(
                out_dir=samples_dir / sample_id,
                sample_id=sample_id,
                condition=condition,
                patch_center=patch_center,
                patch_radius=patch_radius,
                apd_parameter=apd_parameter,
                apd_scale=apd_scale,
                conductivity_scale=cond_scale,
                stimulus_location=stim,
                seed=args.seed + index,
                force=args.force,
            )
            row = manifest_row(metadata, samples_dir / sample_id)
            rows.append(row)
            print(f"{sample_id}: {condition} success={metadata.get('success')}")
    manifest = DATASET_BASE / "manifest.csv"
    fields = [
        "sample_id",
        "condition",
        "model",
        "formulation",
        "geometry",
        "apd_parameter",
        "apd_scale",
        "conductivity_scale",
        "patch_center_x",
        "patch_center_y",
        "patch_radius",
        "stimulus_location",
        "success",
        "output_dir",
        "notes",
    ]
    write_csv(manifest, rows, fields)
    print(f"Wrote {manifest}")
    return 0 if all(row["success"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

