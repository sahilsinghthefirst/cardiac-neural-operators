                      
                                                                      

from __future__ import annotations

import argparse
import json
import math
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v1e_direct_common import (
    FIG_DIR,
    PILOT_DIR,
    ROOT,
    TARGET_DIR,
    ensure_dirs,
    interpolate_to_grid,
    load_node_map,
    load_region_mask,
    patch_params_from_mask,
    read_pts,
    rel,
    scalar_targets_from_fields,
    write_json,
)


OUT_MANIFEST = PILOT_DIR / "target_manifest.csv"
OUT_REPORT = PILOT_DIR / "target_quality_report.json"


def process_sample(row: pd.Series, grid_size: int) -> dict:
    sample_id = row["base_sample_id"]
    out_dir = TARGET_DIR / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pts = read_pts(ROOT / row["mesh_pts_file"])
    lat = load_node_map(ROOT / row["LAT_file"])
    apd80 = load_node_map(ROOT / row["APD80_file"])
    apd90 = load_node_map(ROOT / row["APD90_file"])
    node_mask = load_region_mask(ROOT / row["region_mask_nodes_file"])
    lat_grid, valid_lat = interpolate_to_grid(pts, lat, grid_size, method="linear")
    apd80_grid, valid80 = interpolate_to_grid(pts, apd80, grid_size, method="linear")
    apd90_grid, valid90 = interpolate_to_grid(pts, apd90, grid_size, method="linear")
    patch_float, patch_valid = interpolate_to_grid(pts, node_mask.astype(np.float32), grid_size, method="nearest")
    patch_grid = (patch_float >= 0.5).astype(np.float32)
    if row["condition"] == "homogeneous":
        patch_grid[:] = 0.0
    valid_grid = (valid_lat & valid80 & valid90 & patch_valid).astype(np.float32)
    np.save(out_dir / "LAT_grid.npy", lat_grid.astype(np.float32))
    np.save(out_dir / "APD80_grid.npy", apd80_grid.astype(np.float32))
    np.save(out_dir / "APD90_grid.npy", apd90_grid.astype(np.float32))
    np.save(out_dir / "patch_mask_grid.npy", patch_grid.astype(np.float32))
    np.save(out_dir / "valid_mask_grid.npy", valid_grid.astype(np.float32))
    params = patch_params_from_mask(patch_grid, valid_grid, row["condition"])
    params.update(
        {
            "base_sample_id": sample_id,
            "condition": row["condition"],
            "condition_label": {
                "homogeneous": 0,
                "apd_repolarization_perturbation": 1,
                "conduction_slowing_perturbation": 2,
                "mixed_apd_conduction_perturbation": 3,
            }[row["condition"]],
            "source_patch_center_x_raw": None if pd.isna(row.get("patch_center_x")) else float(row.get("patch_center_x")),
            "source_patch_center_y_raw": None if pd.isna(row.get("patch_center_y")) else float(row.get("patch_center_y")),
            "source_patch_radius_mm": None if pd.isna(row.get("patch_radius_mm")) else float(row.get("patch_radius_mm")),
            "target_grid_size": grid_size,
            "note": "Patch parameters are supervised labels only and must not be used as model inputs.",
        }
    )
    write_json(out_dir / "patch_param_targets.json", params)
    scalars = scalar_targets_from_fields(lat_grid, apd90_grid, patch_grid, valid_grid.astype(bool))
    meta = {
        "base_sample_id": sample_id,
        "condition": row["condition"],
        "grid_size": grid_size,
        "source_mesh_pts": row["mesh_pts_file"],
        "interpolation": "scipy.interpolate.griddata linear with nearest fill; patch mask nearest",
        "valid_fraction": float(np.mean(valid_grid)),
        "patch_fraction": float(params["patch_area_fraction"]),
        "lat_min": float(np.nanmin(lat_grid)),
        "lat_max": float(np.nanmax(lat_grid)),
        "apd90_min": float(np.nanmin(apd90_grid)),
        "apd90_max": float(np.nanmax(apd90_grid)),
        **scalars,
    }
    write_json(out_dir / "target_metadata.json", meta)
    return {
        "base_sample_id": sample_id,
        "condition": row["condition"],
        "target_dir": rel(out_dir),
        "LAT_grid_file": rel(out_dir / "LAT_grid.npy"),
        "APD80_grid_file": rel(out_dir / "APD80_grid.npy"),
        "APD90_grid_file": rel(out_dir / "APD90_grid.npy"),
        "patch_mask_grid_file": rel(out_dir / "patch_mask_grid.npy"),
        "valid_mask_grid_file": rel(out_dir / "valid_mask_grid.npy"),
        "patch_param_targets_file": rel(out_dir / "patch_param_targets.json"),
        "target_metadata_file": rel(out_dir / "target_metadata.json"),
        "grid_size": grid_size,
        "success": True,
        "notes": "",
        **scalars,
        **{k: params[k] for k in ["patch_present", "patch_center_x_norm", "patch_center_y_norm", "patch_area_fraction", "patch_radius_equiv_norm"]},
    }


def make_figures(rows: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    examples = rows.groupby("condition", group_keys=False).head(1)
    fig, axes = plt.subplots(len(examples), 4, figsize=(12, 3 * len(examples)))
    if len(examples) == 1:
        axes = np.array([axes])
    for axrow, (_, row) in zip(axes, examples.iterrows()):
        arrays = [
            (np.load(ROOT / row["LAT_grid_file"]), "LAT"),
            (np.load(ROOT / row["APD90_grid_file"]), "APD90"),
            (np.load(ROOT / row["APD80_grid_file"]), "APD80"),
            (np.load(ROOT / row["patch_mask_grid_file"]), "Patch"),
        ]
        for ax, (arr, title) in zip(axrow, arrays):
            im = ax.imshow(arr, origin="lower", cmap="viridis" if title != "Patch" else "magma")
            ax.set_title(f"{row['condition']}\n{title}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "target_examples.png", dpi=180)
    plt.close(fig)

    non_hom = rows[~rows["condition"].eq("homogeneous")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].hist(non_hom["patch_area_fraction"].dropna(), bins=20)
    axes[0].set_title("Patch Area Fraction")
    axes[1].hist(non_hom["patch_radius_equiv_norm"].dropna(), bins=20)
    axes[1].set_title("Equivalent Radius")
    axes[2].scatter(non_hom["patch_center_x_norm"], non_hom["patch_center_y_norm"], s=12, alpha=0.7)
    axes[2].set_title("Patch Centers")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "patch_param_distributions.png", dpi=180)
    plt.close(fig)

    patch_examples = rows[~rows["condition"].eq("homogeneous")].groupby("condition", group_keys=False).head(3)
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for ax, (_, row) in zip(axes.ravel(), patch_examples.iterrows()):
        ax.imshow(np.load(ROOT / row["patch_mask_grid_file"]), origin="lower", cmap="magma")
        ax.set_title(row["base_sample_id"], fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(patch_examples) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "patch_mask_examples.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=32, choices=[16, 32, 64])
    args = parser.parse_args()
    ensure_dirs()
    pilot = pd.read_csv(PILOT_DIR / "manifest.csv")
    rows = []
    failures = []
    for _, row in pilot.iterrows():
        try:
            out = process_sample(row, args.grid_size)
            rows.append(out)
            print(f"{out['base_sample_id']}: v1e target ok")
        except Exception as exc:
            failures.append({"base_sample_id": row["base_sample_id"], "condition": row["condition"], "error": repr(exc)})
            print(f"{row['base_sample_id']}: target failed {exc!r}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_MANIFEST, index=False)
    if len(df):
        make_figures(df)
    bad_nonempty = []
    bad_hom = []
    bad_param = []
    apd90_ranges = []
    lat_ranges = []
    for _, row in df.iterrows():
        patch = np.load(ROOT / row["patch_mask_grid_file"]) > 0.5
        valid = np.load(ROOT / row["valid_mask_grid_file"]) > 0.5
        if row["condition"] != "homogeneous" and not np.any(patch & valid):
            bad_nonempty.append(row["base_sample_id"])
        if row["condition"] == "homogeneous" and np.any(patch & valid):
            bad_hom.append(row["base_sample_id"])
        if row["condition"] != "homogeneous" and not np.isfinite(float(row["patch_center_x_norm"])):
            bad_param.append(row["base_sample_id"])
        apd = np.load(ROOT / row["APD90_grid_file"])[valid]
        lat = np.load(ROOT / row["LAT_grid_file"])[valid]
        apd90_ranges.append(float(np.nanmax(apd) - np.nanmin(apd)))
        lat_ranges.append(float(np.nanmax(lat) - np.nanmin(lat)))
    report = {
        "grid_size": args.grid_size,
        "selected_samples": int(len(pilot)),
        "successful_targets": int(len(df)),
        "failed_targets": int(len(failures)),
        "condition_counts": df["condition"].value_counts().to_dict() if len(df) else {},
        "failures": failures,
        "non_homogeneous_empty_patch_masks": bad_nonempty,
        "homogeneous_nonempty_patch_masks": bad_hom,
        "non_homogeneous_bad_param_targets": bad_param,
        "apd90_range_std": float(np.nanstd(apd90_ranges)) if apd90_ranges else math.nan,
        "lat_range_std": float(np.nanstd(lat_ranges)) if lat_ranges else math.nan,
        "patch_area_distribution": df.groupby("condition")["patch_area_fraction"].describe().replace({np.nan: None}).to_dict() if len(df) else {},
        "passed": bool(len(df) == len(pilot) and not failures and not bad_nonempty and not bad_hom and not bad_param and np.nanstd(apd90_ranges) > 0 and np.nanstd(lat_ranges) > 0),
    }
    write_json(OUT_REPORT, report)
    print(json.dumps({k: report[k] for k in ["successful_targets", "failed_targets", "passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
