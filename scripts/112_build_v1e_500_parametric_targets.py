                      
                                                                        

from __future__ import annotations

import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v1e_500_common import (
    CONDITION_LABELS,
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
    lat_grid, valid_lat = interpolate_to_grid(pts, lat, grid_size, "linear")
    apd80_grid, valid_apd80 = interpolate_to_grid(pts, apd80, grid_size, "linear")
    apd90_grid, valid_apd = interpolate_to_grid(pts, apd90, grid_size, "linear")
    patch_float, valid_patch = interpolate_to_grid(pts, node_mask.astype(np.float32), grid_size, "nearest")
    patch = (patch_float >= 0.5).astype(np.float32)
    if row["condition"] == "homogeneous":
        patch[:] = 0.0
    valid = (valid_lat & valid_apd80 & valid_apd & valid_patch).astype(np.float32)
    np.save(out_dir / "LAT_grid.npy", lat_grid.astype(np.float32))
    np.save(out_dir / "APD80_grid.npy", apd80_grid.astype(np.float32))
    np.save(out_dir / "APD90_grid.npy", apd90_grid.astype(np.float32))
    np.save(out_dir / "patch_mask_grid.npy", patch.astype(np.float32))
    np.save(out_dir / "valid_mask_grid.npy", valid.astype(np.float32))
    params = patch_params_from_mask(patch, valid, row["condition"])
    params.update(
        {
            "base_sample_id": sample_id,
            "condition": row["condition"],
            "condition_label": CONDITION_LABELS[row["condition"]],
            "grid_size": grid_size,
            "target_note": "Patch parameters and target grids are supervised labels only, not model inputs.",
        }
    )
    write_json(out_dir / "patch_param_targets.json", params)
    meta = {
        "base_sample_id": sample_id,
        "condition": row["condition"],
        "grid_size": grid_size,
        "valid_fraction": float(np.mean(valid)),
        "patch_fraction": float(params["patch_area_fraction"]),
        "lat_range": float(np.nanmax(lat_grid[valid > 0.5]) - np.nanmin(lat_grid[valid > 0.5])),
        "apd90_range": float(np.nanmax(apd90_grid[valid > 0.5]) - np.nanmin(apd90_grid[valid > 0.5])),
    }
    if np.any((patch > 0.5) & (valid > 0.5)) and np.any((patch <= 0.5) & (valid > 0.5)):
        meta["APD90_patch_delta"] = float(np.mean(apd90_grid[(patch > 0.5) & (valid > 0.5)]) - np.mean(apd90_grid[(patch <= 0.5) & (valid > 0.5)]))
    else:
        meta["APD90_patch_delta"] = math.nan
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
        "LAT_range": meta["lat_range"],
        "APD90_patch_delta": meta["APD90_patch_delta"],
        **{k: params[k] for k in ["patch_present", "center_x_norm", "center_y_norm", "patch_area_fraction", "equivalent_radius_norm", "log_area_fraction", "log_radius", "area_bin", "area_bin_label", "condition_label"]},
    }


def make_figures(df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    non = df[~df["condition"].eq("homogeneous")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].hist(non["patch_area_fraction"], bins=30)
    axes[0].set_title("Patch area")
    axes[1].hist(non["equivalent_radius_norm"], bins=30)
    axes[1].set_title("Equivalent radius")
    axes[2].scatter(non["center_x_norm"], non["center_y_norm"], s=8, alpha=0.6)
    axes[2].set_title("Patch centers")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v1e_500_patch_param_distributions.png", dpi=180)
    plt.close(fig)
    examples = df.groupby("condition", group_keys=False).head(2)
    fig, axes = plt.subplots(len(examples), 3, figsize=(9, 2.6 * len(examples)))
    axes = np.atleast_2d(axes)
    for axrow, (_, row) in zip(axes, examples.iterrows()):
        for ax, col, title in zip(axrow, ["LAT_grid_file", "APD90_grid_file", "patch_mask_grid_file"], ["LAT", "APD90", "Patch"]):
            im = ax.imshow(np.load(ROOT / row[col]), origin="lower", cmap="magma" if title == "Patch" else "viridis")
            ax.set_title(f"{row['condition']} {title}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "v1e_500_target_examples.png", dpi=180)
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
            rows.append(process_sample(row, args.grid_size))
        except Exception as exc:
            failures.append({"base_sample_id": row["base_sample_id"], "condition": row["condition"], "error": repr(exc)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_MANIFEST, index=False)
    if len(df):
        make_figures(df)
    bad_non = df[(~df["condition"].eq("homogeneous")) & (df["patch_present"].ne(1))]["base_sample_id"].tolist() if len(df) else []
    bad_hom = df[(df["condition"].eq("homogeneous")) & (df["patch_present"].ne(0))]["base_sample_id"].tolist() if len(df) else []
    non = df[~df["condition"].eq("homogeneous")]
    report = {
        "grid_size": args.grid_size,
        "selected_samples": int(len(pilot)),
        "successful_targets": int(len(df)),
        "failed_targets": len(failures),
        "failures": failures,
        "condition_counts": df["condition"].value_counts().to_dict() if len(df) else {},
        "bad_non_homogeneous_patch_present": bad_non,
        "bad_homogeneous_patch_present": bad_hom,
        "area_bin_counts": non["area_bin_label"].value_counts().to_dict() if len(non) else {},
        "patch_area_std": float(non["patch_area_fraction"].std()) if len(non) else math.nan,
        "passed": bool(len(df) == len(pilot) and not failures and not bad_non and not bad_hom and len(non) and non["patch_area_fraction"].std() > 0),
    }
    write_json(OUT_REPORT, report)
    print(json.dumps({k: report[k] for k in ["successful_targets", "failed_targets", "area_bin_counts", "passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
