                      
                                                                              

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v1d_field_common import (
    FIELD_TARGET_DIR,
    FIG_DIR,
    PILOT_DIR,
    ROOT,
    ensure_dirs,
    interpolate_to_grid,
    load_node_map,
    load_region_mask,
    read_pts,
    rel,
    scalar_targets_from_fields,
    write_json,
)


OUT_MANIFEST = PILOT_DIR / "field_target_manifest.csv"
OUT_REPORT = PILOT_DIR / "field_target_quality_report.json"


def process_sample(row: pd.Series, grid_size: int) -> dict:
    sample_id = row["base_sample_id"]
    out_dir = FIELD_TARGET_DIR / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pts = read_pts(ROOT / row["mesh_pts_file"])
    lat = load_node_map(ROOT / row["LAT_file"])
    apd80 = load_node_map(ROOT / row["APD80_file"])
    apd90 = load_node_map(ROOT / row["APD90_file"])
    rt = load_node_map(ROOT / row["RT_repol_proxy_file"]) if Path(ROOT / row["RT_repol_proxy_file"]).exists() else None
    node_mask = load_region_mask(ROOT / row["region_mask_nodes_file"])
    lat_grid, valid_lat = interpolate_to_grid(pts, lat, grid_size, method="linear")
    apd80_grid, valid80 = interpolate_to_grid(pts, apd80, grid_size, method="linear")
    apd90_grid, valid90 = interpolate_to_grid(pts, apd90, grid_size, method="linear")
    rt_grid = None
    valid_rt = np.ones_like(valid_lat, dtype=bool)
    if rt is not None:
        rt_grid, valid_rt = interpolate_to_grid(pts, rt, grid_size, method="linear")
    patch_float, patch_valid = interpolate_to_grid(pts, node_mask.astype(np.float32), grid_size, method="nearest")
    patch_grid = (patch_float >= 0.5).astype(np.float32)
    if row["condition"] == "homogeneous":
        patch_grid[:] = 0.0
    valid_grid = (valid_lat & valid80 & valid90 & valid_rt & patch_valid).astype(np.float32)
    np.save(out_dir / "LAT_grid.npy", lat_grid)
    np.save(out_dir / "APD80_grid.npy", apd80_grid)
    np.save(out_dir / "APD90_grid.npy", apd90_grid)
    if rt_grid is not None:
        np.save(out_dir / "RT_grid.npy", rt_grid)
    np.save(out_dir / "patch_mask_grid.npy", patch_grid.astype(np.float32))
    np.save(out_dir / "valid_mask_grid.npy", valid_grid.astype(np.float32))
    scalars = scalar_targets_from_fields(lat_grid, apd90_grid, patch_grid, valid_grid.astype(bool))
    meta = {
        "base_sample_id": sample_id,
        "condition": row["condition"],
        "grid_size": grid_size,
        "source_mesh_pts": row["mesh_pts_file"],
        "interpolation": "scipy.interpolate.griddata linear with nearest fill; patch mask nearest",
        "valid_fraction": float(np.mean(valid_grid)),
        "patch_fraction": float(np.mean((patch_grid > 0.5) & (valid_grid > 0.5))),
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
        "RT_grid_file": rel(out_dir / "RT_grid.npy") if rt_grid is not None else "",
        "patch_mask_grid_file": rel(out_dir / "patch_mask_grid.npy"),
        "valid_mask_grid_file": rel(out_dir / "valid_mask_grid.npy"),
        "target_metadata_file": rel(out_dir / "target_metadata.json"),
        "grid_size": grid_size,
        "success": True,
        "notes": "",
        **scalars,
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
    fig.savefig(FIG_DIR / "example_field_targets.png", dpi=180)
    plt.close(fig)

    vals_apd = []
    vals_lat = []
    for _, row in rows.iterrows():
        valid = np.load(ROOT / row["valid_mask_grid_file"]).astype(bool)
        vals_apd.extend(np.load(ROOT / row["APD90_grid_file"])[valid].ravel().tolist())
        vals_lat.extend(np.load(ROOT / row["LAT_grid_file"])[valid].ravel().tolist())
    for vals, fname, title in [
        (vals_apd, "apd90_grid_distribution.png", "APD90 Grid Distribution"),
        (vals_lat, "lat_grid_distribution.png", "LAT Grid Distribution"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(vals, bins=60, color="#4c78a8", alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("ms")
        ax.set_ylabel("grid cells")
        fig.tight_layout()
        fig.savefig(FIG_DIR / fname, dpi=180)
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
    parser.add_argument("--grid-size", type=int, default=32)
    args = parser.parse_args()
    ensure_dirs()
    pilot = pd.read_csv(PILOT_DIR / "manifest.csv")
    rows = []
    failures = []
    for _, row in pilot.iterrows():
        try:
            out = process_sample(row, args.grid_size)
            rows.append(out)
            print(f"{out['base_sample_id']}: target ok")
        except Exception as exc:
            failures.append({"base_sample_id": row["base_sample_id"], "condition": row["condition"], "error": repr(exc)})
            print(f"{row['base_sample_id']}: target failed {exc!r}")
    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_MANIFEST, index=False)
    if len(out_df):
        make_figures(out_df)
    non_hom = out_df[~out_df["condition"].eq("homogeneous")]
    hom = out_df[out_df["condition"].eq("homogeneous")]
    bad_nonempty = []
    bad_hom = []
    apd90_ranges = []
    lat_ranges = []
    for _, row in out_df.iterrows():
        patch = np.load(ROOT / row["patch_mask_grid_file"]) > 0.5
        valid = np.load(ROOT / row["valid_mask_grid_file"]) > 0.5
        if row["condition"] != "homogeneous" and not np.any(patch & valid):
            bad_nonempty.append(row["base_sample_id"])
        if row["condition"] == "homogeneous" and np.any(patch & valid):
            bad_hom.append(row["base_sample_id"])
        apd = np.load(ROOT / row["APD90_grid_file"])[valid]
        lat = np.load(ROOT / row["LAT_grid_file"])[valid]
        apd90_ranges.append(float(np.nanmax(apd) - np.nanmin(apd)))
        lat_ranges.append(float(np.nanmax(lat) - np.nanmin(lat)))
    report = {
        "grid_size": args.grid_size,
        "selected_samples": int(len(pilot)),
        "successful_targets": int(len(out_df)),
        "failed_targets": int(len(failures)),
        "condition_counts": out_df["condition"].value_counts().to_dict() if len(out_df) else {},
        "failures": failures,
        "non_homogeneous_empty_patch_masks": bad_nonempty,
        "homogeneous_nonempty_patch_masks": bad_hom,
        "apd90_range_mean": float(np.mean(apd90_ranges)) if apd90_ranges else math.nan,
        "lat_range_mean": float(np.mean(lat_ranges)) if lat_ranges else math.nan,
        "passed": bool(len(out_df) >= 0.98 * len(pilot) and not bad_nonempty and not bad_hom and np.nanstd(apd90_ranges) > 0 and np.nanstd(lat_ranges) > 0),
        "figure_dir": rel(FIG_DIR),
    }
    write_json(OUT_REPORT, report)
    print(json.dumps({k: report[k] for k in ["successful_targets", "failed_targets", "passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
