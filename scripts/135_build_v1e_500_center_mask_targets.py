from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v1e_500_common import NEURAL_FIG_DIR, PILOT_DIR, ROOT, write_json
from v1e_deeponet_data_common import check_no_leakage


DEEPO_READY = PILOT_DIR / "deeponet_ready"
TARGET_DIR = PILOT_DIR / "deeponet_mask_refine_targets"
MANIFEST_CSV = TARGET_DIR / "target_manifest.csv"
QUALITY_JSON = TARGET_DIR / "target_quality_report.json"


def distance_transform(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(mask)
    except Exception:
        coords = np.argwhere(~mask)
        yy, xx = np.indices(mask.shape)
        out = np.zeros(mask.shape, dtype=np.float32)
        points = np.column_stack([yy.reshape(-1), xx.reshape(-1)])
        if len(coords) == 0:
            return np.zeros(mask.shape, dtype=np.float32)
        for start in range(0, len(points), 512):
            end = min(start + 512, len(points))
            diff = points[start:end, None, :] - coords[None, :, :]
            out.reshape(-1)[start:end] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
        return out


def mask_centroid(patch: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    patch_bool = (patch > 0.5) & valid.astype(bool)
    if not np.any(patch_bool):
        return math.nan, math.nan
    yy, xx = np.indices(patch.shape)
    return float(np.mean(xx[patch_bool])), float(np.mean(yy[patch_bool]))


def build_targets(patch: np.ndarray, valid: np.ndarray, params: dict) -> dict[str, np.ndarray | float]:
    valid_bool = valid.astype(bool)
    patch_bool = (patch > 0.5) & valid_bool
    present = int(params.get("patch_present", 0))
    shape = patch.shape
    diag = math.sqrt(float(shape[0] ** 2 + shape[1] ** 2))
    center_heatmap = np.zeros(shape, dtype=np.float32)
    signed_distance = np.zeros(shape, dtype=np.float32)
    if present and np.any(patch_bool):
        cx, cy = mask_centroid(patch, valid_bool)
        radius_norm = float(params.get("equivalent_radius_norm", 0.0))
        sigma_px = max(1.5, radius_norm * max(shape[0] - 1, shape[1] - 1) * 0.45)
        yy, xx = np.indices(shape)
        center_heatmap = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma_px * sigma_px))).astype(np.float32)
        center_heatmap = np.where(valid_bool, center_heatmap, 0.0).astype(np.float32)

        inside_dist = distance_transform(patch_bool)
        outside_dist = distance_transform(~patch_bool)
        signed_distance = (inside_dist - outside_dist) / max(diag, 1.0)
        signed_distance = np.where(valid_bool, signed_distance, 0.0).astype(np.float32)
    else:
        cx, cy = math.nan, math.nan
        sigma_px = math.nan

    boundary_px = np.abs(signed_distance) * max(diag, 1.0)
    boundary_weight = 1.0 + 4.0 * np.exp(-((boundary_px / 2.5) ** 2))
    if not present:
        boundary_weight = np.ones(shape, dtype=np.float32)
    boundary_weight = np.where(valid_bool, boundary_weight, 0.0).astype(np.float32)
    return {
        "center_heatmap": center_heatmap,
        "signed_distance": signed_distance,
        "boundary_weight": boundary_weight,
        "valid_mask": valid_bool.astype(np.float32),
        "center_x_px": cx,
        "center_y_px": cy,
        "sigma_px": sigma_px,
    }


def load_params(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def make_figures(manifest: pd.DataFrame) -> None:
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    examples = []
    non = manifest[manifest["patch_present"].eq(1)].head(3)
    hom = manifest[manifest["patch_present"].eq(0)].head(1)
    for _, row in pd.concat([non, hom], ignore_index=True).iterrows():
        examples.append(row)
    if not examples:
        return

    fig, axes = plt.subplots(len(examples), 3, figsize=(9, 3 * len(examples)))
    axes = np.atleast_2d(axes)
    for axrow, row in zip(axes, examples):
        patch = np.load(ROOT / row["patch_mask_grid_file"])
        heat = np.load(ROOT / row["center_heatmap_file"])
        valid = np.load(ROOT / row["valid_mask_file"])
        for ax, arr, title, cmap in [
            (axrow[0], patch, "patch", "magma"),
            (axrow[1], heat, "center heatmap", "magma"),
            (axrow[2], valid, "valid mask", "gray"),
        ]:
            im = ax.imshow(arr, origin="lower", cmap=cmap)
            ax.set_title(f"{title}\n{row['condition']}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "mask_refine_center_heatmap_examples.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(examples), 3, figsize=(9, 3 * len(examples)))
    axes = np.atleast_2d(axes)
    for axrow, row in zip(axes, examples):
        patch = np.load(ROOT / row["patch_mask_grid_file"])
        dist = np.load(ROOT / row["signed_distance_target_file"])
        bw = np.load(ROOT / row["boundary_weight_map_file"])
        for ax, arr, title, cmap in [
            (axrow[0], patch, "patch", "magma"),
            (axrow[1], dist, "signed distance", "coolwarm"),
            (axrow[2], bw, "boundary weight", "viridis"),
        ]:
            im = ax.imshow(arr, origin="lower", cmap=cmap)
            ax.set_title(f"{title}\n{row['condition']}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "mask_refine_distance_target_examples.png", dpi=180)
    plt.close(fig)


def main() -> int:
    rows = pd.read_csv(DEEPO_READY / "sample_index.csv")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    target_by_base: dict[str, dict] = {}
    shape_errors = []
    finite_errors = []
    heatmap_errors = []

    for base_id, sub in rows.groupby("base_sample_id", sort=True):
        first = sub.iloc[0]
        base_dir = TARGET_DIR / base_id
        base_dir.mkdir(parents=True, exist_ok=True)
        patch = np.load(ROOT / first["patch_mask_grid_file"]).astype(np.float32)
        valid = np.load(ROOT / first["valid_mask_grid_file"]).astype(bool)
        params = load_params(first["patch_param_targets_file"])
        targets = build_targets(patch, valid, params)
        files = {
            "center_heatmap_file": base_dir / "center_heatmap.npy",
            "signed_distance_target_file": base_dir / "signed_distance_target.npy",
            "boundary_weight_map_file": base_dir / "boundary_weight_map.npy",
            "valid_mask_file": base_dir / "valid_mask.npy",
        }
        np.save(files["center_heatmap_file"], targets["center_heatmap"])
        np.save(files["signed_distance_target_file"], targets["signed_distance"])
        np.save(files["boundary_weight_map_file"], targets["boundary_weight"])
        np.save(files["valid_mask_file"], targets["valid_mask"])
        for key, path in files.items():
            files[key] = str(path.relative_to(ROOT))
        target_by_base[base_id] = {
            **files,
            "center_x_px": targets["center_x_px"],
            "center_y_px": targets["center_y_px"],
            "center_sigma_px": targets["sigma_px"],
        }
        expected = (32, 32)
        for name in ["center_heatmap", "signed_distance", "boundary_weight", "valid_mask"]:
            arr = np.asarray(targets[name])
            if arr.shape != expected:
                shape_errors.append({"base_sample_id": base_id, "target": name, "shape": list(arr.shape)})
            if not np.all(np.isfinite(arr)):
                finite_errors.append({"base_sample_id": base_id, "target": name})
        present = int(params.get("patch_present", 0))
        heat_sum = float(np.sum(targets["center_heatmap"]))
        if present and heat_sum <= 0:
            heatmap_errors.append({"base_sample_id": base_id, "reason": "non_homogeneous_zero_heatmap"})
        if not present and heat_sum != 0:
            heatmap_errors.append({"base_sample_id": base_id, "reason": "homogeneous_positive_heatmap", "heat_sum": heat_sum})

    manifest_rows = []
    for _, row in rows.iterrows():
        target = target_by_base[row["base_sample_id"]]
        manifest_rows.append(
            {
                "row_id": row["row_id"],
                "base_sample_id": row["base_sample_id"],
                "split": row["split"],
                "condition": row["condition"],
                "combined_variant": row["combined_variant"],
                "patch_present": int(row["patch_present"]),
                "patch_mask_grid_file": row["patch_mask_grid_file"],
                "valid_mask_grid_file": row["valid_mask_grid_file"],
                "patch_param_targets_file": row["patch_param_targets_file"],
                **target,
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(MANIFEST_CSV, index=False)
    make_figures(manifest)

    leakage = check_no_leakage(rows)
    quality = {
        "target_manifest": str(MANIFEST_CSV.relative_to(ROOT)),
        "row_count": int(len(manifest)),
        "base_sample_count": int(rows["base_sample_id"].nunique()),
        "unique_target_count": int(len(target_by_base)),
        "split_counts_rows": {k: int(v) for k, v in manifest["split"].value_counts().to_dict().items()},
        "split_counts_base": {k: int(v) for k, v in rows.drop_duplicates("base_sample_id")["split"].value_counts().to_dict().items()},
        "group_leakage": leakage,
        "shape_errors": shape_errors,
        "finite_errors": finite_errors,
        "heatmap_errors": heatmap_errors,
        "passed": not leakage and not shape_errors and not finite_errors and not heatmap_errors and len(manifest) == len(rows),
        "leakage_guardrail": "Targets are labels only. Existing grouped splits by base_sample_id are preserved; no new split assignment is introduced.",
        "figures": [
            "figures/neural_v1e_500/mask_refine_center_heatmap_examples.png",
            "figures/neural_v1e_500/mask_refine_distance_target_examples.png",
        ],
    }
    write_json(QUALITY_JSON, quality)
    print(json.dumps({"manifest": str(MANIFEST_CSV), "quality": str(QUALITY_JSON), "passed": quality["passed"]}, indent=2))
    return 0 if quality["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
