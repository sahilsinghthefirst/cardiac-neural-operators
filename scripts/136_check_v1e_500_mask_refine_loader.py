                      
                                                        

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import pandas as pd

from v1e_500_common import NEURAL_DIR, ROOT, SEED, require_cuda, write_json
from v1e_deeponet_data_common import check_no_leakage, compute_stats, load_descriptors
from v1e_deeponet_mask_refine_data_common import make_mask_refine_loader, merge_rows_with_refine_targets, torch


OUT_JSON = NEURAL_DIR / "mask_refine_loader_check_report.json"


def finite_tensor(x) -> bool:
    return bool(torch.isfinite(x.float()).all().item())


def batch_report(batch, device) -> dict:
    report = {
        "branch_x_shape": list(batch["branch_x"].shape),
        "descriptor_x_shape": list(batch["descriptor_x"].shape),
        "trunk_xy_shape": list(batch["trunk_xy"].shape),
        "y_field_shape": list(batch["y_field"].shape),
        "patch_query_shape": list(batch["patch_query"].shape),
        "center_heatmap_query_shape": list(batch["center_heatmap_query"].shape),
        "distance_query_shape": list(batch["distance_query"].shape),
        "boundary_weight_query_shape": list(batch["boundary_weight_query"].shape),
        "valid_query_mask_shape": list(batch["valid_query_mask"].shape),
        "sampled_all_queries_valid": bool(batch["sampled_all_queries_valid"].all().item()),
    }
    tensors = [
        "branch_x",
        "descriptor_x",
        "trunk_xy",
        "y_field",
        "patch_query",
        "center_heatmap_query",
        "distance_query",
        "boundary_weight_query",
        "valid_query_mask",
        "y_param",
        "raw_area_radius",
    ]
    report["finite"] = {key: finite_tensor(batch[key]) for key in tensors}
    moved = [
        batch["branch_x"].to(device).float(),
        batch["descriptor_x"].to(device).float(),
        batch["trunk_xy"].to(device).float(),
        batch["patch_query"].to(device).float(),
        batch["center_heatmap_query"].to(device).float(),
        batch["distance_query"].to(device).float(),
    ]
    report["cuda_transfer_ok"] = all(x.is_cuda for x in moved)
    report["passed"] = report["cuda_transfer_ok"] and all(report["finite"].values())
    return report


def descriptor_train_only_check(rows: pd.DataFrame, descriptors: pd.DataFrame, stats) -> dict:
    skip = {"row_id", "base_sample_id", "condition", "combined_variant", "measurement_variant", "split"}
    cols = [c for c in descriptors.columns if c not in skip]
    train_ids = set(rows[rows["split"].eq("train")]["row_id"])
    train = descriptors[descriptors["row_id"].isin(train_ids)][cols].to_numpy(np.float32)
    all_vals = descriptors[cols].to_numpy(np.float32)
    train_mean = np.nanmean(train, axis=0)
    all_mean = np.nanmean(all_vals, axis=0)
    forbidden = [c for c in cols if c in {"condition", "condition_label", "row_id", "base_sample_id", "split", "patch_present"}]
    return {
        "descriptor_feature_count": len(cols),
        "descriptor_train_rows": int(train.shape[0]),
        "mean_matches_train_split": bool(np.allclose(train_mean, stats.descriptor_mean, equal_nan=True)),
        "mean_differs_from_all_rows": bool(not np.allclose(all_mean, stats.descriptor_mean, equal_nan=True)),
        "forbidden_descriptor_columns": forbidden,
        "passed": bool(np.allclose(train_mean, stats.descriptor_mean, equal_nan=True) and not forbidden),
    }


def main() -> int:
    require_cuda(torch)
    rows = merge_rows_with_refine_targets()
    desc = load_descriptors()
    stats = compute_stats(rows, desc)
    device = torch.device("cuda")
    errors = []
    leakage = check_no_leakage(rows)
    if leakage:
        errors.append(f"group leakage: {leakage[:5]}")
    split_reports = {}
    for split in ["train", "val", "test"]:
        sub = rows[rows["split"].eq(split)].reset_index(drop=True)
        loader = make_mask_refine_loader(sub, stats, batch_size=4, shuffle=False, mode="random", k_queries=256, descriptors=desc, seed=SEED + 11)
        batch = next(iter(loader))
        split_reports[split] = batch_report(batch, device)
        if not split_reports[split]["sampled_all_queries_valid"]:
            errors.append(f"{split} random queries included invalid tissue points")

    full_loader = make_mask_refine_loader(rows[rows["split"].eq("val")].head(2), stats, batch_size=2, shuffle=False, mode="full", k_queries=1024, descriptors=desc, seed=SEED + 12)
    full_batch = next(iter(full_loader))
    full_report = batch_report(full_batch, device)
    full_report["full_grid_query_count"] = int(full_batch["trunk_xy"].shape[1])
    if full_report["full_grid_query_count"] != 1024:
        errors.append("full-grid loader did not return 1024 query points")
    if not torch.isfinite(full_batch["boundary_weight_query"].float()).all():
        errors.append("non-finite boundary weights in full-grid mode")

    desc_check = descriptor_train_only_check(rows, desc, stats)
    if not desc_check["passed"]:
        errors.append("descriptor train-only scaling check failed")

    payload = {
        "passed": not errors,
        "errors": errors,
        "row_count": int(len(rows)),
        "base_sample_count": int(rows["base_sample_id"].nunique()),
        "split_counts": {k: int(v) for k, v in rows["split"].value_counts().to_dict().items()},
        "group_leakage": leakage,
        "split_reports": split_reports,
        "full_grid_report": full_report,
        "descriptor_train_only_check": desc_check,
        "cuda": {"torch_version": torch.__version__, "torch_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)},
        "input_guardrail": "Loader inputs are B_meas, optional B_meas-derived descriptors, and query coordinates only. Target maps and patch parameters are returned as labels only.",
        "calibration_guardrail": "No calibration is performed in the loader; grouped train/val/test split labels are preserved for downstream validation-only calibration.",
    }
    write_json(OUT_JSON, payload)
    print(json.dumps({"report": str(OUT_JSON), "passed": payload["passed"], "errors": errors}, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
