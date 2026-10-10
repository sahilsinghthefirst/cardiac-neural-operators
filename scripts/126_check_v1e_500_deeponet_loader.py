                      
                                                   

from __future__ import annotations

import json

import numpy as np

from v1e_500_common import NEURAL_DIR, ROOT, require_cuda, write_json
from v1e_deeponet_data_common import check_no_leakage, compute_stats, load_descriptors, load_index, make_loader, torch


REPORT = NEURAL_DIR / "deeponet_loader_check_report.json"


def _batch_finite(batch: dict) -> bool:
    keys = ["branch_x", "descriptor_x", "trunk_xy", "y_query", "valid_query_mask", "y_param", "raw_area_radius"]
    for key in keys:
        tensor = batch[key]
        if key == "raw_area_radius":
            finite_or_nan = torch.isfinite(tensor) | torch.isnan(tensor)
            if not finite_or_nan.all():
                return False
        elif not torch.isfinite(tensor).all():
            return False
    return True


def _cuda_transfer_ok(batch: dict) -> bool:
    moved = [
        batch["branch_x"].cuda(non_blocking=True),
        batch["descriptor_x"].cuda(non_blocking=True),
        batch["trunk_xy"].cuda(non_blocking=True),
        batch["y_query"].cuda(non_blocking=True),
        batch["valid_query_mask"].cuda(non_blocking=True),
        batch["y_param"].cuda(non_blocking=True),
    ]
    return bool(all(t.is_cuda for t in moved))


def _descriptor_train_only_check(rows, desc, stats) -> dict:
    skip = {"row_id", "base_sample_id", "condition", "combined_variant", "measurement_variant", "split"}
    cols = [c for c in desc.columns if c not in skip]
    train_ids = set(rows[rows["split"].eq("train")]["row_id"])
    train_values = desc[desc["row_id"].isin(train_ids)][cols].to_numpy(np.float32)
    all_values = desc[cols].to_numpy(np.float32)
    train_mean = np.nanmean(train_values, axis=0)
    all_mean = np.nanmean(all_values, axis=0)
    mean_matches_train = bool(np.allclose(train_mean, stats.descriptor_mean, rtol=1e-5, atol=1e-7))
    mean_differs_from_all = bool(np.max(np.abs(all_mean - stats.descriptor_mean)) > 0.0)
    forbidden = [c for c in cols if any(token in c.lower() for token in ["target", "patch_param", "lat_grid", "apd", "condition_label", "sample_id", "metadata"])]
    return {
        "descriptor_feature_count": len(cols),
        "descriptor_train_rows": int(len(train_values)),
        "mean_matches_train_split": mean_matches_train,
        "mean_differs_from_all_rows": mean_differs_from_all,
        "forbidden_descriptor_columns": forbidden,
        "passed": bool(mean_matches_train and not forbidden),
    }


def main() -> int:
    require_cuda(torch)
    rows = load_index()
    desc = load_descriptors()
    stats = compute_stats(rows, desc)
    leakage = check_no_leakage(rows)
    split_reports = {}
    errors: list[str] = []

    for split in ["train", "val", "test"]:
        sub = rows[rows["split"].eq(split)].reset_index(drop=True)
        loader = make_loader(
            sub.head(8),
            stats,
            batch_size=4,
            shuffle=False,
            mode="random",
            k_queries=256,
            input_norm="per_sample",
            descriptors=desc,
        )
        batch = next(iter(loader))
        if batch["branch_x"].shape[1:] != (4, 301, 32, 32):
            errors.append(f"{split}:bad branch shape {tuple(batch['branch_x'].shape)}")
        if batch["trunk_xy"].shape[1:] != (256, 2):
            errors.append(f"{split}:bad trunk shape {tuple(batch['trunk_xy'].shape)}")
        if batch["y_query"].shape[1:] != (256, 4):
            errors.append(f"{split}:bad y_query shape {tuple(batch['y_query'].shape)}")
        if batch["valid_query_mask"].shape[1:] != (256,):
            errors.append(f"{split}:bad valid mask shape {tuple(batch['valid_query_mask'].shape)}")
        if not _batch_finite(batch):
            errors.append(f"{split}:nonfinite tensor after safe filling/scaling")
        if not bool(batch["sampled_all_queries_valid"].all()):
            errors.append(f"{split}:random query sampling included invalid points")
        cuda_ok = _cuda_transfer_ok(batch)
        split_reports[split] = {
            "branch_x_shape": list(batch["branch_x"].shape),
            "descriptor_x_shape": list(batch["descriptor_x"].shape),
            "trunk_xy_shape": list(batch["trunk_xy"].shape),
            "y_query_shape": list(batch["y_query"].shape),
            "valid_query_mask_shape": list(batch["valid_query_mask"].shape),
            "sampled_all_queries_valid": bool(batch["sampled_all_queries_valid"].all()),
            "cuda_transfer_ok": cuda_ok,
        }

    full_sub = rows[rows["split"].eq("val")].head(2).reset_index(drop=True)
    full_loader = make_loader(
        full_sub,
        stats,
        batch_size=2,
        shuffle=False,
        mode="full",
        k_queries=1024,
        input_norm="per_sample",
        descriptors=desc,
    )
    full_batch = next(iter(full_loader))
    full_ok = (
        full_batch["trunk_xy"].shape[1:] == (1024, 2)
        and full_batch["y_query"].shape[1:] == (1024, 4)
        and full_batch["valid_query_mask"].shape[1:] == (1024,)
        and torch.isfinite(full_batch["branch_x"]).all()
        and torch.isfinite(full_batch["y_query"]).all()
    )
    if not full_ok:
        errors.append("full:grid evaluation mode failed shape/finite checks")

    descriptor_check = _descriptor_train_only_check(rows, desc, stats)
    if not descriptor_check["passed"]:
        errors.append("descriptor scaling or descriptor column guardrail failed")

    report = {
        "row_count": int(len(rows)),
        "base_sample_count": int(rows["base_sample_id"].nunique()),
        "descriptor_rows": int(len(desc)),
        "split_counts": rows["split"].value_counts().to_dict(),
        "group_leakage": leakage,
        "split_reports": split_reports,
        "full_grid_report": {
            "branch_x_shape": list(full_batch["branch_x"].shape),
            "descriptor_x_shape": list(full_batch["descriptor_x"].shape),
            "trunk_xy_shape": list(full_batch["trunk_xy"].shape),
            "y_query_shape": list(full_batch["y_query"].shape),
            "valid_query_mask_shape": list(full_batch["valid_query_mask"].shape),
            "passed": bool(full_ok),
        },
        "descriptor_train_only_check": descriptor_check,
        "errors": errors,
        "cuda": {"torch_version": torch.__version__, "torch_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)},
        "leakage_guardrail": "Loader returns B_meas/descriptors/query coordinates as inputs; target fields and patch params are labels only; metadata is logging only.",
        "passed": bool(len(rows) == 1500 and len(desc) == 1500 and not leakage and not errors),
    }
    write_json(REPORT, report)
    print(json.dumps({"row_count": report["row_count"], "descriptor_rows": report["descriptor_rows"], "passed": report["passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
