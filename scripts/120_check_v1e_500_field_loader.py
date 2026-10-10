                      
                                                    

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from v1e_500_common import NEURAL_DIR, require_cuda, write_json
from v1e_field_data_common import check_no_leakage, compute_stats, load_descriptors, load_index, make_loader, torch


REPORT = NEURAL_DIR / "field_loader_check_report.json"


def main() -> int:
    require_cuda(torch)
    rows = load_index()
    desc = load_descriptors()
    stats = compute_stats(rows, desc)
    leakage = check_no_leakage(rows)
    split_reports = {}
    errors = []
    for split in ["train", "val", "test"]:
        sub = rows[rows["split"].eq(split)].reset_index(drop=True)
        loader = make_loader(sub.head(8), stats, batch_size=4, shuffle=False, input_norm="per_sample", descriptors=desc)
        for batch in loader:
            x = batch["x"].float()
            d = batch["desc"].float()
            y = batch["y_fields"].float()
            v = batch["valid_mask"].float()
            if x.ndim != 4 or y.shape[1] != 4 or v.ndim != 3:
                errors.append(f"{split}:bad shape x={tuple(x.shape)} y={tuple(y.shape)} valid={tuple(v.shape)}")
            if not torch.isfinite(x).all() or not torch.isfinite(d).all() or not torch.isfinite(y).all():
                errors.append(f"{split}:nonfinite")
            xb = x.cuda(non_blocking=True)
            db = d.cuda(non_blocking=True)
            yb = y.cuda(non_blocking=True)
            split_reports[split] = {
                "batch_x_shape": list(x.shape),
                "batch_desc_shape": list(d.shape),
                "batch_y_shape": list(y.shape),
                "cuda_transfer_ok": bool(xb.is_cuda and db.is_cuda and yb.is_cuda),
            }
            break
    descriptor_train = desc[desc["row_id"].isin(set(rows[rows["split"].eq("train")]["row_id"]))]
    report = {
        "row_count": int(len(rows)),
        "base_sample_count": int(rows["base_sample_id"].nunique()),
        "descriptor_rows": int(len(desc)),
        "descriptor_train_rows": int(len(descriptor_train)),
        "descriptor_feature_count": int(len(stats.descriptor_columns)),
        "split_counts": rows["split"].value_counts().to_dict(),
        "group_leakage": leakage,
        "split_reports": split_reports,
        "errors": errors,
        "cuda": {"torch_version": torch.__version__, "torch_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)},
        "leakage_guardrail": "Loader returns magnetic inputs/descriptors as inputs; target fields and patch params are labels only.",
        "passed": bool(len(rows) == 1500 and len(desc) == 1500 and not leakage and not errors),
    }
    write_json(REPORT, report)
    print(json.dumps({"row_count": report["row_count"], "descriptor_feature_count": report["descriptor_feature_count"], "passed": report["passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
