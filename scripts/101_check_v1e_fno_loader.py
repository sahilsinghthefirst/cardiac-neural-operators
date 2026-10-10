                      
                                                       

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np

from v1e_direct_common import NEURAL_DIR, require_cuda, write_json
from v1e_direct_fno_data_common import check_no_leakage, compute_stats, load_index, make_loader, torch


REPORT = NEURAL_DIR / "v1e_loader_check_report.json"


def main() -> int:
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    require_cuda(torch)
    rows = load_index()
    leaks = check_no_leakage(rows)
    stats = compute_stats(rows)
    split_reports = {}
    errors = []
    device = torch.device("cuda")
    for split in ["train", "val", "test"]:
        sub = rows[rows["split"].eq(split)].reset_index(drop=True)
        loader = make_loader(sub, stats, batch_size=min(8, max(1, len(sub))), shuffle=False)
        try:
            batch = next(iter(loader))
            x = batch["x"].to(device)
            y = batch["y_fields"].to(device)
            split_reports[split] = {
                "rows": int(len(sub)),
                "x_shape": list(x.shape),
                "y_fields_shape": list(y.shape),
                "y_param_shape": list(batch["y_param"].shape),
                "input_finite": bool(torch.isfinite(x).all().item()),
                "target_finite": bool(torch.isfinite(y).all().item()),
                "param_has_finite_present": bool(torch.isfinite(batch["y_param"][:, 0]).all().item()),
            }
        except Exception as exc:
            errors.append(f"{split}: {exc!r}")
    report = {
        "row_count": int(len(rows)),
        "base_sample_count": int(rows["base_sample_id"].nunique()),
        "split_reports": split_reports,
        "group_leakage": leaks,
        "errors": errors,
        "cuda": {"torch_version": torch.__version__, "torch_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)},
        "leakage_guardrail": "Targets and patch params are labels only; loader returns model inputs from B_meas and coordinate channels.",
        "passed": bool(not leaks and not errors and all(v["input_finite"] and v["target_finite"] for v in split_reports.values())),
    }
    write_json(REPORT, report)
    print(json.dumps({k: report[k] for k in ["row_count", "base_sample_count", "split_reports", "passed"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
