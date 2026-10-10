                      
                                                                                    

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fno_field_data_common import compute_stats, load_index, make_loader, save_stats, torch
from v1d_field_common import NEURAL_DIR, require_cuda, write_json


OUT = NEURAL_DIR / "fno_field_loader_check_report.json"
STATS = NEURAL_DIR / "fno_field_train_stats.json"


def main() -> int:
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    require_cuda(torch)
    rows = load_index()
    leakage = []
    for base_id, sub in rows.groupby("base_sample_id"):
        if sub["split"].nunique() > 1:
            leakage.append(base_id)
    stats = compute_stats(rows)
    save_stats(STATS, stats)
    split_reports = {}
    cuda_ok = False
    for split in ["train", "val", "test"]:
        sub = rows[rows["split"].eq(split)].reset_index(drop=True)
        loader = make_loader(sub, stats, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        finite_x = bool(torch.isfinite(batch["x"]).all())
        finite_y = bool(torch.isfinite(batch["y"]).all())
        shape = list(batch["x"].shape)
        target_shape = list(batch["y"].shape)
        if split == "train":
            x_cuda = batch["x"].cuda(non_blocking=False)
            y_cuda = batch["y"].cuda(non_blocking=False)
            cuda_ok = bool(x_cuda.is_cuda and y_cuda.is_cuda and torch.isfinite(x_cuda).all() and torch.isfinite(y_cuda).all())
        split_reports[split] = {
            "rows": int(len(sub)),
            "x_shape": shape,
            "y_shape": target_shape,
            "finite_x": finite_x,
            "finite_y": finite_y,
            "valid_mask_shape": list(batch["valid_mask"].shape),
        }
    report = {
        "row_count": int(len(rows)),
        "base_sample_count": int(rows["base_sample_id"].nunique()),
        "leakage": leakage,
        "stats_file": str(STATS),
        "input_mean": stats.input_mean,
        "input_std": stats.input_std,
        "target_mean": stats.target_mean.tolist(),
        "target_std": stats.target_std.tolist(),
        "split_reports": split_reports,
        "cuda_batch_transfer_ok": cuda_ok,
        "passed": bool(not leakage and cuda_ok and all(v["finite_x"] and v["finite_y"] for v in split_reports.values())),
    }
    write_json(OUT, report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
