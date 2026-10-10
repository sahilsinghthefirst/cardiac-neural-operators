                      
                                                                   

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import pandas as pd
from torch_import_utils import import_torch_prefer_env

torch = import_torch_prefer_env(PROJECT_ROOT)
from torch.utils.data import DataLoader

from neural_data_common import (
    BzTensorDataset,
    check_no_group_leakage,
    load_neural_index,
    select_deployable_feature_columns,
)


READY_DIR = PROJECT_ROOT / "outputs" / "dataset_v1c_5000_augmented_ready"
OUT_DIR = PROJECT_ROOT / "outputs" / "neural_v1c"
REPORT = OUT_DIR / "bz_loader_check_report.json"
FIXED = READY_DIR / "features_mig_deployable_fixed_augmented.csv"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = load_neural_index()
    leakage = check_no_group_leakage(index)
    fixed = pd.read_csv(FIXED)
    feature_cols = select_deployable_feature_columns(fixed)
    train_ids = set(index.loc[index["split"].eq("train"), "augmented_sample_id"])
    train_features = fixed[fixed["augmented_sample_id"].isin(train_ids)][feature_cols].to_numpy(np.float32)
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0

    split_reports = []
    problems = []
    for split in ["train", "val", "test"]:
        rows = index[index["split"].eq(split)].sample(n=64, random_state=20260525).copy()
        dataset = BzTensorDataset(
            rows=rows,
            feature_frame=fixed,
            feature_columns=feature_cols,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        batch_reports = []
        for batch_i, batch in enumerate(loader):
            x = batch["x_bz"]
            features = batch["features"]
            targets = batch["targets"]
            target_mask = batch["target_mask"]
            ok = bool(torch.isfinite(x).all() and torch.isfinite(features).all() and torch.isfinite(targets).all())
            batch_reports.append(
                {
                    "batch": batch_i,
                    "x_bz_shape": list(x.shape),
                    "features_shape": list(features.shape),
                    "targets_shape": list(targets.shape),
                    "target_mask_shape": list(target_mask.shape),
                    "finite": ok,
                }
            )
            if list(x.shape[1:]) != [301, 8, 8] or not ok:
                problems.append(f"{split} batch {batch_i} shape/finite check failed")
            if batch_i >= 2:
                break
        split_reports.append({"split": split, "sampled_rows": int(len(rows)), "batches": batch_reports})

    dropout_rows = index[index["variant"].isin(["dropout_10pct", "dropout_25pct", "combined_medium"])].sample(
        n=24, random_state=20260526
    )
    dropout_dataset = BzTensorDataset(rows=dropout_rows)
    dropout_reports = []
    for item in dropout_dataset:
        arr = item["x_bz"].numpy()
        meta = item["metadata"]
        dropout_reports.append(
            {
                "augmented_sample_id": meta["augmented_sample_id"],
                "variant": meta["variant"],
                "shape": list(arr.shape),
                "finite": bool(np.isfinite(arr).all()),
                "zero_count": int((arr == 0.0).sum()),
            }
        )
        if not np.isfinite(arr).all() or list(arr.shape) != [301, 8, 8]:
            problems.append(f"dropout handling failed for {meta['augmented_sample_id']}")

    if leakage:
        problems.append(f"group leakage detected: {leakage[:10]}")

    report = {
        "passed": not problems,
        "problems": problems,
        "torch_version": torch.__version__,
        "split_reports": split_reports,
        "dropout_reports": dropout_reports,
        "feature_count": int(len(feature_cols)),
        "grouped_split_policy": "base_sample_id groups preserved",
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps({"report": str(REPORT.relative_to(PROJECT_ROOT)), "passed": report["passed"], "problems": problems}, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
