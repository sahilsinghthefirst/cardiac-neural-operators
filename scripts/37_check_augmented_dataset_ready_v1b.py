                      
                                                          

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUG_READY = ROOT / "outputs" / "dataset_v1b_augmented_ready"
EXPECTED_ROWS = 70 * 11


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def numeric_keys(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    return [k for k in rows[0] if k not in {"augmented_sample_id", "base_sample_id", "variant", "condition"}]


def main() -> int:
    problems: list[str] = []
    index_rows = read_rows(AUG_READY / "sample_index_augmented.csv")
    target_rows = read_rows(AUG_READY / "targets_ep_augmented.csv")
    fixed_rows = read_rows(AUG_READY / "features_mig_deployable_fixed_augmented.csv")
    magnetic_rows = read_rows(AUG_READY / "features_mig_deployable_magnetic_augmented.csv")
    split_rows = read_rows(AUG_READY / "splits" / "grouped_train_val_test.csv")
    split_json = AUG_READY / "splits" / "grouped_train_val_test.json"

    for name, rows in [
        ("sample_index_augmented", index_rows),
        ("targets_ep_augmented", target_rows),
        ("features_mig_deployable_fixed_augmented", fixed_rows),
        ("features_mig_deployable_magnetic_augmented", magnetic_rows),
    ]:
        if len(rows) != EXPECTED_ROWS:
            problems.append(f"{name} has {len(rows)} rows, expected {EXPECTED_ROWS}")

    ids = [
        {r["augmented_sample_id"] for r in index_rows},
        {r["augmented_sample_id"] for r in target_rows},
        {r["augmented_sample_id"] for r in fixed_rows},
        {r["augmented_sample_id"] for r in magnetic_rows},
    ]
    sample_ids_align = bool(ids) and all(s == ids[0] for s in ids)
    if not sample_ids_align:
        problems.append("augmented sample IDs do not align")

    if not split_json.exists() or not split_rows:
        problems.append("grouped splits missing")
    else:
        split_by_base: dict[str, set[str]] = {}
        for row in split_rows:
            split_by_base.setdefault(row["base_sample_id"], set()).add(row["split"])
        leaking = sorted(base for base, splits in split_by_base.items() if len(splits) > 1)
        if leaking:
            problems.append(f"base_sample_id leakage across splits: {leaking[:10]}")
        split_ids = {r["augmented_sample_id"] for r in split_rows}
        if split_ids != ids[0]:
            problems.append("split augmented IDs do not match feature/index IDs")

    for table_name, rows in [("fixed", fixed_rows), ("magnetic", magnetic_rows)]:
        for row in rows:
            for key in numeric_keys(rows):
                try:
                    value = float(row[key])
                except ValueError:
                    problems.append(f"{row['augmented_sample_id']}: nonnumeric {table_name} feature {key}")
                    continue
                if not np.isfinite(value):
                    problems.append(f"{row['augmented_sample_id']}: nonfinite {table_name} feature {key}")
            for prefix in [k[:-8] for k in row if k.endswith("_MI_mean")]:
                mi = float(row[f"{prefix}_MI_mean"])
                di = float(row[f"{prefix}_DI_mean"])
                rms = float(row[f"{prefix}_RMS_mean"])
                if mi < -1e-9 or mi > 1.0 + 1e-9:
                    problems.append(f"{row['augmented_sample_id']}: {prefix} MI outside [0,1]")
                if not np.isclose(di, 1.0 - mi, atol=1e-8):
                    problems.append(f"{row['augmented_sample_id']}: {prefix} DI != 1 - MI")
                if rms < -1e-18:
                    problems.append(f"{row['augmented_sample_id']}: {prefix} RMS negative")

    summary = {
        "created_utc": now_utc(),
        "expected_rows": EXPECTED_ROWS,
        "sample_index_rows": len(index_rows),
        "target_rows": len(target_rows),
        "fixed_feature_rows": len(fixed_rows),
        "magnetic_feature_rows": len(magnetic_rows),
        "split_rows": len(split_rows),
        "sample_ids_align": sample_ids_align,
        "passed": not problems,
        "problem_count": len(problems),
        "problems": problems[:200],
    }
    (AUG_READY / "augmented_dataset_quality_report.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
