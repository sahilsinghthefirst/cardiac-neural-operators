                      
                                                                    

from __future__ import annotations

import csv
import json
import random
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUG_READY = ROOT / "outputs" / "dataset_v1b_augmented_ready"
SPLIT_DIR = AUG_READY / "splits"


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["augmented_sample_id", "base_sample_id", "condition", "variant", "split"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def counts_for_group(n: int) -> tuple[int, int, int]:
    train = int(round(0.70 * n))
    val = max(1, int(round(0.15 * n)))
    test = n - train - val
    if test <= 0:
        test = 1
        train = n - val - test
    return train, val, test


def main() -> int:
    rng = random.Random(20260525)
    rows = read_rows(AUG_READY / "sample_index_augmented.csv")
    by_base: dict[str, dict] = {}
    for row in rows:
        by_base.setdefault(row["base_sample_id"], {"condition": row["condition"], "rows": []})["rows"].append(row)
    by_condition: dict[str, list[str]] = {}
    for base_id, data in by_base.items():
        by_condition.setdefault(data["condition"], []).append(base_id)

    base_split: dict[str, str] = {}
    for condition, base_ids in sorted(by_condition.items()):
        base_ids = list(base_ids)
        rng.shuffle(base_ids)
        n_train, n_val, _n_test = counts_for_group(len(base_ids))
        for base_id in base_ids[:n_train]:
            base_split[base_id] = "train"
        for base_id in base_ids[n_train : n_train + n_val]:
            base_split[base_id] = "val"
        for base_id in base_ids[n_train + n_val :]:
            base_split[base_id] = "test"

    split_rows = []
    for row in rows:
        split_rows.append(
            {
                "augmented_sample_id": row["augmented_sample_id"],
                "base_sample_id": row["base_sample_id"],
                "condition": row["condition"],
                "variant": row["variant"],
                "split": base_split[row["base_sample_id"]],
            }
        )
    split_counts = {}
    base_counts = {}
    for row in split_rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    for base_id, split in base_split.items():
        base_counts[split] = base_counts.get(split, 0) + 1

    payload = {
        "created_utc": now_utc(),
        "strategy": "grouped by base_sample_id, stratified by condition with approximate 70/15/15 proportions",
        "base_sample_count": len(by_base),
        "augmented_sample_count": len(split_rows),
        "base_counts_by_split": base_counts,
        "augmented_counts_by_split": split_counts,
        "base_split": base_split,
    }
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    (SPLIT_DIR / "grouped_train_val_test.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(SPLIT_DIR / "grouped_train_val_test.csv", split_rows)
    print(json.dumps({k: payload[k] for k in ["base_counts_by_split", "augmented_counts_by_split"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
