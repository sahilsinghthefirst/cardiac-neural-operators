                      
                                                                       

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from bocf_v1c_common import (
    FULL_MODE,
    PILOT_MODE,
    ROOT,
    TARGET_FIELDS,
    array_file,
    ep_target_row,
    fixed_feature_row,
    fixed_fieldnames,
    grouped_splits,
    load_array,
    load_curves,
    magnetic_feature_row,
    magnetic_fieldnames,
    mode_paths,
    now_utc,
    read_csv_rows,
    rel,
    truthy,
    write_csv_rows,
    write_json,
)


BASE_INDEX_FIELDS = [
    "sample_id",
    "condition",
    "ep_output_dir",
    "magnetic_output_dir",
    "vm_file",
    "lat_file",
    "rt_file",
    "apd80_file",
    "apd90_file",
    "mask_file",
    "Bz_file",
    "mig_curves_file",
    "mig_summary_file",
    "apd_parameter",
    "apd_scale",
    "conductivity_scale",
    "patch_center_x",
    "patch_center_y",
    "patch_radius",
    "patch_radius_mm",
    "nbeats",
    "BCL",
    "conditioning_used",
    "ep_success",
    "magnetic_success",
    "notes",
]


AUG_INDEX_FIELDS = [
    "augmented_sample_id",
    "base_sample_id",
    "condition",
    "variant",
    "ep_output_dir",
    "magnetic_output_dir",
    "Bz_file",
    "mig_curves_file",
    "mig_summary_file",
    "sensor_mask_file",
    "base_target_row",
    "success",
    "notes",
]


def build_base_index(paths: dict[str, Path]) -> list[dict]:
    ep_rows = {r["sample_id"]: r for r in read_csv_rows(paths["ep"] / "manifest.csv")}
    mag_rows = {r["sample_id"]: r for r in read_csv_rows(paths["mag"] / "manifest.csv")}
    rows = []
    for sample_id in sorted(ep_rows):
        ep = ep_rows[sample_id]
        mag = mag_rows.get(sample_id, {})
        ep_dir = ROOT / ep["output_dir"]
        rows.append(
            {
                "sample_id": sample_id,
                "condition": ep["condition"],
                "ep_output_dir": ep["output_dir"],
                "magnetic_output_dir": mag.get("magnetic_output_dir", ""),
                "vm_file": rel(ep_dir / "vm.igb"),
                "lat_file": rel(ep_dir / "LAT.dat"),
                "rt_file": rel(ep_dir / "RT_repol_proxy.dat"),
                "apd80_file": rel(ep_dir / "APD80.dat"),
                "apd90_file": rel(ep_dir / "APD90.dat"),
                "mask_file": rel(ep_dir / "region_mask_nodes.csv"),
                "Bz_file": mag.get("Bz_file", ""),
                "mig_curves_file": mag.get("mig_curves_file", ""),
                "mig_summary_file": mag.get("mig_summary_file", ""),
                "apd_parameter": ep.get("apd_parameter", ""),
                "apd_scale": ep.get("apd_scale", ""),
                "conductivity_scale": ep.get("conductivity_scale", ""),
                "patch_center_x": ep.get("patch_center_x", ""),
                "patch_center_y": ep.get("patch_center_y", ""),
                "patch_radius": ep.get("patch_radius", ""),
                "patch_radius_mm": ep.get("patch_radius_mm", ""),
                "nbeats": ep.get("nbeats", ""),
                "BCL": ep.get("BCL", ""),
                "conditioning_used": ep.get("conditioning_used", ""),
                "ep_success": ep.get("success", ""),
                "magnetic_success": mag.get("success", ""),
                "notes": "; ".join(x for x in [ep.get("notes", ""), mag.get("notes", "")] if x),
            }
        )
    return rows


def build_features(index_rows: list[dict], id_field: str = "sample_id") -> tuple[list[dict], list[dict]]:
    fixed_rows = []
    magnetic_rows = []
    root = Path(__file__).resolve().parents[1]
    for row in index_rows:
        sample_id = row[id_field]
        condition = row["condition"]
        magnetic_dir = root / row["magnetic_output_dir"]
        times = load_array(array_file(magnetic_dir, "time_ms"))
        curves = load_curves(root / row["mig_curves_file"])
        fixed_rows.append(fixed_feature_row(sample_id, condition, curves, times, id_field=id_field))
        magnetic_rows.append(magnetic_feature_row(sample_id, condition, curves, times, id_field=id_field))
    return fixed_rows, magnetic_rows


def build_augmented_index(paths: dict[str, Path], base_index: list[dict], targets: list[dict]) -> list[dict]:
    by_base = {r["sample_id"]: r for r in base_index}
    target_index = {r["sample_id"]: str(i) for i, r in enumerate(targets)}
    rows = []
    for aug in read_csv_rows(paths["aug"] / "manifest.csv"):
        base = by_base.get(aug["base_sample_id"], {})
        rows.append(
            {
                "augmented_sample_id": aug["augmented_sample_id"],
                "base_sample_id": aug["base_sample_id"],
                "condition": aug["condition"],
                "variant": aug["variant"],
                "ep_output_dir": base.get("ep_output_dir", ""),
                "magnetic_output_dir": aug["magnetic_output_dir"],
                "Bz_file": aug["Bz_file"],
                "mig_curves_file": aug["mig_curves_file"],
                "mig_summary_file": aug["mig_summary_file"],
                "sensor_mask_file": aug["sensor_mask_file"],
                "base_target_row": target_index.get(aug["base_sample_id"], ""),
                "success": aug["success"],
                "notes": aug.get("notes", ""),
            }
        )
    return rows


def finite_feature_rows(rows: list[dict], id_field: str) -> tuple[bool, list[str]]:
    problems = []
    for row in rows:
        for key, value in row.items():
            if key in {id_field, "base_sample_id", "variant", "condition"}:
                continue
            try:
                val = float(value)
            except (TypeError, ValueError):
                problems.append(f"{row[id_field]}:{key}:not_float")
                continue
            if not math.isfinite(val):
                problems.append(f"{row[id_field]}:{key}:nonfinite")
    return not problems, problems[:50]


def write_ready_quality(paths: dict[str, Path], base_index: list[dict], targets: list[dict], fixed: list[dict], magnetic: list[dict]) -> bool:
    ids = {r["sample_id"] for r in base_index}
    problems = []
    if ids != {r["sample_id"] for r in targets}:
        problems.append("sample_index and targets sample IDs differ")
    if ids != {r["sample_id"] for r in fixed}:
        problems.append("sample_index and fixed feature sample IDs differ")
    if ids != {r["sample_id"] for r in magnetic}:
        problems.append("sample_index and magnetic feature sample IDs differ")
    ok_fixed, fixed_probs = finite_feature_rows(fixed, "sample_id")
    ok_mag, mag_probs = finite_feature_rows(magnetic, "sample_id")
    if not ok_fixed:
        problems.append(f"fixed feature finite problems: {fixed_probs[:10]}")
    if not ok_mag:
        problems.append(f"magnetic feature finite problems: {mag_probs[:10]}")
    payload = {
        "created_utc": now_utc(),
        "row_count": len(base_index),
        "target_rows": len(targets),
        "fixed_feature_rows": len(fixed),
        "magnetic_feature_rows": len(magnetic),
        "leakage_note": "Deployable features use only time_ms and MIG curves; no LAT/RT/APD/mask windows are used.",
        "passed": not problems,
        "problems": problems,
    }
    write_json(paths["ready"] / "dataset_ready_quality_report.json", payload)
    return not problems


def write_aug_quality(paths: dict[str, Path], aug_index: list[dict], aug_targets: list[dict], fixed: list[dict], magnetic: list[dict], split_rows: list[dict]) -> bool:
    ids = {r["augmented_sample_id"] for r in aug_index}
    problems = []
    if ids != {r["augmented_sample_id"] for r in aug_targets}:
        problems.append("augmented index and targets IDs differ")
    if ids != {r["augmented_sample_id"] for r in fixed}:
        problems.append("augmented index and fixed features IDs differ")
    if ids != {r["augmented_sample_id"] for r in magnetic}:
        problems.append("augmented index and magnetic features IDs differ")
    if ids != {r["augmented_sample_id"] for r in split_rows}:
        problems.append("augmented index and split IDs differ")
    ok_fixed, fixed_probs = finite_feature_rows(fixed, "augmented_sample_id")
    ok_mag, mag_probs = finite_feature_rows(magnetic, "augmented_sample_id")
    if not ok_fixed:
        problems.append(f"fixed feature finite problems: {fixed_probs[:10]}")
    if not ok_mag:
        problems.append(f"magnetic feature finite problems: {mag_probs[:10]}")
    split_by_base = {}
    for row in split_rows:
        split_by_base.setdefault(row["base_sample_id"], set()).add(row["split"])
    leaked = [base for base, splits in split_by_base.items() if len(splits) > 1]
    if leaked:
        problems.append(f"base_sample_id leakage across splits: {leaked[:10]}")
    payload = {
        "created_utc": now_utc(),
        "augmented_rows": len(aug_index),
        "target_rows": len(aug_targets),
        "fixed_feature_rows": len(fixed),
        "magnetic_feature_rows": len(magnetic),
        "split_rows": len(split_rows),
        "leakage_note": "Deployable features use only time_ms and MIG curves; splits are grouped by base_sample_id.",
        "passed": not problems,
        "problems": problems,
    }
    write_json(paths["aug_ready"] / "augmented_dataset_quality_report.json", payload)
    return not problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[PILOT_MODE, FULL_MODE], default=PILOT_MODE)
    parser.add_argument("--seed", type=int, default=20260525)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = mode_paths(args.mode)
    paths["ready"].mkdir(parents=True, exist_ok=True)
    paths["aug_ready"].mkdir(parents=True, exist_ok=True)

    base_index = build_base_index(paths)
    targets = [ep_target_row(r) for r in base_index if truthy(r.get("ep_success"))]
    fixed, magnetic = build_features([r for r in base_index if truthy(r.get("magnetic_success"))], "sample_id")
    write_csv_rows(paths["ready"] / "sample_index.csv", base_index, BASE_INDEX_FIELDS)
    write_json(paths["ready"] / "sample_index.json", {"created_utc": now_utc(), "rows": base_index})
    write_csv_rows(paths["ready"] / "targets_ep.csv", targets, TARGET_FIELDS)
    write_json(paths["ready"] / "targets_ep.json", {"created_utc": now_utc(), "rows": targets})
    write_csv_rows(paths["ready"] / "features_mig_deployable_fixed.csv", fixed, fixed_fieldnames("sample_id"))
    write_json(paths["ready"] / "features_mig_deployable_fixed.json", {"created_utc": now_utc(), "rows": fixed})
    write_csv_rows(paths["ready"] / "features_mig_deployable_magnetic.csv", magnetic, magnetic_fieldnames("sample_id"))
    write_json(paths["ready"] / "features_mig_deployable_magnetic.json", {"created_utc": now_utc(), "rows": magnetic})
    base_ok = write_ready_quality(paths, base_index, targets, fixed, magnetic)

    aug_index = build_augmented_index(paths, base_index, targets)
    target_by_base = {r["sample_id"]: r for r in targets}
    aug_targets = []
    for row in aug_index:
        target = dict(target_by_base[row["base_sample_id"]])
        target["augmented_sample_id"] = row["augmented_sample_id"]
        target["base_sample_id"] = row["base_sample_id"]
        target["variant"] = row["variant"]
        aug_targets.append(target)
    aug_fixed, aug_magnetic = build_features(aug_index, "augmented_sample_id")
    for table, by_id in [(aug_fixed, {r["augmented_sample_id"]: r for r in aug_index}), (aug_magnetic, {r["augmented_sample_id"]: r for r in aug_index})]:
        for row in table:
            aug_row = by_id[row["augmented_sample_id"]]
            row["base_sample_id"] = aug_row["base_sample_id"]
            row["variant"] = aug_row["variant"]
    split_rows, split_payload = grouped_splits(aug_index, seed=args.seed)
    split_dir = paths["aug_ready"] / "splits"
    write_csv_rows(paths["aug_ready"] / "sample_index_augmented.csv", aug_index, AUG_INDEX_FIELDS)
    write_csv_rows(paths["aug_ready"] / "targets_ep_augmented.csv", aug_targets, ["augmented_sample_id", "base_sample_id", "variant", *TARGET_FIELDS])
    write_csv_rows(
        paths["aug_ready"] / "features_mig_deployable_fixed_augmented.csv",
        aug_fixed,
        fixed_fieldnames("augmented_sample_id", ["base_sample_id", "variant"]),
    )
    write_csv_rows(
        paths["aug_ready"] / "features_mig_deployable_magnetic_augmented.csv",
        aug_magnetic,
        magnetic_fieldnames("augmented_sample_id", ["base_sample_id", "variant"]),
    )
    write_json(paths["aug_ready"] / "sample_index_augmented.json", {"created_utc": now_utc(), "rows": aug_index})
    write_json(paths["aug_ready"] / "targets_ep_augmented.json", {"created_utc": now_utc(), "rows": aug_targets})
    write_json(paths["aug_ready"] / "features_mig_deployable_fixed_augmented.json", {"created_utc": now_utc(), "rows": aug_fixed})
    write_json(paths["aug_ready"] / "features_mig_deployable_magnetic_augmented.json", {"created_utc": now_utc(), "rows": aug_magnetic})
    write_csv_rows(split_dir / "grouped_train_val_test.csv", split_rows, ["augmented_sample_id", "base_sample_id", "condition", "variant", "split"])
    write_json(split_dir / "grouped_train_val_test.json", split_payload)
    aug_ok = write_aug_quality(paths, aug_index, aug_targets, aug_fixed, aug_magnetic, split_rows)

    print(json.dumps({"base_ready_passed": base_ok, "augmented_ready_passed": aug_ok, "base_rows": len(base_index), "augmented_rows": len(aug_index)}, indent=2))
    return 0 if base_ok and aug_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
