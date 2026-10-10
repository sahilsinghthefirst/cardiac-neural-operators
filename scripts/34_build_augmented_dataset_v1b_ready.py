                      
                                                                               

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE_READY = ROOT / "outputs" / "dataset_v1b_ready"
AUG_MAG = ROOT / "outputs" / "magnetic_dataset_v1b_augmented"
AUG_READY = ROOT / "outputs" / "dataset_v1b_augmented_ready"

FIXED_WINDOWS = {
    "full": None,
    "early_activation_fixed": (0.0, 25.0),
    "mid_recovery_fixed": (25.0, 150.0),
    "late_repolarization_fixed": (150.0, 280.0),
    "broad_st_t_fixed": (25.0, 280.0),
}
MAGNETIC_WINDOWS = ["activation_magnetic", "repolarization_magnetic", "broad_after_activation"]
SIGNALS = ["MI", "DI", "RMS"]
STATS = ["mean", "max", "auc"]


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_curves(path: Path) -> dict[str, np.ndarray]:
    arr = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.atleast_1d(arr[name]).astype(float) for name in arr.dtype.names}


def clean_bounds(start: float, end: float, times: np.ndarray) -> tuple[float, float]:
    lo = float(np.min(times))
    hi = float(np.max(times))
    start = max(lo, start)
    end = min(hi, end)
    if end < start:
        mid = min(max((start + end) / 2.0, lo), hi)
        return mid, mid
    return start, end


def add_window(out: dict, name: str, curves: dict[str, np.ndarray], times: np.ndarray, start: float, end: float) -> None:
    start, end = clean_bounds(start, end, times)
    mask = (times >= start) & (times <= end)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(times - ((start + end) / 2.0))))
        mask[idx] = True
    t = times[mask]
    out[f"{name}_start_ms"] = start
    out[f"{name}_end_ms"] = end
    out[f"{name}_n_timepoints"] = int(np.sum(mask))
    for signal in SIGNALS:
        vals = curves[signal][mask]
        out[f"{name}_{signal}_mean"] = float(np.mean(vals))
        out[f"{name}_{signal}_max"] = float(np.max(vals))
        out[f"{name}_{signal}_auc"] = float(np.trapezoid(vals, t) if len(t) > 1 else 0.0)


def magnetic_window_bounds(curves: dict[str, np.ndarray], times: np.ndarray) -> tuple[dict[str, tuple[float, float]], float, float]:
    early = (times >= 0.0) & (times <= 60.0)
    if not np.any(early):
        early = np.ones_like(times, dtype=bool)
    early_idx = np.where(early)[0]
    activation_idx = int(early_idx[np.argmax(curves["RMS"][early])])
    activation_peak = float(times[activation_idx])
    late = times >= 80.0
    if not np.any(late):
        late = np.ones_like(times, dtype=bool)
    late_idx = np.where(late)[0]
    rms_late = curves["RMS"][late]
    mi_late = curves["MI"][late]
    score = rms_late / (np.max(rms_late) + 1e-30) + mi_late / (np.max(mi_late) + 1e-30)
    late_peak = float(times[int(late_idx[np.argmax(score)])])
    return {
        "activation_magnetic": (activation_peak - 10.0, activation_peak + 10.0),
        "repolarization_magnetic": (late_peak - 30.0, late_peak + 30.0),
        "broad_after_activation": (activation_peak + 10.0, 280.0),
    }, activation_peak, late_peak


def feature_fields(windows: list[str]) -> list[str]:
    fields = ["augmented_sample_id", "base_sample_id", "variant", "condition"]
    for window in windows:
        fields += [f"{window}_start_ms", f"{window}_end_ms", f"{window}_n_timepoints"]
        for signal in SIGNALS:
            for stat in STATS:
                fields.append(f"{window}_{signal}_{stat}")
    return fields


def main() -> int:
    AUG_READY.mkdir(parents=True, exist_ok=True)
    base_index = {r["sample_id"]: r for r in read_rows(BASE_READY / "sample_index.csv")}
    base_targets = {r["sample_id"]: r for r in read_rows(BASE_READY / "targets_ep.csv")}
    aug_rows = [r for r in read_rows(AUG_MAG / "manifest.csv") if str(r.get("success")).lower() == "true"]

    index_rows = []
    target_rows = []
    fixed_rows = []
    magnetic_rows = []
    for row in aug_rows:
        aug_id = row["augmented_sample_id"]
        base_id = row["base_sample_id"]
        variant = row["variant"]
        base_idx = base_index[base_id]
        index_rows.append({**row, "ep_output_dir": base_idx["ep_output_dir"], "base_condition": base_idx["condition"]})

        target = dict(base_targets[base_id])
        target["augmented_sample_id"] = aug_id
        target["base_sample_id"] = base_id
        target["variant"] = variant
        target_rows.append(target)

        mag_dir = ROOT / row["magnetic_output_dir"]
        times = np.load(mag_dir / "time_ms.npy")
        curves = load_curves(ROOT / row["mig_curves_file"])

        fixed = {"augmented_sample_id": aug_id, "base_sample_id": base_id, "variant": variant, "condition": row["condition"]}
        for name, bounds in FIXED_WINDOWS.items():
            start, end = (float(np.min(times)), float(np.max(times))) if bounds is None else bounds
            add_window(fixed, name, curves, times, start, end)
        fixed_rows.append(fixed)

        bounds, activation_peak, late_peak = magnetic_window_bounds(curves, times)
        magnetic = {
            "augmented_sample_id": aug_id,
            "base_sample_id": base_id,
            "variant": variant,
            "condition": row["condition"],
            "activation_peak_ms": activation_peak,
            "late_peak_ms": late_peak,
        }
        for name, (start, end) in bounds.items():
            add_window(magnetic, name, curves, times, start, end)
        magnetic_rows.append(magnetic)

    index_fields = list(index_rows[0].keys()) if index_rows else []
    target_fields = ["augmented_sample_id", "base_sample_id", "variant"] + [k for k in target_rows[0].keys() if k not in {"augmented_sample_id", "base_sample_id", "variant"}]
    magnetic_fields = ["augmented_sample_id", "base_sample_id", "variant", "condition", "activation_peak_ms", "late_peak_ms"]
    for window in MAGNETIC_WINDOWS:
        magnetic_fields += [f"{window}_start_ms", f"{window}_end_ms", f"{window}_n_timepoints"]
        for signal in SIGNALS:
            for stat in STATS:
                magnetic_fields.append(f"{window}_{signal}_{stat}")
    write_csv(AUG_READY / "sample_index_augmented.csv", index_rows, index_fields)
    write_csv(AUG_READY / "targets_ep_augmented.csv", target_rows, target_fields)
    write_csv(AUG_READY / "features_mig_deployable_fixed_augmented.csv", fixed_rows, feature_fields(list(FIXED_WINDOWS)))
    write_csv(AUG_READY / "features_mig_deployable_magnetic_augmented.csv", magnetic_rows, magnetic_fields)
    (AUG_READY / "build_metadata.json").write_text(json.dumps({"created_utc": now_utc(), "row_count": len(index_rows)}, indent=2, sort_keys=True) + "\n")
    print(f"Wrote augmented ready tables with {len(index_rows)} rows")
    return 0 if len(index_rows) == 770 else 1


if __name__ == "__main__":
    raise SystemExit(main())
