                      
                                                             

                                                                         
                                                                           
        
   

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
READY_BASE = ROOT / "outputs" / "dataset_v1b_ready"

FIXED_WINDOWS = {
    "full": None,
    "early_activation_fixed": (0.0, 25.0),
    "mid_recovery_fixed": (25.0, 150.0),
    "late_repolarization_fixed": (150.0, 280.0),
    "broad_st_t_fixed": (25.0, 280.0),
}
MAGNETIC_WINDOWS = [
    "activation_magnetic",
    "repolarization_magnetic",
    "broad_after_activation",
]
SIGNALS = ["MI", "DI", "RMS"]
STATS = ["mean", "max", "auc"]


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


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


def features_for_window(curves: dict[str, np.ndarray], times: np.ndarray, start: float, end: float) -> tuple[dict, int]:
    start, end = clean_bounds(start, end, times)
    mask = (times >= start) & (times <= end)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(times - ((start + end) / 2.0))))
        mask[idx] = True
    t = times[mask]
    out: dict[str, float] = {}
    for signal in SIGNALS:
        vals = curves[signal][mask]
        out[f"{signal}_mean"] = float(np.mean(vals))
        out[f"{signal}_max"] = float(np.max(vals))
        out[f"{signal}_auc"] = float(np.trapezoid(vals, t) if len(t) > 1 else 0.0)
    return out, int(np.sum(mask))


def add_window(out: dict, name: str, curves: dict[str, np.ndarray], times: np.ndarray, start: float, end: float) -> None:
    start, end = clean_bounds(start, end, times)
    feats, npts = features_for_window(curves, times, start, end)
    out[f"{name}_start_ms"] = start
    out[f"{name}_end_ms"] = end
    out[f"{name}_n_timepoints"] = npts
    for key, value in feats.items():
        out[f"{name}_{key}"] = value


def fixed_fieldnames() -> list[str]:
    fields = ["sample_id", "condition"]
    for window in FIXED_WINDOWS:
        fields += [f"{window}_start_ms", f"{window}_end_ms", f"{window}_n_timepoints"]
        for signal in SIGNALS:
            for stat in STATS:
                fields.append(f"{window}_{signal}_{stat}")
    return fields


def magnetic_fieldnames() -> list[str]:
    fields = ["sample_id", "condition", "activation_peak_ms", "late_peak_ms"]
    for window in MAGNETIC_WINDOWS:
        fields += [f"{window}_start_ms", f"{window}_end_ms", f"{window}_n_timepoints"]
        for signal in SIGNALS:
            for stat in STATS:
                fields.append(f"{window}_{signal}_{stat}")
    return fields


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict], windows: dict | list, kind: str) -> None:
    payload = {
        "created_utc": now_utc(),
        "feature_set": kind,
        "row_count": len(rows),
        "window_definitions": windows,
        "leakage_note": "Windows are defined from fixed time ranges or magnetic curves only, not EP ground-truth maps.",
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
    rms_norm = rms_late / (np.max(rms_late) + 1e-30)
    mi_norm = mi_late / (np.max(mi_late) + 1e-30)
    late_peak_idx = int(late_idx[np.argmax(rms_norm + mi_norm)])
    late_peak = float(times[late_peak_idx])

    bounds = {
        "activation_magnetic": (activation_peak - 10.0, activation_peak + 10.0),
        "repolarization_magnetic": (late_peak - 30.0, late_peak + 30.0),
        "broad_after_activation": (activation_peak + 10.0, 280.0),
    }
    return bounds, activation_peak, late_peak


def main() -> int:
    rows = read_rows(READY_BASE / "sample_index.csv")
    fixed_rows: list[dict] = []
    magnetic_rows: list[dict] = []
    for row in rows:
        sample_id = row["sample_id"]
        condition = row["condition"]
        magnetic_dir = ROOT / row["magnetic_output_dir"]
        times = np.load(magnetic_dir / "time_ms.npy")
        curves = load_curves(ROOT / row["mig_curves_file"])

        fixed = {"sample_id": sample_id, "condition": condition}
        for name, bounds in FIXED_WINDOWS.items():
            start, end = (float(np.min(times)), float(np.max(times))) if bounds is None else bounds
            add_window(fixed, name, curves, times, start, end)
        fixed_rows.append(fixed)

        bounds, activation_peak, late_peak = magnetic_window_bounds(curves, times)
        magnetic = {
            "sample_id": sample_id,
            "condition": condition,
            "activation_peak_ms": activation_peak,
            "late_peak_ms": late_peak,
        }
        for name, (start, end) in bounds.items():
            add_window(magnetic, name, curves, times, start, end)
        magnetic_rows.append(magnetic)

    fixed_csv = READY_BASE / "features_mig_deployable_fixed.csv"
    fixed_json = READY_BASE / "features_mig_deployable_fixed.json"
    mag_csv = READY_BASE / "features_mig_deployable_magnetic_windows.csv"
    mag_json = READY_BASE / "features_mig_deployable_magnetic_windows.json"
    write_csv(fixed_csv, fixed_rows, fixed_fieldnames())
    write_json(fixed_json, fixed_rows, FIXED_WINDOWS, "deployable_fixed")
    write_csv(mag_csv, magnetic_rows, magnetic_fieldnames())
    write_json(
        mag_json,
        magnetic_rows,
        {
            "activation_peak": "time of max RMS in 0-60 ms",
            "activation_magnetic": "activation_peak +/- 10 ms",
            "late_peak": "time of max normalized RMS+MI after 80 ms",
            "repolarization_magnetic": "late_peak +/- 30 ms",
            "broad_after_activation": "activation_peak + 10 ms to 280 ms",
        },
        "deployable_magnetic_windows",
    )
    print(f"Wrote {fixed_csv} with {len(fixed_rows)} rows")
    print(f"Wrote {mag_csv} with {len(magnetic_rows)} rows")
    return 0 if len(fixed_rows) == 70 and len(magnetic_rows) == 70 else 1


if __name__ == "__main__":
    raise SystemExit(main())
