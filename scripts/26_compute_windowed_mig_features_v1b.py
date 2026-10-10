                      
                                                                 

from __future__ import annotations

import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
READY_BASE = ROOT / "outputs" / "dataset_v1b_ready"
WINDOWS = ["full", "activation_like", "post_activation_like", "repolarization_like", "st_t_like"]
SIGNALS = ["MI", "DI", "RMS"]
STATS = ["mean", "max", "auc"]


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def valid(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr) & (arr >= 0)]


def load_curves(path: Path) -> dict[str, np.ndarray]:
    arr = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.atleast_1d(arr[name]).astype(float) for name in arr.dtype.names}


def window_bounds(lat: np.ndarray, rt: np.ndarray | None, apd90: np.ndarray, times: np.ndarray) -> dict[str, tuple[float, float]]:
    lat_v = valid(lat)
    if lat_v.size == 0:
        raise ValueError("LAT has no valid values")
    activation_start = float(np.percentile(lat_v, 5) - 2.0)
    activation_end = float(np.percentile(lat_v, 95) + 2.0)
    if rt is not None and valid(rt).size:
        rt_v = valid(rt)
    else:
        rt_v = valid(lat + np.where(np.isfinite(apd90) & (apd90 >= 0), apd90, np.nan))
    if rt_v.size == 0:
        raise ValueError("RT proxy has no valid values")
    repol_start = float(np.percentile(rt_v, 5) - 5.0)
    repol_end = float(np.percentile(rt_v, 95) + 5.0)
    t0 = float(np.min(times))
    t1 = float(np.max(times))
    return {
        "full": (t0, t1),
        "activation_like": (activation_start, activation_end),
        "post_activation_like": (activation_end, repol_start),
        "repolarization_like": (repol_start, repol_end),
        "st_t_like": (activation_end, repol_end),
    }


def clean_bounds(start: float, end: float, times: np.ndarray) -> tuple[float, float]:
    t0 = float(np.min(times))
    t1 = float(np.max(times))
    start = max(t0, start)
    end = min(t1, end)
    if end < start:
        mid = min(max((start + end) / 2.0, t0), t1)
        return mid, mid
    return start, end


def compute_window_features(curves: dict[str, np.ndarray], times: np.ndarray, start: float, end: float) -> tuple[dict, int]:
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


def json_clean(value):
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return json_clean(value.item())
    return value


def main() -> int:
    index_rows = read_csv(READY_BASE / "sample_index.csv")
    rows: list[dict] = []
    fieldnames = ["sample_id", "condition"]
    for window in WINDOWS:
        fieldnames += [f"{window}_start_ms", f"{window}_end_ms", f"{window}_n_timepoints"]
        for signal in SIGNALS:
            for stat in STATS:
                fieldnames.append(f"{window}_{signal}_{stat}")

    for row in index_rows:
        lat = np.loadtxt(ROOT / row["lat_file"])
        apd90 = np.loadtxt(ROOT / row["apd90_file"])
        rt = np.loadtxt(ROOT / row["rt_file"]) if row.get("rt_file") else None
        times = np.load(ROOT / row["magnetic_output_dir"] / "time_ms.npy")
        curves = load_curves(ROOT / row["mig_curves_file"])
        bounds = window_bounds(lat, rt, apd90, times)
        out = {"sample_id": row["sample_id"], "condition": row["condition"]}
        for window in WINDOWS:
            start, end = clean_bounds(*bounds[window], times)
            feats, npts = compute_window_features(curves, times, start, end)
            out[f"{window}_start_ms"] = start
            out[f"{window}_end_ms"] = end
            out[f"{window}_n_timepoints"] = npts
            for key, value in feats.items():
                out[f"{window}_{key}"] = value
        rows.append(out)

    with (READY_BASE / "features_mig_windowed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "created_utc": now_utc(),
        "row_count": len(rows),
        "window_definitions": {
            "full": "all saved magnetic time points",
            "activation_like": "p5(LAT)-2 ms to p95(LAT)+2 ms",
            "post_activation_like": "activation_end to repol_start",
            "repolarization_like": "p5(RT)-5 ms to p95(RT)+5 ms, with RT fallback LAT+APD90",
            "st_t_like": "activation_end to repol_end",
        },
        "features": "For each window: MI/DI/RMS mean, max, and trapezoidal AUC.",
        "rows": [json_clean(r) for r in rows],
    }
    (READY_BASE / "features_mig_windowed.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {READY_BASE / 'features_mig_windowed.csv'} with {len(rows)} rows")
    return 0 if len(rows) == 70 else 1


if __name__ == "__main__":
    raise SystemExit(main())
