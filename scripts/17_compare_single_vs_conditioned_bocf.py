                      
                                                                         

                                                                            
                                                                             
                                          
   

from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bocf_dataset_common import (
    DEFAULT_APD_PARAMETER,
    ROOT,
    recursive_size_bytes,
    run_conditioned_ep_sample,
    write_csv,
)


BASE = ROOT / "outputs" / "bocf_conditioning_study"
BCL_MS = 1000.0
CONDITIONS = {
    "homogeneous": {
        "patch_center": None,
        "patch_radius": None,
        "apd_parameter": None,
        "apd_scale": None,
        "conductivity_scale": None,
    },
    "apd_repolarization_perturbation": {
        "patch_center": (5000.0, 5000.0),
        "patch_radius": 2200.0,
        "apd_parameter": DEFAULT_APD_PARAMETER,
        "apd_scale": 1.25,
        "conductivity_scale": None,
    },
    "conduction_slowing_perturbation": {
        "patch_center": (5000.0, 5000.0),
        "patch_radius": 2200.0,
        "apd_parameter": None,
        "apd_scale": None,
        "conductivity_scale": 0.35,
    },
    "mixed_apd_conduction_perturbation": {
        "patch_center": (5000.0, 5000.0),
        "patch_radius": 2200.0,
        "apd_parameter": DEFAULT_APD_PARAMETER,
        "apd_scale": 1.25,
        "conductivity_scale": 0.35,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--nbeats", type=int, nargs="+", default=[1, 10, 20])
    parser.add_argument("--conditions", nargs="+", choices=sorted(CONDITIONS), default=list(CONDITIONS))
    return parser.parse_args()


def row_from_metadata(metadata: dict, out_dir) -> dict:
    summary = metadata.get("summary", {})
    vm_debug = metadata.get("vm_debug", {})
    return {
        "condition": metadata.get("condition"),
        "sample_id": metadata.get("sample_id"),
        "nbeats": metadata.get("nbeats"),
        "BCL": metadata.get("BCL"),
        "conditioning_used": metadata.get("conditioning_used"),
        "final_beat_extraction": metadata.get("final_beat_extraction"),
        "success": metadata.get("success"),
        "apd80_mean": summary.get("apd80_mean"),
        "apd80_std": summary.get("apd80_std"),
        "apd90_mean": summary.get("apd90_mean"),
        "apd90_std": summary.get("apd90_std"),
        "lat_range": summary.get("lat_range"),
        "vm_min": vm_debug.get("min_vm"),
        "vm_max": vm_debug.get("max_vm"),
        "propagation_succeeded": summary.get("propagation_stable"),
        "runtime_seconds": metadata.get("run_seconds"),
        "output_size_bytes": metadata.get("output_size_bytes", recursive_size_bytes(out_dir)),
        "output_dir": str(out_dir.relative_to(ROOT)),
        "notes": metadata.get("notes", ""),
    }


def plot_metric(rows: list[dict], key: str, ylabel: str, path) -> None:
    plt.figure(figsize=(9.0, 5.2))
    for condition in CONDITIONS:
        cond_rows = [r for r in rows if r["condition"] == condition and r[key] not in (None, "")]
        if not cond_rows:
            continue
        xs = [int(r["nbeats"]) for r in cond_rows]
        ys = [float(r[key]) for r in cond_rows]
        plt.plot(xs, ys, marker="o", label=condition.replace("_", " "))
    plt.xlabel("nbeats used for state conditioning")
    plt.ylabel(ylabel)
    plt.xticks(sorted({int(r["nbeats"]) for r in rows}))
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> int:
    args = parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for condition in args.conditions:
        params = CONDITIONS[condition]
        condition_lats = None
        for nbeats in args.nbeats:
            sample_id = f"{condition}_nbeats{nbeats:03d}"
            out_dir = BASE / sample_id
            if nbeats > 1:
                if condition_lats is None:
                    single_dir = BASE / f"{condition}_nbeats001"
                    condition_lats = single_dir / "LAT.dat"
                prepacing_lats = condition_lats
            else:
                prepacing_lats = BASE / sample_id / "LAT.dat"
            metadata = run_conditioned_ep_sample(
                out_dir=out_dir,
                sample_id=sample_id,
                condition=condition,
                prepacing_lats=prepacing_lats,
                nbeats=nbeats,
                bcl_ms=BCL_MS,
                stimulus_location="left",
                force=args.force,
                **params,
            )
            if nbeats == 1:
                condition_lats = out_dir / "LAT.dat"
            row = row_from_metadata(metadata, out_dir)
            rows.append(row)
            print(f"{sample_id}: success={row['success']} APD90={row['apd90_mean']} LAT_range={row['lat_range']}")

    fields = [
        "condition",
        "sample_id",
        "nbeats",
        "BCL",
        "conditioning_used",
        "final_beat_extraction",
        "success",
        "apd80_mean",
        "apd80_std",
        "apd90_mean",
        "apd90_std",
        "lat_range",
        "vm_min",
        "vm_max",
        "propagation_succeeded",
        "runtime_seconds",
        "output_size_bytes",
        "output_dir",
        "notes",
    ]
    write_csv(BASE / "conditioning_comparison.csv", rows, fields)
    fig_dir = ROOT / "figures" / "calibration"
    plot_metric(rows, "apd90_mean", "APD90 mean (ms)", fig_dir / "bocf_single_vs_conditioned_apd.png")
    plot_metric(rows, "lat_range", "LAT range (ms)", fig_dir / "bocf_single_vs_conditioned_lat.png")
    print(f"Wrote {BASE / 'conditioning_comparison.csv'}")
    return 0 if all(str(r["success"]) == "True" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
