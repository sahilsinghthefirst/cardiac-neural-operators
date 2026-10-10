                      
                                                                        

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from magnetic_common import (
    MAG_BASE,
    ROOT,
    SIM_IDS,
    compute_mig_curves,
    read_json,
    sim_ep_dir,
    sim_mag_dir,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-id", action="append", choices=SIM_IDS, help="Simulation ID to process; repeatable.")
    return parser.parse_args()


def process_sim(sim_id: str) -> dict:
    out_dir = sim_mag_dir(sim_id)
    b_z = np.load(out_dir / "Bz.npy")
    times = np.load(out_dir / "time_ms.npy") if (out_dir / "time_ms.npy").exists() else np.arange(b_z.shape[0], dtype=float)
    rows, summary = compute_mig_curves(b_z, times)
    write_csv(out_dir / "mig_curves.csv", rows, fieldnames=["time", "MI", "DI", "RMS"])
    magnetic_meta = read_json(out_dir / "magnetic_metadata.json") if (out_dir / "magnetic_metadata.json").exists() else {}
    summary = {
        **summary,
        "sim_id": sim_id,
        "condition": magnetic_meta.get("condition", ""),
        "input_component": "Bz",
        "input_Bz_file": str((out_dir / "Bz.npy").relative_to(ROOT)),
        "scientific_note": "MIG biomarkers from synthetic/proxy magnetic fields.",
    }
    write_json(out_dir / "mig_summary.json", summary)
    return summary


def build_manifest(sim_ids: list[str]) -> None:
    rows = []
    for sim_id in sim_ids:
        out_dir = sim_mag_dir(sim_id)
        ep_dir = sim_ep_dir(sim_id)
        ep_meta = json.loads((ep_dir / "metadata.json").read_text()) if (ep_dir / "metadata.json").exists() else {}
        success = all((out_dir / name).exists() for name in ["Bz.npy", "mig_curves.csv", "mig_summary.json"])
        rows.append(
            {
                "sim_id": sim_id,
                "condition": ep_meta.get("condition", ""),
                "source_ep_dir": str(ep_dir.relative_to(ROOT)),
                "magnetic_output_dir": str(out_dir.relative_to(ROOT)),
                "Bz_file": str((out_dir / "Bz.npy").relative_to(ROOT)) if (out_dir / "Bz.npy").exists() else "",
                "mig_curves_file": str((out_dir / "mig_curves.csv").relative_to(ROOT)) if (out_dir / "mig_curves.csv").exists() else "",
                "mig_summary_file": str((out_dir / "mig_summary.json").relative_to(ROOT)) if (out_dir / "mig_summary.json").exists() else "",
                "success": success,
                "notes": "" if success else "Missing magnetic or biomarker output",
            }
        )
    write_csv(
        MAG_BASE / "manifest.csv",
        rows,
        fieldnames=[
            "sim_id",
            "condition",
            "source_ep_dir",
            "magnetic_output_dir",
            "Bz_file",
            "mig_curves_file",
            "mig_summary_file",
            "success",
            "notes",
        ],
    )


def plot_all_curves(sim_ids: list[str]) -> None:
    fig_dir = ROOT / "figures" / "mig_curves"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.2), sharex=True)
    for sim_id in sim_ids:
        curves = np.genfromtxt(sim_mag_dir(sim_id) / "mig_curves.csv", delimiter=",", names=True)
        label = sim_id.replace("_smoke", "").replace("_", " ")
        axes[0].plot(curves["time"], curves["MI"], label=label)
        axes[1].plot(curves["time"], curves["DI"], label=label)
        axes[2].plot(curves["time"], curves["RMS"], label=label)
    axes[0].set_ylabel("MI")
    axes[1].set_ylabel("DI")
    axes[2].set_ylabel("RMS")
    axes[2].set_xlabel("Time (ms)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "mi_di_rms_all_smoke_cases.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    sim_ids = args.sim_id or SIM_IDS
    MAG_BASE.mkdir(parents=True, exist_ok=True)
    for sim_id in sim_ids:
        summary = process_sim(sim_id)
        print(f"{sim_id}: mean_MI={summary['mean_MI']:.4g}, mean_RMS={summary['mean_RMS']:.4g}")
    build_manifest(SIM_IDS)
    if set(SIM_IDS).issubset(set(sim_ids)):
        plot_all_curves(SIM_IDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

