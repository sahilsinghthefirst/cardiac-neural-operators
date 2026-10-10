                      
                                                                        

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bocf_dataset_common import DATASET_BASE, manifest_row, run_ep_sample, write_csv


SCALES = [1.00, 0.80, 0.65, 0.50, 0.35]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_base = DATASET_BASE / "sweeps" / "conduction_severity_runs"
    rows = []
    for scale in SCALES:
        sample_id = f"conduction_sweep_{scale:.2f}".replace(".", "p")
        metadata = run_ep_sample(
            out_dir=out_base / sample_id,
            sample_id=sample_id,
            condition="conduction_slowing_severity_sweep",
            conductivity_scale=scale,
            patch_center=(5000.0, 5000.0),
            patch_radius=2200.0,
            stimulus_location="left",
            force=args.force,
        )
        row = manifest_row(metadata, out_base / sample_id)
        row.update(metadata.get("summary", {}))
        rows.append(row)
    csv_path = DATASET_BASE / "sweeps" / "conduction_severity_sweep.csv"
    write_csv(csv_path, rows)
    make_plot(rows)
    print(f"Wrote {csv_path}")
    return 0 if all(row["success"] for row in rows) else 1


def make_plot(rows: list[dict]) -> None:
    fig_dir = DATASET_BASE.parents[1] / "figures" / "calibration"
    fig_dir.mkdir(parents=True, exist_ok=True)
    x = [float(r["conductivity_scale"]) for r in rows]
    lat_range = [float(r["lat_range"] or 0.0) for r in rows]
    speed = [float(r["activation_speed_proxy_m_per_s"] or 0.0) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].plot(x, lat_range, marker="o")
    axes[0].set_xlabel("regional conductivity scale")
    axes[0].set_ylabel("LAT range (ms)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(x, speed, marker="o", color="tab:red")
    axes[1].set_xlabel("regional conductivity scale")
    axes[1].set_ylabel("activation-speed proxy (m/s)")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("BOCF conduction severity sweep")
    fig.tight_layout()
    fig.savefig(fig_dir / "bocf_conduction_severity_sweep.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())

