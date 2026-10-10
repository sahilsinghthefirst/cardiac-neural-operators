                      
                                                                

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bocf_dataset_common import DATASET_BASE, manifest_row, run_ep_sample, write_csv


PARAMS = ["tau_si", "tau_so1", "tau_so2", "tau_w_plus"]
SCALES = [0.85, 1.00, 1.15, 1.25, 1.40]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_base = DATASET_BASE / "sweeps" / "apd_parameter_runs"
    rows = []
    for param in PARAMS:
        for scale in SCALES:
            sample_id = f"apd_sweep_{param}_{scale:.2f}".replace(".", "p")
            metadata = run_ep_sample(
                out_dir=out_base / sample_id,
                sample_id=sample_id,
                condition="homogeneous_apd_parameter_sweep",
                apd_parameter=param,
                apd_scale=scale,
                stimulus_location="left",
                force=args.force,
            )
            row = manifest_row(metadata, out_base / sample_id)
            row.update(metadata.get("summary", {}))
            row["sweep_parameter"] = param
            row["sweep_scale"] = scale
            rows.append(row)
    csv_path = DATASET_BASE / "sweeps" / "apd_parameter_sweep.csv"
    write_csv(csv_path, rows)
    make_plot(rows)
    print(f"Wrote {csv_path}")
    return 0 if all(row["success"] for row in rows) else 1


def make_plot(rows: list[dict]) -> None:
    fig_dir = DATASET_BASE.parents[1] / "figures" / "calibration"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 5.2))
    for param in PARAMS:
        subset = [r for r in rows if r["sweep_parameter"] == param]
        x = [float(r["sweep_scale"]) for r in subset]
        y = [float(r["apd90_mean"] or 0.0) for r in subset]
        plt.plot(x, y, marker="o", label=param)
    plt.xlabel("parameter scale")
    plt.ylabel("mean APD90 (ms)")
    plt.title("BOCF APD parameter sensitivity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "bocf_apd_parameter_sweep.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    raise SystemExit(main())
