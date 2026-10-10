                      
                                                                       

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bocf_dataset_common import DATASET_BASE, manifest_row, run_ep_sample, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_base = DATASET_BASE / "calibration" / "control_runs"
    rows = []
    settings = [
        (0.75, "left", "longitudinal"),
        (1.00, "left", "longitudinal"),
        (1.25, "left", "longitudinal"),
        (0.75, "bottom", "transverse"),
        (1.00, "bottom", "transverse"),
        (1.25, "bottom", "transverse"),
    ]
    for base_gil, stim, axis in settings:
        sample_id = f"control_g{base_gil:.2f}_{axis}"
        metadata = run_ep_sample(
            out_dir=out_base / sample_id,
            sample_id=sample_id,
            condition="homogeneous_control_calibration",
            stimulus_location=stim,
            base_gil=base_gil,
            anisotropy_factor=4.0,
            force=args.force,
        )
        row = manifest_row(metadata, out_base / sample_id)
        row.update(metadata.get("summary", {}))
        row["base_gil"] = base_gil
        row["axis_proxy"] = axis
        rows.append(row)
    csv_path = DATASET_BASE / "calibration" / "control_tissue_calibration.csv"
    write_csv(csv_path, rows)
    make_plot(rows)
    print(f"Wrote {csv_path}")
    return 0 if all(row["success"] for row in rows) else 1


def make_plot(rows: list[dict]) -> None:
    fig_dir = DATASET_BASE.parents[1] / "figures" / "calibration"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for axis in ["longitudinal", "transverse"]:
        subset = [r for r in rows if r["axis_proxy"] == axis]
        x = [float(r["base_gil"]) for r in subset]
        cv = [float(r.get("activation_speed_proxy_m_per_s") or 0.0) for r in subset]
        axes[0].plot(x, cv, marker="o", label=axis)
    axes[0].set_xlabel("g_il base")
    axes[0].set_ylabel("activation-speed proxy (m/s)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    x = [float(r["base_gil"]) for r in rows if r["axis_proxy"] == "longitudinal"]
    apd = [float(r.get("apd90_mean") or 0.0) for r in rows if r["axis_proxy"] == "longitudinal"]
    axes[1].plot(x, apd, marker="o", color="tab:green")
    axes[1].set_xlabel("g_il base")
    axes[1].set_ylabel("mean APD90 (ms)")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("BOCF control tissue calibration")
    fig.tight_layout()
    fig.savefig(fig_dir / "bocf_control_cv_apd_calibration.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
