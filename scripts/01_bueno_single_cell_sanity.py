                      
                                                                       

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bocf_common import ROOT, run_command


OUT_DIR = ROOT / "outputs" / "bocf_sweeps" / "single_cell_sanity"


def trace_columns(out_dir: Path) -> list[str]:
    header = out_dir / "Bueno_trace_header.txt"
    if not header.exists():
        return ["time_ms", "col1", "col2", "col3", "col4", "Vm", "col6"]
    return ["time_ms"] + [line.strip() for line in header.read_text().splitlines() if line.strip()]


def compute_apd(trace: np.ndarray, vm_col: int) -> dict[str, float]:
    t = trace[:, 0]
    vm = trace[:, vm_col]
    rest = float(np.mean(vm[t < 1.0])) if np.any(t < 1.0) else float(vm[0])
    peak_idx = int(np.argmax(vm))
    peak = float(vm[peak_idx])
    apa = peak - rest
    summary = {"rest_mV": rest, "peak_mV": peak, "apa_mV": apa, "t_peak_ms": float(t[peak_idx])}
    for frac, label in [(0.80, "APD80_ms"), (0.90, "APD90_ms")]:
        threshold = rest + (1.0 - frac) * apa
        below = np.where(vm[peak_idx:] <= threshold)[0]
        summary[label] = float(t[peak_idx + below[0]] - t[peak_idx]) if below.size else float("nan")
    return summary


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bench",
        "--duration",
        "600",
        "--imp=Bueno",
        "--stim-curr",
        "60",
        "--numstim",
        "1",
        "--bcl",
        "1000",
        "--dt",
        "0.01",
        "--dt-out",
        "0.1",
        "--fout",
        "Bueno",
        "--bin",
        "-v",
    ]
    result = run_command(cmd, OUT_DIR, "bench_single_cell", cwd=OUT_DIR)
    if result.returncode != 0:
        (OUT_DIR / "NOTE.txt").write_text("Single-cell Bueno bench run failed; inspect bench_single_cell.* logs.\n")
        return result.returncode

    trace_file = OUT_DIR / "Trace_0.dat"
    if not trace_file.exists():
        (OUT_DIR / "NOTE.txt").write_text("bench completed but Trace_0.dat was not generated.\n")
        return 2

    trace = np.loadtxt(trace_file)
    columns = trace_columns(OUT_DIR)
    vm_col = columns.index("Vm") if "Vm" in columns else 1
    csv_file = OUT_DIR / "bueno_single_cell_trace.csv"
    with csv_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns[: trace.shape[1]])
        writer.writerows(trace)

    header = OUT_DIR / "Bueno_trace_header.txt"
    if header.exists():
        shutil.copy2(header, OUT_DIR / "bueno_single_cell_trace_header.txt")

    summary = compute_apd(trace, vm_col)
    (OUT_DIR / "apd_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(trace[:, 0], trace[:, vm_col], linewidth=1.6)
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane voltage Vm (mV)")
    plt.title("Bueno Single-Cell Sanity Trace")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "bueno_single_cell_ap.png", dpi=180)
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
