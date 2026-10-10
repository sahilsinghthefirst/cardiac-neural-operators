import numpy as np
from pathlib import Path

base = Path("outputs/bench_ttp_flags")

labels = ["EPI", "ENDO", "MCELL"]

def find_trace(label):
    candidates = list(base.glob(f"ttp_{label}*"))
    for path in candidates:
        if path.is_file() and path.suffix in [".txt", ".dat", ".trc", ""]:
            return path
    raise FileNotFoundError(f"No trace file found for {label}. Candidates: {candidates}")

def analyze_trace(path):
    data = np.loadtxt(path)
    t = data[:, 0]
    vm = data[:, 1]

                                                    
    prestim = t < 10.0
    rmp = np.mean(vm[prestim])

    vmax = np.max(vm)
    apa = vmax - rmp

                                                               
                                                  
    threshold_90 = rmp + 0.1 * apa

    peak_idx = np.argmax(vm)
    t_peak = t[peak_idx]

    after_peak_t = t[peak_idx:]
    after_peak_vm = vm[peak_idx:]

    below = np.where(after_peak_vm <= threshold_90)[0]

    if len(below) == 0:
        apd90 = np.nan
    else:
        repol_idx = below[0]
        t_repol90 = after_peak_t[repol_idx]
        apd90 = t_repol90 - t_peak

    return {
        "RMP_mV": rmp,
        "Vmax_mV": vmax,
        "APA_mV": apa,
        "APD90_ms": apd90,
        "t_peak_ms": t_peak,
    }

print(f"{'Cell':<8} {'RMP (mV)':>12} {'Vmax (mV)':>12} {'APA (mV)':>12} {'APD90 (ms)':>12}")

for label in labels:
    path = find_trace(label)
    result = analyze_trace(path)

    print(
        f"{label:<8} "
        f"{result['RMP_mV']:>12.2f} "
        f"{result['Vmax_mV']:>12.2f} "
        f"{result['APA_mV']:>12.2f} "
        f"{result['APD90_ms']:>12.2f}"
    )
