import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base = Path("outputs/bench_ttp_flags")

files = {
    "EPI": base / "ttp_EPI.txt",
    "ENDO": base / "ttp_ENDO.txt",
    "MCELL": base / "ttp_MCELL.txt",
}

plt.figure(figsize=(8, 4.5))

for label, path in files.items():
    if not path.exists():
        matches = list(base.glob(f"ttp_{label}*"))
        print(f"{label}: expected {path}, found candidates:")
        for m in matches:
            print(" ", m)
        continue

    data = np.loadtxt(path)
    t = data[:, 0]
    vm = data[:, 1]
    plt.plot(t, vm, linewidth=1.5, label=label)

plt.xlabel("Time (ms)")
plt.ylabel("Membrane voltage Vm (mV)")
plt.title("tenTusscherPanfilov Cell-Type Comparison")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
