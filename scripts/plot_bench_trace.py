import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

if len(sys.argv) < 2:
    raise SystemExit("Usage: python scripts/plot_bench_trace.py path/to/trace_file")

path = Path(sys.argv[1])
data = np.loadtxt(path)

t = data[:, 0]
vm = data[:, 1]

plt.figure(figsize=(8, 4.5))
plt.plot(t, vm, linewidth=1.5)
plt.xlabel("Time (ms)")
plt.ylabel("Membrane voltage Vm (mV)")
plt.title("tenTusscherPanfilov Single-Cell Action Potential")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
