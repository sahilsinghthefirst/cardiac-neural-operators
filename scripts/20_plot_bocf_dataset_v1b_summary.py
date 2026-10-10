                      
                                                            

from __future__ import annotations

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bocf_dataset_common import ROOT


DATASET_BASE = ROOT / "outputs" / "bocf_dataset_v1b_conditioned"


def load_rows() -> list[dict]:
    rows = []
    for row in csv.DictReader((DATASET_BASE / "manifest.csv").open()):
        meta = json.loads((ROOT / row["output_dir"] / "metadata.json").read_text())
        row["summary"] = meta.get("summary", {})
        rows.append(row)
    return rows


def plot_box(rows: list[dict], key: str, ylabel: str, path) -> None:
    conditions = sorted({r["condition"] for r in rows})
    data = []
    for cond in conditions:
        vals = [r["summary"].get(key) for r in rows if r["condition"] == cond and r["summary"].get(key) is not None]
        data.append(vals)
    plt.figure(figsize=(10.5, 4.8))
    plt.boxplot(data, tick_labels=[c.replace("_", "\n") for c in conditions], showmeans=True)
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_locations(rows: list[dict], path) -> None:
    plt.figure(figsize=(5.4, 5.0))
    for cond in sorted({r["condition"] for r in rows if r["condition"] != "homogeneous"}):
        xs = [float(r["patch_center_x"]) for r in rows if r["condition"] == cond and r["patch_center_x"]]
        ys = [float(r["patch_center_y"]) for r in rows if r["condition"] == cond and r["patch_center_y"]]
        plt.scatter(xs, ys, label=cond.replace("_", " "), alpha=0.75)
    plt.xlim(0, 10000)
    plt.ylim(0, 10000)
    plt.gca().set_aspect("equal")
    plt.xlabel("patch center x (um)")
    plt.ylabel("patch center y (um)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_examples(rows: list[dict], path) -> None:
    chosen = []
    for cond in ["homogeneous", "apd_repolarization_perturbation", "conduction_slowing_perturbation", "mixed_apd_conduction_perturbation"]:
        match = next((r for r in rows if r["condition"] == cond), None)
        if match:
            chosen.append(match)
    fig, axes = plt.subplots(len(chosen), 3, figsize=(10.5, 3.0 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]
    from carputils.carpio import igb

    for i, row in enumerate(chosen):
        out_dir = ROOT / row["output_dir"]
        pts = np.loadtxt(out_dir / "mesh.pts", skiprows=1)
        apd = np.loadtxt(out_dir / "APD90.dat")
        mask = np.genfromtxt(out_dir / "region_mask_nodes.csv", delimiter=",", names=True)["abnormal"]
        vm, _h, _times = igb.read(str(out_dir / "vm.igb"))
        vm_tn = vm.T
        tidx = int(np.argmax(np.std(vm_tn, axis=1)))
        panels = [
            (vm_tn[tidx], f"{row['condition']} Vm", "turbo"),
            (apd, "APD90", "viridis"),
            (mask, "mask", "gray_r"),
        ]
        for ax, (vals, title, cmap) in zip(axes[i], panels):
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=vals, s=2, cmap=cmap)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    rows = load_rows()
    fig_dir = ROOT / "figures" / "dataset_v1b"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_box(rows, "apd90_mean", "APD90 mean (ms)", fig_dir / "apd_distribution_by_condition.png")
    plot_box(rows, "lat_range", "LAT range (ms)", fig_dir / "lat_range_by_condition.png")
    plot_locations(rows, fig_dir / "patch_locations.png")
    plot_examples(rows, fig_dir / "example_vm_apd_masks.png")
    print(f"Wrote figures to {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
