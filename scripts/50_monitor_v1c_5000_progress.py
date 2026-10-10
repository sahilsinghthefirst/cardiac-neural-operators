                      
                                                             

from __future__ import annotations

import csv
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_BASE = 5000
TARGET_AUG = TARGET_BASE * 11
PATHS = {
    "ep": ROOT / "outputs" / "bocf_dataset_v1c_5000",
    "magnetic": ROOT / "outputs" / "magnetic_dataset_v1c_5000",
    "augmented": ROOT / "outputs" / "magnetic_dataset_v1c_5000_augmented",
    "ready": ROOT / "outputs" / "dataset_v1c_5000_ready",
    "augmented_ready": ROOT / "outputs" / "dataset_v1c_5000_augmented_ready",
}


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def human_size(n: int) -> str:
    value = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TB"


def recent_files(limit: int = 12) -> list[Path]:
    files = []
    for path in PATHS.values():
        if path.exists():
            files.extend(p for p in path.rglob("*") if p.is_file())
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]


def stage_percent() -> tuple[float, str]:
    ep_manifest = PATHS["ep"] / "manifest.csv"
    mag_manifest = PATHS["magnetic"] / "manifest.csv"
    aug_manifest = PATHS["augmented"] / "manifest.csv"
    ready_report = PATHS["ready"] / "dataset_ready_quality_report.json"
    aug_ready_report = PATHS["augmented_ready"] / "augmented_dataset_quality_report.json"

    ep_rows = read_rows(ep_manifest)
    mag_rows = read_rows(mag_manifest)
    aug_rows = read_rows(aug_manifest)
    if aug_ready_report.exists():
        return 100.0, "augmented ready tables/checks complete"
    if ready_report.exists():
        return 92.0, "base ready tables complete; augmented ready pending or writing"
    if aug_rows:
        return min(90.0, 70.0 + 20.0 * len(aug_rows) / TARGET_AUG), f"augmentation manifest rows {len(aug_rows)}/{TARGET_AUG}"
    aug_bz = len(list(PATHS["augmented"].glob("*/*.npy"))) if PATHS["augmented"].exists() else 0
    if aug_bz:
        return min(89.0, 70.0 + 20.0 * aug_bz / TARGET_AUG), f"augmentation output files in progress"
    if mag_rows:
        return min(70.0, 55.0 + 15.0 * len(mag_rows) / TARGET_BASE), f"magnetic manifest rows {len(mag_rows)}/{TARGET_BASE}"
    mag_bz = len(list(PATHS["magnetic"].glob("*/Bz.npy"))) if PATHS["magnetic"].exists() else 0
    if mag_bz:
        return min(69.0, 55.0 + 15.0 * mag_bz / TARGET_BASE), f"magnetic Bz files {mag_bz}/{TARGET_BASE}"
    if ep_rows:
        return min(55.0, 55.0 * len(ep_rows) / TARGET_BASE), f"EP manifest rows {len(ep_rows)}/{TARGET_BASE}"
    ep_dirs = len([p for p in (PATHS["ep"] / "samples").glob("*") if p.is_dir()]) if (PATHS["ep"] / "samples").exists() else 0
    return min(54.0, 55.0 * ep_dirs / TARGET_BASE), f"EP sample directories {ep_dirs}/{TARGET_BASE}"


def main() -> int:
    usage = shutil.disk_usage(ROOT)
    ep_rows = read_rows(PATHS["ep"] / "manifest.csv")
    mag_rows = read_rows(PATHS["magnetic"] / "manifest.csv")
    aug_rows = read_rows(PATHS["augmented"] / "manifest.csv")
    ep_dirs = len([p for p in (PATHS["ep"] / "samples").glob("*") if p.is_dir()]) if (PATHS["ep"] / "samples").exists() else 0
    ep_success = sum(1 for row in ep_rows if truthy(row.get("success")))
    mag_success = sum(1 for row in mag_rows if truthy(row.get("success")))
    aug_success = sum(1 for row in aug_rows if truthy(row.get("success")))
    pct, detail = stage_percent()

    print(f"timestamp_utc: {now()}")
    print(f"working_directory: {ROOT}")
    print(f"disk_free: {human_size(usage.free)} / total {human_size(usage.total)}")
    print("disk_usage:")
    for name, path in PATHS.items():
        print(f"  {name}: {human_size(size_bytes(path))} ({path.relative_to(ROOT)})")
    print("counts:")
    print(f"  ep_sample_directories: {ep_dirs}")
    print(f"  ep_manifest_rows: {len(ep_rows)}")
    print(f"  ep_success_rows: {ep_success}")
    print(f"  magnetic_manifest_rows: {len(mag_rows)}")
    print(f"  magnetic_success_rows: {mag_success}")
    print(f"  augmented_manifest_rows: {len(aug_rows)}")
    print(f"  augmented_success_rows: {aug_success}")
    print(f"estimated_percent_complete: {pct:.1f}")
    print(f"progress_detail: {detail}")
    print("most_recent_files:")
    for path in recent_files():
        mtime = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        print(f"  {mtime} {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
