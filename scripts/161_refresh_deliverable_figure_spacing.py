                      
                                                                            

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = Path(
    str(Path(__file__).resolve().parents[1] / "exports") + "/"
    "BPC_FNO_DeepONet_All_Deliverables_2026-09-01"
)
QA = Path("/tmp/cm_title_qa/figures")
MANIFEST = DEST / "00_README_AND_MANIFESTS/FILE_MANIFEST.csv"
VALIDATION = DEST / "00_README_AND_MANIFESTS/EXPORT_VALIDATION.json"
CHUNK = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".refresh")
    if temporary.exists():
        raise SystemExit(f"Unexpected refresh temporary file: {temporary}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def main() -> int:
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    fields = list(records[0])
    by_destination = {row["destination_path"]: row for row in records}
    current_dir = (
        DEST
        / "02_NEWER_FNO_DEEPONET_BENCHMARK"
        / "05_figures_computer_modern"
    )
    refreshed = []
    for stem in ["current_overall_model_comparison", "current_conduction_failure"]:
        for suffix in [".png", ".pdf"]:
            source = QA / f"{stem}{suffix}"
            destination = current_dir / source.name
            atomic_copy(source, destination)
            relative = destination.relative_to(DEST).as_posix()
            row = by_destination[relative]
            row["bytes"] = destination.stat().st_size
            row["sha256"] = sha256(destination)
            refreshed.append(relative)

    exporter_relative = (
        Path("02_NEWER_FNO_DEEPONET_BENCHMARK")
        / "04_reproducibility_scripts"
        / "159_export_all_project_deliverables.py"
    )
    exporter_destination = DEST / exporter_relative
    atomic_copy(ROOT / "scripts/159_export_all_project_deliverables.py", exporter_destination)
    exporter_row = by_destination[exporter_relative.as_posix()]
    exporter_row["bytes"] = exporter_destination.stat().st_size
    exporter_row["sha256"] = sha256(exporter_destination)
    refreshed.append(exporter_relative.as_posix())

    readme = DEST / "README.md"
    readme_text = readme.read_text(encoding="utf-8").replace(
        "Start with `RESULT_LINEAGE.md`, then use `FILE_MANIFEST.csv` for exact paths,\n"
        "sizes, classifications, and SHA-256 hashes.",
        "Start with `00_README_AND_MANIFESTS/RESULT_LINEAGE.md`, then use\n"
        "`00_README_AND_MANIFESTS/FILE_MANIFEST.csv` for exact paths, sizes,\n"
        "classifications, and SHA-256 hashes.",
    )
    readme.write_text(readme_text, encoding="utf-8")
    readme_row = by_destination[readme.relative_to(DEST).as_posix()]
    readme_row["bytes"] = readme.stat().st_size
    readme_row["sha256"] = sha256(readme)
    refreshed.append(readme.relative_to(DEST).as_posix())

    temporary_manifest = MANIFEST.with_suffix(".csv.refresh")
    with temporary_manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: row["destination_path"]))
    os.replace(temporary_manifest, MANIFEST)

    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    validation["manifest_sha256"] = sha256(MANIFEST)
    validation["manifest_records"] = len(records)
    validation["total_manifest_bytes"] = sum(int(row["bytes"]) for row in records)
    validation["visual_qa_title_spacing_refresh_passed"] = True
    temporary_validation = VALIDATION.with_suffix(".json.refresh")
    temporary_validation.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_validation, VALIDATION)
    print(
        json.dumps(
            {
                "passed": True,
                "refreshed": refreshed,
                "manifest_sha256": validation["manifest_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
