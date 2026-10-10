                      
                                                                          

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = Path(
    str(Path(__file__).resolve().parents[1] / "exports") + "/"
    "BPC_FNO_DeepONet_All_Deliverables_2026-09-01"
)
MANIFEST = DEST / "00_README_AND_MANIFESTS/FILE_MANIFEST.csv"
VALIDATION = DEST / "00_README_AND_MANIFESTS/EXPORT_VALIDATION.json"
README = DEST / "README.md"
CHUNK = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_and_record(source: Path, relative: Path) -> dict[str, object]:
    destination = DEST / relative
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite finalizer target: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        for block in iter(lambda: reader.read(CHUNK), b""):
            writer.write(block)
            digest.update(block)
            count += len(block)
        writer.flush()
        os.fsync(writer.fileno())
    shutil.copystat(source, destination)
    return {
        "destination_path": relative.as_posix(),
        "source_path": str(source),
        "source_repository": "mig-pino-v1",
        "classification": "current_contract_locked_evidence",
        "bytes": count,
        "sha256": digest.hexdigest(),
        "artifact_kind": "copied",
    }


def refresh_existing(source: Path, destination: Path) -> tuple[int, str]:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        raise SystemExit(f"Unexpected finalizer temporary file: {temporary}")
    digest = hashlib.sha256()
    count = 0
    with source.open("rb") as reader, temporary.open("xb") as writer:
        for block in iter(lambda: reader.read(CHUNK), b""):
            writer.write(block)
            digest.update(block)
            count += len(block)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, destination)
    shutil.copystat(source, destination)
    return count, digest.hexdigest()


def main() -> int:
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    fields = list(records[0])
    additions = []
    for name in ["evaluation_contract.json", "artifact_inventory.json"]:
        source = ROOT / "outputs/icbinb_required_analysis" / name
        relative = (
            Path("02_NEWER_FNO_DEEPONET_BENCHMARK")
            / "02_current_locked_evidence"
            / "outputs/icbinb_required_analysis"
            / name
        )
        additions.append(copy_and_record(source, relative))
    additions.append(
        copy_and_record(
            ROOT / "scripts/160_finalize_windows_deliverable_bundle.py",
            Path("02_NEWER_FNO_DEEPONET_BENCHMARK")
            / "04_reproducibility_scripts"
            / "160_finalize_windows_deliverable_bundle.py",
        )
    )
    records.extend(additions)

    exporter_relative = (
        Path("02_NEWER_FNO_DEEPONET_BENCHMARK")
        / "04_reproducibility_scripts"
        / "159_export_all_project_deliverables.py"
    )
    exporter_destination = DEST / exporter_relative
    exporter_bytes, exporter_hash = refresh_existing(
        ROOT / "scripts/159_export_all_project_deliverables.py",
        exporter_destination,
    )
    exporter_row = next(
        row
        for row in records
        if row["destination_path"] == exporter_relative.as_posix()
    )
    exporter_row["bytes"] = exporter_bytes
    exporter_row["sha256"] = exporter_hash

    actual_file_count = len(records) + 2                                                         
    readme_text = README.read_text(encoding="utf-8")
    readme_text = re.sub(
        r"- \d+ copied or generated deliverable files",
        f"- {actual_file_count} copied or generated deliverable files",
        readme_text,
        count=1,
    )
    README.write_text(readme_text, encoding="utf-8")
    readme_relative = README.relative_to(DEST).as_posix()
    readme_row = next(row for row in records if row["destination_path"] == readme_relative)
    readme_row["bytes"] = README.stat().st_size
    readme_row["sha256"] = sha256(README)

    temporary_manifest = MANIFEST.with_suffix(".csv.tmp")
    with temporary_manifest.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(records, key=lambda row: row["destination_path"]))
    os.replace(temporary_manifest, MANIFEST)

    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    validation["manifest_sha256"] = sha256(MANIFEST)
    validation["manifest_records"] = len(records)
    validation["total_manifest_bytes"] = sum(int(row["bytes"]) for row in records)
    validation["actual_bundle_file_count"] = actual_file_count
    validation["module1_contract_and_inventory_explicitly_included"] = True
    temporary_validation = VALIDATION.with_suffix(".json.tmp")
    temporary_validation.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_validation, VALIDATION)
    print(
        json.dumps(
            {
                "passed": True,
                "added": [row["destination_path"] for row in additions],
                "manifest_records": len(records),
                "actual_bundle_file_count": actual_file_count,
                "manifest_sha256": validation["manifest_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
