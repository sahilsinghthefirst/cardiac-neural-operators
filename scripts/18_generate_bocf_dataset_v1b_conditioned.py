                      
                                                      

                                                                           
                                                                           
                                           
   

from __future__ import annotations

import argparse
import csv
import json

from bocf_dataset_common import ROOT, run_conditioned_ep_sample, write_csv, write_json


SOURCE_BASE = ROOT / "outputs" / "bocf_dataset_v1"
DATASET_BASE = ROOT / "outputs" / "bocf_dataset_v1b_conditioned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nbeats", type=int, default=20)
    parser.add_argument("--bcl-ms", type=float, default=1000.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def manifest_fields() -> list[str]:
    return [
        "sample_id",
        "source_v1_sample_id",
        "condition",
        "model",
        "formulation",
        "geometry",
        "apd_parameter",
        "apd_scale",
        "conductivity_scale",
        "patch_center_x",
        "patch_center_y",
        "patch_radius",
        "stimulus_location",
        "nbeats",
        "BCL",
        "final_beat_extraction",
        "success",
        "output_dir",
        "notes",
    ]


def row_from_metadata(metadata: dict, out_dir, source_id: str) -> dict:
    return {
        "sample_id": metadata.get("sample_id"),
        "source_v1_sample_id": source_id,
        "condition": metadata.get("condition"),
        "model": metadata.get("model"),
        "formulation": metadata.get("formulation"),
        "geometry": metadata.get("geometry"),
        "apd_parameter": metadata.get("apd_parameter"),
        "apd_scale": metadata.get("apd_scale"),
        "conductivity_scale": metadata.get("conductivity_scale"),
        "patch_center_x": metadata.get("patch_center_x"),
        "patch_center_y": metadata.get("patch_center_y"),
        "patch_radius": metadata.get("patch_radius"),
        "stimulus_location": metadata.get("stimulus_location"),
        "nbeats": metadata.get("nbeats"),
        "BCL": metadata.get("BCL"),
        "final_beat_extraction": metadata.get("final_beat_extraction"),
        "success": metadata.get("success", False),
        "output_dir": str(out_dir.relative_to(ROOT)),
        "notes": metadata.get("notes", ""),
    }


def main() -> int:
    args = parse_args()
    source_manifest = SOURCE_BASE / "manifest.csv"
    if not source_manifest.exists():
        raise FileNotFoundError(f"Missing source v1 manifest: {source_manifest}")

    rows: list[dict] = []
    DATASET_BASE.mkdir(parents=True, exist_ok=True)
    for index, src_row in enumerate(csv.DictReader(source_manifest.open()), start=1):
        source_dir = ROOT / src_row["output_dir"]
        source_meta = json.loads((source_dir / "metadata.json").read_text())
        source_lats = source_dir / "LAT.dat"
        sample_id = f"bocf_v1b_{index:04d}"
        out_dir = DATASET_BASE / "samples" / sample_id
        patch_center = None
        patch_radius = None
        if source_meta.get("patch_center_x") is not None and source_meta.get("patch_radius") is not None:
            patch_center = (float(source_meta["patch_center_x"]), float(source_meta["patch_center_y"]))
            patch_radius = float(source_meta["patch_radius"])
        metadata = run_conditioned_ep_sample(
            out_dir=out_dir,
            sample_id=sample_id,
            condition=source_meta["condition"],
            prepacing_lats=source_lats,
            patch_center=patch_center,
            patch_radius=patch_radius,
            apd_parameter=source_meta.get("apd_parameter"),
            apd_scale=source_meta.get("apd_scale"),
            conductivity_scale=source_meta.get("conductivity_scale"),
            stimulus_location=source_meta.get("stimulus_location", "left"),
            base_gil=float(source_meta.get("base_gil", 1.0)),
            anisotropy_factor=float(source_meta.get("anisotropy_factor", 4.0)),
            nbeats=args.nbeats,
            bcl_ms=args.bcl_ms,
            seed=source_meta.get("seed"),
            force=args.force,
        )
        metadata["source_v1_sample_id"] = source_meta.get("sample_id")
        metadata["source_v1_output_dir"] = str(source_dir.relative_to(ROOT))
        metadata["conditioning_dataset_version"] = "v1b_conditioned"
        write_json(out_dir / "metadata.json", metadata)
        rows.append(row_from_metadata(metadata, out_dir, source_meta.get("sample_id", src_row["sample_id"])))
        print(f"{sample_id}: source={source_meta.get('sample_id')} condition={source_meta['condition']} success={metadata.get('success')}")

    write_csv(DATASET_BASE / "manifest.csv", rows, manifest_fields())
    print(f"Wrote {DATASET_BASE / 'manifest.csv'}")
    return 0 if len(rows) == 70 and all(row["success"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
