                      
                                                                           

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from v1d_field_common import CONDITION_COUNTS, EP_FILES, PILOT_DIR, ROOT, SEED, V1C_EP_MANIFEST, ensure_dirs, rel


OUT_CSV = PILOT_DIR / "manifest.csv"
OUT_JSON = PILOT_DIR / "manifest.json"


def main() -> int:
    ensure_dirs()
    manifest = pd.read_csv(V1C_EP_MANIFEST)
    manifest = manifest[manifest["success"].astype(str).str.lower().eq("true")].copy()
    selected = []
    rng_seed = SEED
    for condition, count in CONDITION_COUNTS.items():
        sub = manifest[manifest["condition"].eq(condition)].copy()
        if len(sub) < count:
            raise SystemExit(f"Need {count} {condition} samples, found {len(sub)}")
        selected.append(sub.sample(n=count, random_state=rng_seed).sort_values("sample_id"))
        rng_seed += 17
    out = pd.concat(selected, ignore_index=True).sort_values(["condition", "sample_id"]).reset_index(drop=True)
    file_keys = {
        "vm.igb": "vm_file",
        "mesh.pts": "mesh_pts_file",
        "mesh.elem": "mesh_elem_file",
        "mesh.lon": "mesh_lon_file",
        "LAT.dat": "LAT_file",
        "RT_repol_proxy.dat": "RT_repol_proxy_file",
        "APD80.dat": "APD80_file",
        "APD90.dat": "APD90_file",
        "region_mask_nodes.csv": "region_mask_nodes_file",
        "element_region_tags.csv": "element_region_tags_file",
        "metadata.json": "metadata_file",
    }
    for filename in EP_FILES:
        out[file_keys[filename]] = out["output_dir"].map(lambda d, fn=filename: rel(ROOT / d / fn))
    out = out.rename(columns={"sample_id": "base_sample_id"})
    out["field_base_id"] = out["base_sample_id"]
    ordered = [
        "field_base_id",
        "base_sample_id",
        "condition",
        "model",
        "formulation",
        "geometry",
        "apd_parameter",
        "apd_scale",
        "conductivity_scale",
        "patch_shape",
        "patch_center_x",
        "patch_center_y",
        "patch_radius",
        "patch_radius_mm",
        "stimulus_location",
        "base_gil",
        "anisotropy_factor",
        "nbeats",
        "BCL",
        "conditioning_used",
        "success",
        "output_dir",
    ]
    file_cols = [c for c in out.columns if c.endswith("_file")]
    other_cols = [c for c in out.columns if c not in ordered + file_cols]
    out = out[ordered + file_cols + other_cols]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    payload = {
        "seed": SEED,
        "source_manifest": rel(V1C_EP_MANIFEST),
        "output_manifest": rel(OUT_CSV),
        "base_sample_count": int(len(out)),
        "condition_counts": out["condition"].value_counts().to_dict(),
        "rules": [
            "base samples only, no augmented variants",
            "deterministic stratified selection",
            "v1c openCARP outputs are reused; openCARP is not rerun",
        ],
        "rows": json.loads(out.to_json(orient="records")),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ["base_sample_count", "condition_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
