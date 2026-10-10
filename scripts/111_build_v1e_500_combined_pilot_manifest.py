                      
                                                                          

from __future__ import annotations

import json

import pandas as pd

from v1e_500_common import COUNTS, EP_FILES, PILOT_DIR, ROOT, SEED, V1C_EP_MANIFEST, ensure_dirs, rel


OUT_CSV = PILOT_DIR / "manifest.csv"
OUT_JSON = PILOT_DIR / "manifest.json"


def main() -> int:
    ensure_dirs()
    src = pd.read_csv(V1C_EP_MANIFEST)
    src = src[src["success"].astype(str).str.lower().eq("true")].copy()
    selected = []
    rng_seed = SEED
    for condition, count in COUNTS.items():
        sub = src[src["condition"].eq(condition)].copy()
        if len(sub) < count:
            raise SystemExit(f"Need {count} {condition}, found {len(sub)}")
        selected.append(sub.sample(n=count, random_state=rng_seed).sort_values("sample_id"))
        rng_seed += 17
    out = pd.concat(selected, ignore_index=True).sort_values(["condition", "sample_id"]).reset_index(drop=True)
    for filename, col in EP_FILES.items():
        out[col] = out["output_dir"].map(lambda d, fn=filename: rel(ROOT / d / fn))
    out = out.rename(columns={"sample_id": "base_sample_id"})
    ordered = [
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
    file_cols = list(EP_FILES.values())
    other = [c for c in out.columns if c not in ordered + file_cols]
    out = out[ordered + file_cols + other]
    out.to_csv(OUT_CSV, index=False)
    payload = {
        "seed": SEED,
        "source_manifest": rel(V1C_EP_MANIFEST),
        "output_manifest": rel(OUT_CSV),
        "base_sample_count": int(len(out)),
        "condition_counts": out["condition"].value_counts().to_dict(),
        "rules": ["base EP samples only", "deterministic stratified selection", "openCARP is not rerun"],
        "rows": json.loads(out.to_json(orient="records")),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"base_sample_count": len(out), "condition_counts": payload["condition_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
