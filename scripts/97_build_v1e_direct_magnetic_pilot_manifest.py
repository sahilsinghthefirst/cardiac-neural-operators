                      
                                                                 

from __future__ import annotations

import argparse
import json

import pandas as pd

from v1e_direct_common import DEFAULT_COUNTS, EP_FILES, PILOT_DIR, ROOT, SEED, V1C_EP_MANIFEST, ensure_dirs, proportional_counts, rel


OUT_CSV = PILOT_DIR / "manifest.csv"
OUT_JSON = PILOT_DIR / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=100, help="100 by default; 50 is supported for fast fallback.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    ensure_dirs()
    counts = proportional_counts(args.n_samples)
    manifest = pd.read_csv(V1C_EP_MANIFEST)
    manifest = manifest[manifest["success"].astype(str).str.lower().eq("true")].copy()
    selected = []
    rng_seed = args.seed
    for condition, count in counts.items():
        sub = manifest[manifest["condition"].eq(condition)].copy()
        if len(sub) < count:
            raise SystemExit(f"Need {count} {condition} samples, found {len(sub)}")
        selected.append(sub.sample(n=count, random_state=rng_seed).sort_values("sample_id"))
        rng_seed += 17
    out = pd.concat(selected, ignore_index=True).sort_values(["condition", "sample_id"]).reset_index(drop=True)
    for filename, key in EP_FILES.items():
        out[key] = out["output_dir"].map(lambda d, fn=filename: rel(ROOT / d / fn))
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
    other_cols = [c for c in out.columns if c not in ordered + file_cols]
    out = out[ordered + file_cols + other_cols]
    out.to_csv(OUT_CSV, index=False)
    payload = {
        "seed": args.seed,
        "requested_n_samples": args.n_samples,
        "condition_counts_requested": counts,
        "source_manifest": rel(V1C_EP_MANIFEST),
        "output_manifest": rel(OUT_CSV),
        "base_sample_count": int(len(out)),
        "condition_counts": out["condition"].value_counts().to_dict(),
        "rules": [
            "base EP samples only; no augmented rows",
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
