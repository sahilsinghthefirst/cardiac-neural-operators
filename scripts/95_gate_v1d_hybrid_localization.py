                      
                                                  

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if (PROJECT_ROOT / ".codex_deps" / "sklearn").exists():
    sys.path.append(str(PROJECT_ROOT / ".codex_deps" / "sklearn"))

import numpy as np
import pandas as pd

from fno_field_data_common import check_no_leakage, load_index
from v1d_field_common import NEURAL_DIR, ROOT, write_json
from v1d_hybrid_common import HYBRID_MANIFEST, HYBRID_QUALITY


SUCCESS_REPORT = ROOT / "docs" / "V1D_HYBRID_LOCALIZATION_SUCCESS_REPORT.md"
DIAG_REPORT = ROOT / "docs" / "V1D_HYBRID_LOCALIZATION_DIAGNOSTIC_REPORT.md"
DOC_PATH = ROOT / "docs" / "V1D_HYBRID_LOCALIZATION_FNO.md"
FNO_RESULTS = NEURAL_DIR / "fno_hybrid_localizer_results.csv"
CNN_RESULTS = NEURAL_DIR / "cnn_hybrid_localizer_results.csv"
COMPARISON = NEURAL_DIR / "hybrid_localization_comparison.csv"

ORIG_DICE = 0.4492212768074837
ORIG_HOM_FP = 0.0510602678571428
PATCHAWARE_APD_R2 = 0.1756849286354094
PATCHAWARE_AREA_R2 = -0.10382517036987693


def best_fno() -> dict:
    if not FNO_RESULTS.exists():
        return {}
    df = pd.read_csv(FNO_RESULTS)
    sub = df[(df["split"].eq("test")) & (df["group_type"].eq("all")) & (df["group_value"].eq("all"))].copy()
    if len(sub) == 0:
        return {}
    sub["score"] = sub["dice_mean"].fillna(-999) + 0.25 * sub["APD90_delta_pred_mask_pred_apd_r2"].fillna(-999).clip(lower=-5)
    row = sub.sort_values("score", ascending=False).iloc[0]
    return row.to_dict()


def write_report(path: Path, status: str, gates: list[dict], best: dict, cnn: dict) -> None:
    lines = [
        "# V1D Hybrid Localization Success Report" if status == "PASS" else "# V1D Hybrid Localization Diagnostic Report",
        "",
        f"Status: **{status}**",
        "",
        "## Gate Summary",
        "",
        "| Gate | Pass | Detail |",
        "|---|---:|---|",
    ]
    for gate in gates:
        lines.append(f"| {gate['name']} | {'yes' if gate['passed'] else 'no'} | {gate['detail']} |")
    lines.extend(
        [
            "",
            "## Best FNO Hybrid Localizer",
            "",
            f"- Config: `{best.get('config', 'none')}`",
            f"- Geometric Dice: `{best.get('dice_mean', math.nan):.4f}`",
            f"- Homogeneous false-positive area: `{best.get('homogeneous_false_positive_area', math.nan):.4f}`",
            f"- APD90 patch-delta R2: `{best.get('APD90_delta_pred_mask_pred_apd_r2', math.nan):.4f}`",
            f"- Patch area fraction R2: `{best.get('patch_area_fraction_r2', math.nan):.4f}`",
            "",
            "## Matched CNN/UNet",
            "",
            f"- Geometric Dice: `{cnn.get('dice_mean', math.nan):.4f}`",
            f"- Homogeneous false-positive area: `{cnn.get('homogeneous_false_positive_area', math.nan):.4f}`",
            f"- APD90 patch-delta R2: `{cnn.get('APD90_delta_pred_mask_pred_apd_r2', math.nan):.4f}`",
            "",
            "## Interpretation",
            "",
        ]
    )
    if status == "PASS":
        lines.append("The hybrid localization experiment met the quality gate. It is a better candidate for a larger v1d localization pilot than anomaly-only labels.")
    else:
        lines.append("The hybrid localization experiment completed, but did not meet the quality gate. Do not scale v1d yet; inspect the failure modes and tune the hybrid objective or target construction.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `outputs/dataset_v1d_fno_field_pilot/hybrid_target_manifest.csv`",
            "- `outputs/neural_v1d/fno_hybrid_localizer_results.csv`",
            "- `outputs/neural_v1d/cnn_hybrid_localizer_results.csv`",
            "- `outputs/neural_v1d/hybrid_localization_comparison.csv`",
            "- `docs/V1D_HYBRID_LOCALIZATION_FNO.md`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    write_json(NEURAL_DIR / "hybrid_localization_gate_summary.json", {"status": status, "gates": gates, "best_fno": best, "cnn": cnn, "report": str(path.relative_to(ROOT))})


def main() -> int:
    gates = []
    leaked = check_no_leakage(load_index())
    gates.append({"name": "no split leakage", "passed": len(leaked) == 0, "detail": f"{len(leaked)} leaked base IDs"})
    gates.append({"name": "hybrid targets built", "passed": HYBRID_MANIFEST.exists() and HYBRID_QUALITY.exists(), "detail": f"{HYBRID_MANIFEST.exists()=} {HYBRID_QUALITY.exists()=}"})
    gates.append({"name": "FNO hybrid trained", "passed": FNO_RESULTS.exists(), "detail": str(FNO_RESULTS.relative_to(ROOT))})
    gates.append({"name": "matched CNN/UNet trained", "passed": CNN_RESULTS.exists(), "detail": str(CNN_RESULTS.relative_to(ROOT))})
    gates.append({"name": "evaluation completed", "passed": COMPARISON.exists(), "detail": str(COMPARISON.relative_to(ROOT))})
    gates.append({"name": "documentation completed", "passed": DOC_PATH.exists(), "detail": str(DOC_PATH.relative_to(ROOT))})
    best = best_fno()
    cnn = {}
    if CNN_RESULTS.exists():
        cdf = pd.read_csv(CNN_RESULTS)
        sub = cdf[(cdf["split"].eq("test")) & (cdf["group_type"].eq("all")) & (cdf["group_value"].eq("all"))]
        if len(sub):
            cnn = sub.iloc[0].to_dict()
    dice_improved = best.get("dice_mean", -999) > ORIG_DICE + 0.02
    hom_fp_improved = best.get("homogeneous_false_positive_area", 999) < ORIG_HOM_FP - 0.01
    apd_improved = best.get("APD90_delta_pred_mask_pred_apd_r2", -999) > PATCHAWARE_APD_R2 + 0.05
    area_improved = best.get("patch_area_fraction_r2", -999) > PATCHAWARE_AREA_R2 + 0.05
    gates.append(
        {
            "name": "quality improvement",
            "passed": bool((dice_improved or hom_fp_improved) and (apd_improved or area_improved)),
            "detail": f"dice_improved={dice_improved}; hom_fp_improved={hom_fp_improved}; apd_improved={apd_improved}; area_improved={area_improved}",
        }
    )
    status = "PASS" if all(g["passed"] for g in gates) else "FAIL"
    report = SUCCESS_REPORT if status == "PASS" else DIAG_REPORT
    write_report(report, status, gates, best, cnn)
    print(json.dumps({"status": status, "report": str(report), "best_fno": best, "cnn": cnn, "gates": gates}, indent=2, default=str))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
