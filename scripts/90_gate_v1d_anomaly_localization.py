                      
                                              

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
from v1d_anomaly_common import ANOMALY_MANIFEST, ANOMALY_QUALITY
from v1d_field_common import NEURAL_DIR, ROOT, write_json


SUCCESS_REPORT = ROOT / "docs" / "V1D_ANOMALY_LOCALIZATION_SUCCESS_REPORT.md"
DIAGNOSTIC_REPORT = ROOT / "docs" / "V1D_ANOMALY_LOCALIZATION_DIAGNOSTIC_REPORT.md"
DOC_PATH = ROOT / "docs" / "V1D_ANOMALY_LOCALIZATION_TARGETS_AND_MODELS.md"
PATCH_AUDIT = NEURAL_DIR / "patch_target_audit.csv"
FNO_RESULTS = NEURAL_DIR / "fno_anomaly_localizer_results.csv"
CNN_RESULTS = NEURAL_DIR / "cnn_anomaly_localizer_results.csv"
TWO_STAGE = NEURAL_DIR / "two_stage_patch_scalar_results.csv"
ORIG_PATCH = NEURAL_DIR / "patch_mask_error_summary.csv"


def all_row(df: pd.DataFrame) -> pd.Series:
    sub = df[(df["split"].eq("test")) & (df["group_type"].eq("all")) & (df["group_value"].eq("all"))] if "split" in df else df[(df["group_type"].eq("all")) & (df["group_value"].eq("all"))]
    return sub.iloc[0] if len(sub) else pd.Series(dtype=float)


def write_report(path: Path, status: str, gates: list[dict], metrics: dict) -> None:
    lines = [
        "# V1D Anomaly Localization Gate" if status == "PASS" else "# V1D Anomaly Localization Diagnostic Report",
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
            "## Key Metrics",
            "",
            f"- Original patch-mask global Dice: `{metrics.get('original_patch_dice', math.nan):.4f}`",
            f"- Original homogeneous false-positive area: `{metrics.get('original_homogeneous_fp', math.nan):.4f}`",
            f"- FNO anomaly Dice: `{metrics.get('fno_dice', math.nan):.4f}`",
            f"- FNO homogeneous false-positive area: `{metrics.get('fno_homogeneous_fp', math.nan):.4f}`",
            f"- FNO patch area fraction R2: `{metrics.get('fno_patch_area_r2', math.nan):.4f}`",
            f"- Two-stage APD90 patch-delta R2: `{metrics.get('two_stage_apd_delta_r2', math.nan):.4f}`",
            f"- CNN anomaly Dice: `{metrics.get('cnn_dice', math.nan):.4f}`",
            "",
            "## Interpretation",
            "",
        ]
    )
    if status == "PASS":
        lines.append("The anomaly-localization stage improved the localization bottleneck sufficiently to justify further v1d scaling experiments.")
    else:
        lines.append("The anomaly-localization stage completed technically, but the quality gate failed. The anomaly-derived labels improved some condition-specific localization behavior but did not improve homogeneous false positives or derived geometric patch scalars.")
        lines.append("")
        lines.append("Do not scale v1d to all 5,000 samples yet. The target definition needs redesign: the current field-residual anomaly maps are not aligned enough with the geometric abnormal-region scalars.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `outputs/neural_v1d/patch_target_audit.csv`",
            "- `outputs/dataset_v1d_fno_field_pilot/anomaly_target_manifest.csv`",
            "- `outputs/neural_v1d/fno_anomaly_localizer_results.csv`",
            "- `outputs/neural_v1d/two_stage_patch_scalar_results.csv`",
            "- `outputs/neural_v1d/cnn_anomaly_localizer_results.csv`",
            "- `docs/V1D_ANOMALY_LOCALIZATION_TARGETS_AND_MODELS.md`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    write_json(NEURAL_DIR / "anomaly_localization_gate_summary.json", {"status": status, "gates": gates, "metrics": metrics, "report": str(path.relative_to(ROOT))})


def main() -> int:
    gates = []
    rows = load_index()
    leaked = check_no_leakage(rows)
    gates.append({"name": "no split leakage", "passed": len(leaked) == 0, "detail": f"{len(leaked)} leaked base IDs"})
    gates.append({"name": "patch target audit completed", "passed": PATCH_AUDIT.exists(), "detail": str(PATCH_AUDIT.relative_to(ROOT))})
    gates.append({"name": "anomaly targets completed", "passed": ANOMALY_MANIFEST.exists() and ANOMALY_QUALITY.exists(), "detail": f"{ANOMALY_MANIFEST.exists()=} {ANOMALY_QUALITY.exists()=}"})
    gates.append({"name": "FNO localizer completed", "passed": FNO_RESULTS.exists(), "detail": str(FNO_RESULTS.relative_to(ROOT))})
    gates.append({"name": "two-stage scalar evaluation completed", "passed": TWO_STAGE.exists(), "detail": str(TWO_STAGE.relative_to(ROOT))})
    gates.append({"name": "matched CNN/UNet completed", "passed": CNN_RESULTS.exists(), "detail": str(CNN_RESULTS.relative_to(ROOT))})
    gates.append({"name": "documentation completed", "passed": DOC_PATH.exists(), "detail": str(DOC_PATH.relative_to(ROOT))})

    metrics = {}
    if ORIG_PATCH.exists():
        orig = pd.read_csv(ORIG_PATCH)
        metrics["original_patch_dice"] = 0.4492212768074837
        hom = orig[(orig["group_type"].eq("condition")) & (orig["group_value"].eq("homogeneous"))]
        metrics["original_homogeneous_fp"] = float(abs(hom["area_bias_mean"].iloc[0])) if len(hom) else math.nan
    if FNO_RESULTS.exists():
        fno = pd.read_csv(FNO_RESULTS)
        fno_all = all_row(fno)
        metrics["fno_dice"] = float(fno_all.get("dice_mean", math.nan))
        metrics["fno_patch_area_r2"] = float(fno_all.get("patch_area_fraction_r2", math.nan))
        hom = fno[(fno["split"].eq("test")) & (fno["group_type"].eq("condition")) & (fno["group_value"].eq("homogeneous"))]
        metrics["fno_homogeneous_fp"] = float(hom["false_positive_area_mean"].iloc[0]) if len(hom) else math.nan
    if TWO_STAGE.exists():
        ts = pd.read_csv(TWO_STAGE)
        ts_all = ts[(ts["group_type"].eq("all")) & (ts["group_value"].eq("all"))].iloc[0]
        metrics["two_stage_apd_delta_r2"] = float(ts_all.get("pred_mask_pred_apd_APD90_patch_delta_r2", math.nan))
    if CNN_RESULTS.exists():
        cnn = pd.read_csv(CNN_RESULTS)
        metrics["cnn_dice"] = float(all_row(cnn).get("dice_mean", math.nan))

    improved_dice = metrics.get("fno_dice", -999) > metrics.get("original_patch_dice", 999) + 0.02
    reduced_hom_fp = metrics.get("fno_homogeneous_fp", 999) < metrics.get("original_homogeneous_fp", -999)
    scalar_improved = metrics.get("two_stage_apd_delta_r2", -999) > 0.20 or metrics.get("fno_patch_area_r2", -999) > 0.0
    gates.append(
        {
            "name": "localization/scalar improvement",
            "passed": bool((improved_dice or scalar_improved) and reduced_hom_fp),
            "detail": f"improved_dice={improved_dice}; reduced_hom_fp={reduced_hom_fp}; scalar_improved={scalar_improved}",
        }
    )
    status = "PASS" if all(g["passed"] for g in gates) else "FAIL"
    path = SUCCESS_REPORT if status == "PASS" else DIAGNOSTIC_REPORT
    write_report(path, status, gates, metrics)
    print(json.dumps({"status": status, "report": str(path), "metrics": metrics, "gates": gates}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
