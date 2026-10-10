                      
                                                        

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


SUCCESS_REPORT = ROOT / "docs" / "V1D_PATCHAWARE_FNO_SUCCESS_REPORT.md"
DIAGNOSTIC_REPORT = ROOT / "docs" / "V1D_PATCHAWARE_FNO_DIAGNOSTIC_REPORT.md"
AUDIT_JSON = NEURAL_DIR / "field_reconstruction_audit.json"
DECOMP_CSV = NEURAL_DIR / "patch_delta_decomposition.csv"
PATCHAWARE_RESULTS = NEURAL_DIR / "patchaware_fno_results.csv"
PATCHAWARE_JSON = NEURAL_DIR / "patchaware_fno_results.json"
DOC_PATH = ROOT / "docs" / "V1D_FIELD_RECONSTRUCTION_AUDIT_AND_PATCHAWARE_FNO.md"
ORIGINAL_RESULTS = NEURAL_DIR / "fno_field_results.csv"


def metric(df: pd.DataFrame, target: str, scope: str, col: str, config: str | None = None) -> float:
    sub = df[(df["split"].eq("test")) & (df["target"].eq(target)) & (df["metric_scope"].eq(scope)) & (df["variant"].eq("all"))]
    if config is not None and "config" in sub:
        sub = sub[sub["config"].eq(config)]
    if len(sub) == 0 or col not in sub:
        return math.nan
    return float(sub[col].iloc[0])


def best_config(results: pd.DataFrame) -> dict:
    rows = []
    for cfg in sorted(results["config"].dropna().unique()):
        sub = results[results["config"].eq(cfg)]
        rows.append(
            {
                "config": cfg,
                "apd90_field_r2": metric(sub, "APD90", "field", "r2"),
                "patch_dice": metric(sub, "patch_mask", "field", "dice"),
                "apd90_patch_delta_r2": metric(sub, "APD90_patch_delta", "derived_scalar", "r2"),
                "patch_area_fraction_r2": metric(sub, "patch_area_fraction", "derived_scalar", "r2"),
                "lat_range_r2": metric(sub, "LAT_range", "derived_scalar", "r2"),
            }
        )
    if not rows:
        return {}
    return max(rows, key=lambda r: np.nan_to_num(r["apd90_patch_delta_r2"], nan=-999.0) + 0.5 * np.nan_to_num(r["patch_dice"], nan=-999.0))


def write_report(path: Path, title: str, status: str, gates: list[dict], baseline: dict, best: dict, details: dict) -> None:
    passed = sum(1 for g in gates if g["passed"])
    lines = [
        f"# {title}",
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
            "## Original FNO Baseline",
            "",
            f"- APD90 field R2: `{baseline.get('apd90_field_r2', math.nan):.4f}`",
            f"- Patch Dice: `{baseline.get('patch_dice', math.nan):.4f}`",
            f"- APD90 patch-delta R2: `{baseline.get('apd90_patch_delta_r2', math.nan):.4f}`",
            f"- Patch area fraction R2: `{baseline.get('patch_area_fraction_r2', math.nan):.4f}`",
            "",
            "## Best Patch-Aware FNO",
            "",
            f"- Config: `{best.get('config', 'none')}`",
            f"- APD90 field R2: `{best.get('apd90_field_r2', math.nan):.4f}`",
            f"- Patch Dice: `{best.get('patch_dice', math.nan):.4f}`",
            f"- APD90 patch-delta R2: `{best.get('apd90_patch_delta_r2', math.nan):.4f}`",
            f"- Patch area fraction R2: `{best.get('patch_area_fraction_r2', math.nan):.4f}`",
            "",
            "## Interpretation",
            "",
        ]
    )
    if status == "PASS":
        lines.append("The audit and patch-aware training stage completed. At least one patch-aware configuration improved either patch-mask Dice or APD90 patch-delta R2 without using target fields as inputs.")
        lines.append("")
        lines.append("Next recommended step: inspect the audit figures and decide whether the patch-aware objective is strong enough to rerun the 500-sample v1d pilot at true 32x32 direct Biot-Savart resolution before scaling.")
    else:
        lines.append("The patch-aware stage did not satisfy the improvement gate or a required artifact is missing. Do not scale v1d until the listed failures are resolved.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `outputs/neural_v1d/field_reconstruction_audit.csv`",
            "- `outputs/neural_v1d/patch_delta_decomposition.csv`",
            "- `outputs/neural_v1d/patch_mask_error_summary.csv`",
            "- `outputs/neural_v1d/patchaware_fno_results.csv`",
            "- `docs/V1D_FIELD_RECONSTRUCTION_AUDIT_AND_PATCHAWARE_FNO.md`",
            "",
            f"Gates passed: {passed}/{len(gates)}.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    json_path = NEURAL_DIR / "patchaware_fno_gate_summary.json"
    write_json(json_path, {"status": status, "gates": gates, "baseline": baseline, "best": best, "details": details, "report": str(path.relative_to(ROOT))})


def main() -> int:
    gates = []
    details = {}
    rows = load_index()
    leaked = check_no_leakage(rows)
    gates.append({"name": "grouped split leakage", "passed": len(leaked) == 0, "detail": f"{len(leaked)} leaked base IDs"})
    gates.append({"name": "audit completed", "passed": AUDIT_JSON.exists() and DECOMP_CSV.exists(), "detail": f"{AUDIT_JSON.exists()=} {DECOMP_CSV.exists()=}"})
    gates.append({"name": "patch-aware results completed", "passed": PATCHAWARE_RESULTS.exists() and PATCHAWARE_JSON.exists(), "detail": f"{PATCHAWARE_RESULTS.exists()=} {PATCHAWARE_JSON.exists()=}"})
    gates.append({"name": "documentation completed", "passed": DOC_PATH.exists(), "detail": str(DOC_PATH.relative_to(ROOT))})

    baseline = {}
    best = {}
    if ORIGINAL_RESULTS.exists():
        base_df = pd.read_csv(ORIGINAL_RESULTS)
        baseline = {
            "apd90_field_r2": metric(base_df, "APD90", "field", "r2"),
            "patch_dice": metric(base_df, "patch_mask", "field", "dice"),
            "apd90_patch_delta_r2": metric(base_df, "APD90_patch_delta", "derived_scalar", "r2"),
            "patch_area_fraction_r2": metric(base_df, "patch_area_fraction", "derived_scalar", "r2"),
        }
    else:
        gates.append({"name": "original baseline exists", "passed": False, "detail": str(ORIGINAL_RESULTS.relative_to(ROOT))})
    if PATCHAWARE_RESULTS.exists():
        res_df = pd.read_csv(PATCHAWARE_RESULTS)
        best = best_config(res_df)
        patch_improved = best.get("patch_dice", -999) > baseline.get("patch_dice", 999) + 0.01
        delta_improved = best.get("apd90_patch_delta_r2", -999) > baseline.get("apd90_patch_delta_r2", 999) + 0.05
        details["patch_dice_improved_by"] = best.get("patch_dice", math.nan) - baseline.get("patch_dice", math.nan)
        details["apd90_patch_delta_r2_improved_by"] = best.get("apd90_patch_delta_r2", math.nan) - baseline.get("apd90_patch_delta_r2", math.nan)
        gates.append(
            {
                "name": "patch-aware improvement",
                "passed": bool(patch_improved or delta_improved),
                "detail": f"patch Dice delta={details['patch_dice_improved_by']:.4f}; APD90 patch-delta R2 delta={details['apd90_patch_delta_r2_improved_by']:.4f}",
            }
        )
    status = "PASS" if all(g["passed"] for g in gates) else "FAIL"
    report_path = SUCCESS_REPORT if status == "PASS" else DIAGNOSTIC_REPORT
    write_report(report_path, "V1D Patch-Aware FNO Gate", status, gates, baseline, best, details)
    print(json.dumps({"status": status, "report": str(report_path), "best": best, "gates": gates}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
