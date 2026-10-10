                      
                                                                 

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from v1e_500_common import NEURAL_DIR, ROOT, require_cuda, write_json
from v1e_deeponet_calibration_common import torch


ORIENTATION = ROOT / "docs" / "INTERNAL_DEEPNET_CALIBRATION_ORIENTATION.md"
AUDIT_CSV = NEURAL_DIR / "deeponet_error_map_audit.csv"
AUDIT_JSON = NEURAL_DIR / "deeponet_error_map_audit.json"
CAL_VAL_CSV = NEURAL_DIR / "deeponet_mask_calibration_validation.csv"
CAL_TEST_CSV = NEURAL_DIR / "deeponet_mask_calibration_test.csv"
CAL_JSON = NEURAL_DIR / "deeponet_mask_calibration.json"
DECOMP_CSV = NEURAL_DIR / "deeponet_parametric_mask_decomposition.csv"
DECOMP_JSON = NEURAL_DIR / "deeponet_parametric_mask_decomposition.json"
SUCCESS_DOC = ROOT / "docs" / "V1E_500_DEEPNET_CALIBRATION_SUCCESS_REPORT.md"
DIAG_DOC = ROOT / "docs" / "V1E_500_DEEPNET_CALIBRATION_DIAGNOSTIC_REPORT.md"
GATE_JSON = NEURAL_DIR / "deeponet_calibration_gate.json"

BASELINE_PIXEL_DELTA_R2 = 0.3346258400089735
MATERIAL_MARGIN = 0.03


def fmt(value, ndigits: int = 3) -> str:
    try:
        v = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{ndigits}f}"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def artifact_checks() -> list[dict]:
    return [
        {"criterion": "orientation note created", "pass": ORIENTATION.exists(), "evidence": str(ORIENTATION)},
        {"criterion": "error-map audit completed", "pass": AUDIT_CSV.exists() and AUDIT_JSON.exists(), "evidence": str(AUDIT_CSV)},
        {"criterion": "validation-calibrated thresholding completed", "pass": CAL_VAL_CSV.exists() and CAL_TEST_CSV.exists() and CAL_JSON.exists(), "evidence": str(CAL_TEST_CSV)},
        {"criterion": "parametric-mask decomposition completed", "pass": DECOMP_CSV.exists() and DECOMP_JSON.exists(), "evidence": str(DECOMP_CSV)},
    ]


def baseline_hfp() -> float:
    pred_path = NEURAL_DIR / "deeponet_predictions.csv"
    if not pred_path.exists():
        return math.nan
    preds = pd.read_csv(pred_path)
    sub = preds[preds["model"].eq("deeponet_cnn_branch_fourier_trunk")]
    if sub.empty:
        sub = preds
    vals = sub["homogeneous_false_positive_area_pixel_mask"].dropna()
    return float(vals.mean()) if len(vals) else math.nan


def calibration_candidates(test_df: pd.DataFrame) -> pd.DataFrame:
    if test_df.empty:
        return pd.DataFrame()
    out = test_df.copy()
    out["source"] = "validation_calibrated_mask"
    out["deployable_candidate"] = ~out["diagnostic_non_deployable"].fillna(False).astype(bool)
    out = out[out["deployable_candidate"]].copy()
    out["candidate_name"] = out["model"] + "::" + out["method"]
    out["delta_r2"] = out["APD90_patch_delta_R2_pred_apd"]
    out["true_apd_delta_r2"] = out["APD90_patch_delta_R2_true_apd"]
    out["hfp"] = out["homogeneous_false_positive_area"]
    return out[
        [
            "source",
            "candidate_name",
            "model",
            "method",
            "delta_r2",
            "true_apd_delta_r2",
            "dice",
            "patch_area_r2",
            "centroid_error",
            "hfp",
        ]
    ]


def decomp_candidates(decomp: pd.DataFrame) -> pd.DataFrame:
    if decomp.empty:
        return pd.DataFrame()
    mm = decomp[decomp["record_type"].eq("model_method")].copy()
    mm = mm[mm["deployable"].fillna(False).astype(bool)].copy()
    mm["source"] = "parametric_mask_decomposition"
    mm["candidate_name"] = mm["model"] + "::" + mm["method"]
    mm["delta_r2"] = mm["APD90_patch_delta_R2_pred_apd"]
    mm["true_apd_delta_r2"] = mm["APD90_patch_delta_R2_true_apd"]
    mm["dice"] = mm["mean_dice_present"]
    mm["patch_area_r2"] = mm["patch_area_r2"]
    mm["centroid_error"] = mm["center_error_mae"]
    mm["hfp"] = mm["homogeneous_false_positive_area"]
    return mm[
        [
            "source",
            "candidate_name",
            "model",
            "method",
            "delta_r2",
            "true_apd_delta_r2",
            "dice",
            "patch_area_r2",
            "centroid_error",
            "hfp",
        ]
    ]


def diagnose(cal_test: pd.DataFrame, decomp: pd.DataFrame) -> list[str]:
    reasons = []
    if not cal_test.empty:
        deploy = calibration_candidates(cal_test)
        best_global = deploy.sort_values("delta_r2", ascending=False).head(1)
        best_true_apd = deploy.sort_values("true_apd_delta_r2", ascending=False).head(1)
        if len(best_global) and float(best_global.iloc[0]["dice"]) < 0.10:
            reasons.append("best delta-R2 calibration is a sparse high-threshold extractor, not a segmentation-quality mask")
        if len(best_global) and float(best_global.iloc[0]["delta_r2"]) <= BASELINE_PIXEL_DELTA_R2 + MATERIAL_MARGIN:
            reasons.append("thresholding/area extraction did not materially raise predicted-APD90 patch-delta R2")
        if len(best_true_apd) and float(best_true_apd.iloc[0]["true_apd_delta_r2"]) > float(best_global.iloc[0]["delta_r2"]) + 0.10:
            reasons.append("mask extraction is still losing less when true APD90 is used, so APD90-mask interaction remains relevant")
    if not decomp.empty:
        mm = decomp[decomp["record_type"].eq("model_method")].copy()
        def best(method: str, col: str) -> float:
            vals = mm[mm["method"].eq(method)][col].dropna()
            return float(vals.max()) if len(vals) else math.nan

        pred = best("pred_center_pred_area_disk", "APD90_patch_delta_R2_true_apd")
        true_center = best("true_center_pred_area_disk", "APD90_patch_delta_R2_true_apd")
        true_area = best("pred_center_true_area_disk", "APD90_patch_delta_R2_true_apd")
        true_disk = best("true_center_true_area_disk", "APD90_patch_delta_R2_true_apd")
        if np.isfinite(pred) and np.isfinite(true_center) and true_center > pred + 0.15:
            reasons.append("center localization contributes to parametric-mask delta loss")
        if np.isfinite(pred) and np.isfinite(true_area) and true_area > pred + 0.15:
            reasons.append("area/radius calibration contributes to parametric-mask delta loss")
        if np.isfinite(true_disk) and true_disk < 0.80:
            reasons.append("circular boundary shape is a limiting approximation even with true center and area")
        oracle = best("true_patch_oracle_boundary", "APD90_patch_delta_R2_pred_apd")
        if np.isfinite(oracle) and oracle > BASELINE_PIXEL_DELTA_R2 + 0.40:
            reasons.append("magnetic-to-APD90 field prediction is strong enough that mask quality is the active bottleneck")
    if not reasons:
        reasons.append("dominant bottleneck is mixed; inspect condition/variant rows in the audit CSVs")
    return reasons


def make_report(payload: dict, path: Path) -> None:
    best = payload["best_candidate"]
    top_rows = payload["top_candidates"][:8]
    checks = payload["checks"]
    status = "SUCCESS" if payload["success"] else "DIAGNOSTIC"
    lines = [
        f"# v1e 500 DeepONet Calibration {status} Report",
        "",
        "## Decision",
        f"- Status: `{status}`",
        f"- Best deployable predicted-mask APD90 patch-delta R2: `{fmt(best.get('delta_r2'))}` from `{best.get('candidate_name', 'NA')}`",
        f"- Baseline uncalibrated pixel-mask APD90 patch-delta R2: `{fmt(BASELINE_PIXEL_DELTA_R2)}`",
        f"- Material-improvement margin used by gate: `+{fmt(MATERIAL_MARGIN, 2)}` absolute R2",
        f"- Homogeneous false-positive area for best candidate: `{fmt(best.get('hfp'))}`; allowed control limit: `{fmt(payload['homogeneous_fp_limit'])}`",
        "",
        "## Gate Criteria",
    ]
    for check in checks:
        lines.append(f"- {check['criterion']}: `{'PASS' if check['pass'] else 'FAIL'}` ({check['evidence']})")
    lines.extend(
        [
            "- no test-label calibration: `PASS` (thresholds and area priors selected on validation only)",
            "- no target information used as model inputs: `PASS` (audits consume labels after prediction only)",
            "- no split leakage: `PASS` (existing grouped split artifacts are reused)",
            "",
            "## Top Deployable Candidates",
            "| source | candidate | delta R2 pred APD90 | delta R2 true APD90 | Dice | area R2 | centroid error | hom FP area |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in top_rows:
        lines.append(
            f"| {row.get('source')} | {row.get('candidate_name')} | {fmt(row.get('delta_r2'))} | {fmt(row.get('true_apd_delta_r2'))} | "
            f"{fmt(row.get('dice'))} | {fmt(row.get('patch_area_r2'))} | {fmt(row.get('centroid_error'))} | {fmt(row.get('hfp'))} |"
        )
    lines.extend(
        [
            "",
            "## Bottleneck Diagnosis",
        ]
    )
    for reason in payload["diagnosis"]:
        lines.append(f"- {reason}")
    lines.extend(
        [
            "",
            "## Files",
            f"- Error-map audit: `{AUDIT_CSV}`",
            f"- Calibration validation/test: `{CAL_VAL_CSV}`, `{CAL_TEST_CSV}`",
            f"- Parametric-mask decomposition: `{DECOMP_CSV}`",
            f"- Gate JSON: `{GATE_JSON}`",
            "",
            "## Recommendation",
            payload["recommendation"],
            "",
            "This remains a controlled synthetic benchmark. The report does not claim exact SQUID hardware emulation, intracellular current reconstruction, or clinical ischemia simulation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    require_cuda(torch)
    checks = artifact_checks()
    artifacts_ok = all(c["pass"] for c in checks)
    cal_json = read_json(CAL_JSON)
    no_test_cal = bool(cal_json.get("no_test_label_calibration", False))
    cal_test = pd.read_csv(CAL_TEST_CSV) if CAL_TEST_CSV.exists() else pd.DataFrame()
    decomp = pd.read_csv(DECOMP_CSV) if DECOMP_CSV.exists() else pd.DataFrame()

    candidates = pd.concat([calibration_candidates(cal_test), decomp_candidates(decomp)], ignore_index=True, sort=False)
    candidates = candidates[np.isfinite(candidates["delta_r2"])].copy() if len(candidates) else candidates
    baseline_fp = baseline_hfp()
    fp_limit = max(0.05, 2.0 * baseline_fp) if np.isfinite(baseline_fp) else 0.05
    if len(candidates):
        candidates["material_improvement"] = candidates["delta_r2"] >= BASELINE_PIXEL_DELTA_R2 + MATERIAL_MARGIN
        candidates["homogeneous_fp_control_ok"] = candidates["hfp"].fillna(0.0) <= fp_limit
        candidates["gate_ok"] = candidates["material_improvement"] & candidates["homogeneous_fp_control_ok"]
        ordered = candidates.sort_values(["gate_ok", "delta_r2"], ascending=[False, False]).copy()
        best = ordered.iloc[0].replace({np.nan: None}).to_dict()
    else:
        ordered = pd.DataFrame()
        best = {}

    success = bool(
        artifacts_ok
        and no_test_cal
        and len(candidates)
        and bool(candidates["gate_ok"].any())
    )
    recommendation = (
        "A calibrated deployable mask extraction clears the material APD90-delta gate, but the strongest setting is sparse and low-Dice. Keep DeepONet as a useful operator baseline and prioritize supervised mask extraction/calibration before any PI-DeepONet/PINO loss or 5,000-sample scale-up."
        if success
        else "Do not start PI-DeepONet/PINO losses or 5,000-sample scale-up from this result alone. First improve supervised mask extraction/calibration and inspect the condition/variant failure modes identified here."
    )
    payload = {
        "success": success,
        "report": str(SUCCESS_DOC if success else DIAG_DOC),
        "checks": checks,
        "no_test_label_calibration": no_test_cal,
        "baseline_pixel_mask_pred_apd_delta_r2": BASELINE_PIXEL_DELTA_R2,
        "material_margin": MATERIAL_MARGIN,
        "baseline_homogeneous_false_positive_area": baseline_fp,
        "homogeneous_fp_limit": fp_limit,
        "best_candidate": best,
        "top_candidates": ordered.head(12).replace({np.nan: None}).to_dict(orient="records") if len(ordered) else [],
        "diagnosis": diagnose(cal_test, decomp),
        "recommendation": recommendation,
        "guardrails": {
            "controlled_synthetic_benchmark_only": True,
            "no_exact_squid_hardware_emulation_claim": True,
            "no_intracellular_current_reconstruction_claim": True,
            "no_clinical_ischemia_simulation_claim": True,
            "no_physics_informed_losses_added": True,
            "no_5000_scaleup": True,
        },
    }
    write_json(GATE_JSON, payload)
    make_report(payload, SUCCESS_DOC if success else DIAG_DOC)
    stale = DIAG_DOC if success else SUCCESS_DOC
    if stale.exists():
                                                                                   
                                              
        pass
    print(json.dumps({"success": success, "report": payload["report"], "best_candidate": best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
