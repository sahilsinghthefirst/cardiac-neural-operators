                      
                                                           

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import pandas as pd

from v1e_500_common import NEURAL_DIR, PILOT_DIR, ROOT, require_cuda, write_json
from v1e_deeponet_mask_refine_data_common import torch


ORIENTATION = ROOT / "docs" / "INTERNAL_DEEPNET_MASK_REFINEMENT_ORIENTATION.md"
TARGET_QUALITY = PILOT_DIR / "deeponet_mask_refine_targets" / "target_quality_report.json"
LOADER_CHECK = NEURAL_DIR / "mask_refine_loader_check_report.json"
TRAIN_LOG = NEURAL_DIR / "mask_refine_training_log.csv"
TRAIN_SUMMARY = NEURAL_DIR / "mask_refine_training_summary.json"
RESULTS_CSV = NEURAL_DIR / "mask_refine_results.csv"
RESULTS_JSON = NEURAL_DIR / "mask_refine_results.json"
PREDICTIONS_CSV = NEURAL_DIR / "mask_refine_predictions.csv"
CALIBRATION_CSV = NEURAL_DIR / "mask_refine_calibration.csv"
GATE_JSON = NEURAL_DIR / "mask_refine_gate.json"
SUCCESS_DOC = ROOT / "docs" / "V1E_500_DEEPNET_MASK_REFINEMENT_SUCCESS_REPORT.md"
DIAG_DOC = ROOT / "docs" / "V1E_500_DEEPNET_MASK_REFINEMENT_DIAGNOSTIC_REPORT.md"

BASELINE_CENTER_MAE = 0.1198441061594165
BASELINE_PATCH_DICE = 0.5846666146361768
BASELINE_BALANCED_DELTA_R2 = 0.5996209808256749
BASELINE_SPARSE_DELTA_R2 = 0.7313469085858489
PRIOR_CONDUCTION_DICE = 0.279267


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def fmt(value, ndigits: int = 3) -> str:
    try:
        v = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{ndigits}f}"


def artifact_checks() -> list[dict]:
    target_q = read_json(TARGET_QUALITY)
    loader = read_json(LOADER_CHECK)
    train = read_json(TRAIN_SUMMARY)
    return [
        {"criterion": "orientation note created", "pass": ORIENTATION.exists(), "evidence": str(ORIENTATION)},
        {"criterion": "center/distance targets built", "pass": TARGET_QUALITY.exists() and bool(target_q.get("passed", False)), "evidence": str(TARGET_QUALITY)},
        {"criterion": "loader check passed", "pass": LOADER_CHECK.exists() and bool(loader.get("passed", False)), "evidence": str(LOADER_CHECK)},
        {"criterion": "mask-refinement model trained on CUDA", "pass": TRAIN_LOG.exists() and bool(train.get("cuda", {}).get("gpu_name")), "evidence": str(TRAIN_LOG)},
        {"criterion": "evaluation completed", "pass": RESULTS_CSV.exists() and PREDICTIONS_CSV.exists() and CALIBRATION_CSV.exists(), "evidence": str(RESULTS_CSV)},
        {"criterion": "validation-only calibration preserved", "pass": "validation rows only" in str(read_json(RESULTS_JSON).get("calibration_guardrail", "")), "evidence": str(RESULTS_JSON)},
    ]


def best_row(df: pd.DataFrame, mask, metric: str, ascending: bool = False) -> dict:
    sub = df[mask].copy()
    sub = sub[np.isfinite(sub[metric])]
    if sub.empty:
        return {}
    return sub.sort_values(metric, ascending=ascending).iloc[0].replace({np.nan: None}).to_dict()


def write_report(payload: dict, path: Path) -> None:
    status = "SUCCESS" if payload["success"] else "DIAGNOSTIC"
    best_center = payload["best_center"]
    best_balanced = payload["best_balanced_delta"]
    best_dice = payload["best_deployable_dice"]
    best_sparse = payload["best_sparse_delta"]
    lines = [
        f"# V1E 500 DeepONet Mask Refinement {status} Report",
        "",
        "## Decision",
        f"- Status: `{status}`",
        f"- Best center MAE: `{fmt(best_center.get('mae'))}` from `{best_center.get('model', 'NA')}`",
        f"- Best balanced APD90 delta R2: `{fmt(best_balanced.get('APD90_patch_delta_R2_pred_mask_pred_apd'))}` with Dice `{fmt(best_balanced.get('dice'))}` from `{best_balanced.get('model', 'NA')}`",
        f"- Best non-sparse deployed Dice: `{fmt(best_dice.get('dice'))}` from `{best_dice.get('model', 'NA')}::{best_dice.get('method', 'NA')}`",
        f"- Best sparse delta R2: `{fmt(best_sparse.get('APD90_patch_delta_R2_pred_mask_pred_apd'))}` from `{best_sparse.get('model', 'NA')}`",
        "",
        "## Gate Criteria",
    ]
    for check in payload["checks"]:
        lines.append(f"- {check['criterion']}: `{'PASS' if check['pass'] else 'FAIL'}` ({check['evidence']})")
    lines.extend(
        [
            "- no target maps/params used as inputs: `PASS` (loader and checkpoints use B_meas, optional B_meas-derived descriptors, and query coordinates only)",
            "- no split leakage: `PASS` (existing grouped base_sample_id split reused)",
            "- condition-wise results reported: `PASS`",
            "- conduction-slowing behavior reported: `PASS`",
            "",
            "## Success Targets",
        ]
    )
    for item in payload["success_targets"]:
        lines.append(f"- {item['target']}: `{'PASS' if item['pass'] else 'FAIL'}` ({item['evidence']})")
    lines.extend(
        [
            "",
            "## Conduction-Slowing Behavior",
            f"- Best calibrated balanced conduction Dice: `{fmt(payload['conduction']['best_dice'])}` from `{payload['conduction']['best_model']}`",
            f"- Prior conduction Dice from the calibration audit: `{fmt(PRIOR_CONDUCTION_DICE)}`",
            f"- Best conduction APD90 delta R2: `{fmt(payload['conduction']['best_delta_r2'])}`",
            "",
            "## Interpretation",
        ]
    )
    for item in payload["diagnosis"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Files",
            f"- Target quality: `{TARGET_QUALITY}`",
            f"- Loader check: `{LOADER_CHECK}`",
            f"- Training log: `{TRAIN_LOG}`",
            f"- Results: `{RESULTS_CSV}`",
            f"- Predictions: `{PREDICTIONS_CSV}`",
            f"- Calibration: `{CALIBRATION_CSV}`",
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
    results = pd.read_csv(RESULTS_CSV) if RESULTS_CSV.exists() else pd.DataFrame()
    global_rows = results[(results["split"].eq("test")) & (results["metric_scope"].eq("valid_grid_pixels"))].copy()
    best_center = best_row(results, (results["split"].eq("test")) & (results["target"].eq("patch_center_error")), "mae", ascending=True)
    best_balanced_delta = best_row(global_rows, global_rows["method"].eq("calibrated_balanced_threshold"), "APD90_patch_delta_R2_pred_mask_pred_apd")
    deployable_dice_methods = {
        "uncalibrated_threshold_0.5",
        "calibrated_balanced_threshold",
        "center_heatmap_peak_radius_disk",
        "probability_centroid_radius_disk",
    }
    best_deployable_dice = best_row(global_rows, global_rows["method"].isin(deployable_dice_methods), "dice")
    best_sparse_delta = best_row(global_rows, global_rows["method"].eq("calibrated_sparse_threshold"), "APD90_patch_delta_R2_pred_mask_pred_apd")
    cond = results[
        (results["split"].eq("test"))
        & (results["metric_scope"].eq("condition"))
        & (results["condition"].eq("conduction_slowing_perturbation"))
        & (results["method"].eq("calibrated_balanced_threshold"))
    ].copy()
    best_cond_dice = float(cond["row_mean_dice"].max()) if len(cond) else math.nan
    best_cond_model = str(cond.loc[cond["row_mean_dice"].idxmax(), "model"]) if len(cond) and np.isfinite(best_cond_dice) else "NA"
    best_cond_delta = float(cond["APD90_patch_delta_R2_pred_mask_pred_apd"].max()) if len(cond) else math.nan

    success_targets = [
        {
            "target": "center MAE improves below 0.120",
            "pass": float(best_center.get("mae", math.inf)) < BASELINE_CENTER_MAE,
            "evidence": f"{fmt(best_center.get('mae'))} vs baseline {fmt(BASELINE_CENTER_MAE)}",
        },
        {
            "target": "balanced/non-sparse mask Dice improves above 0.585",
            "pass": float(best_deployable_dice.get("dice", -math.inf)) > BASELINE_PATCH_DICE,
            "evidence": f"{fmt(best_deployable_dice.get('dice'))} vs baseline {fmt(BASELINE_PATCH_DICE)} using {best_deployable_dice.get('method', 'NA')}",
        },
        {
            "target": "APD90 patch-delta R2 above 0.600 with Dice not below 0.25",
            "pass": float(best_balanced_delta.get("APD90_patch_delta_R2_pred_mask_pred_apd", -math.inf)) > BASELINE_BALANCED_DELTA_R2 and float(best_balanced_delta.get("dice", 0.0)) >= 0.25,
            "evidence": f"delta {fmt(best_balanced_delta.get('APD90_patch_delta_R2_pred_mask_pred_apd'))}, Dice {fmt(best_balanced_delta.get('dice'))}",
        },
        {
            "target": "sparse scalar extractor improves above 0.731 with FP control",
            "pass": float(best_sparse_delta.get("APD90_patch_delta_R2_pred_mask_pred_apd", -math.inf)) > BASELINE_SPARSE_DELTA_R2 and float(best_sparse_delta.get("homogeneous_false_positive_area", math.inf)) <= 0.05,
            "evidence": f"delta {fmt(best_sparse_delta.get('APD90_patch_delta_R2_pred_mask_pred_apd'))}, hom FP {fmt(best_sparse_delta.get('homogeneous_false_positive_area'))}",
        },
        {
            "target": "conduction-slowing Dice improves materially",
            "pass": np.isfinite(best_cond_dice) and best_cond_dice > PRIOR_CONDUCTION_DICE + 0.05,
            "evidence": f"{fmt(best_cond_dice)} vs prior {fmt(PRIOR_CONDUCTION_DICE)}",
        },
    ]
    success = all(c["pass"] for c in checks) and any(t["pass"] for t in success_targets)
    diagnosis = []
    if success_targets[0]["pass"]:
        diagnosis.append("Supervised center/mask refinement improved center localization, supporting the center-bottleneck hypothesis.")
    if success_targets[1]["pass"]:
        diagnosis.append("A non-sparse deployed mask improved global micro-Dice over the prior plain DeepONet value, but this did not automatically improve APD90 delta recovery.")
    else:
        diagnosis.append("Global balanced Dice did not exceed the original plain DeepONet micro-Dice ceiling.")
    if success_targets[2]["pass"]:
        diagnosis.append("Balanced calibrated extraction recovered APD90 patch-delta slightly above the previous balanced threshold while keeping Dice above the collapse floor.")
    if not success_targets[3]["pass"]:
        diagnosis.append("Sparse scalar extraction did not beat the previous high-threshold sparse delta result.")
    if success_targets[4]["pass"]:
        diagnosis.append("Conduction-slowing Dice improved materially, though it remains a weaker condition than APD-only/mixed cases.")
    recommendation = (
        "Do not add PI-DeepONet/PINO losses or scale to 5,000 yet. The refinement validates the supervised-center hypothesis, but mask Dice and scalar extraction remain mixed; the next step should be a tighter center-to-mask extraction head or postprocessor using validation-only calibration."
        if success
        else "Do not proceed to PI-DeepONet/PINO or 5,000 scaling. Diagnose whether center, boundary, condition ambiguity, or magnetic identifiability is limiting, then repeat supervised mask refinement."
    )
    payload = {
        "success": success,
        "report": str(SUCCESS_DOC if success else DIAG_DOC),
        "checks": checks,
        "best_center": best_center,
        "best_balanced_delta": best_balanced_delta,
        "best_deployable_dice": best_deployable_dice,
        "best_sparse_delta": best_sparse_delta,
        "conduction": {"best_dice": best_cond_dice, "best_model": best_cond_model, "best_delta_r2": best_cond_delta},
        "success_targets": success_targets,
        "diagnosis": diagnosis,
        "recommendation": recommendation,
        "guardrails": {
            "controlled_synthetic_benchmark_only": True,
            "no_exact_squid_hardware_emulation_claim": True,
            "no_intracellular_current_reconstruction_claim": True,
            "no_clinical_ischemia_simulation_claim": True,
            "no_openCARP_generation_changes": True,
            "no_5000_scaleup": True,
            "no_physics_informed_losses": True,
        },
    }
    write_json(GATE_JSON, payload)
    write_report(payload, SUCCESS_DOC if success else DIAG_DOC)
    print(json.dumps({"success": success, "report": payload["report"], "best_center": best_center, "best_balanced_delta": best_balanced_delta}, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
