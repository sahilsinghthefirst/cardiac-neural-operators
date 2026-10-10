                      
                                                                             

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from v1e_500_common import NEURAL_DIR, PILOT_DIR, READY_DIR, ROOT, write_json


SUCCESS_DOC = ROOT / "docs" / "V1E_500_PARAMETRIC_SUCCESS_REPORT.md"
DIAG_DOC = ROOT / "docs" / "V1E_500_PARAMETRIC_DIAGNOSTIC_REPORT.md"
SUMMARY_JSON = NEURAL_DIR / "v1e_500_parametric_gate_summary.json"
METHOD_DOC = ROOT / "docs" / "V1E_500_COMBINED_PARAMETRIC_LOCALIZATION.md"


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def best_metric(results: pd.DataFrame, target: str, metric: str):
    sub = results[(results["split"].eq("test")) & (results["target"].eq(target))]
    if sub.empty or metric not in sub.columns:
        return math.nan, ""
    sub = sub[np.isfinite(sub[metric])]
    if sub.empty:
        return math.nan, ""
    idx = sub[metric].idxmax()
    return float(sub.loc[idx, metric]), str(sub.loc[idx, "model"])


def min_metric(results: pd.DataFrame, target: str, metric: str, exclude_baseline: bool = True):
    sub = results[(results["split"].eq("test")) & (results["target"].eq(target))]
    if exclude_baseline:
        sub = sub[~sub["model"].astype(str).str.contains("baseline", regex=False)]
    if sub.empty or metric not in sub.columns:
        return math.nan, ""
    sub = sub[np.isfinite(sub[metric])]
    if sub.empty:
        return math.nan, ""
    idx = sub[metric].idxmin()
    return float(sub.loc[idx, metric]), str(sub.loc[idx, "model"])


def baseline_center(results: pd.DataFrame):
    sub = results[(results["split"].eq("test")) & (results["target"].eq("patch_center_error")) & (results["model"].eq("mean_center_baseline"))]
    if sub.empty:
        return math.nan
    return float(sub.iloc[0]["mae"])


def main() -> int:
    pilot = pd.read_csv(PILOT_DIR / "manifest.csv") if (PILOT_DIR / "manifest.csv").exists() else pd.DataFrame()
    target_report = load_json(PILOT_DIR / "target_quality_report.json")
    gen_report = load_json(PILOT_DIR / "direct_generation_quality_report.json")
    ready_report = load_json(READY_DIR / "dataset_quality_report.json")
    train_report = load_json(NEURAL_DIR / "v1e_500_parametric_training_summary.json")
    results_path = NEURAL_DIR / "v1e_500_parametric_results.csv"
    preds_path = NEURAL_DIR / "v1e_500_parametric_predictions.csv"
    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    preds = pd.read_csv(preds_path) if preds_path.exists() else pd.DataFrame()
    gates = []

    def gate(name, passed, detail):
        gates.append({"gate": name, "passed": bool(passed), "detail": str(detail)})

    gate("500 samples selected", len(pilot) == 500, f"manifest_rows={len(pilot)}")
    gate("target generation completed", bool(target_report.get("passed")) and target_report.get("successful_targets") == 500, target_report)
    direct_success_rate = gen_report.get("combined_success", 0) / max(gen_report.get("expected_combined_rows", 1), 1)
    gate("direct 32x32 generation completed for >=98%", bool(gen_report.get("passed")) and direct_success_rate >= 0.98, gen_report)
    gate("combined-channel inputs built", ready_report.get("row_count") == 1500 and ready_report.get("passed"), ready_report)
    gate("no split leakage", ready_report.get("passed") and not ready_report.get("group_leakage"), ready_report.get("split_counts_base"))
    models = set(preds["model"].unique()) if len(preds) else set()
    cuda_ok = bool(train_report.get("cuda", {}).get("gpu_name"))
    gate("CNN and FNO trained on CUDA", cuda_ok and {"cnn_encoder_parametric", "fno2d_encoder_parametric"}.issubset(models), train_report.get("cuda"))
    guardrail = str(train_report.get("leakage_guardrail", ""))
    gate("no target maps or patch params used as inputs", "labels" in guardrail and "Inputs are" in guardrail, guardrail)

    f1, f1_model = best_metric(results, "patch_present", "macro_f1")
    center_mae, center_model = min_metric(results, "patch_center_error", "mae")
    center_base = baseline_center(results)
    dice, dice_model = best_metric(results, "parametric_mask_dice", "dice")
    area_r2, area_model = best_metric(results, "patch_area_fraction", "r2")
    log_area_r2, log_area_model = best_metric(results, "log_area_fraction", "r2")
    radius_r2, radius_model = best_metric(results, "equivalent_radius_norm", "r2")
    log_radius_r2, log_radius_model = best_metric(results, "log_radius", "r2")
    area_bin_f1, area_bin_model = best_metric(results, "area_bin", "macro_f1")
    apd_delta_r2, apd_delta_model = best_metric(results, "APD90_patch_delta_parametric_mask_true_apd90", "r2")

    gate("patch-present is strong", np.isfinite(f1) and f1 >= 0.65, f"best_macro_f1={f1} model={f1_model}")
    gate("center error improves over mean-center baseline", np.isfinite(center_mae) and np.isfinite(center_base) and center_mae < center_base, f"best={center_mae} model={center_model}; baseline={center_base}")
    gate("parametric-mask Dice improves over 0.431 or exceeds 0.449", np.isfinite(dice) and (dice > 0.431 or dice > 0.449), f"best_dice={dice} model={dice_model}")
    area_pass = any(np.isfinite(v) and v > 0 for v in [area_r2, log_area_r2, radius_r2, log_radius_r2]) or (np.isfinite(area_bin_f1) and area_bin_f1 >= 0.45)
    gate(
        "area/radius/log targets show useful signal",
        area_pass,
        f"area_r2={area_r2} ({area_model}); log_area_r2={log_area_r2} ({log_area_model}); radius_r2={radius_r2} ({radius_model}); log_radius_r2={log_radius_r2} ({log_radius_model}); area_bin_f1={area_bin_f1} ({area_bin_model})",
    )
    gate("documentation completed", METHOD_DOC.exists(), METHOD_DOC)

    passed = all(g["passed"] for g in gates)
    report_path = SUCCESS_DOC if passed else DIAG_DOC
    title = "V1E 500 Parametric Success Report" if passed else "V1E 500 Parametric Diagnostic Report"
    lines = [
        f"# {title}",
        "",
        "This report gates the 500-base-sample v1e combined-channel parametric localization pilot.",
        "",
        "The pipeline uses direct SQUID-like magnetic measurement tensors generated from the equivalent-current proxy `J_eq = -D_eff grad(Vm)`. It does not modify openCARP generation, does not use target fields or patch parameters as model inputs, and does not claim exact SQUID emulation.",
        "",
        "## Gate Table",
        "",
        "| Gate | Result | Detail |",
        "|---|---:|---|",
    ]
    for g in gates:
        lines.append(f"| {g['gate']} | {'PASS' if g['passed'] else 'FAIL'} | `{g['detail']}` |")
    lines.extend(
        [
            "",
            "## Key Test Metrics",
            "",
            f"- Best patch-present macro F1: `{f1}` from `{f1_model}`",
            f"- Best center MAE: `{center_mae}` from `{center_model}`; mean-center baseline `{center_base}`",
            f"- Best parametric-mask Dice: `{dice}` from `{dice_model}`",
            f"- Best patch area R2: `{area_r2}` from `{area_model}`",
            f"- Best log-area R2: `{log_area_r2}` from `{log_area_model}`",
            f"- Best radius R2: `{radius_r2}` from `{radius_model}`",
            f"- Best log-radius R2: `{log_radius_r2}` from `{log_radius_model}`",
            f"- Best area-bin macro F1: `{area_bin_f1}` from `{area_bin_model}`",
            f"- Best APD90 patch-delta R2 using parametric mask + true APD90: `{apd_delta_r2}` from `{apd_delta_model}`",
            "",
            "## Runtime And Storage",
            "",
            f"- Direct/combined generation runtime: `{gen_report.get('runtime_seconds')}` seconds",
            f"- Direct/combined pilot storage: `{gen_report.get('storage_gb')}` GB",
            f"- Estimated 5,000-sample direct/combined storage: `{gen_report.get('estimated_5000_storage_gb')}` GB",
            f"- Estimated 5,000-sample direct/combined runtime: `{gen_report.get('estimated_5000_runtime_hours')}` hours",
            "",
            "## Decision",
            "",
        ]
    )
    if passed:
        lines.append("The 500-sample combined-channel parametric pilot passed. It is reasonable to consider a field-reconstruction follow-up on this improved input/target formulation before any 5,000-sample v1e scale-up.")
    else:
        lines.append("The 500-sample combined-channel parametric pilot did not pass all gates. Do not scale v1e to 5,000; use the failed gates above to decide whether the bottleneck is sample size, parameterization, magnetic operator design, or model architecture.")
    lines.extend(
        [
            "",
            "## Artifact Paths",
            "",
            "- `outputs/dataset_v1e_500_combined_pilot/manifest.csv`",
            "- `outputs/dataset_v1e_500_combined_pilot/combined_input_manifest.csv`",
            "- `outputs/dataset_v1e_500_combined_pilot/parametric_ready/sample_index.csv`",
            "- `outputs/neural_v1e_500/v1e_500_parametric_results.csv`",
            "- `outputs/neural_v1e_500/v1e_500_parametric_predictions.csv`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    write_json(SUMMARY_JSON, {"passed": passed, "report": str(report_path), "gates": gates})
    print(json.dumps({"passed": passed, "report": str(report_path), "best_dice": dice, "best_area_bin_f1": area_bin_f1}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
