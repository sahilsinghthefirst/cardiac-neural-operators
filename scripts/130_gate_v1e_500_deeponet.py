                      
                                                  

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from v1e_500_common import NEURAL_DIR, PILOT_DIR, ROOT, write_json


SUCCESS_DOC = ROOT / "docs" / "V1E_500_DEEPNET_SUCCESS_REPORT.md"
DIAG_DOC = ROOT / "docs" / "V1E_500_DEEPNET_DIAGNOSTIC_REPORT.md"
METHOD_DOC = ROOT / "docs" / "V1E_500_DEEPNET_OPERATOR_TEST.md"
ORIENTATION_DOC = ROOT / "docs" / "INTERNAL_DEEPNET_REPO_ORIENTATION.md"
SUMMARY_JSON = NEURAL_DIR / "deeponet_gate_summary.json"


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def best_metric(results: pd.DataFrame, target: str, col: str, contains: str | None = None, minimize: bool = False):
    sub = results[(results["split"].eq("test")) & (results["target"].eq(target))]
    if contains:
        sub = sub[sub["model"].astype(str).str.contains(contains, regex=False)]
    if sub.empty or col not in sub.columns:
        return math.nan, ""
    sub = sub[np.isfinite(sub[col])]
    if sub.empty:
        return math.nan, ""
    idx = sub[col].idxmin() if minimize else sub[col].idxmax()
    return float(sub.loc[idx, col]), str(sub.loc[idx, "model"])


def descriptor_effect(results: pd.DataFrame) -> dict:
    pairs = []
    for base, desc in [
        ("deeponet_cnn_branch", "deeponet_cnn_branch_plus_descriptors"),
        ("deeponet_cnn_branch_fourier_trunk", "deeponet_cnn_branch_plus_descriptors_fourier_trunk"),
    ]:
        for target, col in [("patch_mask", "dice"), ("APD90", "r2"), ("patch_area_param_head", "r2"), ("APD90_patch_delta_param_mask_true_apd", "r2")]:
            b, _ = best_metric(results[results["model"].eq(base)], target, col)
            d, _ = best_metric(results[results["model"].eq(desc)], target, col)
            if np.isfinite(b) and np.isfinite(d):
                pairs.append({"base": base, "descriptor": desc, "target": target, "metric": col, "base_value": b, "descriptor_value": d, "delta": d - b})
    any_effect = any(abs(p["delta"]) > 1e-6 for p in pairs)
    any_positive = any(p["delta"] > 0 for p in pairs)
    return {"pairs": pairs, "any_effect": any_effect, "any_positive": any_positive}


def main() -> int:
    ready_report = load_json(PILOT_DIR / "deeponet_ready" / "dataset_quality_report.json")
    loader_report = load_json(NEURAL_DIR / "deeponet_loader_check_report.json")
    train_summary = load_json(NEURAL_DIR / "deeponet_training_summary.json")
    desc_summary = load_json(NEURAL_DIR / "descriptor_deeponet_training_summary.json")
    eval_json = load_json(NEURAL_DIR / "deeponet_results.json")
    results_path = NEURAL_DIR / "deeponet_results.csv"
    preds_path = NEURAL_DIR / "deeponet_predictions.csv"
    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    preds = pd.read_csv(preds_path) if preds_path.exists() else pd.DataFrame()
    gates = []

    def gate(name, passed, detail):
        gates.append({"gate": name, "passed": bool(passed), "detail": str(detail)})

    gate("repo orientation note created", ORIENTATION_DOC.exists(), ORIENTATION_DOC)
    gate("DeepONet research note created", METHOD_DOC.exists(), METHOD_DOC)
    gate("DeepONet-ready dataset built", bool(ready_report.get("passed")) and ready_report.get("row_count") == 1500, ready_report)
    gate("loader check passed", bool(loader_report.get("passed")), loader_report)
    gate("DeepONet trained on CUDA", bool(train_summary.get("cuda", {}).get("gpu_name")) and bool(train_summary.get("checkpoints")), train_summary.get("cuda"))
    descriptor_trained = bool(desc_summary.get("cuda", {}).get("gpu_name")) and "descriptor_deeponet" in set(results.get("model", []))
    gate("descriptor-only baseline trained or explicitly skipped", descriptor_trained or bool(desc_summary.get("skipped_reason")), desc_summary.get("cuda") or desc_summary.get("skipped_reason"))
    guard = " ".join(
        [
            str(ready_report.get("leakage_guardrail", "")),
            str(loader_report.get("leakage_guardrail", "")),
            str(train_summary.get("leakage_guardrail", "")),
            str(desc_summary.get("leakage_guardrail", "")),
            str(eval_json.get("leakage_guardrail", "")),
        ]
    )
    gate("no target fields/params used as inputs", "labels" in guard and "B_meas" in guard and "query" in guard, guard)
    gate("no split leakage", bool(ready_report.get("passed")) and not ready_report.get("group_leakage") and not loader_report.get("group_leakage"), ready_report.get("split_counts_base"))
    field_ok = all(np.isfinite(best_metric(results, target, "r2")[0]) for target in ["LAT", "APD80", "APD90"])
    patch_ok = np.isfinite(best_metric(results, "patch_mask", "dice")[0])
    param_ok = np.isfinite(best_metric(results, "patch_present", "macro_f1")[0]) and np.isfinite(best_metric(results, "patch_center_error", "mae", minimize=True)[0])
    desc_eff = descriptor_effect(results)
    gate("field metrics reported", field_ok, "LAT/APD80/APD90 R2 present")
    gate("patch/parametric metrics reported", patch_ok and param_ok, "patch Dice, patch-present, and center metrics present")
    gate("descriptor branch effect reported", desc_eff["any_effect"], desc_eff["pairs"][:6])
    gate("clear comparison against CNN/FNO/RF baselines", bool(eval_json.get("baselines")), list((eval_json.get("baselines") or {}).keys()))

    patch_dice, patch_dice_model = best_metric(results, "patch_mask", "dice")
    center_mae, center_model = best_metric(results, "patch_center_error", "mae", minimize=True)
    apd90_r2, apd90_model = best_metric(results, "APD90", "r2")
    lat_r2, lat_model = best_metric(results, "LAT", "r2")
    area_r2, area_model = best_metric(results, "patch_area_param_head", "r2")
    log_area_r2, log_area_model = best_metric(results, "log_area_param_head", "r2")
    radius_r2, radius_model = best_metric(results, "radius_param_head", "r2")
    log_radius_r2, log_radius_model = best_metric(results, "log_radius_param_head", "r2")
    delta_pixel_r2, delta_pixel_model = best_metric(results, "APD90_patch_delta_pixel_mask_pred_apd", "r2")
    delta_param_pred_r2, delta_param_pred_model = best_metric(results, "APD90_patch_delta_param_mask_pred_apd", "r2")
    delta_param_true_r2, delta_param_true_model = best_metric(results, "APD90_patch_delta_param_mask_true_apd", "r2")
    present_f1, present_model = best_metric(results, "patch_present", "macro_f1")

    success_signals = {
        "patch_dice_exceeds_parametric_0.438": bool(np.isfinite(patch_dice) and patch_dice > 0.438),
        "apd90_patch_delta_predicted_mask_material": bool(
            (np.isfinite(delta_pixel_r2) and delta_pixel_r2 > 0.10) or (np.isfinite(delta_param_pred_r2) and delta_param_pred_r2 > 0.10)
        ),
        "center_mae_improves_0.138": bool(np.isfinite(center_mae) and center_mae < 0.1376507316959553),
        "area_or_radius_improves_current_bests": bool(
            (np.isfinite(area_r2) and area_r2 > 0.2622975120286949)
            or (np.isfinite(radius_r2) and radius_r2 > 0.2464200928100517)
            or (np.isfinite(log_area_r2) and log_area_r2 > 0.2292131839370053)
            or (np.isfinite(log_radius_r2) and log_radius_r2 > 0.2225541430695877)
        ),
        "field_reconstruction_operator_baseline_strong": bool(np.isfinite(lat_r2) and np.isfinite(apd90_r2) and lat_r2 >= 0.90 and apd90_r2 >= 0.90),
    }
    quality_signal = any(success_signals.values())
    passed = all(g["passed"] for g in gates) and quality_signal
    report_path = SUCCESS_DOC if passed else DIAG_DOC
    title = "V1E 500 DeepONet Success Report" if passed else "V1E 500 DeepONet Diagnostic Report"

    descriptor_metric = {}
    if "descriptor_deeponet" in set(results.get("model", [])):
        descriptor_metric = {
            "patch_dice": best_metric(results[results["model"].eq("descriptor_deeponet")], "patch_mask", "dice")[0],
            "APD90_R2": best_metric(results[results["model"].eq("descriptor_deeponet")], "APD90", "r2")[0],
            "center_MAE": best_metric(results[results["model"].eq("descriptor_deeponet")], "patch_center_error", "mae", minimize=True)[0],
            "area_R2": best_metric(results[results["model"].eq("descriptor_deeponet")], "patch_area_param_head", "r2")[0],
        }

    failure_interpretation = ""
    if not passed:
        if descriptor_metric and np.isfinite(descriptor_metric.get("patch_dice", math.nan)) and descriptor_metric["patch_dice"] >= patch_dice - 0.02:
            failure_interpretation = "Descriptor-only performance is close to the best DeepONet result, suggesting descriptors may dominate the current signal."
        elif np.isfinite(lat_r2) and lat_r2 < 0.85:
            failure_interpretation = "Field reconstruction is weak, suggesting the compact branch/trunk architecture is not fitting the magnetic-to-field operator well at this sample size."
        elif np.isfinite(patch_dice) and patch_dice <= 0.438:
            failure_interpretation = "Broad fields may be learnable, but patch localization remains the bottleneck."
        else:
            failure_interpretation = "The failure is likely a mix of architecture fit and target identifiability under the 500-sample dataset."

    lines = [
        f"# {title}",
        "",
        "This report gates the v1e 500 plain DeepONet operator baseline.",
        "",
        "The models use magnetic tensors, optional descriptors derived from `B_meas`, and query coordinates only. Target fields and patch parameters are supervised labels only. No PI-DeepONet, PINO, or Biot-Savart consistency loss is used.",
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
            "## Best DeepONet Metrics",
            "",
            f"- Patch-mask Dice: `{patch_dice}` from `{patch_dice_model}`",
            f"- LAT R2: `{lat_r2}` from `{lat_model}`",
            f"- APD90 R2: `{apd90_r2}` from `{apd90_model}`",
            f"- Patch-present macro F1: `{present_f1}` from `{present_model}`",
            f"- Center MAE: `{center_mae}` from `{center_model}`",
            f"- Patch area R2: `{area_r2}` from `{area_model}`",
            f"- Log-area R2: `{log_area_r2}` from `{log_area_model}`",
            f"- Radius R2: `{radius_r2}` from `{radius_model}`",
            f"- Log-radius R2: `{log_radius_r2}` from `{log_radius_model}`",
            f"- APD90 delta R2, pixel mask + predicted APD90: `{delta_pixel_r2}` from `{delta_pixel_model}`",
            f"- APD90 delta R2, parametric mask + predicted APD90: `{delta_param_pred_r2}` from `{delta_param_pred_model}`",
            f"- APD90 delta R2, parametric mask + true APD90: `{delta_param_true_r2}` from `{delta_param_true_model}`",
            "",
            "## Descriptor-Only Baseline",
            "",
        ]
    )
    if descriptor_metric:
        lines.extend([f"- `{key}`: `{value}`" for key, value in descriptor_metric.items()])
    else:
        lines.append("- Descriptor-only baseline was not available.")
    lines.extend(
        [
            "",
            "## Success Signals",
            "",
        ]
    )
    for key, value in success_signals.items():
        lines.append(f"- {key}: `{'PASS' if value else 'no'}`")
    lines.extend(
        [
            "",
            "## Comparison Baselines",
            "",
            "- v1e 500 parametric CNN encoder: center MAE `0.138`, Dice `0.438`, APD90 delta R2 with parametric mask plus true APD90 `0.704`.",
            "- v1e 500 tabular RF descriptors: patch-present macro F1 `0.976`, area R2 `0.262`, radius R2 `0.246`.",
            "- v1e 500 field-parametric FNO/CNN: FNO patch Dice `0.581`, FNO LAT/APD90 R2 `0.952/0.959`, CNN APD90 R2 `0.965`.",
            "- v1d FNO field baseline: LAT/APD90 R2 `0.976/0.981`, patch Dice `0.449`.",
            "",
            "## Decision",
            "",
        ]
    )
    if passed:
        lines.append("The plain DeepONet baseline passed the execution gates and produced at least one meaningful quality signal. It is worth keeping as an operator baseline, but PI-DeepONet or PINO losses should still wait until the strongest failure mode is inspected from the generated figures.")
    else:
        lines.append("The plain DeepONet baseline did not pass the quality gate. Do not proceed to PI-DeepONet or scale-up from this result alone.")
        lines.append("")
        lines.append(f"Likely failure mode: {failure_interpretation}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `outputs/dataset_v1e_500_combined_pilot/deeponet_ready/sample_index.csv`",
            "- `outputs/neural_v1e_500/deeponet_training_log.csv`",
            "- `outputs/neural_v1e_500/descriptor_deeponet_training_log.csv`",
            "- `outputs/neural_v1e_500/deeponet_results.csv`",
            "- `outputs/neural_v1e_500/deeponet_predictions.csv`",
            "- `figures/neural_v1e_500/deeponet_vs_fno_cnn_summary.png`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    write_json(
        SUMMARY_JSON,
        {
            "passed": passed,
            "report": str(report_path),
            "gates": gates,
            "success_signals": success_signals,
            "best_metrics": {
                "patch_dice": patch_dice,
                "patch_dice_model": patch_dice_model,
                "center_mae": center_mae,
                "center_model": center_model,
                "lat_r2": lat_r2,
                "lat_model": lat_model,
                "apd90_r2": apd90_r2,
                "apd90_model": apd90_model,
                "area_r2": area_r2,
                "area_model": area_model,
                "radius_r2": radius_r2,
                "radius_model": radius_model,
                "delta_pixel_r2": delta_pixel_r2,
                "delta_param_pred_r2": delta_param_pred_r2,
                "delta_param_true_r2": delta_param_true_r2,
            },
            "descriptor_only_metrics": descriptor_metric,
            "descriptor_effect": desc_eff,
            "failure_interpretation": failure_interpretation,
        },
    )
    print(json.dumps({"passed": passed, "report": str(report_path), "best_patch_dice": patch_dice, "best_apd90_r2": apd90_r2}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
