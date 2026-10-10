                      
                                                                 

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from v1e_direct_common import COMBINED_READY_DIR, PILOT_DIR, ROOT, write_json


SUCCESS_DOC = ROOT / "docs" / "V1E_PARAMETRIC_SIGNAL_SUCCESS_REPORT.md"
DIAG_DOC = ROOT / "docs" / "V1E_PARAMETRIC_SIGNAL_DIAGNOSTIC_REPORT.md"
SUMMARY_JSON = ROOT / "outputs" / "neural_v1e" / "v1e_parametric_signal_gate_summary.json"


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def best_metric(results: pd.DataFrame, target: str, metric: str, model_contains: str | None = None) -> tuple[float, str]:
    sub = results[(results["split"].eq("test")) & (results["target"].eq(target))]
    if model_contains:
        sub = sub[sub["model"].astype(str).str.contains(model_contains, regex=False)]
    if sub.empty or metric not in sub:
        return math.nan, ""
    sub = sub[np.isfinite(sub[metric])]
    if sub.empty:
        return math.nan, ""
    idx = sub[metric].idxmax()
    return float(sub.loc[idx, metric]), str(sub.loc[idx, "model"])


def min_metric(results: pd.DataFrame, target: str, metric: str) -> tuple[float, str]:
    sub = results[(results["split"].eq("test")) & (results["target"].eq(target))]
    if sub.empty or metric not in sub:
        return math.nan, ""
    sub = sub[np.isfinite(sub[metric])]
    if sub.empty:
        return math.nan, ""
    idx = sub[metric].idxmin()
    return float(sub.loc[idx, metric]), str(sub.loc[idx, "model"])


def center_baseline(preds: pd.DataFrame) -> float:
    train = preds[(preds["model"].eq("tabular_rf_signal_descriptors")) & (preds["split"].eq("train")) & (preds["patch_present"].eq(1))]
    test = preds[(preds["model"].eq("tabular_rf_signal_descriptors")) & (preds["split"].eq("test")) & (preds["patch_present"].eq(1))]
    if train.empty or test.empty:
        return math.nan
    cx = float(train["patch_center_x_norm"].mean())
    cy = float(train["patch_center_y_norm"].mean())
    err = np.sqrt((test["patch_center_x_norm"] - cx) ** 2 + (test["patch_center_y_norm"] - cy) ** 2)
    return float(np.mean(err))


def main() -> int:
    audit_report = load_json(ROOT / "outputs" / "neural_v1e" / "direct_signal_audit_report.json")
    combined_report = load_json(PILOT_DIR / "combined_input_quality_report.json")
    ready_report = load_json(COMBINED_READY_DIR / "dataset_quality_report.json")
    param_json = load_json(ROOT / "outputs" / "neural_v1e" / "parametric_only_results.json")
    results_path = ROOT / "outputs" / "neural_v1e" / "parametric_only_results.csv"
    preds_path = ROOT / "outputs" / "neural_v1e" / "parametric_only_predictions.csv"
    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    preds = pd.read_csv(preds_path) if preds_path.exists() else pd.DataFrame()
    gates = []

    def gate(name, passed, detail):
        gates.append({"gate": name, "passed": bool(passed), "detail": detail})

    gate("direct signal audit completed", bool(audit_report.get("passed")), f"rows={audit_report.get('rows')}")
    gate("combined-channel inputs built", bool(combined_report.get("passed")), f"{combined_report.get('rows_generated')}/{combined_report.get('expected_rows')}")
    gate("grouped split no leakage", bool(ready_report.get("passed")) and not ready_report.get("group_leakage"), str(ready_report.get("split_counts")))
    gate("parametric FNO and CNN trained on CUDA", bool(param_json.get("cuda", {}).get("gpu_name")), str(param_json.get("cuda", {})))
    gate("no target inputs documented", "target fields/patch params are labels" in str(param_json.get("leakage_guardrail", "")), str(param_json.get("leakage_guardrail", "")))
    f1, f1_model = best_metric(results, "patch_present", "macro_f1")
    dice, dice_model = best_metric(results, "parametric_mask_dice", "dice")
    area_r2, area_model = best_metric(results, "patch_area_fraction", "r2")
    radius_r2, radius_model = best_metric(results, "patch_radius_equiv_norm", "r2")
    center_err, center_model = min_metric(results, "patch_center_error", "mae")
    base_center = center_baseline(preds)
    gate("patch_present above baseline", np.isfinite(f1) and f1 > 0.55, f"best_macro_f1={f1} model={f1_model}")
    gate("center error below mean-center baseline", np.isfinite(center_err) and np.isfinite(base_center) and center_err < base_center, f"best_center_mae={center_err} model={center_model}; mean_center_baseline={base_center}")
    gate("area or radius positive R2", (np.isfinite(area_r2) and area_r2 > 0) or (np.isfinite(radius_r2) and radius_r2 > 0), f"area_r2={area_r2} model={area_model}; radius_r2={radius_r2} model={radius_model}")
    gate("parametric-mask Dice improves over v1e FNO", np.isfinite(dice) and dice > 0.206, f"best_dice={dice} model={dice_model}")
    passed = all(g["passed"] for g in gates)
    report_path = SUCCESS_DOC if passed else DIAG_DOC
    title = "V1E Parametric Signal Success Report" if passed else "V1E Parametric Signal Diagnostic Report"
    lines = [
        f"# {title}",
        "",
        "This diagnostic tests whether direct combined-channel SQUID-like magnetic measurements contain learnable information about controlled geometric patch parameters before further field-reconstruction training.",
        "",
        "No openCARP generation was modified. No PINO or Biot-Savart consistency loss was added. Target fields and patch parameters were labels only, not model inputs.",
        "",
        "## Gate Table",
        "",
        "| Gate | Result | Detail |",
        "|---|---:|---|",
    ]
    for g in gates:
        lines.append(f"| {g['gate']} | {'PASS' if g['passed'] else 'FAIL'} | {g['detail']} |")
    lines.extend(
        [
            "",
            "## Key Metrics",
            "",
            f"- Best patch-present macro F1: `{f1}` from `{f1_model}`",
            f"- Best patch center MAE: `{center_err}` from `{center_model}`; mean-center baseline `{base_center}`",
            f"- Best patch area fraction R2: `{area_r2}` from `{area_model}`",
            f"- Best patch radius R2: `{radius_r2}` from `{radius_model}`",
            f"- Best parametric-mask Dice: `{dice}` from `{dice_model}`",
            "",
            "## Decision",
            "",
        ]
    )
    if passed:
        lines.append("The combined-channel parametric diagnostic passed. A larger 500-sample direct pilot is justified before any 5,000-sample scale-up.")
    else:
        lines.append("The combined-channel parametric diagnostic did not pass all gates. Do not scale v1e yet; diagnose whether the limitation is sample size, target design, direct measurement channel design, or architecture.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `outputs/neural_v1e/direct_signal_audit.csv`",
            "- `outputs/neural_v1e/direct_signal_audit_correlations.csv`",
            "- `outputs/dataset_v1e_direct_magnetic_pilot/combined_input_manifest.csv`",
            "- `outputs/dataset_v1e_direct_magnetic_pilot/combined_fno_ready/sample_index.csv`",
            "- `outputs/neural_v1e/parametric_only_results.csv`",
            "- `outputs/neural_v1e/parametric_only_predictions.csv`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    write_json(SUMMARY_JSON, {"passed": passed, "report": str(report_path), "gates": gates})
    print(json.dumps({"passed": passed, "report": str(report_path)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
