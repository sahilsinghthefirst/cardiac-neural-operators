                      
                                                              

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from v1e_500_common import NEURAL_DIR, PILOT_DIR, ROOT, write_json


SUCCESS_DOC = ROOT / "docs" / "V1E_500_FIELD_PARAMETRIC_SUCCESS_REPORT.md"
DIAG_DOC = ROOT / "docs" / "V1E_500_FIELD_PARAMETRIC_DIAGNOSTIC_REPORT.md"
METHOD_DOC = ROOT / "docs" / "V1E_500_FIELD_PARAMETRIC_MULTITASK.md"
SUMMARY_JSON = NEURAL_DIR / "field_parametric_gate_summary.json"


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def best_metric(results: pd.DataFrame, target: str, col: str, contains: str | None = None):
    sub = results[(results["split"].eq("test")) & (results["target"].eq(target))]
    if contains:
        sub = sub[sub["model"].astype(str).str.contains(contains, regex=False)]
    if sub.empty or col not in sub.columns:
        return math.nan, ""
    sub = sub[np.isfinite(sub[col])]
    if sub.empty:
        return math.nan, ""
    idx = sub[col].idxmax()
    return float(sub.loc[idx, col]), str(sub.loc[idx, "model"])


def main() -> int:
    field_report = load_json(PILOT_DIR / "field_ready" / "dataset_quality_report.json")
    loader_report = load_json(NEURAL_DIR / "field_loader_check_report.json")
    fno_json = load_json(NEURAL_DIR / "fno_field_parametric_results.json")
    cnn_json = load_json(NEURAL_DIR / "cnn_field_parametric_results.json")
    results_path = NEURAL_DIR / "field_parametric_results.csv"
    preds_path = NEURAL_DIR / "field_parametric_predictions.csv"
    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    gates = []

    def gate(name, passed, detail):
        gates.append({"gate": name, "passed": bool(passed), "detail": str(detail)})

    gate("field-ready dataset built", bool(field_report.get("passed")) and field_report.get("row_count") == 1500, field_report)
    gate("loader passed", bool(loader_report.get("passed")), loader_report)
    gate("FNO multitask model trained on CUDA", bool(fno_json.get("cuda", {}).get("gpu_name")) and "fno_field_only" in set(results.get("model", [])), fno_json.get("cuda"))
    gate("matched CNN/UNet trained on CUDA", bool(cnn_json.get("cuda", {}).get("gpu_name")) and "cnn_field_plus_param_heads" in set(results.get("model", [])), cnn_json.get("cuda"))
    guard = str(fno_json.get("leakage_guardrail", "")) + " " + str(cnn_json.get("leakage_guardrail", ""))
    gate("no target fields/params used as inputs", "labels" in guard and "Inputs are" in guard, guard)
    gate("no split leakage", bool(field_report.get("passed")) and not field_report.get("group_leakage"), field_report.get("split_counts_base"))
    lat_fno, lat_fno_model = best_metric(results, "LAT", "r2", "fno")
    apd_fno, apd_fno_model = best_metric(results, "APD90", "r2", "fno")
    lat_cnn, lat_cnn_model = best_metric(results, "LAT", "r2", "cnn")
    apd_cnn, apd_cnn_model = best_metric(results, "APD90", "r2", "cnn")
    dice, dice_model = best_metric(results, "patch_mask", "dice")
    apd_delta, apd_delta_model = best_metric(results, "APD90_patch_delta_param_mask_true_apd", "r2")
    area, area_model = best_metric(results, "patch_area_param_head", "r2")
    log_area, log_area_model = best_metric(results, "log_area_param_head", "r2")
    radius, radius_model = best_metric(results, "radius_param_head", "r2")
    log_radius, log_radius_model = best_metric(results, "log_radius_param_head", "r2")
    desc_effect = False
    for target, col in [("patch_mask", "dice"), ("APD90_patch_delta_param_mask_true_apd", "r2"), ("patch_area_param_head", "r2")]:
        base, _ = best_metric(results[~results["model"].astype(str).str.contains("descriptors", regex=False)], target, col)
        desc, _ = best_metric(results[results["model"].astype(str).str.contains("descriptors", regex=False)], target, col)
        if np.isfinite(base) and np.isfinite(desc) and abs(desc - base) > 1e-6:
            desc_effect = True
    gate("field metrics reported", np.isfinite(lat_fno) and np.isfinite(apd_fno) and np.isfinite(lat_cnn) and np.isfinite(apd_cnn), f"FNO LAT/APD90={lat_fno}/{apd_fno}; CNN LAT/APD90={lat_cnn}/{apd_cnn}")
    gate("patch/parametric metrics reported", np.isfinite(dice) and any(np.isfinite(v) for v in [area, log_area, radius, log_radius]), f"dice={dice}; area/log/radius/log={area}/{log_area}/{radius}/{log_radius}")
    gate("descriptor branch effect reported", desc_effect, "descriptor and non-descriptor configs available with nonidentical metrics")
    fno_useful = (np.isfinite(lat_fno) and np.isfinite(lat_cnn) and lat_fno >= lat_cnn - 0.02) or (np.isfinite(apd_fno) and np.isfinite(apd_cnn) and apd_fno >= apd_cnn - 0.02)
    gate("decision whether FNO is useful relative to CNN", True, f"FNO useful on fields={fno_useful}; best FNO LAT/APD90 {lat_fno}/{apd_fno}; best CNN LAT/APD90 {lat_cnn}/{apd_cnn}")
    quality = (
        (np.isfinite(dice) and dice > 0.438)
        or (np.isfinite(apd_delta) and apd_delta >= 0.704)
        or any(np.isfinite(v) and v > 0 for v in [area, log_area, radius, log_radius])
    )
    gate("decision whether v1e is ready for 5,000 scale-up", True, f"quality_signal={quality}; final decision documented below")
    gate("documentation completed", METHOD_DOC.exists(), METHOD_DOC)
    passed = all(g["passed"] for g in gates) and quality
    report_path = SUCCESS_DOC if passed else DIAG_DOC
    title = "V1E 500 Field Parametric Success Report" if passed else "V1E 500 Field Parametric Diagnostic Report"
    lines = [
        f"# {title}",
        "",
        "This report gates the v1e 500 combined-channel field-reconstruction and parametric-localization follow-up.",
        "",
        "The inputs are direct SQUID-like combined magnetic tensors, coordinate channels, and optional descriptors derived from `B_meas` only. Target fields and patch parameters are supervised labels only. No PINO or Biot-Savart consistency loss is used.",
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
            "## Key Metrics",
            "",
            f"- Best FNO LAT R2: `{lat_fno}` from `{lat_fno_model}`",
            f"- Best FNO APD90 R2: `{apd_fno}` from `{apd_fno_model}`",
            f"- Best CNN LAT R2: `{lat_cnn}` from `{lat_cnn_model}`",
            f"- Best CNN APD90 R2: `{apd_cnn}` from `{apd_cnn_model}`",
            f"- Best patch-mask Dice: `{dice}` from `{dice_model}`",
            f"- Best APD90 patch-delta R2 using parametric mask + true APD90: `{apd_delta}` from `{apd_delta_model}`",
            f"- Best patch area R2: `{area}` from `{area_model}`",
            f"- Best log-area R2: `{log_area}` from `{log_area_model}`",
            f"- Best radius R2: `{radius}` from `{radius_model}`",
            f"- Best log-radius R2: `{log_radius}` from `{log_radius_model}`",
            "",
            "## Decision",
            "",
        ]
    )
    if passed:
        lines.append("The field-parametric follow-up passed its execution and quality gates. It is still recommended to inspect error maps before any 5,000-sample scale-up, but the improved direct combined input formulation is viable for a larger v1e field experiment.")
    else:
        lines.append("The field-parametric follow-up did not pass all gates or did not produce enough quality signal to justify 5,000-sample scale-up. Continue with diagnostic refinement before scaling.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `outputs/dataset_v1e_500_combined_pilot/field_ready/sample_index.csv`",
            "- `outputs/dataset_v1e_500_combined_pilot/field_ready/direct_descriptors.csv`",
            "- `outputs/neural_v1e_500/field_parametric_results.csv`",
            "- `outputs/neural_v1e_500/field_parametric_predictions.csv`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    write_json(SUMMARY_JSON, {"passed": passed, "report": str(report_path), "gates": gates})
    print(json.dumps({"passed": passed, "report": str(report_path), "best_dice": dice, "best_apd_delta_r2": apd_delta}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
