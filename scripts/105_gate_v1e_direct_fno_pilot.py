                      
                                                               

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from v1e_direct_common import PILOT_DIR, ROOT, write_json


SUCCESS_DOC = ROOT / "docs" / "V1E_DIRECT_FNO_PARAMETRIC_SUCCESS_REPORT.md"
DIAG_DOC = ROOT / "docs" / "V1E_DIRECT_FNO_PARAMETRIC_DIAGNOSTIC_REPORT.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def metric_value(df: pd.DataFrame, model_contains: str, target: str, metric: str = "r2") -> float:
    if df.empty:
        return math.nan
    sub = df[df["model"].astype(str).str.contains(model_contains, regex=False) & df["split"].eq("test") & df["target"].eq(target)]
    if sub.empty or metric not in sub.columns:
        return math.nan
    vals = sub[metric].dropna()
    return float(vals.iloc[0]) if len(vals) else math.nan


def main() -> int:
    manifest = pd.read_csv(PILOT_DIR / "manifest.csv") if (PILOT_DIR / "manifest.csv").exists() else pd.DataFrame()
    target_report = load_json(PILOT_DIR / "target_quality_report.json")
    mag_report = load_json(PILOT_DIR / "direct_magnetic_quality_report.json")
    ready_report = load_json(PILOT_DIR / "fno_ready" / "dataset_quality_report.json")
    loader_report = load_json(ROOT / "outputs" / "neural_v1e" / "v1e_loader_check_report.json")
    eval_path = ROOT / "outputs" / "neural_v1e" / "v1e_parametric_evaluation.csv"
    eval_df = pd.read_csv(eval_path) if eval_path.exists() else pd.DataFrame()
    fno_json = load_json(ROOT / "outputs" / "neural_v1e" / "v1e_fno_field_parametric_results.json")
    cnn_json = load_json(ROOT / "outputs" / "neural_v1e" / "v1e_cnn_field_parametric_results.json")
    gates = []

    def gate(name: str, passed: bool, detail: str):
        gates.append({"gate": name, "passed": bool(passed), "detail": detail})

    gate("100 base samples selected", len(manifest) == 100, f"{len(manifest)}/100")
    gate("target generation passed", bool(target_report.get("passed")), f"{target_report.get('successful_targets')}/{target_report.get('selected_samples')}")
    gate("direct magnetic generation >=95%", bool(mag_report.get("passed")) and mag_report.get("direct_not_upsampled") is True, f"{mag_report.get('rows_generated')}/{mag_report.get('expected_rows')}, direct_not_upsampled={mag_report.get('direct_not_upsampled')}")
    gate("ready dataset passed", bool(ready_report.get("passed")), str(ready_report.get("split_counts")))
    gate("loader CUDA check passed", bool(loader_report.get("passed")), str(loader_report.get("cuda", {})))
    gate("FNO trained on CUDA", bool(fno_json.get("cuda", {}).get("gpu_name")), str(fno_json.get("cuda", {})))
    gate("CNN baseline trained or skipped", bool(cnn_json.get("cuda", {}).get("gpu_name")), str(cnn_json.get("cuda", {})))
    gate("no split leakage", not ready_report.get("group_leakage"), str(ready_report.get("group_leakage")))
    gate("field metrics reported", not eval_df.empty and eval_df["target"].isin(["LAT", "APD90", "APD80"]).any(), f"rows={len(eval_df)}")
    gate("parametric metrics reported", not eval_df.empty and eval_df["target"].astype(str).str.contains("patch_center|patch_area|patch_present").any(), f"rows={len(eval_df)}")
    fno_dice = metric_value(eval_df, "direct_bz_only_fno_parametric", "patch_mask", "dice")
    fno_param_area = metric_value(eval_df, "direct_bz_only_fno_parametric", "patch_area_param_head", "r2")
    fno_apd_param = metric_value(eval_df, "direct_bz_only_fno_parametric", "APD90_patch_delta_param_mask", "r2")
    fno_apd_pixel = metric_value(eval_df, "direct_bz_only_fno_parametric", "APD90_patch_delta_pixel_mask", "r2")
    localization_improved = any(np.isfinite(v) and v > thresh for v, thresh in [(fno_dice, 0.449), (fno_param_area, 0.320), (fno_apd_param, 0.269), (fno_apd_pixel, 0.269)])
    gate("localization/scalar improvement target", localization_improved, f"fno_dice={fno_dice}, param_area_r2={fno_param_area}, apd_param_r2={fno_apd_param}, apd_pixel_r2={fno_apd_pixel}")
    passed = all(g["passed"] for g in gates)
    report_path = SUCCESS_DOC if passed else DIAG_DOC
    report_path.parent.mkdir(parents=True, exist_ok=True)
    title = "V1E Direct FNO Parametric Success Report" if passed else "V1E Direct FNO Parametric Diagnostic Report"
    lines = [
        f"# {title}",
        "",
        "This pilot tests direct high-resolution SQUID-like magnetic measurements and parametric geometric patch targets. It is controlled synthetic benchmarking, not exact SQUID hardware emulation, not intracellular current reconstruction, and not clinical ischemia simulation.",
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
            "## Runtime And Storage",
            "",
            f"- Direct magnetic runtime seconds: `{mag_report.get('runtime_seconds')}`",
            f"- Direct magnetic storage GB: `{mag_report.get('storage_gb')}`",
            f"- Estimated 500-sample storage GB: `{mag_report.get('estimated_500_sample_storage_gb')}`",
            f"- Estimated 5,000-sample storage GB: `{mag_report.get('estimated_5000_sample_storage_gb')}`",
            f"- Estimated 500-sample runtime hours: `{mag_report.get('estimated_500_sample_runtime_hours')}`",
            f"- Estimated 5,000-sample runtime hours: `{mag_report.get('estimated_5000_sample_runtime_hours')}`",
            "",
            "## Key Metrics",
            "",
            f"- FNO direct Bz patch Dice: `{fno_dice}`",
            f"- FNO direct Bz patch area param-head R2: `{fno_param_area}`",
            f"- FNO direct Bz APD90 patch-delta R2 using parametric mask: `{fno_apd_param}`",
            f"- FNO direct Bz APD90 patch-delta R2 using pixel mask: `{fno_apd_pixel}`",
            "",
            "## Decision",
            "",
        ]
    )
    if passed:
        lines.append("The v1e pilot passed the configured feasibility gates. Direct high-resolution inputs and/or parametric targets improved the localization/scalar bottleneck enough to justify a larger pilot before any 5,000-sample scale-up.")
    else:
        lines.append("The v1e pilot did not pass all configured gates. Do not scale this pipeline yet. Inspect direct measurement quality, parametric head behavior, and whether 32x32 direct inputs materially improved localization compared with v1d.")
    lines.extend(
        [
            "",
            "## Artifact Paths",
            "",
            "- `outputs/dataset_v1e_direct_magnetic_pilot/manifest.csv`",
            "- `outputs/dataset_v1e_direct_magnetic_pilot/target_manifest.csv`",
            "- `outputs/dataset_v1e_direct_magnetic_pilot/direct_magnetic_manifest.csv`",
            "- `outputs/dataset_v1e_direct_magnetic_pilot/fno_ready/sample_index.csv`",
            "- `outputs/neural_v1e/v1e_parametric_evaluation.csv`",
            "- `figures/v1e_direct_magnetic_pilot/`",
            "- `figures/neural_v1e/`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")
    write_json(ROOT / "outputs" / "neural_v1e" / "v1e_gate_summary.json", {"passed": passed, "report": str(report_path), "gates": gates})
    print(json.dumps({"passed": passed, "report": str(report_path)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
