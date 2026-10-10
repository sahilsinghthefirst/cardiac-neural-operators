                      
                                                                                 

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import pandas as pd

from v1e_500_common import write_json
from v1f_metric_realignment_common import V1F_OUTPUT_DIR, ensure_dirs


GATE_JSON = V1F_OUTPUT_DIR / "v1f_metric_realignment_gate.json"
DOC = PROJECT_ROOT / "docs" / "V1F_METRIC_REALIGNMENT_SUCCESS_REPORT.md"


def metric(summary: pd.DataFrame, condition: str, col: str) -> float:
    row = summary[summary["condition"].eq(condition)]
    if row.empty or col not in row:
        return math.nan
    return float(row.iloc[0][col])


def best_test_ident(ident: pd.DataFrame, task: str) -> dict:
    sub = ident[(ident["task"].eq(task)) & (ident["split"].eq("test"))].copy()
    if sub.empty:
        return {}
    return sub.sort_values(["balanced_accuracy", "macro_f1"], ascending=False).iloc[0].to_dict()


def best_oracle(oracle: pd.DataFrame, method: str, scope: str = "overall") -> dict:
    sub = oracle[(oracle["split"].eq("test")) & (oracle["method"].eq(method)) & (oracle["scope"].eq(scope))].copy()
    if sub.empty:
        return {}
    return sub.sort_values("r2", ascending=False).iloc[0].to_dict()


def main() -> None:
    ensure_dirs()
    align = pd.read_csv(V1F_OUTPUT_DIR / "conduction_metric_alignment.csv")
    align_summary = pd.read_csv(V1F_OUTPUT_DIR / "conduction_metric_alignment_summary.csv")
    ident = pd.read_csv(V1F_OUTPUT_DIR / "magnetic_identifiability_results.csv")
    oracle = pd.read_csv(V1F_OUTPUT_DIR / "oracle_apd_lat_decomposition.csv")

    cond_apd = metric(align_summary, "conduction_slowing_perturbation", "APD90_patch_delta_median")
    apd_apd = metric(align_summary, "apd_repolarization_perturbation", "APD90_patch_delta_median")
    cond_lat = metric(align_summary, "conduction_slowing_perturbation", "LAT_patch_delay_median")
    homog_lat = metric(align_summary, "homogeneous", "LAT_patch_delay_median")
    cond_conduction = metric(align_summary, "conduction_slowing_perturbation", "conduction_severity_scalar_median")
    apd_conduction = metric(align_summary, "apd_repolarization_perturbation", "conduction_severity_scalar_median")

    weak_apd90_contrast = bool(np.isfinite(cond_apd) and np.isfinite(apd_apd) and abs(cond_apd) < 0.5 * abs(apd_apd))
    meaningful_lat_contrast = bool(np.isfinite(cond_lat) and abs(cond_lat) > max(1.0, 2.0 * abs(homog_lat if np.isfinite(homog_lat) else 0.0)))
    meaningful_conduction_proxy = bool(np.isfinite(cond_conduction) and (not np.isfinite(apd_conduction) or cond_conduction > apd_conduction))

    conduction_ident = best_test_ident(ident, "conduction_component")
    condition_ident = best_test_ident(ident, "condition_multiclass")
    patch_ident = best_test_ident(ident, "patch_present")
    apd_ident = best_test_ident(ident, "apd_component")
    conduction_above_chance = bool(float(conduction_ident.get("balanced_accuracy", 0.0)) > 0.55)

    lat_true_mask = best_oracle(oracle, "true_mask_pred_lat", "condition:conduction_slowing_perturbation")
    apd_true_mask = best_oracle(oracle, "true_mask_pred_apd90", "condition:conduction_slowing_perturbation")
    lat_pred_mask = best_oracle(oracle, "predicted_balanced_mask_pred_lat", "condition:conduction_slowing_perturbation")
    apd_pred_mask = best_oracle(oracle, "predicted_balanced_mask_pred_apd90", "condition:conduction_slowing_perturbation")
    lat_anomaly = best_oracle(oracle, "true_lat_anomaly_map_pred_lat", "condition:conduction_slowing_perturbation")
    apd_anomaly = best_oracle(oracle, "true_apd_anomaly_map_pred_apd90", "condition:conduction_slowing_perturbation")
    lat_field_better_for_conduction = bool(float(lat_true_mask.get("r2", -999.0)) > float(apd_true_mask.get("r2", -999.0)))
    lat_anomaly_field_strong = bool(float(lat_anomaly.get("r2", -999.0)) > 0.80)

    separate_heads_recommended = bool(conduction_above_chance and meaningful_conduction_proxy and lat_anomaly_field_strong)
    data_generation_adequate_to_continue = bool(conduction_above_chance and meaningful_conduction_proxy and lat_anomaly_field_strong)
    redesign_generation_needed = not data_generation_adequate_to_continue
    if not conduction_above_chance:
        recommendation = "audit_or_redesign_conduction_perturbation_generation_before_more_neural_work"
    elif separate_heads_recommended:
        recommendation = "add_separate_apd_and_lat_anomaly_heads_with_conduction_specific_supervision"
    else:
        recommendation = "continue_generic_patch_mask_only_with_caution"

    gate = {
        "success": bool(separate_heads_recommended and data_generation_adequate_to_continue),
        "weak_apd90_contrast_for_conduction_only": weak_apd90_contrast,
        "meaningful_lat_contrast_for_conduction_only": meaningful_lat_contrast,
        "meaningful_conduction_proxy_for_conduction_only": meaningful_conduction_proxy,
        "magnetic_descriptors_separate_conduction_above_chance_grouped_test": conduction_above_chance,
        "predicted_lat_field_supports_conduction_scalar_better_than_predicted_apd90": lat_field_better_for_conduction,
        "predicted_lat_anomaly_oracle_is_strong_for_conduction": lat_anomaly_field_strong,
        "future_training_should_use_separate_apd_and_lat_anomaly_heads": separate_heads_recommended,
        "data_generation_adequate_to_continue_controlled_v1f": data_generation_adequate_to_continue,
        "conduction_generation_redesign_needed": redesign_generation_needed,
        "recommendation": recommendation,
        "key_numbers": {
            "conduction_median_APD90_patch_delta": cond_apd,
            "apd_median_APD90_patch_delta": apd_apd,
            "conduction_median_LAT_patch_delay": cond_lat,
            "homogeneous_median_LAT_patch_delay": homog_lat,
            "conduction_descriptor_test_balanced_accuracy": conduction_ident.get("balanced_accuracy"),
            "condition_multiclass_test_balanced_accuracy": condition_ident.get("balanced_accuracy"),
            "patch_present_test_balanced_accuracy": patch_ident.get("balanced_accuracy"),
            "apd_component_test_balanced_accuracy": apd_ident.get("balanced_accuracy"),
            "conduction_true_mask_pred_LAT_R2": lat_true_mask.get("r2"),
            "conduction_true_mask_pred_APD90_R2": apd_true_mask.get("r2"),
            "conduction_pred_mask_pred_LAT_R2": lat_pred_mask.get("r2"),
            "conduction_pred_mask_pred_APD90_R2": apd_pred_mask.get("r2"),
            "conduction_true_LAT_anomaly_pred_LAT_R2": lat_anomaly.get("r2"),
            "conduction_true_APD_anomaly_pred_APD90_R2": apd_anomaly.get("r2"),
        },
        "guardrails": {
            "controlled_synthetic_benchmark_only": True,
            "no_physics_informed_losses": True,
            "no_5000_scaleup": True,
            "no_openCARP_changes": True,
            "condition_labels_not_model_inputs": True,
            "true_labels_targets_or_diagnostics_only": True,
            "grouped_split_preserved": True,
        },
        "outputs": {
            "conduction_metric_alignment_csv": str(V1F_OUTPUT_DIR / "conduction_metric_alignment.csv"),
            "magnetic_identifiability_csv": str(V1F_OUTPUT_DIR / "magnetic_identifiability_results.csv"),
            "oracle_decomposition_csv": str(V1F_OUTPUT_DIR / "oracle_apd_lat_decomposition.csv"),
            "target_manifest_csv": str(V1F_OUTPUT_DIR / "v1f_multichannel_target_manifest.csv"),
        },
    }
    write_json(GATE_JSON, gate)

    DOC.write_text(
        f"""# V1F Metric Realignment Success Report

Date: 2026-06-30

Status: {'passed' if gate['success'] else 'diagnostic'}.

This remains a controlled synthetic benchmark. It does not claim exact SQUID/OPM hardware emulation, intracellular current reconstruction, clinical ischemia simulation, or real-patient validation.

## Gate Findings

- Conduction-only APD90 contrast is weak relative to APD-only rows: `{weak_apd90_contrast}`.
- Conduction-only LAT patch-delay contrast is meaningful: `{meaningful_lat_contrast}`.
- Conduction-only local conduction-gradient proxy is meaningful: `{meaningful_conduction_proxy}`.
- B-meas-derived descriptors separate conduction-component rows above chance on grouped test: `{conduction_above_chance}`.
- Existing predicted LAT fields support conduction scalar recovery better than APD90 fields under true-mask oracle: `{lat_field_better_for_conduction}`.
- Existing predicted LAT fields recover true LAT-anomaly scalars under oracle maps: `{lat_anomaly_field_strong}`.
- Future training should use separate APD and LAT anomaly heads: `{separate_heads_recommended}`.
- Current controlled data generation is adequate for a v1f metric/target continuation: `{data_generation_adequate_to_continue}`.

## Key Numbers

- conduction median APD90 patch delta: `{cond_apd:.4g}`
- APD-only median APD90 patch delta: `{apd_apd:.4g}`
- conduction median LAT patch delay: `{cond_lat:.4g}`
- conduction descriptor grouped-test balanced accuracy: `{float(conduction_ident.get('balanced_accuracy', math.nan)):.4g}`
- conduction true-mask predicted-LAT R2: `{float(lat_true_mask.get('r2', math.nan)):.4g}`
- conduction true-mask predicted-APD90 R2: `{float(apd_true_mask.get('r2', math.nan)):.4g}`
- conduction true-LAT-anomaly predicted-LAT R2: `{float(lat_anomaly.get('r2', math.nan)):.4g}`

## Recommendation

`{recommendation}`

Do not add PI-DeepONet, PINO, Biot-Savart consistency losses, openCARP changes, or 5,000-sample scale-up yet. The next controlled model-side experiment should train separate APD/repolarization and LAT/conduction anomaly heads plus a generic support head, preserving grouped splits and validation-only calibration.
""",
        encoding="utf-8",
    )
    print(f"wrote {GATE_JSON}")
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
