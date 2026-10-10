                      
                                                                 
 
                                                                              
                                                                          
                                                                         
                                                
   

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "icbinb_required_analysis"

SPLIT_JSON = Path(
    "outputs/dataset_v1e_500_combined_pilot/deeponet_ready/splits/"
    "grouped_train_val_test.json"
)
SAMPLE_INDEX = Path(
    "outputs/dataset_v1e_500_combined_pilot/deeponet_ready/sample_index.csv"
)
COMBINED_MANIFEST = Path(
    "outputs/dataset_v1e_500_combined_pilot/combined_input_manifest.csv"
)
NEURAL_DIR = Path("outputs/neural_v1e_500")
CHECKPOINT_DIR = NEURAL_DIR / "checkpoints"
V1F_DIR = Path("outputs/v1f_metric_realignment")
MODULE4_DIR = Path("outputs/icbinb_sensor_height_ood")
MODULE6_DIR = Path("outputs/icbinb_paper_evidence_bundle")

PRIMARY_VARIANT = "combined_bz_axial"
EXPECTED_VARIANTS = [
    "combined_all_available",
    "combined_bz_axial",
    "combined_bz_axial_noise",
]

FIELD_FNO = "fno_field_only"
FIELD_DEEPONET = "deeponet_cnn_branch_fourier_trunk"
LOCALIZATION_FNO = "fno_field_plus_param_heads_plus_descriptors_area_calibrated"
LOCALIZATION_DEEPONET = "deeponet_cnn_branch_plus_descriptors_fourier_trunk"
REFINEMENT_DEEPONET = "deeponet_cnn_branch_fourier_trunk_mask_refine"
REFINEMENT_BALANCED_THRESHOLD = 0.62


ARTIFACT_SPECS = [
    {
        "id": "grouped_split_json",
        "category": "split",
        "path": SPLIT_JSON,
        "purpose": "Canonical grouped train/validation/test assignment.",
    },
    {
        "id": "deeponet_sample_index",
        "category": "dataset_index",
        "path": SAMPLE_INDEX,
        "purpose": "Row-level model-ready index with base IDs, variants, and targets.",
    },
    {
        "id": "combined_input_manifest",
        "category": "dataset_index",
        "path": COMBINED_MANIFEST,
        "purpose": "Measurement-channel and input-file manifest.",
    },
    {
        "id": "dataset_quality_report",
        "category": "dataset_validation",
        "path": Path(
            "outputs/dataset_v1e_500_combined_pilot/deeponet_ready/"
            "dataset_quality_report.json"
        ),
        "purpose": "Existing 500-base/1,500-row grouped-dataset quality report.",
    },
    {
        "id": "fno_training_statistics",
        "category": "normalization",
        "path": NEURAL_DIR / "v1e_500_fno_field_parametric_stats.json",
        "purpose": "Saved train-only normalization statistics required for exact FNO checkpoint replay.",
    },
    {
        "id": "fno_checkpoint_field",
        "category": "checkpoint",
        "path": CHECKPOINT_DIR / "v1e_500_fno_field_only.pt",
        "purpose": "Validation-selected FNO global-field representative.",
    },
    {
        "id": "fno_checkpoint_localization",
        "category": "checkpoint",
        "path": CHECKPOINT_DIR
        / "v1e_500_fno_field_plus_param_heads_plus_descriptors_area_calibrated.pt",
        "purpose": "Validation-selected FNO localization representative.",
    },
    {
        "id": "deeponet_checkpoint_field",
        "category": "checkpoint",
        "path": CHECKPOINT_DIR
        / "deeponet_deeponet_cnn_branch_fourier_trunk.pt",
        "purpose": "Validation-selected DeepONet global-field representative.",
    },
    {
        "id": "deeponet_checkpoint_localization",
        "category": "checkpoint",
        "path": CHECKPOINT_DIR
        / "deeponet_deeponet_cnn_branch_plus_descriptors_fourier_trunk.pt",
        "purpose": "Validation-selected plain DeepONet localization representative.",
    },
    {
        "id": "deeponet_checkpoint_refinement",
        "category": "checkpoint",
        "path": CHECKPOINT_DIR
        / "deeponet_mask_refine_deeponet_cnn_branch_fourier_trunk_mask_refine.pt",
        "purpose": "Validation-selected mask-refinement recovery attempt.",
    },
    {
        "id": "fno_predictions_scalar",
        "category": "prediction",
        "path": NEURAL_DIR / "fno_field_parametric_predictions.csv",
        "purpose": "Saved FNO row-level scalar and metadata predictions.",
    },
    {
        "id": "fno_map_cache_index",
        "category": "derived_prediction",
        "path": Path(
            "outputs/icbinb_required_analysis/fno_full_grid_prediction_index.csv"
        ),
        "purpose": "Inference-only row alignment for the locked FNO full-grid cache.",
        "required": False,
    },
    {
        "id": "fno_map_cache_arrays",
        "category": "derived_prediction",
        "path": Path(
            "outputs/icbinb_required_analysis/fno_full_grid_prediction_maps.npz"
        ),
        "purpose": "Inference-only LAT/APD90/patch maps for the locked FNO representatives.",
        "required": False,
    },
    {
        "id": "fno_map_cache_metadata",
        "category": "derived_prediction",
        "path": Path(
            "outputs/icbinb_required_analysis/fno_full_grid_prediction_cache.json"
        ),
        "purpose": "Checkpoint, normalization, coverage, and parity provenance for the FNO cache.",
        "required": False,
    },
    {
        "id": "paired_test_cohort_table",
        "category": "derived_analysis_input",
        "path": Path("outputs/icbinb_required_analysis/paired_test_cohort.csv"),
        "purpose": "Canonical one-row-per-base paired primary-variant test cohort.",
        "required": False,
    },
    {
        "id": "paired_test_cohort_maps",
        "category": "derived_analysis_input",
        "path": Path("outputs/icbinb_required_analysis/paired_test_cohort_maps.npz"),
        "purpose": "Aligned truth and locked-model prediction maps for the 75-case cohort.",
        "required": False,
    },
    {
        "id": "paired_test_cohort_validation",
        "category": "derived_analysis_validation",
        "path": Path("outputs/icbinb_required_analysis/cohort_validation.json"),
        "purpose": "Pairing, coverage, truth-alignment, tolerance, and provenance checks for the canonical cohort.",
        "required": False,
    },
    {
        "id": "module2_case_statistics",
        "category": "derived_analysis",
        "path": Path("outputs/icbinb_required_analysis/module2_case_statistics.csv"),
        "purpose": "Per-base sufficient statistics and deployed-support diagnostics for all locked endpoints.",
        "required": False,
    },
    {
        "id": "module2_point_estimates",
        "category": "derived_analysis",
        "path": Path("outputs/icbinb_required_analysis/module2_point_estimates.csv"),
        "purpose": "Overall point estimates for the six locked endpoints and two model families.",
        "required": False,
    },
    {
        "id": "module2_condition_specific",
        "category": "derived_analysis",
        "path": Path("outputs/icbinb_required_analysis/module2_condition_specific.csv"),
        "purpose": "Condition-specific point estimates on the paired held-out cohort.",
        "required": False,
    },
    {
        "id": "module2_bootstrap_intervals",
        "category": "derived_analysis",
        "path": Path("outputs/icbinb_required_analysis/module2_bootstrap_intervals.csv"),
        "purpose": "Base-sample percentile intervals for overall and condition-specific endpoints.",
        "required": False,
    },
    {
        "id": "module2_paired_differences",
        "category": "derived_analysis",
        "path": Path("outputs/icbinb_required_analysis/module2_paired_differences.csv"),
        "purpose": "Paired FNO-minus-DeepONet differences and bootstrap intervals.",
        "required": False,
    },
    {
        "id": "module2_support_diagnostics",
        "category": "derived_analysis",
        "path": Path("outputs/icbinb_required_analysis/module2_support_diagnostics.csv"),
        "purpose": "Degenerate-support counts and complete-case APD90 recovery sensitivity.",
        "required": False,
    },
    {
        "id": "module2_bootstrap_replicates",
        "category": "derived_analysis",
        "path": Path("outputs/icbinb_required_analysis/module2_bootstrap_replicates.npz"),
        "purpose": "Exact paired base-case draws, metric replicates, and paired differences.",
        "required": False,
    },
    {
        "id": "module2_summary",
        "category": "derived_analysis",
        "path": Path("outputs/icbinb_required_analysis/module2_summary.json"),
        "purpose": "Machine-readable Module 2 result summary for paper preparation.",
        "required": False,
    },
    {
        "id": "module2_validation",
        "category": "derived_analysis_validation",
        "path": Path("outputs/icbinb_required_analysis/module2_validation.json"),
        "purpose": "Pairing, bootstrap replay, metric parity, provenance, and safety checks.",
        "required": False,
    },
    {
        "id": "module3_identifiability_predictions",
        "category": "failure_decomposition",
        "path": Path("outputs/icbinb_required_analysis/module3_identifiability_predictions.csv"),
        "purpose": "Reproduced descriptor-only conduction predictions with a primary-variant marker.",
        "required": False,
    },
    {
        "id": "module3_failure_decomposition",
        "category": "failure_decomposition",
        "path": Path("outputs/icbinb_required_analysis/module3_failure_decomposition.csv"),
        "purpose": "Controlled magnetic-identifiability to oracle-field to deployed-support decomposition.",
        "required": False,
    },
    {
        "id": "module3_secondary_diagnostics",
        "category": "failure_decomposition",
        "path": Path("outputs/icbinb_required_analysis/module3_secondary_diagnostics.csv"),
        "purpose": "Secondary APD90 oracle and deployed-support recovery diagnostics.",
        "required": False,
    },
    {
        "id": "module3_bootstrap_replicates",
        "category": "failure_decomposition",
        "path": Path("outputs/icbinb_required_analysis/module3_bootstrap_replicates.npz"),
        "purpose": "Exact grouped draws and bootstrap values for the Module 3 decomposition.",
        "required": False,
    },
    {
        "id": "module3_summary",
        "category": "failure_decomposition",
        "path": Path("outputs/icbinb_required_analysis/module3_summary.json"),
        "purpose": "Machine-readable Module 3 thesis, values, limitations, and recommendation.",
        "required": False,
    },
    {
        "id": "module3_validation",
        "category": "derived_analysis_validation",
        "path": Path("outputs/icbinb_required_analysis/module3_validation.json"),
        "purpose": "Model-selection, primary-variant, bootstrap, provenance, and safety checks for Module 3.",
        "required": False,
    },
    {
        "id": "module3_figure_png",
        "category": "paper_figure",
        "path": Path("figures/icbinb_required_analysis/module3_failure_decomposition.png"),
        "purpose": "Raster preview of the compact paper-ready failure-decomposition figure.",
        "required": False,
    },
    {
        "id": "module3_figure_pdf",
        "category": "paper_figure",
        "path": Path("figures/icbinb_required_analysis/module3_failure_decomposition.pdf"),
        "purpose": "Vector paper-ready failure-decomposition figure.",
        "required": False,
    },
    {
        "id": "module4_sensor_height_manifest",
        "category": "robustness_input",
        "path": MODULE4_DIR / "sensor_height_manifest.csv",
        "purpose": "Isolated 35/50/65 mm held-out sensor-height cohort manifest.",
        "required": False,
    },
    {
        "id": "module4_generation_validation",
        "category": "derived_analysis_validation",
        "path": MODULE4_DIR / "generation_validation.json",
        "purpose": "Biot-Savart reprojection coverage, nominal bit-parity, and no-EP-rerun validation.",
        "required": False,
    },
    {
        "id": "module4_nominal_projection_parity",
        "category": "robustness_input_validation",
        "path": MODULE4_DIR / "nominal_50mm_parity.csv",
        "purpose": "Per-base bit-exact 50 mm reprojection checks for both active measurement channels.",
        "required": False,
    },
    {
        "id": "module4_prediction_index",
        "category": "derived_prediction",
        "path": MODULE4_DIR / "module4_prediction_index.csv",
        "purpose": "Frozen-model prediction alignment for 75 cases, two models, and three heights.",
        "required": False,
    },
    {
        "id": "module4_prediction_maps",
        "category": "derived_prediction",
        "path": MODULE4_DIR / "module4_prediction_maps.npz",
        "purpose": "Full-grid frozen FNO and DeepONet predictions at 35, 50, and 65 mm.",
        "required": False,
    },
    {
        "id": "module4_case_statistics",
        "category": "robustness_analysis",
        "path": MODULE4_DIR / "module4_case_statistics.csv",
        "purpose": "Per-base sufficient statistics for the sensor-height sensitivity analysis.",
        "required": False,
    },
    {
        "id": "module4_point_estimates",
        "category": "robustness_analysis",
        "path": MODULE4_DIR / "module4_point_estimates.csv",
        "purpose": "Frozen-model point estimates at each sensor height.",
        "required": False,
    },
    {
        "id": "module4_bootstrap_intervals",
        "category": "robustness_analysis",
        "path": MODULE4_DIR / "module4_bootstrap_intervals.csv",
        "purpose": "Base-sample bootstrap intervals for all sensor-height endpoints.",
        "required": False,
    },
    {
        "id": "module4_height_deltas",
        "category": "robustness_analysis",
        "path": MODULE4_DIR / "module4_height_deltas.csv",
        "purpose": "Paired 35/65 mm changes from the nominal 50 mm acquisition height.",
        "required": False,
    },
    {
        "id": "module4_architecture_differences",
        "category": "robustness_analysis",
        "path": MODULE4_DIR / "module4_architecture_differences.csv",
        "purpose": "Paired raw FNO-minus-DeepONet differences at each height.",
        "required": False,
    },
    {
        "id": "module4_condition_specific",
        "category": "robustness_analysis",
        "path": MODULE4_DIR / "module4_condition_specific.csv",
        "purpose": "Condition-specific sensor-height sensitivity point estimates.",
        "required": False,
    },
    {
        "id": "module4_support_diagnostics",
        "category": "robustness_analysis",
        "path": MODULE4_DIR / "module4_support_diagnostics.csv",
        "purpose": "Nondegenerate-support and zero-fallback counts by model and height.",
        "required": False,
    },
    {
        "id": "module4_bootstrap_replicates",
        "category": "robustness_analysis",
        "path": MODULE4_DIR / "module4_bootstrap_replicates.npz",
        "purpose": "Saved 5,000 grouped bootstrap values for all Module 4 endpoints.",
        "required": False,
    },
    {
        "id": "module4_summary",
        "category": "robustness_analysis",
        "path": MODULE4_DIR / "module4_summary.json",
        "purpose": "Machine-readable Module 4 results, caveats, and claim boundaries.",
        "required": False,
    },
    {
        "id": "module4_validation",
        "category": "derived_analysis_validation",
        "path": MODULE4_DIR / "module4_validation.json",
        "purpose": "Cohort, nominal replay, checkpoint provenance, and safety validation for Module 4.",
        "required": False,
    },
    {
        "id": "module4_figure_png",
        "category": "paper_figure",
        "path": MODULE4_DIR / "module4_sensor_height_ood.png",
        "purpose": "Raster preview of the sensor-height sensitivity figure.",
        "required": False,
    },
    {
        "id": "module4_figure_pdf",
        "category": "paper_figure",
        "path": MODULE4_DIR / "module4_sensor_height_ood.pdf",
        "purpose": "Vector paper-ready sensor-height sensitivity figure.",
        "required": False,
    },
    {
        "id": "module6_readme",
        "category": "paper_evidence_bundle",
        "path": MODULE6_DIR / "README.md",
        "purpose": "Scope, endpoint-selection, and statistical-use guide for the frozen paper evidence bundle.",
        "required": False,
    },
    {
        "id": "module6_table1_csv",
        "category": "paper_table",
        "path": MODULE6_DIR / "table1_overall_model_comparison.csv",
        "purpose": "Exact overall FNO/DeepONet estimates, intervals, and paired differences.",
        "required": False,
    },
    {
        "id": "module6_table1_tex",
        "category": "paper_table",
        "path": MODULE6_DIR / "table1_overall_model_comparison.tex",
        "purpose": "Paper-ready LaTeX rendering of the overall comparison table.",
        "required": False,
    },
    {
        "id": "module6_table2_csv",
        "category": "paper_table",
        "path": MODULE6_DIR / "table2_conduction_failure.csv",
        "purpose": "Exact conduction-slowing subset field and localization endpoints.",
        "required": False,
    },
    {
        "id": "module6_table2_tex",
        "category": "paper_table",
        "path": MODULE6_DIR / "table2_conduction_failure.tex",
        "purpose": "Paper-ready LaTeX rendering of the conduction-failure table.",
        "required": False,
    },
    {
        "id": "module6_table3_csv",
        "category": "paper_table",
        "path": MODULE6_DIR / "table3_v1f_failure_decomposition.csv",
        "purpose": "Exact magnetic-identifiability, oracle-support, and deployed-support decomposition.",
        "required": False,
    },
    {
        "id": "module6_table3_tex",
        "category": "paper_table",
        "path": MODULE6_DIR / "table3_v1f_failure_decomposition.tex",
        "purpose": "Paper-ready LaTeX rendering of the failure-decomposition table.",
        "required": False,
    },
    {
        "id": "module6_table4_csv",
        "category": "paper_table",
        "path": MODULE6_DIR / "table4_sensor_height_sensitivity.csv",
        "purpose": "Exact optional sensor-standoff sensitivity endpoints and paired changes.",
        "required": False,
    },
    {
        "id": "module6_table4_tex",
        "category": "paper_table",
        "path": MODULE6_DIR / "table4_sensor_height_sensitivity.tex",
        "purpose": "Paper-ready LaTeX rendering of the sensor-standoff sensitivity table.",
        "required": False,
    },
    {
        "id": "module6_figure1_png",
        "category": "paper_evidence_figure",
        "path": MODULE6_DIR / "figure1_failure_decomposition.png",
        "purpose": "Byte-identical raster copy of the validated failure-decomposition figure.",
        "required": False,
    },
    {
        "id": "module6_figure1_pdf",
        "category": "paper_evidence_figure",
        "path": MODULE6_DIR / "figure1_failure_decomposition.pdf",
        "purpose": "Byte-identical vector copy of the validated failure-decomposition figure.",
        "required": False,
    },
    {
        "id": "module6_figure2_png",
        "category": "paper_evidence_figure",
        "path": MODULE6_DIR / "figure2_sensor_height_sensitivity.png",
        "purpose": "Byte-identical raster copy of the validated sensor-standoff figure.",
        "required": False,
    },
    {
        "id": "module6_figure2_pdf",
        "category": "paper_evidence_figure",
        "path": MODULE6_DIR / "figure2_sensor_height_sensitivity.pdf",
        "purpose": "Byte-identical vector copy of the validated sensor-standoff figure.",
        "required": False,
    },
    {
        "id": "module6_paper_claims",
        "category": "paper_evidence_bundle",
        "path": MODULE6_DIR / "paper_claims.json",
        "purpose": "Source-linked, scope-bounded candidate claims generated from the exact tables.",
        "required": False,
    },
    {
        "id": "module6_captions",
        "category": "paper_evidence_bundle",
        "path": MODULE6_DIR / "captions.json",
        "purpose": "Scope-bounded candidate table and figure captions.",
        "required": False,
    },
    {
        "id": "module6_source_manifest",
        "category": "paper_evidence_provenance",
        "path": MODULE6_DIR / "source_manifest.json",
        "purpose": "Source paths, sizes, hashes, and frozen-contract provenance for the bundle.",
        "required": False,
    },
    {
        "id": "module6_bundle_manifest",
        "category": "paper_evidence_provenance",
        "path": MODULE6_DIR / "bundle_manifest.json",
        "purpose": "Hashes and sizes for all content artifacts in the paper evidence bundle.",
        "required": False,
    },
    {
        "id": "module6_validation",
        "category": "derived_analysis_validation",
        "path": MODULE6_DIR / "validation_report.json",
        "purpose": "Numeric, provenance, scope-language, figure-copy, and safety validation for Module 6.",
        "required": False,
    },
    {
        "id": "source_module6_bundle_builder",
        "category": "source_code",
        "path": Path("scripts/157_build_icbinb_module6_paper_bundle.py"),
        "purpose": "Deterministic no-training builder and fail-closed validator for the paper evidence bundle.",
    },
    {
        "id": "fno_results",
        "category": "result",
        "path": NEURAL_DIR / "fno_field_parametric_results.csv",
        "purpose": "Existing FNO train/validation/test aggregate metrics.",
    },
    {
        "id": "deeponet_predictions_scalar",
        "category": "prediction",
        "path": NEURAL_DIR / "deeponet_predictions.csv",
        "purpose": "Saved plain-DeepONet test scalar and metadata predictions.",
    },
    {
        "id": "deeponet_results",
        "category": "result",
        "path": NEURAL_DIR / "deeponet_results.csv",
        "purpose": "Existing plain-DeepONet test aggregate metrics.",
    },
    {
        "id": "deeponet_training_eval",
        "category": "selection_evidence",
        "path": NEURAL_DIR / "deeponet_training_eval.csv",
        "purpose": "Train/validation/test evaluation used for validation-only selection.",
    },
    {
        "id": "deeponet_map_cache_metadata",
        "category": "prediction",
        "path": NEURAL_DIR / "deeponet_full_grid_prediction_cache.json",
        "purpose": "Metadata for saved DeepONet full-grid validation/test maps.",
    },
    {
        "id": "deeponet_map_cache_index",
        "category": "prediction",
        "path": NEURAL_DIR / "deeponet_full_grid_prediction_index.csv",
        "purpose": "Row alignment for saved DeepONet full-grid maps.",
    },
    {
        "id": "deeponet_map_cache_arrays",
        "category": "prediction",
        "path": NEURAL_DIR / "deeponet_full_grid_prediction_maps.npz",
        "purpose": "Saved DeepONet LAT/APD90/patch full-grid maps.",
    },
    {
        "id": "deeponet_mask_calibration",
        "category": "calibration",
        "path": NEURAL_DIR / "deeponet_mask_calibration.json",
        "purpose": "Existing validation-only plain-DeepONet threshold audit.",
    },
    {
        "id": "mask_refine_predictions",
        "category": "prediction",
        "path": NEURAL_DIR / "mask_refine_predictions.csv",
        "purpose": "Saved test predictions for mask-refinement models and extractors.",
    },
    {
        "id": "mask_refine_results",
        "category": "result",
        "path": NEURAL_DIR / "mask_refine_results.csv",
        "purpose": "Validation/test mask-refinement metrics and threshold evidence.",
    },
    {
        "id": "mask_refine_results_json",
        "category": "calibration",
        "path": NEURAL_DIR / "mask_refine_results.json",
        "purpose": "Saved validation-selected balanced and sparse thresholds.",
    },
    {
        "id": "v1f_conduction_alignment_rows",
        "category": "failure_decomposition",
        "path": V1F_DIR / "conduction_metric_alignment.csv",
        "purpose": "V1F deployed conduction-metric alignment rows.",
    },
    {
        "id": "v1f_magnetic_identifiability",
        "category": "failure_decomposition",
        "path": V1F_DIR / "magnetic_identifiability_results.csv",
        "purpose": "Grouped magnetic-descriptor identifiability results.",
    },
    {
        "id": "v1f_oracle_rows",
        "category": "failure_decomposition",
        "path": V1F_DIR / "oracle_apd_lat_decomposition_rows.csv",
        "purpose": "Row-level oracle/deployed support decomposition.",
    },
    {
        "id": "v1f_oracle_summary",
        "category": "failure_decomposition",
        "path": V1F_DIR / "oracle_apd_lat_decomposition_summary.json",
        "purpose": "Source summary for the V1F failure-decomposition result.",
    },
    {
        "id": "source_fno_evaluator",
        "category": "source_code",
        "path": Path("scripts/123_evaluate_v1e_500_field_parametric.py"),
        "purpose": "Legacy FNO metric implementation for parity checks.",
    },
    {
        "id": "source_deeponet_evaluator",
        "category": "source_code",
        "path": Path("scripts/129_evaluate_v1e_500_deeponet.py"),
        "purpose": "Legacy plain-DeepONet metric implementation for parity checks.",
    },
    {
        "id": "source_refinement_evaluator",
        "category": "source_code",
        "path": Path("scripts/138_evaluate_v1e_500_deeponet_mask_refinement.py"),
        "purpose": "Mask-refinement extraction and validation selection implementation.",
    },
]


def rel(path: Path) -> str:
    return path.as_posix()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with absolute(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with absolute(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def summarize_csv(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    columns = list(rows[0]) if rows else []
    summary: dict[str, Any] = {"row_count": len(rows), "columns": columns}
    for key in ("model", "split", "combined_variant", "measurement_variant"):
        if key in columns:
            summary[f"counts_by_{key}"] = dict(sorted(Counter(row[key] for row in rows).items()))
    if "base_sample_id" in columns:
        summary["unique_base_sample_ids"] = len({row["base_sample_id"] for row in rows})
        if "split" in columns:
            by_split: dict[str, set[str]] = {}
            for row in rows:
                by_split.setdefault(row["split"], set()).add(row["base_sample_id"])
            summary["unique_base_sample_ids_by_split"] = {
                key: len(value) for key, value in sorted(by_split.items())
            }
    return summary


def metric_by_model(
    rows: list[dict[str, str]],
    *,
    split: str,
    target: str,
    metric: str,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in rows:
        if row.get("split") != split or row.get("target") != target:
            continue
        value = finite_float(row.get(metric))
        if value is not None:
            values[row["model"]] = value
    return values


def select_field_models() -> dict[str, Any]:
    fno_rows = read_csv(NEURAL_DIR / "fno_field_parametric_results.csv")
    deep_rows = read_csv(NEURAL_DIR / "deeponet_training_eval.csv")
    evidence: dict[str, Any] = {}
    for family, rows in (("fno", fno_rows), ("deeponet", deep_rows)):
        lat = metric_by_model(rows, split="val", target="LAT", metric="r2")
        apd90 = metric_by_model(rows, split="val", target="APD90", metric="r2")
        common = sorted(set(lat) & set(apd90))
        scores = {model: (lat[model] + apd90[model]) / 2.0 for model in common}
        winner = max(scores, key=scores.get)
        evidence[family] = {
            "criterion": "maximum validation mean of LAT R2 and APD90 R2",
            "winner": winner,
            "scores": scores,
            "validation_LAT_R2": lat,
            "validation_APD90_R2": apd90,
        }
    return evidence


def select_localization_models() -> dict[str, Any]:
    fno_rows = read_csv(NEURAL_DIR / "fno_field_parametric_results.csv")
    deep_rows = read_csv(NEURAL_DIR / "deeponet_training_eval.csv")
    evidence: dict[str, Any] = {}
    for family, rows in (("fno", fno_rows), ("deeponet", deep_rows)):
        scores = metric_by_model(rows, split="val", target="patch_mask", metric="dice")
        winner = max(scores, key=scores.get)
        evidence[family] = {
            "criterion": "maximum validation patch-mask micro-Dice at fixed threshold 0.5",
            "winner": winner,
            "scores": scores,
        }
    return evidence


def select_refinement_model() -> dict[str, Any]:
    rows = read_csv(NEURAL_DIR / "mask_refine_results.csv")
    candidates: dict[str, dict[str, float]] = {}
    for row in rows:
        if not (
            row.get("split") == "val"
            and row.get("method") == "calibrated_balanced_threshold"
            and row.get("metric_scope") == "valid_grid_pixels"
        ):
            continue
        dice = finite_float(row.get("dice"))
        delta_r2 = finite_float(row.get("APD90_patch_delta_R2_pred_mask_pred_apd"))
        false_positive = finite_float(row.get("homogeneous_false_positive_area"))
        threshold = finite_float(row.get("threshold"))
        if None in (dice, delta_r2, false_positive, threshold):
            continue
        eligible = dice >= 0.25 and false_positive <= 0.05
        score = delta_r2 + 0.35 * dice - 0.8 * false_positive if eligible else -math.inf
        candidates[row["model"]] = {
            "threshold": threshold,
            "validation_dice": dice,
            "validation_APD90_patch_delta_R2": delta_r2,
            "validation_homogeneous_false_positive_area": false_positive,
            "selection_score_balanced": score,
        }
    winner = max(candidates, key=lambda model: candidates[model]["selection_score_balanced"])
    return {
        "criterion": (
            "validation APD90 patch-delta R2 + 0.35*Dice - 0.8*homogeneous "
            "false-positive area, constrained to Dice >= 0.25 and false-positive area <= 0.05"
        ),
        "winner": winner,
        "winner_threshold": candidates[winner]["threshold"],
        "candidates": candidates,
    }


def validate_split(errors: list[str]) -> dict[str, Any]:
    payload = read_json(SPLIT_JSON)
    expected_base = {"train": 350, "val": 75, "test": 75}
    expected_rows = {"train": 1050, "val": 225, "test": 225}
    if payload.get("base_counts_by_split") != expected_base:
        errors.append(
            f"Grouped split base counts differ: {payload.get('base_counts_by_split')}"
        )
    if payload.get("row_counts_by_split") != expected_rows:
        errors.append(
            f"Grouped split row counts differ: {payload.get('row_counts_by_split')}"
        )
    assignments = payload.get("base_split", {})
    if len(assignments) != 500 or len(set(assignments)) != 500:
        errors.append("Grouped split does not contain exactly 500 unique base IDs.")
    return {
        "source": rel(SPLIT_JSON),
        "independent_unit": "base_sample_id",
        "base_counts_by_split": payload.get("base_counts_by_split"),
        "row_counts_by_split": payload.get("row_counts_by_split"),
        "expected_test_base_count": 75,
        "expected_test_rows_all_variants": 225,
        "expected_test_rows_primary_variant": 75,
        "grouping_rule": "All measurement variants from one base_sample_id remain in one split.",
    }


def validate_primary_variant(errors: list[str]) -> dict[str, Any]:
    rows = read_csv(SAMPLE_INDEX)
    variants = Counter(row["combined_variant"] for row in rows)
    if dict(sorted(variants.items())) != {variant: 500 for variant in sorted(EXPECTED_VARIANTS)}:
        errors.append(f"Unexpected sample-index variant counts: {dict(variants)}")
    primary_test = [
        row
        for row in rows
        if row.get("split") == "test" and row.get("combined_variant") == PRIMARY_VARIANT
    ]
    unique_ids = {row["base_sample_id"] for row in primary_test}
    condition_counts = Counter(row["condition"] for row in primary_test)
    if len(primary_test) != 75 or len(unique_ids) != 75:
        errors.append(
            "Primary variant does not yield exactly one row for each of 75 test base IDs."
        )
    return {
        "primary_variant": PRIMARY_VARIANT,
        "primary_active_channels": [
            "direct_bz_50mm",
            "direct_axial_grad_30mm_baseline",
        ],
        "rationale": (
            "Predeclared clean two-channel magnetic configuration; it yields one deterministic "
            "row per base case and avoids treating augmented measurement rows as independent."
        ),
        "primary_test_row_count": len(primary_test),
        "primary_test_unique_base_count": len(unique_ids),
        "primary_test_condition_counts": dict(sorted(condition_counts.items())),
        "secondary_variant_policy": (
            "Report the three variants separately as sensitivity analyses. Do not pool them as "
            "independent biological observations. If jointly bootstrapped, carry all variants "
            "inside the sampled base_sample_id cluster."
        ),
        "available_variants": dict(sorted(variants.items())),
    }


def validate_prediction_coverage(errors: list[str]) -> dict[str, Any]:
    sources = {
        "fno_scalar": {
            "path": NEURAL_DIR / "fno_field_parametric_predictions.csv",
            "models": [FIELD_FNO, LOCALIZATION_FNO],
            "splits": ["test"],
        },
        "deeponet_scalar": {
            "path": NEURAL_DIR / "deeponet_predictions.csv",
            "models": [FIELD_DEEPONET, LOCALIZATION_DEEPONET],
            "splits": ["test"],
        },
        "deeponet_full_grid": {
            "path": NEURAL_DIR / "deeponet_full_grid_prediction_index.csv",
            "models": [FIELD_DEEPONET, LOCALIZATION_DEEPONET],
            "splits": ["val", "test"],
        },
        "mask_refinement": {
            "path": NEURAL_DIR / "mask_refine_predictions.csv",
            "models": [REFINEMENT_DEEPONET],
            "splits": ["test"],
        },
    }
    coverage: dict[str, Any] = {}
    for source_id, spec in sources.items():
        rows = read_csv(spec["path"])
        source_report: dict[str, Any] = {"path": rel(spec["path"]), "groups": {}}
        for model in spec["models"]:
            for split in spec["splits"]:
                selected = [
                    row
                    for row in rows
                    if row.get("model") == model
                    and row.get("split") == split
                    and row.get("combined_variant") == PRIMARY_VARIANT
                ]
                base_ids = [row.get("base_sample_id", "") for row in selected]
                expected = 75
                group_id = f"{model}|{split}|{PRIMARY_VARIANT}"
                source_report["groups"][group_id] = {
                    "row_count": len(selected),
                    "unique_base_sample_ids": len(set(base_ids)),
                    "duplicate_base_sample_ids": len(base_ids) - len(set(base_ids)),
                    "expected_rows": expected,
                }
                if len(selected) != expected or len(set(base_ids)) != expected:
                    errors.append(
                        f"Prediction coverage failure for {source_id} {group_id}: "
                        f"rows={len(selected)}, unique_base_ids={len(set(base_ids))}, expected={expected}"
                    )
        coverage[source_id] = source_report

    cache_metadata = read_json(NEURAL_DIR / "deeponet_full_grid_prediction_cache.json")
    cache_models = set(cache_metadata.get("models", []))
    missing_cache_models = sorted({FIELD_DEEPONET, LOCALIZATION_DEEPONET} - cache_models)
    if missing_cache_models:
        errors.append(
            f"Selected DeepONet models are missing from the full-grid cache: {missing_cache_models}"
        )
    coverage["deeponet_full_grid"]["cache_metadata_models"] = sorted(cache_models)
    coverage["deeponet_full_grid"]["selected_models_missing_from_cache"] = missing_cache_models
    return coverage


def inventory_artifacts(errors: list[str]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    csv_summary_ids = {
        "deeponet_sample_index",
        "fno_predictions_scalar",
        "deeponet_predictions_scalar",
        "deeponet_map_cache_index",
        "mask_refine_predictions",
        "paired_test_cohort_table",
        "module2_case_statistics",
        "module2_point_estimates",
        "module2_condition_specific",
        "module2_bootstrap_intervals",
        "module2_paired_differences",
        "module2_support_diagnostics",
        "module3_identifiability_predictions",
        "module3_failure_decomposition",
        "module3_secondary_diagnostics",
        "module4_sensor_height_manifest",
        "module4_nominal_projection_parity",
        "module4_prediction_index",
        "module4_case_statistics",
        "module4_point_estimates",
        "module4_bootstrap_intervals",
        "module4_height_deltas",
        "module4_architecture_differences",
        "module4_condition_specific",
        "module4_support_diagnostics",
        "module6_table1_csv",
        "module6_table2_csv",
        "module6_table3_csv",
        "module6_table4_csv",
    }
    for spec in ARTIFACT_SPECS:
        path = absolute(spec["path"])
        required = bool(spec.get("required", True))
        record: dict[str, Any] = {
            "id": spec["id"],
            "category": spec["category"],
            "path": rel(spec["path"]),
            "purpose": spec["purpose"],
            "required": required,
            "exists": path.is_file(),
        }
        if not path.is_file():
            if required:
                errors.append(f"Missing required artifact: {rel(spec['path'])}")
        else:
            record["bytes"] = path.stat().st_size
            record["sha256"] = sha256(path)
            if spec["id"] in csv_summary_ids:
                record["csv_summary"] = summarize_csv(spec["path"])
        inventory.append(record)
    return inventory


def build_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    split = validate_split(errors)
    measurement = validate_primary_variant(errors)
    prediction_coverage = validate_prediction_coverage(errors)
    field_selection = select_field_models()
    localization_selection = select_localization_models()
    refinement_selection = select_refinement_model()

    expected = {
        "field_fno": FIELD_FNO,
        "field_deeponet": FIELD_DEEPONET,
        "localization_fno": LOCALIZATION_FNO,
        "localization_deeponet": LOCALIZATION_DEEPONET,
        "refinement_deeponet": REFINEMENT_DEEPONET,
    }
    observed = {
        "field_fno": field_selection["fno"]["winner"],
        "field_deeponet": field_selection["deeponet"]["winner"],
        "localization_fno": localization_selection["fno"]["winner"],
        "localization_deeponet": localization_selection["deeponet"]["winner"],
        "refinement_deeponet": refinement_selection["winner"],
    }
    if observed != expected:
        errors.append(f"Validation-only selections changed: expected {expected}, observed {observed}")
    if not math.isclose(
        refinement_selection["winner_threshold"],
        REFINEMENT_BALANCED_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        errors.append(
            "Validation-selected refinement threshold changed: "
            f"expected {REFINEMENT_BALANCED_THRESHOLD}, "
            f"observed {refinement_selection['winner_threshold']}"
        )

    inventory = inventory_artifacts(errors)
    inventory_by_id = {record["id"]: record for record in inventory}
    fno_cache_artifact_ids = [
        "fno_map_cache_index",
        "fno_map_cache_arrays",
        "fno_map_cache_metadata",
    ]
    fno_cache_available = all(
        inventory_by_id[artifact_id]["exists"] for artifact_id in fno_cache_artifact_ids
    )
    cohort_artifact_ids = [
        "paired_test_cohort_table",
        "paired_test_cohort_maps",
        "paired_test_cohort_validation",
    ]
    cohort_artifact_presence = {
        artifact_id: inventory_by_id[artifact_id]["exists"]
        for artifact_id in cohort_artifact_ids
    }
    cohort_available = all(cohort_artifact_presence.values())
    if any(cohort_artifact_presence.values()) and not cohort_available:
        errors.append(
            f"Canonical cohort artifact set is incomplete: {cohort_artifact_presence}"
        )
    cohort_validation: dict[str, Any] | None = None
    if cohort_available:
        cohort_validation = read_json(
            Path("outputs/icbinb_required_analysis/cohort_validation.json")
        )
        if not cohort_validation.get("passed"):
            errors.append("Canonical cohort validation does not pass.")
        if cohort_validation.get("row_count") != 75:
            errors.append(
                "Canonical cohort validation does not report exactly 75 rows."
            )
        if cohort_validation.get("unique_base_sample_ids") != 75:
            errors.append(
                "Canonical cohort validation does not report exactly 75 unique base IDs."
            )
    module2_artifact_ids = [
        "module2_case_statistics",
        "module2_point_estimates",
        "module2_condition_specific",
        "module2_bootstrap_intervals",
        "module2_paired_differences",
        "module2_support_diagnostics",
        "module2_bootstrap_replicates",
        "module2_summary",
        "module2_validation",
    ]
    module2_artifact_presence = {
        artifact_id: inventory_by_id[artifact_id]["exists"]
        for artifact_id in module2_artifact_ids
    }
    module2_available = all(module2_artifact_presence.values())
    if any(module2_artifact_presence.values()) and not module2_available:
        errors.append(f"Module 2 artifact set is incomplete: {module2_artifact_presence}")
    module2_validation: dict[str, Any] | None = None
    if module2_available:
        module2_validation = read_json(
            Path("outputs/icbinb_required_analysis/module2_validation.json")
        )
        if not module2_validation.get("passed"):
            errors.append("Module 2 validation does not pass.")
        if module2_validation.get("cohort_rows") != 75:
            errors.append("Module 2 validation does not report exactly 75 cohort rows.")
        if module2_validation.get("unique_base_sample_ids") != 75:
            errors.append("Module 2 validation does not report exactly 75 unique base IDs.")
        observed_bootstrap = module2_validation.get("bootstrap", {})
        if observed_bootstrap.get("replicates") != 5000:
            errors.append("Module 2 validation does not report 5,000 bootstrap replicates.")
        if observed_bootstrap.get("random_seed") != 20260901:
            errors.append("Module 2 validation reports an unexpected bootstrap seed.")
    module3_artifact_ids = [
        "module3_identifiability_predictions",
        "module3_failure_decomposition",
        "module3_secondary_diagnostics",
        "module3_bootstrap_replicates",
        "module3_summary",
        "module3_validation",
        "module3_figure_png",
        "module3_figure_pdf",
    ]
    module3_artifact_presence = {
        artifact_id: inventory_by_id[artifact_id]["exists"]
        for artifact_id in module3_artifact_ids
    }
    module3_available = all(module3_artifact_presence.values())
    if any(module3_artifact_presence.values()) and not module3_available:
        errors.append(f"Module 3 artifact set is incomplete: {module3_artifact_presence}")
    module3_validation: dict[str, Any] | None = None
    if module3_available:
        module3_validation = read_json(
            Path("outputs/icbinb_required_analysis/module3_validation.json")
        )
        if not module3_validation.get("passed"):
            errors.append("Module 3 validation does not pass.")
        if module3_validation.get("primary_variant") != PRIMARY_VARIANT:
            errors.append("Module 3 validation reports an unexpected primary variant.")
        observed_bootstrap = module3_validation.get("bootstrap", {})
        if observed_bootstrap.get("replicates") != 5000:
            errors.append("Module 3 validation does not report 5,000 bootstrap replicates.")
        ident = module3_validation.get("identifiability", {})
        if ident.get("primary_test_unique_base_ids") != 75:
            errors.append("Module 3 identifiability does not report 75 primary test bases.")
        oracle = module3_validation.get("oracle_decomposition", {})
        if oracle.get("conduction_test_unique_base_ids") != 18:
            errors.append("Module 3 oracle decomposition does not report 18 conduction bases.")
    module4_artifact_ids = [
        "module4_sensor_height_manifest",
        "module4_generation_validation",
        "module4_nominal_projection_parity",
        "module4_prediction_index",
        "module4_prediction_maps",
        "module4_case_statistics",
        "module4_point_estimates",
        "module4_bootstrap_intervals",
        "module4_height_deltas",
        "module4_architecture_differences",
        "module4_condition_specific",
        "module4_support_diagnostics",
        "module4_bootstrap_replicates",
        "module4_summary",
        "module4_validation",
        "module4_figure_png",
        "module4_figure_pdf",
    ]
    module4_artifact_presence = {
        artifact_id: inventory_by_id[artifact_id]["exists"]
        for artifact_id in module4_artifact_ids
    }
    module4_available = all(module4_artifact_presence.values())
    if any(module4_artifact_presence.values()) and not module4_available:
        errors.append(f"Module 4 artifact set is incomplete: {module4_artifact_presence}")
    module4_validation: dict[str, Any] | None = None
    module4_summary: dict[str, Any] | None = None
    module4_generation: dict[str, Any] | None = None
    if module4_available:
        module4_validation = read_json(MODULE4_DIR / "module4_validation.json")
        module4_summary = read_json(MODULE4_DIR / "module4_summary.json")
        module4_generation = read_json(MODULE4_DIR / "generation_validation.json")
        if not module4_validation.get("passed"):
            errors.append("Module 4 validation does not pass.")
        if not module4_summary.get("passed") or module4_summary.get("status") != "complete":
            errors.append("Module 4 summary is not complete and passing.")
        if not module4_generation.get("passed"):
            errors.append("Module 4 input-generation validation does not pass.")
        if module4_validation.get("case_rows") != 450:
            errors.append("Module 4 validation does not report 450 model-height-case rows.")
        if module4_validation.get("grouped_base_cases_per_model_height") != 75:
            errors.append("Module 4 validation does not report 75 grouped cases per model-height.")
        if module4_generation.get("nominal_bit_exact_cases") != 75:
            errors.append("Module 4 generation does not report 75 bit-exact nominal reprojections.")
        if not module4_validation.get("nominal_prediction_replay_passed"):
            errors.append("Module 4 nominal frozen-prediction replay does not pass.")
        if module4_summary.get("heights_mm") != [35, 50, 65]:
            errors.append("Module 4 summary reports unexpected sensor heights.")
        bootstrap = module4_summary.get("bootstrap", {})
        if bootstrap.get("replicates") != 5000 or bootstrap.get("seed") != 20260901:
            errors.append("Module 4 summary reports an unexpected grouped-bootstrap configuration.")
    module6_error_count_before = len(errors)
    module6_artifact_ids = [
        "module6_readme",
        "module6_table1_csv",
        "module6_table1_tex",
        "module6_table2_csv",
        "module6_table2_tex",
        "module6_table3_csv",
        "module6_table3_tex",
        "module6_table4_csv",
        "module6_table4_tex",
        "module6_figure1_png",
        "module6_figure1_pdf",
        "module6_figure2_png",
        "module6_figure2_pdf",
        "module6_paper_claims",
        "module6_captions",
        "module6_source_manifest",
        "module6_bundle_manifest",
        "module6_validation",
    ]
    module6_artifact_presence = {
        artifact_id: inventory_by_id[artifact_id]["exists"]
        for artifact_id in module6_artifact_ids
    }
    module6_available = all(module6_artifact_presence.values())
    if any(module6_artifact_presence.values()) and not module6_available:
        errors.append(f"Module 6 artifact set is incomplete: {module6_artifact_presence}")
    module6_validation: dict[str, Any] | None = None
    module6_manifest: dict[str, Any] | None = None
    module6_source_manifest: dict[str, Any] | None = None
    if module6_available:
        module6_validation = read_json(MODULE6_DIR / "validation_report.json")
        module6_manifest = read_json(MODULE6_DIR / "bundle_manifest.json")
        module6_source_manifest = read_json(MODULE6_DIR / "source_manifest.json")
        if (
            not module6_validation.get("passed")
            or module6_validation.get("module") != 6
            or not module6_validation.get("ready_for_paper_writing")
        ):
            errors.append("Module 6 validation is not complete, passing, and paper-ready.")
        source_validation = module6_validation.get("source_validation", {})
        table_validation = module6_validation.get("table_validation", {})
        scope_audit = module6_validation.get("scope_language_audit", {})
        figure_validation = module6_validation.get("figure_validation", {})
        safety = module6_validation.get("safety", {})
        if not source_validation.get("all_inventory_hash_checks_passed"):
            errors.append("Module 6 source inventory-hash validation does not pass.")
        if not table_validation.get("passed") or not table_validation.get(
            "csv_roundtrip_matches_generated_tables"
        ):
            errors.append("Module 6 paper-table validation does not pass.")
        if not scope_audit.get("passed"):
            errors.append("Module 6 scope-language audit does not pass.")
        if len(figure_validation) != 4 or not all(
            row.get("byte_identical") for row in figure_validation.values()
        ):
            errors.append("Module 6 figure-copy validation does not pass for all four files.")
        if any(
            safety.get(key)
            for key in (
                "training_performed",
                "model_inference_performed",
                "simulation_performed",
                "legacy_outputs_overwritten",
            )
        ):
            errors.append("Module 6 safety report contains a prohibited action.")
        expected_content_names = {
            Path(inventory_by_id[artifact_id]["path"]).name
            for artifact_id in module6_artifact_ids
            if artifact_id not in {"module6_bundle_manifest", "module6_validation"}
        }
        manifest_rows = module6_manifest.get("content_artifacts", [])
        manifest_names = {str(row.get("path")) for row in manifest_rows}
        if (
            module6_manifest.get("content_artifact_count") != len(expected_content_names)
            or manifest_names != expected_content_names
        ):
            errors.append("Module 6 bundle manifest does not enumerate the exact content set.")
        for row in manifest_rows:
            bundled_path = absolute(MODULE6_DIR / str(row.get("path")))
            if not bundled_path.is_file():
                errors.append(f"Module 6 manifest file is missing: {bundled_path}")
                continue
            if bundled_path.stat().st_size != row.get("bytes") or sha256(bundled_path) != row.get("sha256"):
                errors.append(f"Module 6 manifest hash or size mismatch: {bundled_path}")
        source_hash_exemptions = {"evaluation_contract", "artifact_inventory"}
        for row in module6_source_manifest.get("sources", []):
            if row.get("source_id") in source_hash_exemptions:
                continue
            source_path = absolute(Path(str(row.get("path"))))
            if not source_path.is_file() or sha256(source_path) != row.get("sha256"):
                errors.append(
                    f"Module 6 frozen source is missing or changed: {row.get('source_id')}"
                )
    module6_integrity_passed = bool(
        module6_available and len(errors) == module6_error_count_before
    )
    if not fno_cache_available:
        warnings.append(
            "No complete saved FNO full-grid map cache was found. Module 2 must run "
            "frozen-checkpoint inference for the two locked FNO representatives into a new "
            "isolated cache before base-level bootstrap of global field R2 or patch Dice. "
            "This is inference only, not training."
        )
    warnings.append(
        "Historical headline values selected different test-best checkpoints by metric. The "
        "confirmatory analysis instead uses the validation-selected endpoint representatives "
        "locked here; historical family maxima are descriptive only."
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    contract: dict[str, Any] = {
        "schema_version": "1.0.0",
        "generated_at_utc": generated_at,
        "purpose": "Locked evaluation contract for the mandatory ICBINB-BIO tiny-paper analysis.",
        "scope": {
            "repository": ".",
            "benchmark_description": "controlled synthetic operator-learning benchmark",
            "paper_role": "primary empirical system",
            "excluded_claims": [
                "patient-specific",
                "clinically validated",
                "real-SQUID validated",
                "intracellular-current reconstruction",
                "physics-informed training",
            ],
        },
        "split": split,
        "measurement_policy": measurement,
        "prediction_coverage": prediction_coverage,
        "model_roles": {
            "global_field_comparison": {
                "selection_partition": "validation",
                "selection_criterion": field_selection["fno"]["criterion"],
                "fno": {
                    "model_id": FIELD_FNO,
                    "checkpoint_artifact_id": "fno_checkpoint_field",
                    "saved_scalar_predictions_artifact_id": "fno_predictions_scalar",
                    "saved_full_grid_maps": fno_cache_available,
                    "map_artifact_ids": (
                        ["fno_map_cache_index", "fno_map_cache_arrays"]
                        if fno_cache_available
                        else []
                    ),
                },
                "deeponet": {
                    "model_id": FIELD_DEEPONET,
                    "checkpoint_artifact_id": "deeponet_checkpoint_field",
                    "saved_scalar_predictions_artifact_id": "deeponet_predictions_scalar",
                    "saved_full_grid_maps": True,
                    "map_artifact_ids": [
                        "deeponet_map_cache_index",
                        "deeponet_map_cache_arrays",
                    ],
                },
                "selection_evidence": field_selection,
            },
            "localization_comparison": {
                "selection_partition": "validation",
                "selection_criterion": localization_selection["fno"]["criterion"],
                "mask_extractor": {
                    "method": "pixel_probability_threshold",
                    "threshold": 0.5,
                    "threshold_origin": "fixed legacy evaluation threshold; not test-calibrated",
                },
                "fno": {
                    "model_id": LOCALIZATION_FNO,
                    "checkpoint_artifact_id": "fno_checkpoint_localization",
                    "saved_scalar_predictions_artifact_id": "fno_predictions_scalar",
                    "saved_full_grid_maps": fno_cache_available,
                    "map_artifact_ids": (
                        ["fno_map_cache_index", "fno_map_cache_arrays"]
                        if fno_cache_available
                        else []
                    ),
                },
                "deeponet": {
                    "model_id": LOCALIZATION_DEEPONET,
                    "checkpoint_artifact_id": "deeponet_checkpoint_localization",
                    "saved_scalar_predictions_artifact_id": "deeponet_predictions_scalar",
                    "saved_full_grid_maps": True,
                    "map_artifact_ids": [
                        "deeponet_map_cache_index",
                        "deeponet_map_cache_arrays",
                    ],
                },
                "selection_evidence": localization_selection,
            },
            "mask_refinement_recovery_attempt": {
                "comparison_role": "secondary recovery attempt; not part of paired FNO-versus-DeepONet architecture differences",
                "selection_partition": "validation",
                "model_id": REFINEMENT_DEEPONET,
                "checkpoint_artifact_id": "deeponet_checkpoint_refinement",
                "predictions_artifact_id": "mask_refine_predictions",
                "extractor": {
                    "method": "calibrated_balanced_threshold",
                    "threshold": REFINEMENT_BALANCED_THRESHOLD,
                    "threshold_origin": "validation-only sweep",
                },
                "selection_evidence": refinement_selection,
            },
        },
        "primary_metrics": [
            {
                "id": "LAT_global_R2",
                "direction": "higher_is_better",
                "population": "75 test base cases, primary measurement variant, all finite valid-grid pixels",
                "definition": "1 - sum((true-pred)^2) / sum((true-mean(true))^2), recomputed within each bootstrap replicate",
                "model_role": "global_field_comparison",
            },
            {
                "id": "APD90_global_R2",
                "direction": "higher_is_better",
                "population": "75 test base cases, primary measurement variant, all finite valid-grid pixels",
                "definition": "1 - sum((true-pred)^2) / sum((true-mean(true))^2), recomputed within each bootstrap replicate",
                "model_role": "global_field_comparison",
            },
            {
                "id": "patch_Dice_nonhomogeneous_micro",
                "direction": "higher_is_better",
                "population": "test base cases with true_patch_present >= 0.5, primary measurement variant",
                "definition": "2*TP/(2*TP+FP+FN) over concatenated valid-grid pixels after the locked extractor",
                "model_role": "localization_comparison",
            },
            {
                "id": "patch_center_error_mean",
                "direction": "lower_is_better",
                "population": "test base cases with true_patch_present >= 0.5, primary measurement variant",
                "definition": "mean Euclidean distance between predicted and true centers in normalized [0,1] x-y coordinates",
                "model_role": "localization_comparison",
            },
            {
                "id": "homogeneous_false_positive_area_mean",
                "direction": "lower_is_better",
                "population": "homogeneous test base cases, primary measurement variant",
                "definition": "mean fraction of valid-grid pixels marked positive by the locked extractor",
                "model_role": "localization_comparison",
            },
            {
                "id": "APD90_patch_delta_deployed_R2",
                "direction": "higher_is_better",
                "population": "test base cases with true_patch_present >= 0.5, primary measurement variant",
                "definition": (
                    "R2 of true versus predicted mean APD90 inside-minus-outside the deployed "
                    "predicted support. A degenerate empty or full predicted support is assigned "
                    "0 ms recovered contrast so localization failures remain in the paired "
                    "63-case population; report degenerate-support counts and complete-case "
                    "sensitivity separately."
                ),
                "model_role": "localization_comparison",
            },
        ],
        "condition_specific_reporting": {
            "conditions": [
                "apd_repolarization_perturbation",
                "conduction_slowing_perturbation",
                "homogeneous",
                "mixed_apd_conduction_perturbation",
            ],
            "primary_failure_focus": "conduction_slowing_perturbation",
            "rule": "Use the same locked metrics and base-level resampling; report base-case counts and wider intervals.",
        },
        "bootstrap": {
            "independent_unit": "base_sample_id",
            "test_base_count": 75,
            "replicates": 5000,
            "random_seed": 20260901,
            "confidence_level": 0.95,
            "interval": "percentile",
            "paired_draws": True,
            "paired_difference_direction": "FNO_minus_DeepONet",
            "resampling_rule": (
                "Sample 75 base IDs with replacement. Preserve model, condition, target, and "
                "measurement pairing. Retain duplicate draws as distinct bootstrap clusters."
            ),
            "prohibited": [
                "resampling the 225 augmented test rows as independent cases",
                "deduplicating repeated base IDs within a bootstrap replicate",
                "test-set threshold or checkpoint selection",
            ],
        },
        "comparability": {
            "paired_endpoints": [
                "LAT_global_R2",
                "APD90_global_R2",
                "patch_Dice_nonhomogeneous_micro",
                "patch_center_error_mean",
                "homogeneous_false_positive_area_mean",
                "APD90_patch_delta_deployed_R2",
            ],
            "pairing_keys": ["base_sample_id", "combined_variant"],
            "only_shared_population_and_definition": True,
            "excluded_from_paired_architecture_claims": [
                "mask_refinement_recovery_attempt",
                "V1F oracle-support decomposition",
                "historical test-best metric-by-metric family maxima",
                "older bpc_fno_v2 repository metrics",
            ],
        },
        "module_2_status": {
            "complete": module2_available and bool(
                module2_validation is not None and module2_validation.get("passed")
            ),
            "fno_full_grid_cache_available": fno_cache_available,
            "canonical_paired_cohort_available": cohort_available,
            "canonical_paired_cohort_artifact_ids": (
                cohort_artifact_ids if cohort_available else []
            ),
            "canonical_paired_cohort_validation_passed": (
                bool(cohort_validation.get("passed"))
                if cohort_validation is not None
                else False
            ),
            "paired_bootstrap_available": module2_available,
            "paired_bootstrap_artifact_ids": (
                module2_artifact_ids if module2_available else []
            ),
            "paired_bootstrap_validation_passed": (
                bool(module2_validation.get("passed"))
                if module2_validation is not None
                else False
            ),
        },
        "required_followup_for_module_2": ([] if module2_available else ([
            {
                "id": (
                    "run_paired_base_case_bootstrap"
                    if cohort_available
                    else "build_canonical_primary_variant_cohort"
                ),
                "action": (
                    "Recompute the six locked endpoints and run the 5,000-replicate paired base-case bootstrap."
                    if cohort_available
                    else "Join the locked model outputs to the grouped split and retain exactly one combined_bz_axial row for each of 75 test base IDs."
                ),
                "training_allowed": False,
                "overwrite_existing_outputs": False,
            },
        ] if fno_cache_available else [
            {
                "id": "build_fno_full_grid_cache",
                "action": "Run frozen inference for the two locked FNO checkpoints on validation/test rows and write a new isolated cache.",
                "reason": "The saved FNO CSV contains scalar summaries but not per-pixel LAT/APD90/patch maps needed for grouped field and Dice bootstrap.",
                "training_allowed": False,
                "overwrite_existing_outputs": False,
            },
            {
                "id": "build_canonical_primary_variant_cohort",
                "action": "Join the locked model outputs to the grouped split and retain exactly one combined_bz_axial row for each of 75 test base IDs.",
                "training_allowed": False,
                "overwrite_existing_outputs": False,
            },
        ])),
        "module_3_status": {
            "complete": module3_available and bool(
                module3_validation is not None and module3_validation.get("passed")
            ),
            "failure_decomposition_available": module3_available,
            "failure_decomposition_artifact_ids": (
                module3_artifact_ids if module3_available else []
            ),
            "failure_decomposition_validation_passed": (
                bool(module3_validation.get("passed"))
                if module3_validation is not None
                else False
            ),
            "selection_policy": (
                "validation-preferred descriptor classifier and contract-locked refinement model; "
                "single combined_bz_axial variant"
            ),
        },
        "required_followup_for_module_3": ([] if module3_available else [
            {
                "id": "build_controlled_failure_decomposition",
                "action": (
                    "Recompute the primary-variant magnetic identifiability, oracle-support, "
                    "and deployed-support chain with grouped base-case uncertainty."
                ),
                "training_allowed": "descriptor audit reproduction only; no neural training",
                "overwrite_existing_outputs": False,
            }
        ]),
        "module_4_status": {
            "complete": module4_available and bool(
                module4_validation is not None and module4_validation.get("passed")
            ),
            "sensor_height_ood_available": module4_available,
            "sensor_height_ood_artifact_ids": (
                module4_artifact_ids if module4_available else []
            ),
            "sensor_height_ood_validation_passed": (
                bool(module4_validation.get("passed"))
                if module4_validation is not None
                else False
            ),
            "heights_mm": (
                module4_summary.get("heights_mm", [])
                if module4_summary is not None
                else []
            ),
            "independent_base_cases": (
                module4_summary.get("base_cases", 0)
                if module4_summary is not None
                else 0
            ),
            "experiment_label": "zero-shot synthetic sensor-tissue standoff sensitivity",
            "claim_boundary": (
                "not patient-specific validation, clinical validation, real-SQUID validation, "
                "or learned geometry conditioning"
            ),
        },
        "required_followup_for_module_4": ([] if module4_available else [
            {
                "id": "run_sensor_height_ood_stress_test",
                "action": (
                    "Reproject the 75 held-out combined_bz_axial cases at 35/50/65 mm, "
                    "run frozen validation-selected field models, and bootstrap by base_sample_id."
                ),
                "training_allowed": False,
                "overwrite_existing_outputs": False,
            }
        ]),
        "module_6_status": {
            "complete": module6_integrity_passed,
            "paper_evidence_bundle_available": module6_available,
            "paper_evidence_bundle_artifact_ids": (
                module6_artifact_ids if module6_available else []
            ),
            "validation_passed": bool(
                module6_validation is not None and module6_validation.get("passed")
            ),
            "ready_for_paper_writing": bool(
                module6_validation is not None
                and module6_validation.get("ready_for_paper_writing")
                and module6_integrity_passed
            ),
            "table_count": 4 if module6_integrity_passed else 0,
            "figure_count": 2 if module6_integrity_passed else 0,
            "content_artifact_count": (
                module6_manifest.get("content_artifact_count", 0)
                if module6_manifest is not None
                else 0
            ),
        },
        "required_followup_for_module_6": ([] if module6_integrity_passed else [
            {
                "id": "build_validated_paper_evidence_bundle",
                "action": (
                    "Export the locked result tables, verified figure copies, source-linked claims, "
                    "captions, manifests, and fail-closed validation report into an isolated directory."
                ),
                "training_allowed": False,
                "overwrite_existing_outputs": False,
            }
        ]),
        "canonical_original_module_status": {
            "numbering_note": (
                "This mapping preserves the six pre-writing modules originally agreed with the user. "
                "Legacy internal module_2/module_3/module_4 keys aggregate several items or describe the optional standoff experiment."
            ),
            "1_evaluation_contract_and_artifact_inventory": {
                "complete": not errors,
                "internal_evidence": ["evaluation_contract.json", "artifact_inventory.json"],
            },
            "2_canonical_paired_test_cohort_builder": {
                "complete": bool(
                    cohort_available
                    and cohort_validation is not None
                    and cohort_validation.get("passed")
                ),
                "internal_evidence": cohort_artifact_ids if cohort_available else [],
            },
            "3_metric_reproduction_and_definition_lock": {
                "complete": bool(
                    module2_available
                    and module2_validation is not None
                    and module2_validation.get("passed")
                ),
                "internal_evidence": [
                    "module2_point_estimates",
                    "module2_condition_specific",
                    "module2_support_diagnostics",
                    "module2_validation",
                ] if module2_available else [],
            },
            "4_grouped_paired_bootstrap": {
                "complete": bool(
                    module2_available
                    and module2_validation is not None
                    and module2_validation.get("passed")
                ),
                "internal_evidence": [
                    "module2_bootstrap_intervals",
                    "module2_paired_differences",
                    "module2_bootstrap_replicates",
                    "module2_validation",
                ] if module2_available else [],
            },
            "5_v1f_failure_decomposition_result": {
                "complete": bool(
                    module3_available
                    and module3_validation is not None
                    and module3_validation.get("passed")
                ),
                "internal_evidence": module3_artifact_ids if module3_available else [],
            },
            "6_analysis_bundle_validation_and_exports": {
                "complete": module6_integrity_passed,
                "internal_evidence": module6_artifact_ids if module6_available else [],
            },
            "optional_extra_sensor_standoff_robustness": {
                "complete": bool(
                    module4_available
                    and module4_validation is not None
                    and module4_validation.get("passed")
                ),
                "internal_evidence": module4_artifact_ids if module4_available else [],
            },
        },
        "artifact_inventory_file": "outputs/icbinb_required_analysis/artifact_inventory.json",
        "validation": {
            "passed": not errors,
            "errors": errors,
            "warnings": warnings,
        },
    }

    inventory_payload = {
        "schema_version": "1.0.0",
        "generated_at_utc": generated_at,
        "repository": ".",
        "artifact_count": len(inventory),
        "artifacts": inventory,
        "known_missing_derived_artifacts": (
            ([] if fno_cache_available else [
                {
                    "id": "fno_full_grid_prediction_cache",
                    "status": "not_present",
                    "required_by": "Module 2 before grouped bootstrap of field R2 and patch Dice",
                    "safe_creation_rule": "Frozen inference only; write under outputs/icbinb_required_analysis/ and never overwrite legacy outputs.",
                }
            ])
            + ([] if cohort_available else [
                {
                    "id": "canonical_paired_test_cohort",
                    "status": "not_present",
                    "required_by": "Module 2 before grouped metric reproduction and bootstrap",
                    "safe_creation_rule": "Join only the locked combined_bz_axial test predictions at base_sample_id level; never overwrite legacy outputs.",
                }
            ])
            + ([] if module2_available else [
                {
                    "id": "module2_paired_bootstrap_analysis",
                    "status": "not_present",
                    "required_by": "Module 2 completion before paper writing",
                    "safe_creation_rule": "Use only the locked 75-case cohort, 5,000 paired base-sample draws, seed 20260901, and new isolated outputs.",
                }
            ])
            + ([] if module3_available else [
                {
                    "id": "module3_failure_decomposition",
                    "status": "not_present",
                    "required_by": "Module 3 completion before paper writing",
                    "safe_creation_rule": "Use the primary variant, validation-fixed model choices, grouped base-case bootstrap, and diagnostic-only oracle supports.",
                }
            ])
            + ([] if module4_available else [
                {
                    "id": "module4_sensor_height_ood",
                    "status": "not_present",
                    "required_by": "Module 4 acquisition-geometry robustness boundary",
                    "safe_creation_rule": (
                        "Recompute only Biot-Savart measurements for the canonical 75 held-out bases; "
                        "use frozen checkpoints, isolated outputs, and grouped base-sample bootstrap."
                    ),
                }
            ])
            + ([] if module6_integrity_passed else [
                {
                    "id": "module6_validated_paper_evidence_bundle",
                    "status": "not_present_or_not_validated",
                    "required_by": "Original Module 6 completion before paper writing",
                    "safe_creation_rule": (
                        "Package only locked analyses into isolated outputs; verify numeric parity, "
                        "source and output hashes, figure copies, scope language, and safety before publication."
                    ),
                }
            ])
        ),
        "validation": {
            "passed": not errors,
            "errors": errors,
            "warnings": warnings,
        },
    }
    return contract, inventory_payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> int:
    contract, inventory = build_contract()
    write_json(OUTPUT_DIR / "evaluation_contract.json", contract)
    write_json(OUTPUT_DIR / "artifact_inventory.json", inventory)
    summary = {
        "contract": rel((OUTPUT_DIR / "evaluation_contract.json").relative_to(ROOT)),
        "inventory": rel((OUTPUT_DIR / "artifact_inventory.json").relative_to(ROOT)),
        "passed": contract["validation"]["passed"],
        "errors": contract["validation"]["errors"],
        "warnings": contract["validation"]["warnings"],
    }
    print(json.dumps(summary, indent=2))
    return 0 if contract["validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
