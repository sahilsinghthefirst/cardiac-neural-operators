                      
                                                                    
 
                                                                                
                                                                             
                                                                        
                                                                             
                                  
   

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "outputs" / "icbinb_required_analysis"
OOD_DIR = ROOT / "outputs" / "icbinb_sensor_height_ood"
FIGURE_DIR = ROOT / "figures" / "icbinb_required_analysis"
FINAL_DIR = ROOT / "outputs" / "icbinb_paper_evidence_bundle"

CONTRACT_PATH = ANALYSIS_DIR / "evaluation_contract.json"
INVENTORY_PATH = ANALYSIS_DIR / "artifact_inventory.json"

SOURCE_PATHS = {
    "evaluation_contract": CONTRACT_PATH,
    "artifact_inventory": INVENTORY_PATH,
    "cohort_validation": ANALYSIS_DIR / "cohort_validation.json",
    "module2_point_estimates": ANALYSIS_DIR / "module2_point_estimates.csv",
    "module2_bootstrap_intervals": ANALYSIS_DIR / "module2_bootstrap_intervals.csv",
    "module2_paired_differences": ANALYSIS_DIR / "module2_paired_differences.csv",
    "module2_condition_specific": ANALYSIS_DIR / "module2_condition_specific.csv",
    "module2_support_diagnostics": ANALYSIS_DIR / "module2_support_diagnostics.csv",
    "module2_summary": ANALYSIS_DIR / "module2_summary.json",
    "module2_validation": ANALYSIS_DIR / "module2_validation.json",
    "module3_failure_decomposition": ANALYSIS_DIR / "module3_failure_decomposition.csv",
    "module3_secondary_diagnostics": ANALYSIS_DIR / "module3_secondary_diagnostics.csv",
    "module3_summary": ANALYSIS_DIR / "module3_summary.json",
    "module3_validation": ANALYSIS_DIR / "module3_validation.json",
    "module3_figure_png": FIGURE_DIR / "module3_failure_decomposition.png",
    "module3_figure_pdf": FIGURE_DIR / "module3_failure_decomposition.pdf",
    "module4_point_estimates": OOD_DIR / "module4_point_estimates.csv",
    "module4_bootstrap_intervals": OOD_DIR / "module4_bootstrap_intervals.csv",
    "module4_height_deltas": OOD_DIR / "module4_height_deltas.csv",
    "module4_summary": OOD_DIR / "module4_summary.json",
    "module4_validation": OOD_DIR / "module4_validation.json",
    "module4_figure_png": OOD_DIR / "module4_sensor_height_ood.png",
    "module4_figure_pdf": OOD_DIR / "module4_sensor_height_ood.pdf",
}

METRIC_LABELS = {
    "LAT_global_R2": "LAT global R2",
    "APD90_global_R2": "APD90 global R2",
    "patch_Dice_nonhomogeneous_micro": "Non-homogeneous patch Dice",
    "patch_center_error_mean": "Patch-center error",
    "homogeneous_false_positive_area_mean": "Homogeneous false-positive area",
    "APD90_patch_delta_deployed_R2": "Deployed APD90 patch-delta R2",
}

PUBLIC_TEXT_FORBIDDEN = {
    "patient_specific": re.compile(r"patient[- ]specific", re.IGNORECASE),
    "real_squid": re.compile(r"real[- ]SQUID", re.IGNORECASE),
    "clinical_validation": re.compile(r"clinic(?:al|ally)[- ]validat", re.IGNORECASE),
    "intracellular_current": re.compile(r"intracellular[- ]current", re.IGNORECASE),
    "physics_informed": re.compile(r"physics[- ]informed", re.IGNORECASE),
    "pino_branding": re.compile(r"\bPINO\b", re.IGNORECASE),
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def require_unique(table: pd.DataFrame, selector: dict[str, Any], label: str) -> pd.Series:
    selected = table
    for column, value in selector.items():
        selected = selected[selected[column].eq(value)]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one source row for {label}, found {len(selected)}: {selector}")
    return selected.iloc[0]


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Nonfinite value for {label}: {value}")
    return result


def format_value(value: float, digits: int = 3) -> str:
    numeric = float(value)
    if abs(numeric) < 0.5 * 10 ** (-digits):
        numeric = 0.0
    return f"{numeric:.{digits}f}"


def format_interval(value: float, low: float, high: float, digits: int = 3) -> str:
    return f"{format_value(value, digits)} [{format_value(low, digits)}, {format_value(high, digits)}]"


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def tex_metric_label(text: str) -> str:
                                                                                  
    return tex_escape(text).replace("R2", r"$R^2$")


def latex_table(headers: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    columns = "l" + "c" * (len(headers) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def validate_prerequisites() -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    missing = [str(path) for path in SOURCE_PATHS.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing Module 6 sources: {missing}")
    if FINAL_DIR.exists():
        raise SystemExit(f"Refusing to overwrite existing paper evidence bundle: {FINAL_DIR}")
    contract = read_json(CONTRACT_PATH)
    inventory = read_json(INVENTORY_PATH)
    if not contract.get("validation", {}).get("passed"):
        raise SystemExit("The evaluation contract is not passing.")
    blocking_missing = [
        row
        for row in inventory.get("known_missing_derived_artifacts", [])
        if row.get("id") != "module6_validated_paper_evidence_bundle"
    ]
    if blocking_missing:
        raise SystemExit(
            f"The artifact inventory still reports prerequisite derived artifacts: {blocking_missing}"
        )
    for key in ("module_2_status", "module_3_status", "module_4_status"):
        if not contract.get(key, {}).get("complete"):
            raise SystemExit(f"Prerequisite status is not complete: {key}")
    for key in ("module2_validation", "module3_validation", "module4_validation"):
        if not read_json(SOURCE_PATHS[key]).get("passed"):
            raise SystemExit(f"Source validation does not pass: {key}")
    inventory_by_path = {str(row["path"]): row for row in inventory["artifacts"]}
    return contract, inventory, inventory_by_path


def verify_inventory_hashes(inventory_by_path: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    exempt = {"evaluation_contract", "artifact_inventory"}
    for source_id, path in SOURCE_PATHS.items():
        observed = sha256(path)
        record = {
            "source_id": source_id,
            "path": rel(path),
            "bytes": path.stat().st_size,
            "sha256": observed,
            "inventory_hash_checked": False,
            "inventory_hash_matches": None,
        }
        inventory_record = inventory_by_path.get(rel(path))
        if inventory_record is not None:
            record["inventory_hash_checked"] = True
            record["inventory_hash_matches"] = observed == inventory_record.get("sha256")
            if not record["inventory_hash_matches"]:
                raise RuntimeError(f"Inventory hash mismatch for {source_id}: {path}")
        elif source_id not in exempt:
            raise RuntimeError(f"Source artifact is absent from the locked inventory: {source_id}")
        records.append(record)
    return records


def build_model_comparison(
    intervals: pd.DataFrame, paired: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    records = []
    tex_rows: list[list[str]] = []
    for metric, label in METRIC_LABELS.items():
        fno = require_unique(
            intervals,
            {"scope": "overall", "metric": metric, "model_family": "fno"},
            f"overall FNO {metric}",
        )
        deep = require_unique(
            intervals,
            {"scope": "overall", "metric": metric, "model_family": "deeponet"},
            f"overall DeepONet {metric}",
        )
        difference = require_unique(
            paired, {"scope": "overall", "metric": metric}, f"overall paired {metric}"
        )
        for row, model in ((fno, "fno"), (deep, "deeponet")):
            if int(row["bootstrap_replicates_valid"]) != 5000 or row["resampling_unit"] != "base_sample_id":
                raise RuntimeError(f"Invalid bootstrap provenance for {model}/{metric}")
        record = {
            "metric": metric,
            "display_label": label,
            "direction": str(fno["direction"]),
            "population_base_cases": int(fno["n_base_cases"]),
            "fno_model_id": str(fno["model_id"]),
            "fno_estimate": finite(fno["estimate"], f"FNO {metric}"),
            "fno_ci_lower": finite(fno["ci_lower_2p5"], f"FNO {metric} lower"),
            "fno_ci_upper": finite(fno["ci_upper_97p5"], f"FNO {metric} upper"),
            "deeponet_model_id": str(deep["model_id"]),
            "deeponet_estimate": finite(deep["estimate"], f"DeepONet {metric}"),
            "deeponet_ci_lower": finite(deep["ci_lower_2p5"], f"DeepONet {metric} lower"),
            "deeponet_ci_upper": finite(deep["ci_upper_97p5"], f"DeepONet {metric} upper"),
            "paired_difference_definition": str(difference["difference_definition"]),
            "paired_difference": finite(difference["paired_difference"], f"paired {metric}"),
            "paired_ci_lower": finite(difference["ci_lower_2p5"], f"paired {metric} lower"),
            "paired_ci_upper": finite(difference["ci_upper_97p5"], f"paired {metric} upper"),
            "bootstrap_replicates": 5000,
            "resampling_unit": "base_sample_id",
        }
        records.append(record)
        tex_rows.append(
            [
                tex_metric_label(label),
                format_interval(record["fno_estimate"], record["fno_ci_lower"], record["fno_ci_upper"]),
                format_interval(record["deeponet_estimate"], record["deeponet_ci_lower"], record["deeponet_ci_upper"]),
                format_interval(record["paired_difference"], record["paired_ci_lower"], record["paired_ci_upper"]),
            ]
        )
    caption = (
        "Grouped held-out results with 95\\% base-sample bootstrap intervals. "
        "Field and localization endpoints use their separately validation-selected representatives; "
        "the final column is the raw FNO-minus-DeepONet difference."
    )
    tex = latex_table(
        ["Metric", "FNO", "DeepONet", r"$\Delta$"], tex_rows, caption, "tab:overall_results"
    )
    return pd.DataFrame(records), tex


def build_conduction_table(
    intervals: pd.DataFrame, paired: pd.DataFrame, support: pd.DataFrame
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    scope = "conduction_slowing_perturbation"
    metric_order = [
        "LAT_global_R2",
        "APD90_global_R2",
        "patch_Dice_nonhomogeneous_micro",
        "patch_center_error_mean",
        "APD90_patch_delta_deployed_R2",
    ]
    records = []
    tex_rows: list[list[str]] = []
    for metric in metric_order:
        fno = require_unique(
            intervals, {"scope": scope, "metric": metric, "model_family": "fno"}, f"conduction FNO {metric}"
        )
        deep = require_unique(
            intervals, {"scope": scope, "metric": metric, "model_family": "deeponet"}, f"conduction DeepONet {metric}"
        )
        difference = require_unique(paired, {"scope": scope, "metric": metric}, f"conduction paired {metric}")
        record = {
            "metric": metric,
            "display_label": METRIC_LABELS[metric],
            "population_base_cases": int(fno["n_base_cases"]),
            "fno_estimate": finite(fno["estimate"], metric),
            "fno_ci_lower": finite(fno["ci_lower_2p5"], metric),
            "fno_ci_upper": finite(fno["ci_upper_97p5"], metric),
            "deeponet_estimate": finite(deep["estimate"], metric),
            "deeponet_ci_lower": finite(deep["ci_lower_2p5"], metric),
            "deeponet_ci_upper": finite(deep["ci_upper_97p5"], metric),
            "paired_difference": finite(difference["paired_difference"], metric),
            "paired_ci_lower": finite(difference["ci_lower_2p5"], metric),
            "paired_ci_upper": finite(difference["ci_upper_97p5"], metric),
            "bootstrap_replicates": 5000,
            "resampling_unit": "base_sample_id",
        }
        records.append(record)
        tex_rows.append(
            [
                tex_metric_label(METRIC_LABELS[metric]),
                format_interval(record["fno_estimate"], record["fno_ci_lower"], record["fno_ci_upper"]),
                format_interval(record["deeponet_estimate"], record["deeponet_ci_lower"], record["deeponet_ci_upper"]),
            ]
        )
    support_counts = {}
    for model in ("fno", "deeponet"):
        row = require_unique(
            support, {"scope": scope, "model_family": model}, f"conduction support {model}"
        )
        support_counts[model] = {
            "eligible_cases": int(row["eligible_nonhomogeneous_cases"]),
            "nondegenerate_cases": int(row["nondegenerate_support_cases"]),
            "degenerate_cases": int(row["degenerate_support_cases"]),
        }
    caption = (
        "Conduction-slowing subset (18 independent base cases). Global fields remain accurate while "
        "support-dependent endpoints deteriorate. Intervals use grouped base-sample resampling."
    )
    tex = latex_table(
        ["Metric", "FNO", "DeepONet"], tex_rows, caption, "tab:conduction_failure"
    )
    return pd.DataFrame(records), tex, support_counts


def build_v1f_table(decomposition: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    required_stages = [
        "magnetic_conduction_identifiability",
        "oracle_true_patch_predicted_LAT",
        "oracle_true_LAT_anomaly_predicted_LAT",
        "deployed_support_predicted_LAT",
        "deployed_support_nondegenerate_rate",
    ]
    rows = []
    tex_rows: list[list[str]] = []
    for stage_id in required_stages:
        source = require_unique(decomposition, {"stage_id": stage_id}, stage_id)
        if int(source["bootstrap_valid_replicates"]) != 5000 or source["resampling_unit"] != "base_sample_id":
            raise RuntimeError(f"Invalid V1F bootstrap provenance for {stage_id}")
        record = {
            "stage_id": stage_id,
            "stage_order": int(source["stage_order"]),
            "stage_label": str(source["stage_label"]),
            "metric": str(source["metric"]),
            "support_type": str(source["support_type"]),
            "population": str(source["population"]),
            "n_base_cases": int(source["n"]),
            "estimate": finite(source["estimate"], stage_id),
            "ci_lower": finite(source["ci_lower_2p5"], stage_id),
            "ci_upper": finite(source["ci_upper_97p5"], stage_id),
            "bootstrap_replicates": 5000,
            "resampling_unit": "base_sample_id",
        }
        rows.append(record)
        tex_rows.append(
            [
                tex_escape(record["stage_label"]),
                tex_metric_label(record["metric"]),
                format_interval(record["estimate"], record["ci_lower"], record["ci_upper"]),
                str(record["n_base_cases"]),
            ]
        )
    caption = (
        "Failure decomposition from magnetic identifiability through oracle-support recovery to deployed-support failure. "
        "Oracle supports are diagnostic probes, and stages with different populations or metrics are not direct comparisons."
    )
    tex = latex_table(
        ["Stage", "Metric", "Estimate", "$n$"], tex_rows, caption, "tab:failure_decomposition"
    )
    return pd.DataFrame(rows).sort_values("stage_order"), tex


def build_standoff_table(
    intervals: pd.DataFrame, deltas: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    metrics = ["LAT_global_R2", "APD90_global_R2", "patch_Dice_nonhomogeneous_micro"]
    records = []
    tex_rows: list[list[str]] = []
    for model in ("fno", "deeponet"):
        for height in (35, 50, 65):
            record: dict[str, Any] = {
                "model_family": model,
                "height_mm": height,
                "reference_height_mm": 50,
            }
            for metric in metrics:
                source = require_unique(
                    intervals,
                    {"model_family": model, "height_mm": height, "metric": metric},
                    f"standoff {model}/{height}/{metric}",
                )
                if int(source["finite_bootstrap_replicates"]) != 5000:
                    raise RuntimeError(f"Invalid standoff bootstrap count for {model}/{height}/{metric}")
                prefix = {
                    "LAT_global_R2": "lat",
                    "APD90_global_R2": "apd90",
                    "patch_Dice_nonhomogeneous_micro": "dice",
                }[metric]
                record[f"{prefix}_estimate"] = finite(source["estimate"], metric)
                record[f"{prefix}_ci_lower"] = finite(source["ci_lower"], metric)
                record[f"{prefix}_ci_upper"] = finite(source["ci_upper"], metric)
                if height == 50:
                    record[f"{prefix}_change_from_50"] = 0.0
                    record[f"{prefix}_change_ci_lower"] = 0.0
                    record[f"{prefix}_change_ci_upper"] = 0.0
                else:
                    delta = require_unique(
                        deltas,
                        {"model_family": model, "height_mm": height, "metric": metric},
                        f"standoff delta {model}/{height}/{metric}",
                    )
                    record[f"{prefix}_change_from_50"] = finite(delta["performance_change"], metric)
                    record[f"{prefix}_change_ci_lower"] = finite(delta["performance_change_ci_lower"], metric)
                    record[f"{prefix}_change_ci_upper"] = finite(delta["performance_change_ci_upper"], metric)
            records.append(record)
            tex_rows.append(
                [
                    "FNO" if model == "fno" else "DeepONet",
                    str(height),
                    format_value(record["lat_estimate"]),
                    format_value(record["apd90_estimate"]),
                    format_value(record["dice_estimate"]),
                ]
            )
    caption = (
        "Frozen field-model sensitivity to sensor-tissue standoff. The 50 mm condition is nominal; "
        "both models use per-sample measurement normalization, and only FNO receives the training-constant height coordinate."
    )
    tex = latex_table(
        ["Model", "Height (mm)", "LAT $R^2$", "APD90 $R^2$", "Patch Dice"],
        tex_rows,
        caption,
        "tab:standoff",
    )
    return pd.DataFrame(records), tex


def build_claims(
    overall: pd.DataFrame,
    conduction: pd.DataFrame,
    support_counts: dict[str, Any],
    v1f: pd.DataFrame,
    standoff: pd.DataFrame,
) -> list[dict[str, Any]]:
    def metric_row(table: pd.DataFrame, metric: str) -> pd.Series:
        return require_unique(table, {"metric": metric}, f"claim metric {metric}")

    lat = metric_row(overall, "LAT_global_R2")
    apd = metric_row(overall, "APD90_global_R2")
    dice = metric_row(overall, "patch_Dice_nonhomogeneous_micro")
    c_lat = metric_row(conduction, "LAT_global_R2")
    c_apd = metric_row(conduction, "APD90_global_R2")
    c_dice = metric_row(conduction, "patch_Dice_nonhomogeneous_micro")
    c_delta = metric_row(conduction, "APD90_patch_delta_deployed_R2")
    ident = require_unique(v1f, {"stage_id": "magnetic_conduction_identifiability"}, "claim identifiability")
    oracle = require_unique(v1f, {"stage_id": "oracle_true_patch_predicted_LAT"}, "claim oracle")
    deployed = require_unique(v1f, {"stage_id": "deployed_support_predicted_LAT"}, "claim deployed")
    fno65 = require_unique(standoff, {"model_family": "fno", "height_mm": 65}, "claim FNO 65")
    deep65 = require_unique(standoff, {"model_family": "deeponet", "height_mm": 65}, "claim DeepONet 65")
    return [
        {
            "claim_id": "overall_field_vs_localization",
            "paper_ready_text": (
                f"On 75 held-out base cases, FNO and DeepONet achieved LAT R2 values of "
                f"{lat.fno_estimate:.3f} and {lat.deeponet_estimate:.3f}, and APD90 R2 values of "
                f"{apd.fno_estimate:.3f} and {apd.deeponet_estimate:.3f}; non-homogeneous patch Dice was "
                f"lower at {dice.fno_estimate:.3f} and {dice.deeponet_estimate:.3f}."
            ),
            "evidence_files": ["table1_overall_model_comparison.csv"],
            "interpretation_limit": "Field and localization metrics use separately validation-selected representatives.",
        },
        {
            "claim_id": "conduction_failure",
            "paper_ready_text": (
                f"For 18 conduction-slowing cases, LAT/APD90 R2 remained {c_lat.fno_estimate:.3f}/{c_apd.fno_estimate:.3f} "
                f"for FNO and {c_lat.deeponet_estimate:.3f}/{c_apd.deeponet_estimate:.3f} for DeepONet, while patch Dice "
                f"fell to {c_dice.fno_estimate:.3f} and {c_dice.deeponet_estimate:.3f} and deployed APD90 patch-delta R2 "
                f"was {c_delta.fno_estimate:.3f} and {c_delta.deeponet_estimate:.3f}."
            ),
            "evidence_files": ["table2_conduction_failure.csv"],
            "support_diagnostic": {
                "fno_degenerate_supports": support_counts["fno"]["degenerate_cases"],
                "deeponet_degenerate_supports": support_counts["deeponet"]["degenerate_cases"],
                "eligible_cases": 18,
            },
        },
        {
            "claim_id": "failure_decomposition",
            "paper_ready_text": (
                f"Magnetic descriptors detected the conduction component with balanced accuracy {ident.estimate:.3f}; "
                f"predicted LAT recovered the conduction scalar under true support with R2 {oracle.estimate:.3f}, "
                "but deployed-support recovery was near zero."
            ),
            "evidence_files": ["table3_v1f_failure_decomposition.csv", "figure1_failure_decomposition.pdf"],
            "interpretation_limit": "Oracle support is diagnostic only; the stages are not a paired model comparison.",
        },
        {
            "claim_id": "standoff_boundary",
            "paper_ready_text": (
                f"At 65 mm versus the nominal 50 mm standoff, FNO LAT and APD90 R2 changed by "
                f"{fno65.lat_change_from_50:.4f} and {fno65.apd90_change_from_50:.4f}, whereas the corresponding "
                f"DeepONet changes were {deep65.lat_change_from_50:.4f} and {deep65.apd90_change_from_50:.4f}."
            ),
            "evidence_files": ["table4_sensor_height_sensitivity.csv", "figure2_sensor_height_sensitivity.pdf"],
            "interpretation_limit": "This is a zero-shot synthetic acquisition shift with per-sample normalization, not learned subject conditioning.",
        },
    ]


def scan_public_text(paths: list[Path]) -> dict[str, Any]:
    findings = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for code, pattern in PUBLIC_TEXT_FORBIDDEN.items():
            for match in pattern.finditer(text):
                findings.append(
                    {
                        "file": path.name,
                        "rule": code,
                        "matched_text": match.group(0),
                    }
                )
    return {"passed": not findings, "findings": findings, "files_scanned": [p.name for p in paths]}


def main() -> int:
    contract, inventory, inventory_by_path = validate_prerequisites()
    source_manifest = verify_inventory_hashes(inventory_by_path)

    m2_intervals = pd.read_csv(SOURCE_PATHS["module2_bootstrap_intervals"])
    m2_paired = pd.read_csv(SOURCE_PATHS["module2_paired_differences"])
    m2_support = pd.read_csv(SOURCE_PATHS["module2_support_diagnostics"])
    m3_decomposition = pd.read_csv(SOURCE_PATHS["module3_failure_decomposition"])
    m4_intervals = pd.read_csv(SOURCE_PATHS["module4_bootstrap_intervals"])
    m4_deltas = pd.read_csv(SOURCE_PATHS["module4_height_deltas"])

    overall, overall_tex = build_model_comparison(m2_intervals, m2_paired)
    conduction, conduction_tex, support_counts = build_conduction_table(
        m2_intervals, m2_paired, m2_support
    )
    v1f, v1f_tex = build_v1f_table(m3_decomposition)
    standoff, standoff_tex = build_standoff_table(m4_intervals, m4_deltas)
    claims = build_claims(overall, conduction, support_counts, v1f, standoff)

    captions = {
        "table1_overall_model_comparison": (
            "Grouped held-out comparison with 95% base-sample bootstrap intervals. Field and localization endpoints "
            "use separately validation-selected representatives."
        ),
        "table2_conduction_failure": (
            "Conduction-slowing results on 18 independent base cases, showing strong global field recovery but weaker support-dependent endpoints."
        ),
        "figure1_failure_decomposition": (
            "Conduction failure decomposition on the primary measurement configuration. Oracle supports are diagnostic probes, not deployable inputs."
        ),
        "figure2_sensor_height_sensitivity": (
            "Frozen-checkpoint sensitivity to synthetic sensor-tissue standoff. Error bars are 95% base-sample bootstrap intervals."
        ),
    }

    temp_name = tempfile.mkdtemp(prefix=".icbinb_paper_bundle_", dir=ROOT / "outputs")
    temp_dir = Path(temp_name)
    published = False
    try:
        overall.to_csv(temp_dir / "table1_overall_model_comparison.csv", index=False)
        conduction.to_csv(temp_dir / "table2_conduction_failure.csv", index=False)
        v1f.to_csv(temp_dir / "table3_v1f_failure_decomposition.csv", index=False)
        standoff.to_csv(temp_dir / "table4_sensor_height_sensitivity.csv", index=False)
        exported_tables = {
            "table1_overall_model_comparison.csv": overall,
            "table2_conduction_failure.csv": conduction,
            "table3_v1f_failure_decomposition.csv": v1f,
            "table4_sensor_height_sensitivity.csv": standoff,
        }
        for filename, expected in exported_tables.items():
            observed = pd.read_csv(temp_dir / filename)
            pd.testing.assert_frame_equal(
                observed,
                expected.reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-15,
            )
        (temp_dir / "table1_overall_model_comparison.tex").write_text(overall_tex, encoding="utf-8")
        (temp_dir / "table2_conduction_failure.tex").write_text(conduction_tex, encoding="utf-8")
        (temp_dir / "table3_v1f_failure_decomposition.tex").write_text(v1f_tex, encoding="utf-8")
        (temp_dir / "table4_sensor_height_sensitivity.tex").write_text(standoff_tex, encoding="utf-8")

        figure_copies = {
            "figure1_failure_decomposition.png": SOURCE_PATHS["module3_figure_png"],
            "figure1_failure_decomposition.pdf": SOURCE_PATHS["module3_figure_pdf"],
            "figure2_sensor_height_sensitivity.png": SOURCE_PATHS["module4_figure_png"],
            "figure2_sensor_height_sensitivity.pdf": SOURCE_PATHS["module4_figure_pdf"],
        }
        for name, source in figure_copies.items():
            shutil.copyfile(source, temp_dir / name)
            if sha256(source) != sha256(temp_dir / name):
                raise RuntimeError(f"Figure copy hash mismatch: {name}")

        write_json(temp_dir / "paper_claims.json", claims)
        write_json(temp_dir / "captions.json", captions)
        write_json(
            temp_dir / "source_manifest.json",
            {
                "schema_version": "1.0.0",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "sources": source_manifest,
                "frozen_contract_sha256": sha256(CONTRACT_PATH),
                "frozen_inventory_sha256": sha256(INVENTORY_PATH),
            },
        )
        readme = """# ICBINB paper evidence bundle

This directory is the frozen pre-writing evidence package for the four-page tiny paper.

- `table1_*`: overall validation-selected FNO and DeepONet endpoints.
- `table2_*`: the 18-case conduction-slowing failure subset.
- `table3_*` and `figure1_*`: the magnetic-identifiability to oracle-support to deployed-support decomposition.
- `table4_*` and `figure2_*`: the optional zero-shot sensor-standoff boundary test.
- `paper_claims.json`: concise text whose numbers are generated directly from the verified tables.
- `captions.json`: scope-safe candidate captions.
- `source_manifest.json`: immutable source paths and SHA-256 values used to build this package.
- `bundle_manifest.json` and `validation_report.json`: output integrity and validation results.

Statistical unit: `base_sample_id`. All displayed intervals use 5,000 grouped bootstrap replicates. The benchmark uses a fixed synthetic two-dimensional sheet, an equivalent-current proxy, and synthetic SQUID-like measurements. The evaluated models are supervised predictors. No human measurements or multiple anatomies are included.

Table 1 deliberately uses different validation-selected checkpoints for field and localization endpoints. Table 4 uses only the field representatives; its patch output must not be substituted for Table 1's localization comparison.
"""
        (temp_dir / "README.md").write_text(readme, encoding="utf-8")

        public_paths = [
            temp_dir / "README.md",
            temp_dir / "paper_claims.json",
            temp_dir / "captions.json",
            *sorted(temp_dir.glob("*.tex")),
        ]
        language_audit = scan_public_text(public_paths)
        if not language_audit["passed"]:
            raise RuntimeError(f"Scope-language audit failed: {language_audit['findings']}")

        table_validation = {
            "overall_rows": int(len(overall)),
            "overall_expected_rows": 6,
            "conduction_rows": int(len(conduction)),
            "conduction_expected_rows": 5,
            "v1f_rows": int(len(v1f)),
            "v1f_expected_rows": 5,
            "standoff_rows": int(len(standoff)),
            "standoff_expected_rows": 6,
            "all_numeric_values_finite": bool(
                all(
                    np.all(np.isfinite(table.select_dtypes(include=[np.number]).to_numpy()))
                    for table in (overall, conduction, v1f, standoff)
                )
            ),
            "all_bootstrap_counts_5000": bool(
                (overall["bootstrap_replicates"] == 5000).all()
                and (conduction["bootstrap_replicates"] == 5000).all()
                and (v1f["bootstrap_replicates"] == 5000).all()
            ),
            "all_resampling_units_base_sample_id": bool(
                overall["resampling_unit"].eq("base_sample_id").all()
                and conduction["resampling_unit"].eq("base_sample_id").all()
                and v1f["resampling_unit"].eq("base_sample_id").all()
            ),
            "csv_roundtrip_matches_generated_tables": True,
        }
        table_validation["passed"] = bool(
            table_validation["overall_rows"] == table_validation["overall_expected_rows"]
            and table_validation["conduction_rows"] == table_validation["conduction_expected_rows"]
            and table_validation["v1f_rows"] == table_validation["v1f_expected_rows"]
            and table_validation["standoff_rows"] == table_validation["standoff_expected_rows"]
            and table_validation["all_numeric_values_finite"]
            and table_validation["all_bootstrap_counts_5000"]
            and table_validation["all_resampling_units_base_sample_id"]
            and table_validation["csv_roundtrip_matches_generated_tables"]
        )
        if not table_validation["passed"]:
            raise RuntimeError(f"Paper table validation failed: {table_validation}")

        content_files = sorted(
            path for path in temp_dir.iterdir() if path.name not in {"bundle_manifest.json", "validation_report.json"}
        )
        output_records = [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in content_files
        ]
        write_json(
            temp_dir / "bundle_manifest.json",
            {
                "schema_version": "1.0.0",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "content_artifact_count": len(output_records),
                "content_artifacts": output_records,
            },
        )
        validation = {
            "schema_version": "1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "module": 6,
            "passed": True,
            "errors": [],
            "source_validation": {
                "contract_passed": bool(contract["validation"]["passed"]),
                "inventory_known_missing_count": len(inventory["known_missing_derived_artifacts"]),
                "module2_complete": bool(contract["module_2_status"]["complete"]),
                "module3_complete": bool(contract["module_3_status"]["complete"]),
                "module4_extra_complete": bool(contract["module_4_status"]["complete"]),
                "source_artifact_count": len(source_manifest),
                "all_inventory_hash_checks_passed": all(
                    (not row["inventory_hash_checked"]) or row["inventory_hash_matches"]
                    for row in source_manifest
                ),
            },
            "table_validation": table_validation,
            "figure_validation": {
                name: {
                    "source": rel(source),
                    "source_sha256": sha256(source),
                    "bundle_sha256": sha256(temp_dir / name),
                    "byte_identical": sha256(source) == sha256(temp_dir / name),
                }
                for name, source in figure_copies.items()
            },
            "scope_language_audit": language_audit,
            "claim_count": len(claims),
            "caption_count": len(captions),
            "safety": {
                "training_performed": False,
                "model_inference_performed": False,
                "simulation_performed": False,
                "legacy_outputs_overwritten": False,
                "isolated_output_directory": rel(FINAL_DIR),
            },
            "ready_for_paper_writing": True,
        }
        write_json(temp_dir / "validation_report.json", validation)
        temp_dir.replace(FINAL_DIR)
        published = True
    finally:
        if not published and temp_dir.exists():
            shutil.rmtree(temp_dir)

    print(
        json.dumps(
            {
                "passed": True,
                "module": 6,
                "ready_for_paper_writing": True,
                "output_dir": str(FINAL_DIR),
                "table_count": 4,
                "figure_count": 2,
                "claim_count": len(claims),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
