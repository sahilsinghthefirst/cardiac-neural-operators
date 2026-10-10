from __future__ import annotations

import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "outputs" / "icbinb_required_analysis"
FIGURE_DIR = ROOT / "figures" / "icbinb_required_analysis"
V1F_DIR = ROOT / "outputs" / "v1f_metric_realignment"
DESCRIPTOR_PATH = (
    ROOT
    / "outputs"
    / "dataset_v1e_500_combined_pilot"
    / "deeponet_ready"
    / "direct_descriptors.csv"
)
CONTRACT_PATH = ANALYSIS_DIR / "evaluation_contract.json"
INVENTORY_PATH = ANALYSIS_DIR / "artifact_inventory.json"

FINAL_FILES = {
    "identifiability_predictions": ANALYSIS_DIR / "module3_identifiability_predictions.csv",
    "decomposition": ANALYSIS_DIR / "module3_failure_decomposition.csv",
    "secondary": ANALYSIS_DIR / "module3_secondary_diagnostics.csv",
    "bootstrap": ANALYSIS_DIR / "module3_bootstrap_replicates.npz",
    "summary": ANALYSIS_DIR / "module3_summary.json",
    "validation": ANALYSIS_DIR / "module3_validation.json",
    "figure_png": FIGURE_DIR / "module3_failure_decomposition.png",
    "figure_pdf": FIGURE_DIR / "module3_failure_decomposition.pdf",
}

PRIMARY_VARIANT = "combined_bz_axial"
CONDUCTION = "conduction_slowing_perturbation"
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260903
RF_SEED = 20260525


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def verify_inventory_artifact(
    inventory_by_id: dict[str, dict[str, Any]], artifact_id: str
) -> dict[str, Any]:
    if artifact_id not in inventory_by_id:
        raise RuntimeError(f"Inventory lacks source artifact {artifact_id!r}.")
    record = inventory_by_id[artifact_id]
    path = resolve(record["path"])
    if not path.is_file():
        raise RuntimeError(f"Source artifact is missing: {path}")
    observed = sha256(path)
    if observed != record.get("sha256"):
        raise RuntimeError(
            f"Source hash changed for {artifact_id}: expected {record.get('sha256')}, "
            f"observed {observed}"
        )
    return {"path": path, "sha256": observed, "bytes": path.stat().st_size}


def descriptor_features(desc: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    skip = {
        "row_id",
        "base_sample_id",
        "condition",
        "combined_variant",
        "measurement_variant",
        "split",
    }
    columns = [
        column
        for column in desc.columns
        if column not in skip and pd.api.types.is_numeric_dtype(desc[column])
    ]
    features = desc[columns].replace([np.inf, -np.inf], np.nan)
    return features.fillna(features.median(numeric_only=True)).fillna(0.0), columns


def conduction_labels(frame: pd.DataFrame) -> np.ndarray:
    return frame["condition"].isin(
        [CONDUCTION, "mixed_apd_conduction_perturbation"]
    ).astype(int).to_numpy()


def classification_metrics(
    truth: np.ndarray, prediction: np.ndarray, probability: np.ndarray
) -> dict[str, float | int]:
    return {
        "n": int(len(truth)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "auroc": float(roc_auc_score(truth, probability)),
    }


def r2_score(true: np.ndarray, prediction: np.ndarray) -> float:
    true = np.asarray(true, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    finite = np.isfinite(true) & np.isfinite(prediction)
    if int(np.sum(finite)) < 2:
        return math.nan
    true = true[finite]
    prediction = prediction[finite]
    sst = float(np.sum((true - np.mean(true)) ** 2, dtype=np.float64))
    if sst <= 0.0 or not math.isfinite(sst):
        return math.nan
    return float(
        1.0 - np.sum((true - prediction) ** 2, dtype=np.float64) / sst
    )


def bootstrap_balanced_accuracy(
    truth: np.ndarray, prediction: np.ndarray, draws: np.ndarray
) -> np.ndarray:
    truth_draw = truth[draws]
    prediction_draw = prediction[draws]
    positive = truth_draw == 1
    negative = truth_draw == 0
    positive_count = positive.sum(axis=1)
    negative_count = negative.sum(axis=1)
    true_positive = ((prediction_draw == 1) & positive).sum(axis=1)
    true_negative = ((prediction_draw == 0) & negative).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = 0.5 * (
            true_positive / positive_count + true_negative / negative_count
        )
    return np.where((positive_count > 0) & (negative_count > 0), values, np.nan)


def bootstrap_r2(
    true: np.ndarray, prediction: np.ndarray, draws: np.ndarray
) -> np.ndarray:
    true_draw = np.asarray(true, dtype=np.float64)[draws]
    prediction_draw = np.asarray(prediction, dtype=np.float64)[draws]
    means = np.mean(true_draw, axis=1, keepdims=True)
    sst = np.sum((true_draw - means) ** 2, axis=1, dtype=np.float64)
    sse = np.sum((true_draw - prediction_draw) ** 2, axis=1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = 1.0 - sse / sst
    return np.where(sst > 0.0, values, np.nan)


def interval(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return math.nan, math.nan, 0
    lower, upper = np.percentile(finite, [2.5, 97.5])
    return float(lower), float(upper), int(finite.size)


def figure_value(value: float, digits: int = 3) -> str:
    numeric = float(value)
    if abs(numeric) < 0.5 * 10 ** (-digits):
        numeric = 0.0
    return f"{numeric:.{digits}f}"


def make_figure(decomposition: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    stage_ids = [
        "magnetic_conduction_identifiability",
        "oracle_true_patch_predicted_LAT",
        "oracle_true_LAT_anomaly_predicted_LAT",
        "deployed_support_predicted_LAT",
    ]
    stage_rows = {
        row["stage_id"]: row for row in decomposition.to_dict(orient="records")
    }
    selected = [stage_rows[stage_id] for stage_id in stage_ids]
    colors = ["#DCEBFA", "#DDF3E4", "#DDF3E4", "#FBE0DE"]
    edges = ["#3978B8", "#2D8A4B", "#2D8A4B", "#C8443A"]
    headings = [
        "Magnetic signal",
        "Predicted LAT",
        "Predicted LAT",
        "Deployed support",
    ]
    support_row = stage_rows["deployed_support_nondegenerate_rate"]

    fig, ax = plt.subplots(figsize=(11.2, 3.45), dpi=220)
    ax.set_xlim(-0.65, 3.65)
    ax.set_ylim(-0.95, 1.05)
    ax.axis("off")
    for index in range(3):
        ax.add_patch(
            FancyArrowPatch(
                (index + 0.43, 0.18),
                (index + 0.72, 0.18),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.5,
                color="#6B7280",
            )
        )
    for index, (row, color, edge, heading) in enumerate(
        zip(selected, colors, edges, headings)
    ):
        box = FancyBboxPatch(
            (index - 0.43, -0.42),
            0.86,
            1.16,
            boxstyle="round,pad=0.025,rounding_size=0.045",
            linewidth=1.6,
            edgecolor=edge,
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(
            index,
            0.59,
            heading,
            ha="center",
            va="center",
            fontsize=10.4,
            fontweight="bold",
            color="#111827",
        )
        if index == 0:
            support_text = "Descriptor-only RF\n75 held-out bases"
        elif index == 1:
            support_text = "True patch support\n18 conduction bases"
        elif index == 2:
            support_text = "True LAT-anomaly support\n18 conduction bases"
        else:
            support_text = (
                f"Predicted support\n{int(support_row['numerator'])}/"
                f"{int(support_row['denominator'])} nondegenerate"
            )
        ax.text(
            index,
            0.27,
            support_text,
            ha="center",
            va="center",
            fontsize=8.7,
            linespacing=1.25,
            color="#374151",
        )
        metric_label = "BA" if index == 0 else "R²"
        ax.text(
            index,
            -0.07,
            f"{metric_label} = {figure_value(row['estimate'])}",
            ha="center",
            va="center",
            fontsize=13.2,
            fontweight="bold",
            color=edge,
        )
        ax.text(
            index,
            -0.29,
            (
                f"95% CI [{figure_value(row['ci_lower_2p5'])}, "
                f"{figure_value(row['ci_upper_97p5'])}]"
            ),
            ha="center",
            va="center",
            fontsize=7.7,
            color="#4B5563",
        )
    ax.text(
        1.5,
        0.93,
        "Conduction information survives field reconstruction but collapses at deployed localization",
        ha="center",
        va="center",
        fontsize=13.2,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        1.5,
        -0.76,
        "BA = balanced accuracy. Primary combined_bz_axial variant; grouped test bases. Oracle supports are diagnostics only and are not model inputs.",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#4B5563",
    )
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    existing = [str(path) for path in FINAL_FILES.values() if path.exists()]
    if existing:
        raise SystemExit(
            "Refusing to overwrite existing Module 3 artifacts: " + ", ".join(existing)
        )

    contract = read_json(CONTRACT_PATH)
    inventory = read_json(INVENTORY_PATH)
    if not contract.get("validation", {}).get("passed"):
        raise SystemExit("Evaluation contract does not pass.")
    if not contract.get("module_2_status", {}).get("complete"):
        raise SystemExit("Module 2 is not recorded as complete.")
    if contract["measurement_policy"]["primary_variant"] != PRIMARY_VARIANT:
        raise SystemExit("Primary measurement variant changed.")
    selected_refinement_model = contract["model_roles"][
        "mask_refinement_recovery_attempt"
    ]["model_id"]

    inventory_by_id = {record["id"]: record for record in inventory["artifacts"]}
    source_artifacts = {
        artifact_id: verify_inventory_artifact(inventory_by_id, artifact_id)
        for artifact_id in (
            "v1f_magnetic_identifiability",
            "v1f_oracle_rows",
            "v1f_oracle_summary",
        )
    }
    if not DESCRIPTOR_PATH.is_file():
        raise SystemExit(f"Descriptor source is missing: {DESCRIPTOR_PATH}")
    source_artifacts["direct_magnetic_descriptors"] = {
        "path": DESCRIPTOR_PATH,
        "sha256": sha256(DESCRIPTOR_PATH),
        "bytes": DESCRIPTOR_PATH.stat().st_size,
    }

    errors: list[str] = []
    descriptor_rows = pd.read_csv(DESCRIPTOR_PATH)
    features, feature_columns = descriptor_features(descriptor_rows)
    if int(features.isna().sum().sum()) != 0:
        errors.append("Descriptor feature matrix contains missing values after preprocessing.")
    truth = conduction_labels(descriptor_rows)
    train_mask = descriptor_rows["split"].eq("train").to_numpy()
    test_mask = descriptor_rows["split"].eq("test").to_numpy()
    classifier = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RF_SEED,
        n_jobs=1,
    )
    classifier.fit(features.loc[train_mask, feature_columns], truth[train_mask])
    test_prediction = classifier.predict(features.loc[test_mask, feature_columns])
    test_probability = classifier.predict_proba(
        features.loc[test_mask, feature_columns]
    )[:, 1]
    test_rows = descriptor_rows.loc[test_mask].copy().reset_index(drop=True)
    test_rows["true_conduction_component"] = truth[test_mask]
    test_rows["pred_conduction_component"] = test_prediction
    test_rows["pred_conduction_probability"] = test_probability
    test_rows["is_primary_variant"] = test_rows["combined_variant"].eq(PRIMARY_VARIANT)
    all_test_metrics = classification_metrics(
        truth[test_mask], test_prediction, test_probability
    )

    ident_results = pd.read_csv(source_artifacts["v1f_magnetic_identifiability"]["path"])
    legacy_row = ident_results[
        ident_results["task"].eq("conduction_component")
        & ident_results["model"].eq("random_forest")
        & ident_results["split"].eq("test")
    ]
    if len(legacy_row) != 1:
        errors.append("Could not identify exactly one legacy RF conduction-test result.")
        legacy_metrics: dict[str, float] = {}
    else:
        legacy_metrics = {
            key: float(legacy_row.iloc[0][key])
            for key in ("accuracy", "balanced_accuracy", "macro_f1", "auroc")
        }
        for key, expected in legacy_metrics.items():
            if not math.isclose(
                float(all_test_metrics[key]), expected, rel_tol=0.0, abs_tol=1e-12
            ):
                errors.append(
                    f"RF reproduction differs from legacy {key}: "
                    f"{all_test_metrics[key]} vs {expected}"
                )
    validation_candidates = ident_results[
        ident_results["task"].eq("conduction_component")
        & ident_results["split"].eq("val")
    ].sort_values(["balanced_accuracy", "macro_f1"], ascending=False)
    if validation_candidates.empty or validation_candidates.iloc[0]["model"] != "random_forest":
        errors.append("Random forest is no longer validation-preferred for conduction detection.")

    primary_predictions = test_rows[test_rows["is_primary_variant"]].copy()
    primary_predictions = primary_predictions.sort_values("base_sample_id").reset_index(drop=True)
    if len(primary_predictions) != 75 or primary_predictions["base_sample_id"].nunique() != 75:
        errors.append("Primary identifiability evaluation is not exactly 75 unique test bases.")
    primary_metrics = classification_metrics(
        primary_predictions["true_conduction_component"].to_numpy(dtype=int),
        primary_predictions["pred_conduction_component"].to_numpy(dtype=int),
        primary_predictions["pred_conduction_probability"].to_numpy(dtype=float),
    )

    oracle_rows = pd.read_csv(source_artifacts["v1f_oracle_rows"]["path"])
    oracle = oracle_rows[
        oracle_rows["model"].eq(selected_refinement_model)
        & oracle_rows["split"].eq("test")
        & oracle_rows["condition"].eq(CONDUCTION)
        & oracle_rows["combined_variant"].eq(PRIMARY_VARIANT)
    ].copy()
    oracle = oracle.sort_values("base_sample_id").reset_index(drop=True)
    if len(oracle) != 18 or oracle["base_sample_id"].nunique() != 18:
        errors.append("Oracle decomposition is not exactly 18 unique conduction test bases.")
    if set(oracle["base_sample_id"]) - set(primary_predictions["base_sample_id"]):
        errors.append("Oracle conduction cases are not a subset of primary test base IDs.")

    true_lat_patch = oracle["true_LAT_patch_delay"].to_numpy(dtype=float)
    true_lat_anomaly = oracle["true_LAT_anomaly_delay"].to_numpy(dtype=float)
    true_apd90 = oracle["true_APD90_patch_delta"].to_numpy(dtype=float)
    pred_true_patch_lat = oracle["pred_true_mask_LAT_delay"].to_numpy(dtype=float)
    pred_true_anomaly_lat = oracle["pred_true_LAT_anomaly_LAT_delay"].to_numpy(dtype=float)
    pred_true_patch_apd90 = oracle["pred_true_mask_APD90_delta"].to_numpy(dtype=float)
    raw_deployed_lat = oracle["pred_balanced_mask_LAT_delay"].to_numpy(dtype=float)
    raw_deployed_apd90 = oracle["pred_balanced_mask_APD90_delta"].to_numpy(dtype=float)
    deployed_lat = np.where(np.isfinite(raw_deployed_lat), raw_deployed_lat, 0.0)
    deployed_apd90 = np.where(np.isfinite(raw_deployed_apd90), raw_deployed_apd90, 0.0)
    support_nondegenerate = np.isfinite(raw_deployed_lat) & np.isfinite(raw_deployed_apd90)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ident_draws = rng.integers(
        0, 75, size=(BOOTSTRAP_REPLICATES, 75), dtype=np.int16
    )
    oracle_draws = rng.integers(
        0, 18, size=(BOOTSTRAP_REPLICATES, 18), dtype=np.int16
    )
    ident_bootstrap = bootstrap_balanced_accuracy(
        primary_predictions["true_conduction_component"].to_numpy(dtype=int),
        primary_predictions["pred_conduction_component"].to_numpy(dtype=int),
        ident_draws,
    )
    bootstrap_values = {
        "magnetic_conduction_identifiability": ident_bootstrap,
        "oracle_true_patch_predicted_LAT": bootstrap_r2(
            true_lat_patch, pred_true_patch_lat, oracle_draws
        ),
        "oracle_true_LAT_anomaly_predicted_LAT": bootstrap_r2(
            true_lat_anomaly, pred_true_anomaly_lat, oracle_draws
        ),
        "deployed_support_predicted_LAT": bootstrap_r2(
            true_lat_patch, deployed_lat, oracle_draws
        ),
        "oracle_true_patch_predicted_APD90": bootstrap_r2(
            true_apd90, pred_true_patch_apd90, oracle_draws
        ),
        "deployed_support_predicted_APD90": bootstrap_r2(
            true_apd90, deployed_apd90, oracle_draws
        ),
        "deployed_support_nondegenerate_rate": np.mean(
            support_nondegenerate[oracle_draws], axis=1
        ),
    }
    point_values = {
        "magnetic_conduction_identifiability": float(
            primary_metrics["balanced_accuracy"]
        ),
        "oracle_true_patch_predicted_LAT": r2_score(
            true_lat_patch, pred_true_patch_lat
        ),
        "oracle_true_LAT_anomaly_predicted_LAT": r2_score(
            true_lat_anomaly, pred_true_anomaly_lat
        ),
        "deployed_support_predicted_LAT": r2_score(true_lat_patch, deployed_lat),
        "oracle_true_patch_predicted_APD90": r2_score(
            true_apd90, pred_true_patch_apd90
        ),
        "deployed_support_predicted_APD90": r2_score(
            true_apd90, deployed_apd90
        ),
        "deployed_support_nondegenerate_rate": float(
            np.mean(support_nondegenerate)
        ),
    }

    definitions = {
        "magnetic_conduction_identifiability": {
            "stage_order": 1,
            "stage_label": "Magnetic descriptor identifiability",
            "metric": "balanced_accuracy",
            "population": "75 primary-variant grouped-test base cases",
            "support_type": "B-measurement-derived descriptors only",
            "model": "random_forest",
            "selection_policy": "model family preferred by validation balanced accuracy",
            "interpretation": "Conduction-component information is present in the controlled magnetic measurements.",
            "n": 75,
        },
        "oracle_true_patch_predicted_LAT": {
            "stage_order": 2,
            "stage_label": "Predicted LAT under true patch support",
            "metric": "R2",
            "population": "18 primary-variant conduction-only grouped-test base cases",
            "support_type": "true patch support; diagnostic oracle only",
            "model": selected_refinement_model,
            "selection_policy": "refinement model locked before Module 3 by validation criterion",
            "interpretation": "The predicted LAT field contains recoverable conduction information under the true support.",
            "n": 18,
        },
        "oracle_true_LAT_anomaly_predicted_LAT": {
            "stage_order": 3,
            "stage_label": "Predicted LAT under true LAT-anomaly support",
            "metric": "R2",
            "population": "18 primary-variant conduction-only grouped-test base cases",
            "support_type": "true LAT-anomaly support; diagnostic oracle only",
            "model": selected_refinement_model,
            "selection_policy": "refinement model locked before Module 3 by validation criterion",
            "interpretation": "A conduction-aligned oracle target remains strongly recoverable from predicted LAT.",
            "n": 18,
        },
        "deployed_support_predicted_LAT": {
            "stage_order": 4,
            "stage_label": "Predicted LAT under deployed support",
            "metric": "R2",
            "population": "18 primary-variant conduction-only grouped-test base cases",
            "support_type": "existing V1F center-constrained probability support",
            "model": selected_refinement_model,
            "selection_policy": "same locked model; existing diagnostic support extractor",
            "interpretation": "Deployed support extraction collapses despite useful predicted LAT fields.",
            "n": 18,
        },
        "deployed_support_nondegenerate_rate": {
            "stage_order": 5,
            "stage_label": "Nondegenerate deployed support rate",
            "metric": "fraction",
            "population": "18 primary-variant conduction-only grouped-test base cases",
            "support_type": "existing V1F center-constrained probability support",
            "model": selected_refinement_model,
            "selection_policy": "same locked model; existing diagnostic support extractor",
            "interpretation": "A zero rate means every deployed conduction support was empty or otherwise unusable.",
            "n": 18,
        },
    }
    decomposition_records: list[dict[str, Any]] = []
    for stage_id, definition in definitions.items():
        lower, upper, valid_replicates = interval(bootstrap_values[stage_id])
        record = {
            "stage_id": stage_id,
            **definition,
            "estimate": point_values[stage_id],
            "ci_lower_2p5": lower,
            "ci_upper_97p5": upper,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_valid_replicates": valid_replicates,
            "resampling_unit": "base_sample_id",
            "measurement_variant": PRIMARY_VARIANT,
            "numerator": int(np.sum(support_nondegenerate))
            if stage_id == "deployed_support_nondegenerate_rate"
            else math.nan,
            "denominator": int(len(support_nondegenerate))
            if stage_id == "deployed_support_nondegenerate_rate"
            else math.nan,
        }
        decomposition_records.append(record)
        if valid_replicates < BOOTSTRAP_REPLICATES:
            errors.append(
                f"Stage {stage_id} has {valid_replicates}/{BOOTSTRAP_REPLICATES} valid replicates."
            )
    decomposition = pd.DataFrame(decomposition_records).sort_values("stage_order")

    secondary_specs = [
        (
            "oracle_true_patch_predicted_APD90",
            "Predicted APD90 under true patch support",
            "R2",
            "Diagnostic oracle",
        ),
        (
            "deployed_support_predicted_APD90",
            "Predicted APD90 under deployed support",
            "R2",
            "Degenerate supports assigned 0 ms recovered contrast",
        ),
    ]
    secondary_records: list[dict[str, Any]] = []
    for stage_id, label, metric, policy in secondary_specs:
        lower, upper, valid_replicates = interval(bootstrap_values[stage_id])
        secondary_records.append(
            {
                "diagnostic_id": stage_id,
                "label": label,
                "metric": metric,
                "estimate": point_values[stage_id],
                "ci_lower_2p5": lower,
                "ci_upper_97p5": upper,
                "n_base_cases": 18,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_valid_replicates": valid_replicates,
                "policy": policy,
                "model": selected_refinement_model,
                "measurement_variant": PRIMARY_VARIANT,
            }
        )
        if valid_replicates < BOOTSTRAP_REPLICATES:
            errors.append(
                f"Secondary diagnostic {stage_id} has too few valid replicates."
            )
    secondary = pd.DataFrame(secondary_records)

    if not np.all(np.isfinite(decomposition["estimate"].to_numpy(dtype=float))):
        errors.append("Module 3 decomposition contains a nonfinite estimate.")
    if not np.all(np.isfinite(secondary["estimate"].to_numpy(dtype=float))):
        errors.append("Module 3 secondary diagnostics contain a nonfinite estimate.")
    if int(np.sum(support_nondegenerate)) != 0:
        errors.append(
            "The locked primary-variant decomposition no longer has 0/18 nondegenerate deployed supports."
        )

    validation_core = {
        "passed": not errors,
        "errors": errors,
        "primary_variant": PRIMARY_VARIANT,
        "independent_unit": "base_sample_id",
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "random_seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "interval_method": "percentile",
        },
        "identifiability": {
            "feature_count": int(len(feature_columns)),
            "training_rows": int(np.sum(train_mask)),
            "training_unique_base_ids": int(
                descriptor_rows.loc[train_mask, "base_sample_id"].nunique()
            ),
            "all_test_rows": int(np.sum(test_mask)),
            "all_test_unique_base_ids": int(
                descriptor_rows.loc[test_mask, "base_sample_id"].nunique()
            ),
            "primary_test_rows": int(len(primary_predictions)),
            "primary_test_unique_base_ids": int(
                primary_predictions["base_sample_id"].nunique()
            ),
            "legacy_all_variant_reproduction": {
                "observed": all_test_metrics,
                "expected": legacy_metrics,
                "passed": not any(
                    message.startswith("RF reproduction differs") for message in errors
                ),
            },
            "validation_selection_winner": (
                str(validation_candidates.iloc[0]["model"])
                if not validation_candidates.empty
                else None
            ),
            "primary_test_metrics": primary_metrics,
            "features_are_measurement_derived_only": True,
        },
        "oracle_decomposition": {
            "selected_model": selected_refinement_model,
            "selection_partition": "validation",
            "conduction_test_rows": int(len(oracle)),
            "conduction_test_unique_base_ids": int(oracle["base_sample_id"].nunique()),
            "nondegenerate_deployed_supports": int(np.sum(support_nondegenerate)),
            "degenerate_support_policy": "0 ms recovered contrast; never silently dropped",
            "true_supports_used_as_model_inputs": False,
        },
        "legacy_headline_exclusion": {
            "excluded_from_confirmatory_table": True,
            "reason": (
                "The legacy gate selected separate test-best rows across models and pooled "
                "three augmented measurement variants per base case."
            ),
            "historical_values_remain_available": True,
        },
        "safety": {
            "neural_training_run": False,
            "neural_inference_run": False,
            "ep_simulation_run": False,
            "test_partition_used_for_model_selection": False,
            "condition_labels_used_as_model_features": False,
            "legacy_outputs_modified": False,
        },
    }
    if errors:
        print(json.dumps(sanitize(validation_core), indent=2), flush=True)
        raise SystemExit("Module 3 validation failed; no outputs were published.")

    summary_payload = {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "thesis": (
            "Conduction information is identifiable from controlled magnetic descriptors and "
            "recoverable from predicted LAT under oracle supports, but the deployed generic "
            "support extractor fails on the same primary-variant conduction cases."
        ),
        "decomposition": decomposition.to_dict(orient="records"),
        "secondary_diagnostics": secondary.to_dict(orient="records"),
        "recommended_next_model_change": (
            "Separate LAT/conduction-anomaly and APD/repolarization-anomaly supervision, while "
            "retaining a generic support head and validation-only calibration."
        ),
        "claim_boundary": (
            "Controlled synthetic diagnostic only; not patient-specific, clinical, real-SQUID, "
            "or physics-informed validation."
        ),
    }
    bootstrap_arrays = {
        "identifiability_draws": ident_draws,
        "oracle_draws": oracle_draws,
        **{
            f"metric__{key}": value.astype(np.float64)
            for key, value in bootstrap_values.items()
        },
    }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".module3_", dir=ANALYSIS_DIR) as temp_name:
        temp_dir = Path(temp_name)
        temp_paths = {key: temp_dir / path.name for key, path in FINAL_FILES.items()}
        test_rows.to_csv(temp_paths["identifiability_predictions"], index=False)
        decomposition.to_csv(temp_paths["decomposition"], index=False)
        secondary.to_csv(temp_paths["secondary"], index=False)
        np.savez_compressed(temp_paths["bootstrap"], **bootstrap_arrays)
        write_json(temp_paths["summary"], summary_payload)
        make_figure(
            decomposition,
            temp_paths["figure_png"],
            temp_paths["figure_pdf"],
        )
        output_hashes = {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for key, path in temp_paths.items()
            if key != "validation"
        }
        validation_payload = {
            "schema_version": "1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            **validation_core,
            "source_contract": {
                "path": str(CONTRACT_PATH.relative_to(ROOT)),
                "sha256": sha256(CONTRACT_PATH),
            },
            "source_inventory": {
                "path": str(INVENTORY_PATH.relative_to(ROOT)),
                "sha256": sha256(INVENTORY_PATH),
            },
            "source_artifacts": {
                artifact_id: {
                    "path": str(record["path"].relative_to(ROOT)),
                    "sha256": record["sha256"],
                    "bytes": record["bytes"],
                }
                for artifact_id, record in source_artifacts.items()
            },
            "outputs": output_hashes,
        }
        write_json(temp_paths["validation"], validation_payload)
        for key, final_path in FINAL_FILES.items():
            temp_paths[key].replace(final_path)

    print(
        json.dumps(
            {
                "passed": True,
                "status": "complete",
                "primary_identifiability_balanced_accuracy": point_values[
                    "magnetic_conduction_identifiability"
                ],
                "oracle_true_patch_LAT_R2": point_values[
                    "oracle_true_patch_predicted_LAT"
                ],
                "oracle_true_LAT_anomaly_R2": point_values[
                    "oracle_true_LAT_anomaly_predicted_LAT"
                ],
                "deployed_LAT_R2": point_values[
                    "deployed_support_predicted_LAT"
                ],
                "nondegenerate_deployed_supports": int(
                    np.sum(support_nondegenerate)
                ),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
