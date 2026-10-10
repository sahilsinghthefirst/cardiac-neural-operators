from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from v1e_500_common import ROOT, SEED, write_json
from v1f_metric_realignment_common import CONDITION_ORDER, CONDITION_SHORT, DEEPO_READY, V1F_FIG_DIR, V1F_OUTPUT_DIR, ensure_dirs, load_index


OUT_CSV = V1F_OUTPUT_DIR / "magnetic_identifiability_results.csv"
SUMMARY_JSON = V1F_OUTPUT_DIR / "magnetic_identifiability_summary.json"


def descriptor_features(desc: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    skip = {"row_id", "base_sample_id", "condition", "combined_variant", "measurement_variant", "split"}
    cols = [c for c in desc.columns if c not in skip and pd.api.types.is_numeric_dtype(desc[c])]
    x = desc[cols].replace([np.inf, -np.inf], np.nan)
    return x.fillna(x.median(numeric_only=True)).fillna(0.0), cols


def task_labels(df: pd.DataFrame, task: str):
    if task == "condition_multiclass":
        return df["condition"].astype(str).to_numpy()
    if task == "patch_present":
        return (df["condition"].astype(str) != "homogeneous").astype(int).to_numpy()
    if task == "conduction_component":
        return df["condition"].isin(["conduction_slowing_perturbation", "mixed_apd_conduction_perturbation"]).astype(int).to_numpy()
    if task == "apd_component":
        return df["condition"].isin(["apd_repolarization_perturbation", "mixed_apd_conduction_perturbation"]).astype(int).to_numpy()
    raise ValueError(task)


def classifier_specs():
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def evaluate_predictions(y_true, y_pred, prob=None) -> dict[str, float]:
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    classes = np.unique(y_true)
    out["auroc"] = math.nan
    if prob is not None and len(classes) == 2:
        try:
            if prob.ndim == 2 and prob.shape[1] == 2:
                out["auroc"] = float(roc_auc_score(y_true, prob[:, 1]))
        except Exception:
            pass
    return out


def plot_feature_separability(desc: pd.DataFrame, x: pd.DataFrame, cols: list[str]) -> None:
    train = desc["split"].eq("train")
    scaler = StandardScaler().fit(x.loc[train, cols])
    z = scaler.transform(x[cols])
    pca = PCA(n_components=2, random_state=SEED).fit(z[train])
    xy = pca.transform(z)
    fig, ax = plt.subplots(figsize=(6.8, 5.4), dpi=160)
    colors = {
        "homogeneous": "#666666",
        "apd_repolarization_perturbation": "#3b82f6",
        "conduction_slowing_perturbation": "#ef4444",
        "mixed_apd_conduction_perturbation": "#10b981",
    }
    for cond in CONDITION_ORDER:
        m = desc["condition"].eq(cond).to_numpy()
        ax.scatter(xy[m, 0], xy[m, 1], s=12, alpha=0.45, label=CONDITION_SHORT[cond], color=colors[cond], linewidths=0)
    ax.set_xlabel("descriptor PCA 1")
    ax.set_ylabel("descriptor PCA 2")
    ax.set_title("APD-only vs Conduction-only Descriptor Separability")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(V1F_FIG_DIR / "apd_only_vs_conduction_only_feature_separability.png")
    plt.close(fig)


def plot_magnetic_examples(index: pd.DataFrame) -> None:
    chosen = []
    for cond in CONDITION_ORDER:
        sub = index[index["condition"].eq(cond) & index["combined_variant"].eq("combined_bz_axial")]
        if sub.empty:
            sub = index[index["condition"].eq(cond)]
        if not sub.empty:
            chosen.append(sub.iloc[0])
    if not chosen:
        return
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.2), dpi=160, sharex=True)
    for row in chosen:
        b = np.load(ROOT / row["B_meas_file"]).astype(np.float32)
        mask = np.load(ROOT / row["sensor_mask_file"]).astype(bool)
        time = np.load(ROOT / row["time_ms_file"]).astype(np.float32) if Path(ROOT / row["time_ms_file"]).exists() else np.arange(b.shape[0])
        ch0 = b[:, :, :, 0]
        valid = mask[:, :, 0] if mask.ndim == 3 else np.ones(ch0.shape[1:], dtype=bool)
        rms = np.sqrt(np.nanmean(np.square(np.where(valid[None, :, :], ch0, np.nan)), axis=(1, 2)))
        peak = np.nanmax(np.abs(np.where(valid[None, :, :], ch0, np.nan)), axis=(1, 2))
        label = CONDITION_SHORT[row["condition"]]
        axes[0].plot(time, rms, label=label)
        axes[1].plot(time, peak, label=label)
    axes[0].set_ylabel("Bz RMS")
    axes[1].set_ylabel("Bz peak abs")
    axes[1].set_xlabel("time (ms)")
    axes[0].set_title("Conduction-only Magnetic Temporal Examples")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(V1F_FIG_DIR / "conduction_only_magnetic_examples.png")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    desc = pd.read_csv(DEEPO_READY / "direct_descriptors.csv")
    x, cols = descriptor_features(desc)
    rows = []
    for task in ["condition_multiclass", "patch_present", "conduction_component", "apd_component"]:
        y_raw = task_labels(desc, task)
        le = LabelEncoder().fit(y_raw)
        y = le.transform(y_raw)
        train = desc["split"].eq("train").to_numpy()
        for model_name, clf in classifier_specs().items():
            clf.fit(x.loc[train, cols], y[train])
            for split in ["train", "val", "test"]:
                m = desc["split"].eq(split).to_numpy()
                pred = clf.predict(x.loc[m, cols])
                prob = clf.predict_proba(x.loc[m, cols]) if hasattr(clf, "predict_proba") else None
                metrics = evaluate_predictions(y[m], pred, prob)
                metrics.update(
                    {
                        "task": task,
                        "model": model_name,
                        "split": split,
                        "classes": "|".join(str(c) for c in le.classes_),
                        "feature_count": len(cols),
                        "guardrail": "features are B_meas-derived descriptors only; IDs, condition labels, split labels, and metadata are excluded from X",
                    }
                )
                rows.append(metrics)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    plot_feature_separability(desc, x, cols)
    plot_magnetic_examples(load_index())

    test = out[out["split"].eq("test")]
    best = {}
    for task, sub in test.groupby("task"):
        best[task] = sub.sort_values(["balanced_accuracy", "macro_f1"], ascending=False).iloc[0].to_dict()
    write_json(
        SUMMARY_JSON,
        {
            "results_csv": OUT_CSV,
            "descriptor_rows": int(len(desc)),
            "feature_count": len(cols),
            "best_test_by_task": best,
            "conduction_component_above_chance": bool(best.get("conduction_component", {}).get("balanced_accuracy", 0.0) > 0.55),
            "guardrails": {
                "features_exclude_condition_and_metadata": True,
                "grouped_split_reused": True,
                "no_neural_training": True,
            },
        },
    )
    print(f"wrote {OUT_CSV}")
    print(f"wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
