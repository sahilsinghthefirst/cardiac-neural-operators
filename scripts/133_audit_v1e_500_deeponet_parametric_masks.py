                      
                                                                      

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v1e_500_common import NEURAL_DIR, NEURAL_FIG_DIR, require_cuda, write_json
from v1e_deeponet_calibration_common import (
    area_fraction,
    disk_mask,
    load_or_build_full_grid_cache,
    patch_metrics,
    regression_metrics,
    scalar_targets,
    torch,
)


OUT_CSV = NEURAL_DIR / "deeponet_parametric_mask_decomposition.csv"
OUT_JSON = NEURAL_DIR / "deeponet_parametric_mask_decomposition.json"


METHODS = [
    {
        "method": "pred_center_pred_radius",
        "deployable": True,
        "oracle_component": False,
        "description": "Predicted center, predicted equivalent-radius head, predicted patch-present gate.",
    },
    {
        "method": "pred_center_pred_area_disk",
        "deployable": True,
        "oracle_component": False,
        "description": "Predicted center, disk radius from predicted area head, predicted patch-present gate.",
    },
    {
        "method": "pred_center_pred_area_radius_mean",
        "deployable": True,
        "oracle_component": False,
        "description": "Predicted center, average of predicted radius head and predicted-area disk radius, predicted patch-present gate.",
    },
    {
        "method": "pred_center_true_radius",
        "deployable": False,
        "oracle_component": True,
        "description": "Predicted center with true equivalent radius and true patch-present label; isolates center localization.",
    },
    {
        "method": "pred_center_true_area_disk",
        "deployable": False,
        "oracle_component": True,
        "description": "Predicted center with disk radius from true area and true patch-present label; isolates center localization with true size.",
    },
    {
        "method": "true_center_pred_radius",
        "deployable": False,
        "oracle_component": True,
        "description": "True center with predicted radius and true patch-present label; isolates radius calibration.",
    },
    {
        "method": "true_center_pred_area_disk",
        "deployable": False,
        "oracle_component": True,
        "description": "True center with disk radius from predicted area and true patch-present label; isolates area calibration.",
    },
    {
        "method": "true_center_true_area_disk",
        "deployable": False,
        "oracle_component": True,
        "description": "True center with disk radius from true area and true patch-present label; estimates circular-parametric boundary ceiling.",
    },
    {
        "method": "true_patch_oracle_boundary",
        "deployable": False,
        "oracle_component": True,
        "description": "True patch mask; estimates APD90-field upper bound independent of mask reconstruction.",
    },
]


def radius_from_area(area: float) -> float:
    return math.sqrt(max(float(area), 0.0) / math.pi) if np.isfinite(area) else math.nan


def center_error(cx: float, cy: float, true_cx: float, true_cy: float) -> float:
    if not all(np.isfinite(v) for v in [cx, cy, true_cx, true_cy]):
        return math.nan
    return float(math.sqrt((cx - true_cx) ** 2 + (cy - true_cy) ** 2))


def mask_and_params(row: pd.Series, true_patch: np.ndarray, method: str) -> tuple[np.ndarray, dict]:
    pred_present = float(row.get("pred_patch_present_prob", 0.0))
    true_present = float(row.get("true_patch_present", 0.0))
    pred_cx = float(row.get("pred_center_x_norm", math.nan))
    pred_cy = float(row.get("pred_center_y_norm", math.nan))
    true_cx = float(row.get("true_center_x_norm", math.nan))
    true_cy = float(row.get("true_center_y_norm", math.nan))
    pred_radius = float(row.get("pred_radius_param", math.nan))
    true_radius = float(row.get("true_radius_param", math.nan))
    pred_area = float(row.get("pred_patch_area_param", math.nan))
    true_area = float(row.get("true_patch_area_fraction", math.nan))

    if method == "true_patch_oracle_boundary":
        return true_patch.astype(np.float32), {
            "cx": true_cx,
            "cy": true_cy,
            "radius": true_radius,
            "size_source": "true_patch_boundary",
            "present_source": "true",
        }

    if method == "pred_center_pred_radius":
        return disk_mask(pred_cx, pred_cy, pred_radius, pred_present, true_patch.shape), {
            "cx": pred_cx,
            "cy": pred_cy,
            "radius": pred_radius,
            "size_source": "pred_radius",
            "present_source": "pred",
        }
    if method == "pred_center_pred_area_disk":
        radius = radius_from_area(pred_area)
        return disk_mask(pred_cx, pred_cy, radius, pred_present, true_patch.shape), {
            "cx": pred_cx,
            "cy": pred_cy,
            "radius": radius,
            "size_source": "pred_area",
            "present_source": "pred",
        }
    if method == "pred_center_pred_area_radius_mean":
        radius_area = radius_from_area(pred_area)
        if np.isfinite(radius_area) and np.isfinite(pred_radius):
            radius = 0.5 * (radius_area + pred_radius)
        elif np.isfinite(radius_area):
            radius = radius_area
        else:
            radius = pred_radius
        return disk_mask(pred_cx, pred_cy, radius, pred_present, true_patch.shape), {
            "cx": pred_cx,
            "cy": pred_cy,
            "radius": radius,
            "size_source": "pred_area_radius_mean",
            "present_source": "pred",
        }
    if method == "pred_center_true_radius":
        return disk_mask(pred_cx, pred_cy, true_radius, true_present, true_patch.shape), {
            "cx": pred_cx,
            "cy": pred_cy,
            "radius": true_radius,
            "size_source": "true_radius",
            "present_source": "true",
        }
    if method == "pred_center_true_area_disk":
        radius = radius_from_area(true_area)
        return disk_mask(pred_cx, pred_cy, radius, true_present, true_patch.shape), {
            "cx": pred_cx,
            "cy": pred_cy,
            "radius": radius,
            "size_source": "true_area",
            "present_source": "true",
        }
    if method == "true_center_pred_radius":
        return disk_mask(true_cx, true_cy, pred_radius, true_present, true_patch.shape), {
            "cx": true_cx,
            "cy": true_cy,
            "radius": pred_radius,
            "size_source": "pred_radius",
            "present_source": "true",
        }
    if method == "true_center_pred_area_disk":
        radius = radius_from_area(pred_area)
        return disk_mask(true_cx, true_cy, radius, true_present, true_patch.shape), {
            "cx": true_cx,
            "cy": true_cy,
            "radius": radius,
            "size_source": "pred_area",
            "present_source": "true",
        }
    if method == "true_center_true_area_disk":
        radius = radius_from_area(true_area)
        return disk_mask(true_cx, true_cy, radius, true_present, true_patch.shape), {
            "cx": true_cx,
            "cy": true_cy,
            "radius": radius,
            "size_source": "true_area",
            "present_source": "true",
        }
    raise ValueError(f"Unknown parametric-mask method: {method}")


def row_decomposition(index: pd.DataFrame, maps: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    method_meta = {m["method"]: m for m in METHODS}
    for i, row in index.iterrows():
        true_patch = maps["true_patch"][i]
        valid = maps["valid_mask"][i]
        true_apd90 = maps["true_apd90"][i]
        pred_apd90 = maps["pred_apd90"][i]
        true_area = area_fraction(true_patch, valid)
        true_delta = scalar_targets(true_apd90, true_patch, valid)
        for method in method_meta:
            pred_mask, params = mask_and_params(row, true_patch, method)
            pm = patch_metrics(true_patch, pred_mask, valid)
            pred_area = area_fraction(pred_mask, valid)
            pred_delta_true_apd = scalar_targets(true_apd90, pred_mask, valid)
            pred_delta_pred_apd = scalar_targets(pred_apd90, pred_mask, valid)
            c_err = center_error(
                float(params.get("cx", math.nan)),
                float(params.get("cy", math.nan)),
                float(row.get("true_center_x_norm", math.nan)),
                float(row.get("true_center_y_norm", math.nan)),
            )
            rows.append(
                {
                    "record_type": "row",
                    "model": row["model"],
                    "method": method,
                    "split": row["split"],
                    "row_id": row["row_id"],
                    "base_sample_id": row["base_sample_id"],
                    "condition": row["condition"],
                    "combined_variant": row["combined_variant"],
                    "true_patch_present": float(row["true_patch_present"]),
                    "deployable": method_meta[method]["deployable"],
                    "oracle_component": method_meta[method]["oracle_component"],
                    "size_source": params["size_source"],
                    "present_source": params["present_source"],
                    "center_error": c_err,
                    "true_area": true_area,
                    "pred_area": pred_area,
                    "area_error": pred_area - true_area if np.isfinite(pred_area) and np.isfinite(true_area) else math.nan,
                    "abs_area_error": abs(pred_area - true_area) if np.isfinite(pred_area) and np.isfinite(true_area) else math.nan,
                    "mask_radius": float(params.get("radius", math.nan)),
                    "dice": pm["dice"],
                    "iou": pm["iou"],
                    "precision": pm["precision"],
                    "recall": pm["recall"],
                    "true_apd90_patch_delta": true_delta,
                    "pred_apd90_patch_delta_true_apd": pred_delta_true_apd,
                    "pred_apd90_patch_delta_pred_apd": pred_delta_pred_apd,
                    "abs_delta_error_true_apd": abs(pred_delta_true_apd - true_delta) if np.isfinite(pred_delta_true_apd) and np.isfinite(true_delta) else math.nan,
                    "abs_delta_error_pred_apd": abs(pred_delta_pred_apd - true_delta) if np.isfinite(pred_delta_pred_apd) and np.isfinite(true_delta) else math.nan,
                    "homogeneous_false_positive_area": pred_area if float(row["true_patch_present"]) < 0.5 else math.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for group_cols, record_type in [
        (["model", "method"], "model_method"),
        (["model", "method", "condition"], "condition"),
    ]:
        for keys, sub in rows.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            rec = {"record_type": record_type, "split": "test", "n": int(len(sub))}
            for col, value in zip(group_cols, keys):
                rec[col] = value
            non = sub[sub["true_patch_present"].eq(1)]
            rec.update(
                {
                    "deployable": bool(sub["deployable"].iloc[0]),
                    "oracle_component": bool(sub["oracle_component"].iloc[0]),
                    "mean_dice_all": float(sub["dice"].mean(skipna=True)),
                    "mean_dice_present": float(non["dice"].mean(skipna=True)) if len(non) else math.nan,
                    "mean_iou_present": float(non["iou"].mean(skipna=True)) if len(non) else math.nan,
                    "mean_precision_present": float(non["precision"].mean(skipna=True)) if len(non) else math.nan,
                    "mean_recall_present": float(non["recall"].mean(skipna=True)) if len(non) else math.nan,
                    "center_error_mae": float(non["center_error"].mean(skipna=True)) if len(non) else math.nan,
                    "area_error_mae": float(non["abs_area_error"].mean(skipna=True)) if len(non) else math.nan,
                    "area_bias": float(non["area_error"].mean(skipna=True)) if len(non) else math.nan,
                    "homogeneous_false_positive_area": float(sub["homogeneous_false_positive_area"].mean(skipna=True)),
                    "patch_area_r2": regression_metrics(non["true_area"], non["pred_area"])["r2"] if len(non) else math.nan,
                    "APD90_patch_delta_R2_true_apd": regression_metrics(non["true_apd90_patch_delta"], non["pred_apd90_patch_delta_true_apd"])["r2"] if len(non) else math.nan,
                    "APD90_patch_delta_R2_pred_apd": regression_metrics(non["true_apd90_patch_delta"], non["pred_apd90_patch_delta_pred_apd"])["r2"] if len(non) else math.nan,
                    "APD90_patch_delta_MAE_true_apd": regression_metrics(non["true_apd90_patch_delta"], non["pred_apd90_patch_delta_true_apd"])["mae"] if len(non) else math.nan,
                    "APD90_patch_delta_MAE_pred_apd": regression_metrics(non["true_apd90_patch_delta"], non["pred_apd90_patch_delta_pred_apd"])["mae"] if len(non) else math.nan,
                }
            )
            records.append(rec)
    return pd.DataFrame(records)


def make_figures(rows: pd.DataFrame, summary: pd.DataFrame) -> None:
    NEURAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    mm = summary[summary["record_type"].eq("model_method")].copy()
    deployable = mm[mm["deployable"].eq(True)].copy()
    best_model = deployable.sort_values("APD90_patch_delta_R2_pred_apd", ascending=False).iloc[0]["model"] if len(deployable) else str(mm["model"].iloc[0])
    sub = mm[mm["model"].eq(best_model)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(sub))
    axes[0].bar(x, sub["mean_dice_present"])
    axes[0].set_title(f"Parametric-mask Dice: {best_model}")
    axes[0].set_ylabel("mean Dice, patch-present rows")
    axes[1].bar(x - 0.18, sub["APD90_patch_delta_R2_true_apd"], width=0.36, label="true APD90")
    axes[1].bar(x + 0.18, sub["APD90_patch_delta_R2_pred_apd"], width=0.36, label="predicted APD90")
    axes[1].set_title("APD90 Patch-delta R2")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(sub["method"], rotation=80, fontsize=7)
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "deeponet_parametric_mask_decomposition.png", dpi=180)
    plt.close(fig)

    row_sub = rows[
        rows["model"].eq(best_model)
        & rows["method"].isin(["pred_center_pred_radius", "pred_center_pred_area_disk"])
        & rows["true_patch_present"].eq(1)
    ].copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(row_sub):
        sc = ax.scatter(
            row_sub["center_error"],
            row_sub["abs_area_error"],
            c=row_sub["abs_delta_error_true_apd"],
            s=24,
            alpha=0.75,
            cmap="viridis",
        )
        fig.colorbar(sc, ax=ax, label="abs delta error with true APD90")
    ax.set_xlabel("center error")
    ax.set_ylabel("absolute area error")
    ax.set_title(f"Center vs Area Failure: {best_model}")
    fig.tight_layout()
    fig.savefig(NEURAL_FIG_DIR / "deeponet_center_vs_area_failure.png", dpi=180)
    plt.close(fig)


def main() -> int:
    require_cuda(torch)
    index, maps = load_or_build_full_grid_cache(splits=("val", "test"), batch_size=4)
    mask = index["split"].eq("test")
    test_index = index[mask].reset_index(drop=True)
    test_maps = {k: v[np.flatnonzero(mask.to_numpy())] for k, v in maps.items()}
    rows = row_decomposition(test_index, test_maps)
    summary = summarize(rows)
    out = pd.concat([rows, summary], ignore_index=True, sort=False)
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    make_figures(rows, summary)

    mm = summary[summary["record_type"].eq("model_method")].copy()
    best_deployable = mm[mm["deployable"].eq(True)].sort_values("APD90_patch_delta_R2_pred_apd", ascending=False).head(5)
    best_oracle = mm[mm["oracle_component"].eq(True)].sort_values("APD90_patch_delta_R2_pred_apd", ascending=False).head(8)
    payload = {
        "decomposition_csv": str(OUT_CSV),
        "method_descriptions": METHODS,
        "best_deployable_by_predicted_apd_delta": best_deployable.replace({np.nan: None}).to_dict(orient="records"),
        "oracle_component_ceilings": best_oracle.replace({np.nan: None}).to_dict(orient="records"),
        "ellipse_bbox_approximation": {
            "available": False,
            "reason": "v1e 500 patch target files store center, area fraction, equivalent radius, and area bin only; no ellipse axis, orientation, or bounding-box labels are available.",
        },
        "figures": [
            "figures/neural_v1e_500/deeponet_parametric_mask_decomposition.png",
            "figures/neural_v1e_500/deeponet_center_vs_area_failure.png",
        ],
        "leakage_guardrail": "Oracle rows use target parameters only for diagnostic decomposition. Deployable rows use model-predicted center/area/radius/presence only.",
    }
    write_json(OUT_JSON, payload)
    print(
        json.dumps(
            {
                "decomposition": str(OUT_CSV),
                "best_deployable": payload["best_deployable_by_predicted_apd_delta"][:3],
                "ellipse_bbox_available": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
