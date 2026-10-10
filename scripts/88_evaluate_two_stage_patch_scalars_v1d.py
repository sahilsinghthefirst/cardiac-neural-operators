                      
                                                                                   

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if (PROJECT_ROOT / ".codex_deps" / "sklearn").exists():
    sys.path.append(str(PROJECT_ROOT / ".codex_deps" / "sklearn"))

import numpy as np
import pandas as pd

from fno_field_data_common import load_stats, torch
from scripts_78_bridge import FNOFieldReconstructor
from v1d_anomaly_common import ANOMALY_STATS_JSON, dice_iou_precision_recall, finite_regression, load_rows_with_anomaly, make_anomaly_loader
from v1d_field_common import CHECKPOINT_DIR, NEURAL_DIR, ROOT, scalar_targets_from_fields, write_json


RESULTS_CSV = NEURAL_DIR / "two_stage_patch_scalar_results.csv"
RESULTS_JSON = NEURAL_DIR / "two_stage_patch_scalar_results.json"
PRED_CSV = NEURAL_DIR / "two_stage_patch_scalar_predictions.csv"
FIELD_STATS = NEURAL_DIR / "fno_field_train_stats.json"


def load_anomaly_model(device):
    path = PROJECT_ROOT / "scripts" / "87_train_fno_anomaly_localizer_v1d.py"
    spec = importlib.util.spec_from_file_location("_fno_anomaly_localizer", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ckpt = torch.load(CHECKPOINT_DIR / "fno_anomaly_localizer.pt", map_location="cpu")
    args = ckpt["args"]
    model = module.FNOAnomalyLocalizer(int(ckpt["in_channels"]), int(args["width"]), int(args["modes"]), int(args["layers"]), float(args["dropout"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, args


def load_field_model(device):
    ckpt = torch.load(CHECKPOINT_DIR / "fno_field_reconstructor.pt", map_location="cpu")
    args = ckpt["args"]
    model = FNOFieldReconstructor(int(ckpt["in_channels"]), int(args["width"]), int(args["modes"]), int(args["layers"]), float(args["dropout"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, args


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v1d two-stage evaluation")
    device = torch.device("cuda")
    rows = load_rows_with_anomaly()
    test_rows = rows[rows["split"].eq("test")].reset_index(drop=True)
    anomaly_stats = load_stats(ANOMALY_STATS_JSON)
    field_stats = load_stats(FIELD_STATS)
    anomaly_model, anomaly_args = load_anomaly_model(device)
    field_model, field_args = load_field_model(device)
    args = SimpleNamespace(batch_size=int(anomaly_args.get("batch_size", 12)), input_norm=anomaly_args.get("input_norm", "per_sample"))
    loader = make_anomaly_loader(test_rows, anomaly_stats, args.batch_size, False, args.input_norm)
    pred_rows = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device).float()
            anom = anomaly_model(x).cpu().numpy()
            field = field_model(x).cpu().numpy()
            y = batch["y"].numpy()
            valid = batch["valid_mask"].numpy().astype(bool)
            true_anom = batch["anomaly_binary"].numpy() > 0.5
            prob = 1.0 / (1.0 + np.exp(-anom[:, 0]))
            for b in range(prob.shape[0]):
                src = test_rows.iloc[offset + b]
                pred_mask = prob[b] > 0.5
                true_patch = y[b, 3] > 0.5
                true_lat = y[b, 0] * anomaly_stats.target_std[0] + anomaly_stats.target_mean[0]
                true_apd = y[b, 2] * anomaly_stats.target_std[2] + anomaly_stats.target_mean[2]
                pred_lat = field[b, 0] * field_stats.target_std[0] + field_stats.target_mean[0]
                pred_apd = field[b, 2] * field_stats.target_std[2] + field_stats.target_mean[2]
                true_scalar = scalar_targets_from_fields(true_lat, true_apd, true_patch, valid[b])
                rows_for_eval = {
                    "pred_mask_pred_apd": scalar_targets_from_fields(pred_lat, pred_apd, pred_mask, valid[b]),
                    "pred_mask_true_apd": scalar_targets_from_fields(true_lat, true_apd, pred_mask, valid[b]),
                    "true_mask_pred_apd": scalar_targets_from_fields(pred_lat, pred_apd, true_patch, valid[b]),
                    "true_anomaly_mask_true_apd": scalar_targets_from_fields(true_lat, true_apd, true_anom[b], valid[b]),
                }
                                                  
                v = valid[b] & np.isfinite(pred_apd)
                soft = prob[b]
                inside_w = soft * v
                outside_w = (1.0 - soft) * v
                soft_delta = math.nan
                if inside_w.sum() > 1e-6 and outside_w.sum() > 1e-6:
                    soft_delta = float((pred_apd * inside_w).sum() / inside_w.sum() - (pred_apd * outside_w).sum() / outside_w.sum())
                mm = dice_iou_precision_recall(true_patch, pred_mask, valid[b])
                rec = {
                    "field_sample_id": src["field_sample_id"],
                    "base_sample_id": src["base_sample_id"],
                    "condition": src["condition"],
                    "measurement_variant": src["measurement_variant"],
                    "true_APD90_patch_delta": true_scalar["APD90_patch_delta"],
                    "true_patch_area_fraction": true_scalar["patch_area_fraction"],
                    "soft_anomaly_weighted_APD90_patch_delta": soft_delta,
                    **{f"{k}_APD90_patch_delta": v2["APD90_patch_delta"] for k, v2 in rows_for_eval.items()},
                    **{f"{k}_patch_area_fraction": v2["patch_area_fraction"] for k, v2 in rows_for_eval.items()},
                    **{f"mask_{k}": v for k, v in mm.items()},
                }
                pred_rows.append(rec)
            offset += prob.shape[0]
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(PRED_CSV, index=False)
    result_rows = []
    for group_type, col in [("all", None), ("condition", "condition"), ("measurement_variant", "measurement_variant")]:
        groups = [("all", pred_df)] if col is None else list(pred_df.groupby(col))
        for group_value, sub in groups:
            row = {"group_type": group_type, "group_value": group_value, "n": int(len(sub))}
            row["dice_mean"] = float(sub["mask_dice"].mean())
            row["homogeneous_false_positive_area"] = float(sub[sub["condition"].eq("homogeneous")]["mask_false_positive_area"].mean()) if group_type == "all" else math.nan
            for pred_col in [
                "pred_mask_pred_apd_APD90_patch_delta",
                "pred_mask_true_apd_APD90_patch_delta",
                "true_mask_pred_apd_APD90_patch_delta",
                "soft_anomaly_weighted_APD90_patch_delta",
            ]:
                metrics = finite_regression(sub["true_APD90_patch_delta"].to_numpy(), sub[pred_col].to_numpy())
                for k, v in metrics.items():
                    row[f"{pred_col}_{k}"] = v
            metrics = finite_regression(sub["true_patch_area_fraction"].to_numpy(), sub["pred_mask_true_apd_patch_area_fraction"].to_numpy())
            for k, v in metrics.items():
                row[f"patch_area_fraction_{k}"] = v
            result_rows.append(row)
    results = pd.DataFrame(result_rows)
    results.to_csv(RESULTS_CSV, index=False)
    payload = {
        "anomaly_checkpoint": "outputs/neural_v1d/checkpoints/fno_anomaly_localizer.pt",
        "field_checkpoint": "outputs/neural_v1d/checkpoints/fno_field_reconstructor.pt",
        "results": results.replace({np.nan: None}).to_dict(orient="records"),
        "notes": "Two-stage evaluation uses anomaly predictions as masks and FNO field predictions for APD/LAT; target fields are evaluation labels only.",
    }
    write_json(RESULTS_JSON, payload)
    print(json.dumps({"results_csv": str(RESULTS_CSV), "all": results[results["group_type"].eq("all")].to_dict(orient="records")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
