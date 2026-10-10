                      
                                                                        
 
                                                                             
                                                                        
                                                                             
                                                                    
   

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from fno_elementary_common import FNOBlock2d
from v1e_field_data_common import (
    PARAM_KEYS,
    FieldStats,
    load_descriptors,
    load_index,
    make_loader,
    torch,
)


ANALYSIS_DIR = PROJECT_ROOT / "outputs" / "icbinb_required_analysis"
CONTRACT_PATH = ANALYSIS_DIR / "evaluation_contract.json"
INVENTORY_PATH = ANALYSIS_DIR / "artifact_inventory.json"
LEGACY_RESULTS_PATH = (
    PROJECT_ROOT / "outputs" / "neural_v1e_500" / "fno_field_parametric_results.csv"
)

FINAL_INDEX = ANALYSIS_DIR / "fno_full_grid_prediction_index.csv"
FINAL_MAPS = ANALYSIS_DIR / "fno_full_grid_prediction_maps.npz"
FINAL_METADATA = ANALYSIS_DIR / "fno_full_grid_prediction_cache.json"

ARRAY_DTYPES = {
    "true_lat": np.float32,
    "true_apd90": np.float32,
    "true_patch": np.float32,
    "valid_mask": np.bool_,
    "pred_lat": np.float32,
    "pred_apd90": np.float32,
    "pred_patch_prob": np.float32,
}


class FNOFieldParametric(torch.nn.Module):
                                                                                

    def __init__(
        self,
        in_channels: int,
        desc_dim: int,
        width: int,
        modes: int,
        layers: int,
        dropout: float,
        use_descriptors: bool,
    ) -> None:
        super().__init__()
        self.use_descriptors = use_descriptors
        self.lift = torch.nn.Conv2d(in_channels, width, 1)
        self.blocks = torch.nn.ModuleList(
            [FNOBlock2d(width, modes, modes, dropout) for _ in range(layers)]
        )
        self.field_head = torch.nn.Sequential(
            torch.nn.Conv2d(width, width, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(dropout),
            torch.nn.Conv2d(width, 4, 1),
        )
        if use_descriptors and desc_dim:
            self.desc_encoder = torch.nn.Sequential(
                torch.nn.Linear(desc_dim, 128),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(128, 64),
                torch.nn.GELU(),
            )
            head_in = width + 64
        else:
            self.desc_encoder = None
            head_in = width
        self.param_head = torch.nn.Sequential(
            torch.nn.Linear(head_in, width),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(width, len(PARAM_KEYS)),
        )
        self.area_head = torch.nn.Linear(head_in, 3)
        self.condition_head = torch.nn.Linear(head_in, 4)

    def forward(self, x, desc=None):
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        pooled = h.mean(dim=(2, 3))
        if self.desc_encoder is not None and desc is not None and desc.shape[1] > 0:
            pooled = torch.cat([pooled, self.desc_encoder(desc)], dim=1)
        return (
            self.field_head(h),
            self.param_head(pooled),
            self.area_head(pooled),
            self.condition_head(pooled),
        )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_stats(path: Path) -> FieldStats:
    payload = read_json(path)
    return FieldStats(
        input_mean=float(payload["input_mean"]),
        input_std=float(payload["input_std"]),
        field_mean=np.asarray(payload["field_mean"], dtype=np.float32),
        field_std=np.asarray(payload["field_std"], dtype=np.float32),
        param_mean=np.asarray(payload["param_mean"], dtype=np.float32),
        param_std=np.asarray(payload["param_std"], dtype=np.float32),
        descriptor_mean=np.asarray(payload["descriptor_mean"], dtype=np.float32),
        descriptor_std=np.asarray(payload["descriptor_std"], dtype=np.float32),
        descriptor_columns=list(payload["descriptor_columns"]),
    )


def locked_models(contract: dict[str, Any], inventory: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = {
        record["id"]: resolve_repo_path(record["path"])
        for record in inventory["artifacts"]
    }
    roles = [
        ("global_field_comparison", contract["model_roles"]["global_field_comparison"]["fno"]),
        ("localization_comparison", contract["model_roles"]["localization_comparison"]["fno"]),
    ]
    models: list[dict[str, Any]] = []
    for role_name, role in roles:
        artifact_id = role["checkpoint_artifact_id"]
        if artifact_id not in artifacts:
            raise RuntimeError(f"Checkpoint artifact {artifact_id!r} is absent from the inventory.")
        models.append(
            {
                "role": role_name,
                "model_id": role["model_id"],
                "checkpoint_artifact_id": artifact_id,
                "checkpoint_path": artifacts[artifact_id],
            }
        )
    if len({item["model_id"] for item in models}) != len(models):
        raise RuntimeError("Locked FNO roles unexpectedly resolve to duplicate model IDs.")
    return models


def build_model(checkpoint_path: Path, stats: FieldStats, device) -> tuple[Any, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    args = checkpoint.get("args", {})
    weights = checkpoint.get("weights", {})
    lift_weight = state["lift.weight"]
    in_channels = int(lift_weight.shape[1])
    width = int(lift_weight.shape[0])
    layers = int(args.get("layers", 4))
    modes = min(int(args.get("modes", 12)), 16)
    dropout = float(args.get("dropout", 0.08))
    use_descriptors = bool(weights.get("descriptors", False))
    model = FNOFieldParametric(
        in_channels=in_channels,
        desc_dim=len(stats.descriptor_columns),
        width=width,
        modes=modes,
        layers=layers,
        dropout=dropout,
        use_descriptors=use_descriptors,
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    replay = {
        "checkpoint_args": args,
        "checkpoint_weights": weights,
        "in_channels": in_channels,
        "descriptor_dim": len(stats.descriptor_columns),
        "width": width,
        "layers": layers,
        "modes": modes,
        "dropout": dropout,
        "use_descriptors": use_descriptors,
        "input_norm": str(args.get("input_norm", "per_sample")),
    }
    return model, replay


def open_array_memmaps(temp_dir: Path, count: int, grid_shape: tuple[int, int]):
    arrays: dict[str, np.memmap] = {}
    paths: dict[str, Path] = {}
    for name, dtype in ARRAY_DTYPES.items():
        path = temp_dir / f"{name}.npy"
        paths[name] = path
        arrays[name] = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=dtype,
            shape=(count, *grid_shape),
        )
    return arrays, paths


def finite_r2(true: np.ndarray, pred: np.ndarray) -> float:
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    valid = np.isfinite(true) & np.isfinite(pred)
    if valid.sum() < 2 or np.allclose(true[valid], true[valid][0]):
        return math.nan
    residual = float(np.sum((true[valid] - pred[valid]) ** 2))
    total = float(np.sum((true[valid] - np.mean(true[valid])) ** 2))
    return 1.0 - residual / total if total > 0 else math.nan


def micro_dice(true: np.ndarray, probability: np.ndarray, valid: np.ndarray) -> float:
    truth = (true > 0.5) & valid
    predicted = (probability > 0.5) & valid
    tp = float(np.sum(truth & predicted))
    fp = float(np.sum(~truth & predicted & valid))
    fn = float(np.sum(truth & ~predicted))
    return (2.0 * tp) / max(2.0 * tp + fp + fn, 1.0)


def legacy_metric(
    legacy: pd.DataFrame,
    model: str,
    split: str,
    target: str,
    column: str,
) -> float:
    selected = legacy[
        legacy["model"].eq(model)
        & legacy["split"].eq(split)
        & legacy["target"].eq(target)
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"Expected one legacy metric for {model}/{split}/{target}, found {len(selected)}."
        )
    value = float(selected.iloc[0][column])
    if not math.isfinite(value):
        raise RuntimeError(f"Legacy metric is nonfinite for {model}/{split}/{target}/{column}.")
    return value


def validate_cache(
    index: pd.DataFrame,
    arrays: dict[str, np.memmap],
    models: list[dict[str, Any]],
    rows_per_model: int,
    parity_tolerance: float,
) -> dict[str, Any]:
    errors: list[str] = []
    expected_total = rows_per_model * len(models)
    if len(index) != expected_total:
        errors.append(f"Index has {len(index)} rows; expected {expected_total}.")
    if index["array_index"].tolist() != list(range(len(index))):
        errors.append("array_index is not contiguous and order-preserving.")
    duplicate_count = int(index.duplicated(["model", "row_id"]).sum())
    if duplicate_count:
        errors.append(f"Found {duplicate_count} duplicate (model,row_id) pairs.")
    if set(index["split"]) != {"val", "test"}:
        errors.append(f"Unexpected split set: {sorted(set(index['split']))}")

    coverage: dict[str, Any] = {}
    for item in models:
        model = item["model_id"]
        for split in ("val", "test"):
            for variant in (
                "combined_all_available",
                "combined_bz_axial",
                "combined_bz_axial_noise",
            ):
                selected = index[
                    index["model"].eq(model)
                    & index["split"].eq(split)
                    & index["combined_variant"].eq(variant)
                ]
                key = f"{model}|{split}|{variant}"
                coverage[key] = {
                    "rows": int(len(selected)),
                    "unique_base_sample_ids": int(selected["base_sample_id"].nunique()),
                }
                if len(selected) != 75 or selected["base_sample_id"].nunique() != 75:
                    errors.append(f"Coverage failure for {key}: {coverage[key]}")

    if len(models) == 2:
        first = slice(0, rows_per_model)
        second = slice(rows_per_model, 2 * rows_per_model)
        for name in ("true_lat", "true_apd90", "true_patch", "valid_mask"):
            if not np.array_equal(arrays[name][first], arrays[name][second], equal_nan=True):
                errors.append(f"Truth array {name} differs between locked model blocks.")

    valid = np.asarray(arrays["valid_mask"], dtype=bool)
    for name in ("pred_lat", "pred_apd90", "pred_patch_prob"):
        nonfinite = int(np.sum(~np.isfinite(arrays[name][valid])))
        if nonfinite:
            errors.append(f"Found {nonfinite} nonfinite valid-pixel values in {name}.")

    legacy = pd.read_csv(LEGACY_RESULTS_PATH)
    parity: list[dict[str, Any]] = []
    for item in models:
        model = item["model_id"]
        for split in ("val", "test"):
            positions = index.index[
                index["model"].eq(model) & index["split"].eq(split)
            ].to_numpy(dtype=int)
            valid_sub = np.asarray(arrays["valid_mask"][positions], dtype=bool)
            metrics = [
                (
                    "LAT",
                    "r2",
                    finite_r2(
                        arrays["true_lat"][positions][valid_sub],
                        arrays["pred_lat"][positions][valid_sub],
                    ),
                ),
                (
                    "APD90",
                    "r2",
                    finite_r2(
                        arrays["true_apd90"][positions][valid_sub],
                        arrays["pred_apd90"][positions][valid_sub],
                    ),
                ),
                (
                    "patch_mask",
                    "dice",
                    micro_dice(
                        arrays["true_patch"][positions],
                        arrays["pred_patch_prob"][positions],
                        valid_sub,
                    ),
                ),
            ]
            for target, column, observed in metrics:
                expected = legacy_metric(legacy, model, split, target, column)
                delta = abs(observed - expected)
                parity.append(
                    {
                        "model": model,
                        "split": split,
                        "target": target,
                        "metric": column,
                        "observed": observed,
                        "legacy": expected,
                        "absolute_difference": delta,
                        "tolerance": parity_tolerance,
                        "passed": delta <= parity_tolerance,
                    }
                )
                if delta > parity_tolerance:
                    errors.append(
                        f"Parity failure for {model}/{split}/{target}: "
                        f"observed={observed}, legacy={expected}, delta={delta}"
                    )
    return {
        "passed": not errors,
        "errors": errors,
        "coverage": coverage,
        "legacy_metric_parity": parity,
        "max_legacy_metric_absolute_difference": max(
            (row["absolute_difference"] for row in parity), default=math.nan
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parity-tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 6:
        raise SystemExit("Use a batch size from 1 through 6; 6 matches the legacy evaluation.")
    if any(path.exists() for path in (FINAL_INDEX, FINAL_MAPS, FINAL_METADATA)):
        raise SystemExit(
            "Refusing to overwrite an existing FNO cache. Move or explicitly inspect the existing files first."
        )
    if not CONTRACT_PATH.is_file() or not INVENTORY_PATH.is_file():
        raise SystemExit("Run script 150 successfully before exporting the FNO cache.")

    contract = read_json(CONTRACT_PATH)
    inventory = read_json(INVENTORY_PATH)
    if not contract.get("validation", {}).get("passed"):
        raise SystemExit("The locked evaluation contract is not valid.")
    artifacts = {row["id"]: row for row in inventory["artifacts"]}
    if "fno_training_statistics" not in artifacts:
        raise SystemExit(
            "The artifact inventory does not yet contain fno_training_statistics; rerun script 150."
        )
    stats_path = resolve_repo_path(artifacts["fno_training_statistics"]["path"])
    if sha256(stats_path) != artifacts["fno_training_statistics"]["sha256"]:
        raise SystemExit("Saved FNO normalization statistics no longer match the locked inventory.")
    stats = load_stats(stats_path)
    models = locked_models(contract, inventory)
    for item in models:
        expected_hash = artifacts[item["checkpoint_artifact_id"]]["sha256"]
        if sha256(item["checkpoint_path"]) != expected_hash:
            raise SystemExit(f"Checkpoint hash changed for {item['model_id']}.")

    rows = load_index()
    rows = rows[rows["split"].isin(["val", "test"])].reset_index(drop=True)
    if len(rows) != 450:
        raise SystemExit(f"Expected 450 validation/test measurement rows, found {len(rows)}.")
    if rows["row_id"].duplicated().any():
        raise SystemExit("FNO validation/test index contains duplicate row_id values.")
    expected_variant_counts = {
        "combined_all_available": 150,
        "combined_bz_axial": 150,
        "combined_bz_axial_noise": 150,
    }
    observed_variant_counts = dict(sorted(Counter(rows["combined_variant"]).items()))
    if observed_variant_counts != expected_variant_counts:
        raise SystemExit(
            f"Unexpected validation/test variant counts: {observed_variant_counts}"
        )

    descriptors = load_descriptors()
    sample_shape = tuple(
        np.load(PROJECT_ROOT / rows.iloc[0]["LAT_grid_file"], mmap_mode="r").shape
    )
    if sample_shape != (32, 32):
        raise SystemExit(f"Unexpected target grid shape: {sample_shape}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    total_rows = len(rows) * len(models)
    index_records: list[dict[str, Any]] = []
    replay_metadata: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix=".fno_cache_", dir=ANALYSIS_DIR) as temp_name:
        temp_dir = Path(temp_name)
        arrays, array_paths = open_array_memmaps(temp_dir, total_rows, sample_shape)
        array_index = 0
        for model_number, item in enumerate(models, start=1):
            model, replay = build_model(item["checkpoint_path"], stats, device)
            replay_metadata[item["model_id"]] = replay
            loader = make_loader(
                rows,
                stats,
                batch_size=args.batch_size,
                shuffle=False,
                input_norm=replay["input_norm"],
                descriptors=descriptors,
            )
            print(
                f"Exporting {item['model_id']} ({model_number}/{len(models)}), "
                f"{len(rows)} rows at batch size {args.batch_size}...",
                flush=True,
            )
            with torch.inference_mode():
                row_cursor = 0
                for batch in loader:
                    field, _params, _area, _condition = model(
                        batch["x"].to(device).float(),
                        batch["desc"].to(device).float(),
                    )
                    field_np = field.detach().cpu().numpy()
                    patch_probability = torch.sigmoid(field[:, 3]).detach().cpu().numpy()
                    truth_np = batch["y_fields"].numpy()
                    valid_np = batch["valid_mask"].numpy().astype(bool)
                    current_batch = int(field_np.shape[0])

                    for batch_offset in range(current_batch):
                        arrays["true_lat"][array_index] = (
                            truth_np[batch_offset, 0] * stats.field_std[0] + stats.field_mean[0]
                        )
                        arrays["true_apd90"][array_index] = (
                            truth_np[batch_offset, 2] * stats.field_std[2] + stats.field_mean[2]
                        )
                        arrays["true_patch"][array_index] = truth_np[batch_offset, 3]
                        arrays["valid_mask"][array_index] = valid_np[batch_offset]
                        arrays["pred_lat"][array_index] = (
                            field_np[batch_offset, 0] * stats.field_std[0] + stats.field_mean[0]
                        )
                        arrays["pred_apd90"][array_index] = (
                            field_np[batch_offset, 2] * stats.field_std[2] + stats.field_mean[2]
                        )
                        arrays["pred_patch_prob"][array_index] = patch_probability[batch_offset]

                        source = rows.iloc[row_cursor + batch_offset]
                        index_records.append(
                            {
                                "array_index": array_index,
                                "model": item["model_id"],
                                "model_role": item["role"],
                                "row_id": source["row_id"],
                                "base_sample_id": source["base_sample_id"],
                                "split": source["split"],
                                "condition": source["condition"],
                                "combined_variant": source["combined_variant"],
                            }
                        )
                        array_index += 1
                    row_cursor += current_batch
                    if row_cursor % 100 < current_batch or row_cursor == len(rows):
                        print(
                            f"  {item['model_id']}: {row_cursor}/{len(rows)}",
                            flush=True,
                        )
            del loader, model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        for array in arrays.values():
            array.flush()
        index = pd.DataFrame(index_records)
        validation = validate_cache(
            index=index,
            arrays=arrays,
            models=models,
            rows_per_model=len(rows),
            parity_tolerance=args.parity_tolerance,
        )
        if not validation["passed"]:
            print(json.dumps(validation, indent=2), flush=True)
            raise SystemExit("FNO cache validation failed; no final cache files were published.")

        temp_index = temp_dir / FINAL_INDEX.name
        temp_maps = temp_dir / FINAL_MAPS.name
        temp_metadata = temp_dir / FINAL_METADATA.name
        index.to_csv(temp_index, index=False)
        packed = {
            name: np.load(path, mmap_mode="r") for name, path in array_paths.items()
        }
        np.savez_compressed(temp_maps, **packed)
        del packed

        metadata = {
            "schema_version": "1.0.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Frozen-checkpoint FNO full-grid cache for grouped ICBINB analysis.",
            "training_performed": False,
            "inference_only": True,
            "device": str(device),
            "batch_size": args.batch_size,
            "source_contract": str(CONTRACT_PATH.relative_to(PROJECT_ROOT)),
            "source_contract_sha256": sha256(CONTRACT_PATH),
            "source_inventory": str(INVENTORY_PATH.relative_to(PROJECT_ROOT)),
            "source_inventory_sha256": sha256(INVENTORY_PATH),
            "source_statistics": str(stats_path.relative_to(PROJECT_ROOT)),
            "source_statistics_sha256": sha256(stats_path),
            "models": [item["model_id"] for item in models],
            "model_roles": {item["model_id"]: item["role"] for item in models},
            "checkpoints": {
                item["model_id"]: {
                    "path": str(item["checkpoint_path"].relative_to(PROJECT_ROOT)),
                    "sha256": sha256(item["checkpoint_path"]),
                }
                for item in models
            },
            "replay_configuration": replay_metadata,
            "splits": ["val", "test"],
            "variants": [
                "combined_all_available",
                "combined_bz_axial",
                "combined_bz_axial_noise",
            ],
            "row_count": int(len(index)),
            "rows_per_model": int(len(rows)),
            "grid_shape": list(sample_shape),
            "arrays": {
                name: {"shape": [total_rows, *sample_shape], "dtype": np.dtype(dtype).name}
                for name, dtype in ARRAY_DTYPES.items()
            },
            "index_file": str(FINAL_INDEX.relative_to(PROJECT_ROOT)),
            "index_sha256": sha256(temp_index),
            "maps_file": str(FINAL_MAPS.relative_to(PROJECT_ROOT)),
            "maps_sha256": sha256(temp_maps),
            "validation": validation,
            "safety": {
                "legacy_outputs_overwritten": False,
                "one_model_loaded_at_a_time": True,
                "legacy_evaluation_batch_size_replayed": args.batch_size == 6,
                "full_dataset_held_in_memory": False,
            },
        }
        write_json(temp_metadata, metadata)
        temp_index.replace(FINAL_INDEX)
        temp_maps.replace(FINAL_MAPS)
        temp_metadata.replace(FINAL_METADATA)

    print(
        json.dumps(
            {
                "passed": True,
                "index": str(FINAL_INDEX),
                "maps": str(FINAL_MAPS),
                "metadata": str(FINAL_METADATA),
                "row_count": total_rows,
                "max_legacy_metric_absolute_difference": validation[
                    "max_legacy_metric_absolute_difference"
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
