                      
                                                                      

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from v1e_500_common import CHECKPOINT_DIR, NEURAL_DIR, NEURAL_FIG_DIR, require_cuda, write_json
from v1e_field_data_common import compute_stats, load_descriptors, load_index, make_loader, save_stats, torch


SPEC = importlib.util.spec_from_file_location("_v1e_500_fno_field", PROJECT_ROOT / "scripts" / "121_train_v1e_500_fno_field_parametric_multitask.py")
if SPEC is None or SPEC.loader is None:
    raise ImportError("Could not load FNO field helpers")
FNO_TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FNO_TRAIN)


RESULTS_CSV = NEURAL_DIR / "cnn_field_parametric_results.csv"
RESULTS_JSON = NEURAL_DIR / "cnn_field_parametric_results.json"
PRED_CSV = NEURAL_DIR / "cnn_field_parametric_predictions.csv"

CNN_CONFIGS = {
    "cnn_field_plus_param_heads": {"descriptors": False, "param": 0.75, "area": 0.4, "scalar": 0.0, "consistency": 0.02},
    "cnn_field_plus_param_heads_plus_descriptors": {"descriptors": True, "param": 0.75, "area": 0.4, "scalar": 0.0, "consistency": 0.02},
}


class ConvBlock(torch.nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(cin, cout, 3, padding=1),
            torch.nn.BatchNorm2d(cout),
            torch.nn.GELU(),
            torch.nn.Dropout2d(dropout),
            torch.nn.Conv2d(cout, cout, 3, padding=1),
            torch.nn.BatchNorm2d(cout),
            torch.nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNetParametric(torch.nn.Module):
    def __init__(self, in_channels: int, desc_dim: int, base: int, dropout: float, use_descriptors: bool) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base, dropout)
        self.down1 = torch.nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc2 = ConvBlock(base * 2, base * 2, dropout)
        self.down2 = torch.nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.mid = ConvBlock(base * 4, base * 4, dropout)
        self.up2 = torch.nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2, dropout)
        self.up1 = torch.nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base, dropout)
        self.field_head = torch.nn.Conv2d(base, 4, 1)
        if use_descriptors and desc_dim:
            self.desc_encoder = torch.nn.Sequential(torch.nn.Linear(desc_dim, 128), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(128, 64), torch.nn.GELU())
            head_in = base * 4 + 64
        else:
            self.desc_encoder = None
            head_in = base * 4
        self.param_head = torch.nn.Sequential(torch.nn.Linear(head_in, base * 2), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(base * 2, 5))
        self.area_head = torch.nn.Linear(head_in, 3)
        self.condition_head = torch.nn.Linear(head_in, 4)

    def forward(self, x, desc=None):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        mid = self.mid(self.down2(e2))
        pooled = mid.mean(dim=(2, 3))
        if self.desc_encoder is not None and desc is not None and desc.shape[1] > 0:
            pooled = torch.cat([pooled, self.desc_encoder(desc)], dim=1)
        d2 = self.up2(mid)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.field_head(d1), self.param_head(pooled), self.area_head(pooled), self.condition_head(pooled)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", choices=list(CNN_CONFIGS), default=list(CNN_CONFIGS))
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-norm", choices=["per_sample", "train_global"], default="per_sample")
    args = parser.parse_args()
    FNO_TRAIN.seed_all()
    require_cuda(torch)
    rows = load_index()
    desc = load_descriptors()
    stats = compute_stats(rows, desc)
    save_stats(NEURAL_DIR / "v1e_500_cnn_field_parametric_stats.json", stats)
    train_rows = rows[rows["split"].eq("train")].reset_index(drop=True)
    first = next(iter(make_loader(train_rows.head(1), stats, 1, False, args.input_norm, desc)))
    in_channels = int(first["x"].shape[1])
    desc_dim = int(first["desc"].shape[1])
    device = torch.device("cuda")
    all_results = []
    all_preds = []
    histories = {}
    checkpoints = {}
    for config in args.configs:
        weights = {"field": 1.0, "patch": 1.0, "present": 0.5, "condition": 0.12, **CNN_CONFIGS[config]}
        model = SmallUNetParametric(in_channels, desc_dim, args.base, args.dropout, weights["descriptors"]).to(device)
        model, hist, ckpt = FNO_TRAIN.train_model(config, model, rows, desc, stats, args, device, weights)
        res, pred = FNO_TRAIN.evaluate_model(model, stats, rows, desc, args, device, config)
        all_results.append(res)
        all_preds.append(pred)
        histories[config] = hist
        checkpoints[config] = str(ckpt)
    results = pd.concat(all_results, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    NEURAL_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_CSV, index=False)
    preds.to_csv(PRED_CSV, index=False)
    FNO_TRAIN.plot_training(histories, "v1e_500_cnn_field_training_curves.png")
    payload = {
        "architecture": "small_unet_field_parametric_multitask",
        "configs": CNN_CONFIGS,
        "checkpoints": checkpoints,
        "histories": histories,
        "input_channels": in_channels,
        "descriptor_dim": desc_dim,
        "results": results.replace({np.nan: None}).to_dict(orient="records"),
        "cuda": {"torch_version": torch.__version__, "torch_cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0)},
        "leakage_guardrail": "Inputs are combined magnetic tensors, coordinate channels, and optional B_meas-derived descriptors only; target fields and patch params are labels.",
    }
    write_json(RESULTS_JSON, payload)
    print(json.dumps({"results": str(RESULTS_CSV), "predictions": str(PRED_CSV), "configs": args.configs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
