                      
                                                                       

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fno_elementary_common import SpatiotemporalFNO3d, parameter_count, torch


OUT = PROJECT_ROOT / "outputs" / "neural_v1c" / "fno3d_layer_sanity_report.json"


def cuda_memory() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def run_once(device: torch.device) -> dict[str, object]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = SpatiotemporalFNO3d(hidden_width=24, modes_t=16, modes_x=4, modes_y=4, layers=3, dropout=0.05).to(device)
    x = torch.randn(2, 1, 301, 8, 8, device=device)
    logits, reg = model(x)
    loss = logits.square().mean() + reg.square().mean()
    loss.backward()
    grad_ok = []
    for name, param in model.named_parameters():
        if "weights_" in name:
            grad_ok.append(bool(param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum().item() > 0))
    return {
        "device": str(device),
        "input_shape": list(x.shape),
        "logits_shape": list(logits.shape),
        "regression_shape": list(reg.shape),
        "finite_output": bool(torch.isfinite(logits).all() and torch.isfinite(reg).all()),
        "spectral_weight_gradients_ok": bool(grad_ok and all(grad_ok)),
        "parameter_count": parameter_count(model),
        "memory": cuda_memory() if device.type == "cuda" else None,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cuda_available = bool(torch.cuda.is_available())
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "torch_file": str(Path(torch.__file__).resolve()),
        "torch_version": torch.__version__,
        "torch_version_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "modes_t": 16,
        "modes_x": 4,
        "modes_y": 4,
        "grid_t": 301,
        "grid_x": 8,
        "grid_y": 8,
        "modes_within_nyquist": bool(16 <= 301 // 2 and 4 <= 8 // 2 and 4 <= 8 // 2),
        "cpu_check": run_once(torch.device("cpu")),
        "cuda_check": None,
        "passed": False,
    }
    if cuda_available:
        report["cuda_check"] = run_once(torch.device("cuda"))
    cpu_ok = report["cpu_check"]["finite_output"] and report["cpu_check"]["spectral_weight_gradients_ok"]
    cuda_ok = bool(report["cuda_check"] and report["cuda_check"]["finite_output"] and report["cuda_check"]["spectral_weight_gradients_ok"])
    report["passed"] = bool(cpu_ok and cuda_available and cuda_ok and report["modes_within_nyquist"])
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("FNO3d layer sanity check failed or CUDA unavailable")


if __name__ == "__main__":
    main()
