                      
                                                                         

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from torch_import_utils import import_torch_prefer_env

torch = import_torch_prefer_env(PROJECT_ROOT)


OUT_DIR = PROJECT_ROOT / "outputs" / "neural_v1c"
REPORT = OUT_DIR / "torch_cuda_check.json"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    tensor_device = "cpu"
    tensor_test_ok = False
    tensor_error = None
    try:
        x = torch.ones((2, 3), dtype=torch.float32)
        if cuda_available:
            x = x.to("cuda")
            tensor_device = str(x.device)
        y = (x + 1).sum().item()
        tensor_test_ok = bool(y == 12.0)
    except Exception as exc:                    
        tensor_error = repr(exc)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "torch_path": str(Path(torch.__file__).resolve()),
        "torch_version_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "gpu_name": gpu_name,
        "tensor_device": tensor_device,
        "tensor_test_ok": tensor_test_ok,
        "tensor_error": tensor_error,
        "warning": None
        if cuda_available
        else "CUDA is not available; CNN training will use CPU and should be rerun with CUDA-enabled PyTorch for larger sweeps.",
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
