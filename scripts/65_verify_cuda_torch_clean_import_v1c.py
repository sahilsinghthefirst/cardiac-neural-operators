                      
                                                                        
 
                                                                               
                                              
   

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from torch_import_utils import import_torch_prefer_env


OUT_DIR = PROJECT_ROOT / "outputs" / "neural_v1c"
REPORT = OUT_DIR / "cuda_clean_import_report.json"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch = import_torch_prefer_env(PROJECT_ROOT)
    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    tensor_test_ok = False
    tensor_device = "cpu"
    tensor_error = None
    try:
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        if cuda_available:
            x = x.to("cuda")
            tensor_device = str(x.device)
        y = (x * 2).sum().item()
        tensor_test_ok = bool(y == 132.0)
    except Exception as exc:                    
        tensor_error = repr(exc)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "torch_file": str(Path(torch.__file__).resolve()),
        "torch_version": torch.__version__,
        "torch_version_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "gpu_name": gpu_name,
        "tensor_device": tensor_device,
        "tensor_test_ok": tensor_test_ok,
        "tensor_error": tensor_error,
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(f"sys.executable: {report['python_executable']}")
    print(f"torch.__file__: {report['torch_file']}")
    print(f"torch.__version__: {report['torch_version']}")
    print(f"torch.version.cuda: {report['torch_version_cuda']}")
    print(f"torch.cuda.is_available(): {report['cuda_available']}")
    if gpu_name:
        print(f"torch.cuda.get_device_name(0): {gpu_name}")
    print(f"tensor_device: {tensor_device}")
    print(f"tensor_test_ok: {tensor_test_ok}")
    if tensor_error:
        print(f"tensor_error: {tensor_error}")
    print(f"wrote: {REPORT}")


if __name__ == "__main__":
    main()
