                      
                                                          
 
                                                                               
                                                                              
                                              
   

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from torch_import_utils import import_torch_prefer_env

torch = import_torch_prefer_env(PROJECT_ROOT)


OUT_DIR = PROJECT_ROOT / "outputs" / "neural_v1c"
REPORT = OUT_DIR / "cuda_setup_report.json"


def run_nvidia_smi() -> dict[str, object]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "path": None, "stdout": "", "stderr": "nvidia-smi not found"}
    proc = subprocess.run([exe], text=True, capture_output=True, timeout=15)
    return {
        "available": proc.returncode == 0,
        "path": exe,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cuda_available = bool(torch.cuda.is_available())
    tensor_test_ok = False
    tensor_device = "cpu"
    tensor_error = None
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    try:
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        if cuda_available:
            x = x.to("cuda")
            tensor_device = str(x.device)
        y = (x * 2).sum().item()
        tensor_test_ok = bool(y == 30.0)
    except Exception as exc:                    
        tensor_error = repr(exc)

    nvidia_smi = run_nvidia_smi()
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
        "nvidia_smi": nvidia_smi,
        "warning": None
        if cuda_available
        else (
            "CUDA is unavailable in the current PyTorch import. Fusion training "
            "will fall back to CPU if run now. For WSL/Linux CUDA setup, use the "
            "official PyTorch install selector at https://pytorch.org/get-started/locally/ "
            "and choose the CUDA build matching the installed NVIDIA driver."
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
