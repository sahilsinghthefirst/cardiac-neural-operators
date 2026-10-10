                      
                                                      
 
                                                                               
                                                                                
                                                                               
   

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "neural_v1c"
REPORT = OUT_DIR / "cuda_environment_report.json"
LOCAL_TORCH = PROJECT_ROOT / ".local_deps" / "torch"


def import_torch_with_fallback():
                                                                            
    initial_spec = importlib.util.find_spec("torch")
    used_fallback = False
    if initial_spec is None and LOCAL_TORCH.exists():
        sys.path.insert(0, str(LOCAL_TORCH))
        used_fallback = True
    import torch                

    return torch, initial_spec, used_fallback


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
    torch, initial_spec, used_fallback = import_torch_with_fallback()
    cuda_available = bool(torch.cuda.is_available())
    tensor_test_ok = False
    tensor_device = "cpu"
    tensor_error = None
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    try:
        x = torch.linspace(0, 1, 16, dtype=torch.float32)
        if cuda_available:
            x = x.to("cuda")
            tensor_device = str(x.device)
        y = (x * x).sum().item()
        tensor_test_ok = bool(y > 0)
    except Exception as exc:                    
        tensor_error = repr(exc)

    nvidia_smi = run_nvidia_smi()
    diagnostic = None
    if not cuda_available:
        diagnostic = (
            "The active PyTorch import is CPU-only or cannot access CUDA. "
            "Do not assume GPU training is active. Install a CUDA-enabled "
            "PyTorch build using the official Linux/WSL selector at "
            "https://pytorch.org/get-started/locally/ and then rerun this check."
        )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "initial_torch_spec": str(initial_spec.origin) if initial_spec else None,
        "used_local_torch_fallback": used_fallback,
        "local_torch_fallback_path": str(LOCAL_TORCH) if LOCAL_TORCH.exists() else None,
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
        "diagnostic": diagnostic,
        "official_pytorch_selector": "https://pytorch.org/get-started/locally/",
    }
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
