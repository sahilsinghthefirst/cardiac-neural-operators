from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def import_torch_prefer_env(project_root: Path, *, log: bool = True) -> ModuleType:

    used_fallback = False
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        fallback = project_root / ".local_deps" / "torch"
        if not fallback.exists():
            raise
        sys.path.insert(0, str(fallback))
        used_fallback = True
        torch = importlib.import_module("torch")
    if log:
        log_torch_runtime(torch, used_fallback=used_fallback)
    return torch


def log_torch_runtime(torch: ModuleType, *, used_fallback: bool = False) -> None:

    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    cuda_available = bool(torch.cuda.is_available())
    msg = (
        "[torch] "
        f"file={Path(torch.__file__).resolve()} "
        f"version={torch.__version__} "
        f"torch.version.cuda={cuda_version} "
        f"cuda_available={cuda_available} "
        f"used_local_deps_fallback={used_fallback}"
    )
    print(msg, flush=True)
