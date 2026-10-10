                                                                       

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "78_train_fno_field_reconstruction_v1d.py"
SPEC = importlib.util.spec_from_file_location("_train_fno_field_reconstruction_v1d", PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Could not load {PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

evaluate = MODULE.evaluate
field_loss = MODULE.field_loss
seed_all = MODULE.seed_all
FNOFieldReconstructor = MODULE.FNOFieldReconstructor
regression_metrics = MODULE.regression_metrics
patch_metrics = MODULE.patch_metrics
dice_loss = MODULE.dice_loss
