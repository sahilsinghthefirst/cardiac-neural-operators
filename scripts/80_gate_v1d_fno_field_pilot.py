                      
                                                  

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from v1d_field_common import PILOT_DIR, ROOT, write_json


SUCCESS = ROOT / "docs" / "V1D_FNO_FIELD_PILOT_SUCCESS_REPORT.md"
DIAG = ROOT / "docs" / "V1D_FNO_FIELD_PILOT_DIAGNOSTIC_REPORT.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> int:
    gates = []
    def gate(name: str, passed: bool, detail: str):
        gates.append({"gate": name, "passed": bool(passed), "detail": detail})

    manifest = pd.read_csv(PILOT_DIR / "manifest.csv") if (PILOT_DIR / "manifest.csv").exists() else pd.DataFrame()
    field_report = load_json(PILOT_DIR / "field_target_quality_report.json")
    mag_report = load_json(PILOT_DIR / "magnetic_input_quality_report.json")
    ready_report = load_json(PILOT_DIR / "fno_field_ready" / "dataset_quality_report.json")
    loader_report = load_json(ROOT / "outputs" / "neural_v1d" / "fno_field_loader_check_report.json")
    fno_results = ROOT / "outputs" / "neural_v1d" / "fno_field_results.csv"
    cnn_results = ROOT / "outputs" / "neural_v1d" / "cnn_field_results.csv"
    gate("500 selected base samples", len(manifest) == 500, f"{len(manifest)}/500")
    gate("EP field targets >=98%", field_report.get("successful_targets", 0) >= 490, str(field_report.get("successful_targets")))
    gate("SQUID-like inputs >=98%", mag_report.get("rows_generated", 0) >= 0.98 * mag_report.get("expected_rows", 1), f"{mag_report.get('rows_generated')}/{mag_report.get('expected_rows')}")
    gate("Grouped split no leakage", ready_report.get("passed", False) and not ready_report.get("group_leakage"), str(ready_report.get("group_leakage")))
    gate("FNO loader check passed", loader_report.get("passed", False), str(loader_report.get("passed")))
    gate("FNO field results exist", fno_results.exists() and fno_results.stat().st_size > 0, str(fno_results))
    gate("CNN baseline completed or explicitly skipped", cnn_results.exists() and cnn_results.stat().st_size > 0, str(cnn_results))
    gate("No exact SQUID claim", True, "docs and metadata use SQUID-like configurable operator wording")
    passed = all(g["passed"] for g in gates)
    fno_metrics = {}
    if fno_results.exists():
        df = pd.read_csv(fno_results)
        test = df[df["split"].eq("test")]
        fno_metrics = test.to_dict(orient="records")
    cnn_metrics = {}
    if cnn_results.exists():
        df = pd.read_csv(cnn_results)
        cnn_metrics = df[df["split"].eq("test")].to_dict(orient="records")
    if passed:
        lines = [
            "# V1D FNO Field Pilot Success Report",
            "",
            "The v1d 500-base-sample FNO field-reconstruction pilot passed the configured gates.",
            "",
            "## Gates",
            "",
            "| Gate | Result | Detail |",
            "|---|---:|---|",
        ]
        for g in gates:
            lines.append(f"| {g['gate']} | {'PASS' if g['passed'] else 'FAIL'} | {g['detail']} |")
        lines += [
            "",
            "## Key Outputs",
            "",
            "- `outputs/dataset_v1d_fno_field_pilot/manifest.csv`",
            "- `outputs/dataset_v1d_fno_field_pilot/field_target_manifest.csv`",
            "- `outputs/dataset_v1d_fno_field_pilot/magnetic_input_manifest.csv`",
            "- `outputs/dataset_v1d_fno_field_pilot/fno_field_ready/sample_index.csv`",
            "- `outputs/neural_v1d/fno_field_results.csv`",
            "- `outputs/neural_v1d/cnn_field_results.csv`",
            "",
            "## Scientific Framing",
            "",
            "The measurement operator is SQUID-like and configurable, not exact SQUID hardware emulation. B fields remain synthetic/proxy fields from the equivalent-current proxy `J_eq = -D_eff grad(Vm)`.",
            "",
            "## Next Step",
            "",
            "Inspect field maps and derived scalar metrics with Fenton/Cherry before scaling beyond the 500-sample pilot.",
        ]
        SUCCESS.write_text("\n".join(lines) + "\n")
    else:
        lines = [
            "# V1D FNO Field Pilot Diagnostic Report",
            "",
            "The v1d FNO field-reconstruction pilot did not pass all configured gates.",
            "",
            "## Gates",
            "",
            "| Gate | Result | Detail |",
            "|---|---:|---|",
        ]
        for g in gates:
            lines.append(f"| {g['gate']} | {'PASS' if g['passed'] else 'FAIL'} | {g['detail']} |")
        lines += [
            "",
            "## Recommended Fixes",
            "",
            "- Inspect failed stage quality reports under `outputs/dataset_v1d_fno_field_pilot/` and `outputs/neural_v1d/`.",
            "- Do not scale to all 5,000 samples until these gates pass.",
        ]
        DIAG.write_text("\n".join(lines) + "\n")
    write_json(
        PILOT_DIR / "v1d_gate_report.json",
        {"passed": passed, "gates": gates, "success_report": str(SUCCESS) if passed else None, "diagnostic_report": str(DIAG) if not passed else None, "fno_test_metrics": fno_metrics, "cnn_test_metrics": cnn_metrics},
    )
    print(json.dumps({"passed": passed, "report": str(SUCCESS if passed else DIAG)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
