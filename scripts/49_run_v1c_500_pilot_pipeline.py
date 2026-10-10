                      
                                          
 
                                                                               
                                          
   

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from bocf_v1c_common import FULL_MODE, PILOT_MODE, mode_paths, now_utc, write_json


ROOT = Path(__file__).resolve().parents[1]


def run_stage(name: str, cmd: list[str], log_dir: Path) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    print(f"\n=== {name} ===")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    runtime = time.perf_counter() - start
    (log_dir / f"{name}.command.txt").write_text(" ".join(cmd) + "\n")
    (log_dir / f"{name}.stdout").write_text(result.stdout)
    (log_dir / f"{name}.stderr").write_text(result.stderr)
    (log_dir / f"{name}.returncode.txt").write_text(f"{result.returncode}\n")
    if result.stdout:
        print(result.stdout[-4000:])
    if result.stderr:
        print(result.stderr[-4000:], file=sys.stderr)
    status = {"stage": name, "returncode": result.returncode, "runtime_seconds": runtime, "command": " ".join(cmd)}
    print(f"{name}: returncode={result.returncode} runtime={runtime:.1f}s")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[PILOT_MODE, FULL_MODE], default=PILOT_MODE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--skip-existing", action="store_true", help="Default resume behavior; successful existing samples are skipped.")
    parser.add_argument("--run-full-5000", action="store_true")
    parser.add_argument("--compress-arrays", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == FULL_MODE and not args.run_full_5000:
        print("Refusing full5000 pipeline without --run-full-5000")
        return 2
    if args.run_full_5000 and args.mode != FULL_MODE:
        print("To launch 5,000, use --mode full5000 --run-full-5000. Running pilot mode only.")
    paths = mode_paths(args.mode)
    log_dir = paths["ep"] / "pipeline_logs"
    py = sys.executable
    force = ["--force"] if args.force else []
    compact = ["--compact"] if args.compact else []
    compress = ["--compress-arrays"] if args.compress_arrays else []
    full_flag = ["--run-full-5000"] if args.mode == FULL_MODE and args.run_full_5000 else []
    stages = [
        (
            "40_generate_bocf_dataset_v1c",
            [
                py,
                "scripts/40_generate_bocf_dataset_v1c.py",
                "--mode",
                args.mode,
                "--workers",
                str(args.workers),
                "--seed",
                str(args.seed),
                *compact,
                *force,
                *full_flag,
            ],
        ),
        ("41_check_bocf_dataset_v1c", [py, "scripts/41_check_bocf_dataset_v1c.py", "--mode", args.mode]),
        (
            "42_run_magnetic_forward_v1c_compact",
            [
                py,
                "scripts/42_run_magnetic_forward_v1c_compact.py",
                "--mode",
                args.mode,
                "--workers",
                str(args.workers),
                *compact,
                *force,
                *compress,
            ],
        ),
        ("43_check_magnetic_dataset_v1c", [py, "scripts/43_check_magnetic_dataset_v1c.py", "--mode", args.mode]),
        (
            "44_generate_magnetic_variants_v1c_compact",
            [
                py,
                "scripts/44_generate_magnetic_variants_v1c_compact.py",
                "--mode",
                args.mode,
                "--workers",
                str(args.workers),
                *force,
                *compress,
            ],
        ),
        ("45_check_magnetic_variants_v1c", [py, "scripts/45_check_magnetic_variants_v1c.py", "--mode", args.mode]),
        ("46_build_dataset_v1c_ready", [py, "scripts/46_build_dataset_v1c_ready.py", "--mode", args.mode, "--seed", str(args.seed)]),
        ("47_plot_v1c_500_pilot_summary", [py, "scripts/47_plot_v1c_500_pilot_summary.py", "--mode", args.mode]),
    ]
    if args.mode == PILOT_MODE:
        stages.append(("48_gate_v1c_500_pilot", [py, "scripts/48_gate_v1c_500_pilot.py"]))
    else:
        stages.append(("51_gate_v1c_5000", [py, "scripts/51_gate_v1c_5000.py"]))
    start = time.perf_counter()
    results = [run_stage(name, cmd, log_dir) for name, cmd in stages]
    payload = {
        "created_utc": now_utc(),
        "mode": args.mode,
        "workers": args.workers,
        "seed": args.seed,
        "force": args.force,
        "compact": args.compact,
        "compress_arrays": args.compress_arrays,
        "runtime_seconds": time.perf_counter() - start,
        "stages": results,
        "note": "Nonzero intermediate stages are preserved for gate diagnostics; 5,000 is never launched from pilot mode.",
    }
    write_json(paths["ep"] / "pipeline_run_summary.json", payload)
    failed = [r for r in results if r["returncode"] != 0]
    print(json.dumps({"failed_stages": [r["stage"] for r in failed], "summary": str(paths["ep"] / "pipeline_run_summary.json")}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
