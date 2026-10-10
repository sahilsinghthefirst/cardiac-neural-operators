                      
                                                                      

from __future__ import annotations

import json
import re
from pathlib import Path

from bocf_common import ROOT, run_command


OUT_DIR = ROOT / "outputs" / "model_info"


def parse_imp_info(text: str) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for key in ["Authors", "Year", "Title", "Journal", "DOI", "Comment", "Name"]:
        match = re.search(rf"^\s*{re.escape(key)}:\s*(.+)$", text, flags=re.MULTILINE)
        if match:
            metadata[key.lower()] = match.group(1).strip()
    state_match = re.search(r"State variables:\n(?P<body>(?:\s+\S+\n)+)", text)
    if state_match:
        metadata["state_variables"] = [line.strip() for line in state_match.group("body").splitlines() if line.strip()]
    params = re.findall(r"^\s+([A-Za-z0-9_]+)\s+([-+A-Za-z0-9_.|]+)\s*$", text, flags=re.MULTILINE)
    metadata["parameters"] = {name: value for name, value in params}
    metadata["default_vm_scale_note"] = metadata.get("comment", "")
    return metadata


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = run_command(["bench", "--imp=Bueno", "--imp-info"], OUT_DIR, "bench_bueno_imp_info", cwd=ROOT)
    combined = result.stdout + result.stderr
    (OUT_DIR / "bueno_imp_info.txt").write_text(combined)
    (OUT_DIR / "bueno_imp_info_metadata.json").write_text(json.dumps(parse_imp_info(combined), indent=2) + "\n")
    if result.returncode != 0:
        raise SystemExit(f"bench --imp=Bueno --imp-info failed; see {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

