"""Read a SupeRR Selector data pack (zip or folder) into a raw/ directory, stripping folder names and prefixes.
Used by scripts/setup_data.py and the dashboard's "Run on a data pack" page."""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path
from typing import IO

EXPECTED = {"ball_by_ball": 200_999, "matches": 883, "competitions": 141, "teams": 353, "playing_eleven": 28_381,
            "competition_team_player_mapping": 6_413}


def canonical(name: str) -> str | None:
    stem = Path(name).name.lower()
    if stem.startswith("._") or not stem.endswith(".csv"):
        return None
    for k in sorted(EXPECTED, key=len, reverse=True):
        if re.search(rf"(^|[_\-]){k}\.csv$", stem):
            return k
    return None


def extract(src: str | Path | IO[bytes], raw: Path) -> dict[str, dict]:
    """Copy the six CSVs from a zip (path or file object) or folder into ``raw``. Returns per-file status."""
    raw.mkdir(parents=True, exist_ok=True)
    if isinstance(src, (str, Path)) and Path(src).is_dir():
        for f in Path(src).rglob("*.csv"):
            k = canonical(f.name)
            if k:
                shutil.copy2(f, raw / f"{k}.csv")
    else:
        with zipfile.ZipFile(src) as z:
            for info in z.infolist():
                k = None if "__MACOSX" in info.filename else canonical(info.filename)
                if k:
                    with z.open(info) as fin, open(raw / f"{k}.csv", "wb") as fout:
                        shutil.copyfileobj(fin, fout)
                elif info.filename.lower().endswith(".pdf") and "__MACOSX" not in info.filename:
                    with z.open(info) as fin, open(raw / "problem_statement.pdf", "wb") as fout:
                        shutil.copyfileobj(fin, fout)
    status = {}
    for k, n in EXPECTED.items():
        p = raw / f"{k}.csv"
        if not p.exists():
            status[k] = {"present": False, "rows": 0, "expected": n}
            continue
        with open(p, encoding="utf-8", errors="ignore") as fh:
            rows = sum(1 for _ in fh) - 1
        status[k] = {"present": True, "rows": rows, "expected": n, "matches_official_pack": rows == n}
    return status
