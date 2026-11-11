"""Put the SupeRR Selector data pack in data/raw/ (gitignored), stripping folder names and upload prefixes.

    python scripts/setup_data.py "path/to/superr_selector_season_2_problem_statement_data_pack.zip"
    python scripts/setup_data.py path/to/folder/with/the/csv/files
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pressure.datapack import extract  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2 or not Path(sys.argv[1]).expanduser().exists():
        print(__doc__)
        return 2
    st = extract(Path(sys.argv[1]).expanduser(), REPO / "data" / "raw")
    ok = True
    for k, s in st.items():
        good = s["present"] and s["matches_official_pack"]
        ok &= good
        print(f"{'ok ' if good else 'CHECK'}  {k}.csv  {s['rows']:,} rows (expected {s['expected']:,})"
              if s["present"] else f"MISSING  {k}.csv")
    print("data/raw is ready." if ok else "Some files are missing or differ - check the pack.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
