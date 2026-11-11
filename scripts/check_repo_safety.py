"""Fail if anything confidential is tracked or staged: the data pack, derived outputs, secrets, raw tables.

    python scripts/check_repo_safety.py            # check all tracked files (CI)
    python scripts/check_repo_safety.py --staged   # check staged files (pre-commit hook)
"""
from __future__ import annotations

import re
import subprocess
import sys

BLOCKED_PATHS = re.compile(r"^(data/|outputs/|\.env$|\.streamlit/secrets\.toml$|.*\.(csv|parquet|xlsx|xls|zip|pkl|pickle)$)")
ALLOWED = {".streamlit/secrets.toml.example"}
SECRET = re.compile(r"(APP_PASSWORD\s*=\s*['\"](?!choose-a-strong-password)[^'\"]{4,}|GEMINI_API_KEY\s*=\s*\S{10,}|"
                    r"GROQ_API_KEY\s*=\s*\S{10,}|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})")
MAX_BYTES = 2_000_000


def files(staged: bool) -> list[str]:
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"] if staged else ["git", "ls-files"]
    return [f for f in subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.splitlines() if f]


def main() -> int:
    staged = "--staged" in sys.argv
    bad = []
    for f in files(staged):
        if f in ALLOWED:
            continue
        if BLOCKED_PATHS.match(f):
            bad.append(f"{f}: confidential path or raw table type")
            continue
        try:
            if staged:
                blob = subprocess.run(["git", "show", f":{f}"], capture_output=True, check=True).stdout
            else:
                with open(f, "rb") as fh:          # plain Python read: works on Windows, macOS and Linux
                    blob = fh.read()
        except (subprocess.CalledProcessError, OSError):
            continue
        if len(blob) > MAX_BYTES:
            bad.append(f"{f}: larger than 2 MB (data?)")
        elif SECRET.search(blob.decode("utf-8", "ignore")):
            bad.append(f"{f}: looks like it contains a secret")
    if bad:
        print("BLOCKED - confidential content would be committed:\n  " + "\n  ".join(bad))
        return 1
    print(f"ok - {'staged' if staged else 'tracked'} files are clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
