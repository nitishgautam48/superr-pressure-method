"""Decision log: every change to weights, thresholds, eligibility, toggles or overrides
is recorded with a timestamp, before -> after and a reason. Changes without a reason are refused."""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path
from typing import Any

COLUMNS = ["timestamp", "setting", "before", "after", "reason", "source"]
AUTOPILOT_REASON = "autopilot default - review"


class DecisionRefused(ValueError):
    pass


def _path(cfg: Any) -> Path:
    return cfg.path("decision_log")


def read_log(cfg: Any) -> list[dict[str, str]]:
    p = _path(cfg)
    if not p.exists():
        return []
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def log_decision(cfg: Any, setting: str, before: Any, after: Any, reason: str, source: str = "analyst",
                 dedupe: bool = False) -> dict[str, str]:
    """Append one decision. Reason must have at least analyst.min_reason_chars characters."""
    reason = (reason or "").strip()
    if len(reason) < int(cfg.get("analyst.min_reason_chars", 10)):
        raise DecisionRefused(f"a reason of at least {cfg.get('analyst.min_reason_chars', 10)} characters is required")
    row = {"timestamp": dt.datetime.now().isoformat(timespec="seconds"), "setting": setting,
           "before": str(before), "after": str(after), "reason": reason, "source": source}
    if dedupe and any(r["setting"] == setting and r["after"] == str(after) for r in read_log(cfg)):
        return row
    p = _path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with p.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def log_autopilot_default(cfg: Any, setting: str, value: Any, note: str = "") -> None:
    """Record a default the autopilot used for a choice that is the analyst's (logged once per value)."""
    log_decision(cfg, setting, "(unset)", value, AUTOPILOT_REASON + (f": {note}" if note else ""),
                 source="autopilot", dedupe=True)
