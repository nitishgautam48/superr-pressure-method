"""Analyst layer: the analyst's pin / drop / swap_rank decisions on top of the model top 10s.

Both model_top10 and final_top10 are kept; each final pick carries a provenance:
model | analyst-confirmed | analyst-moved | analyst-add. Overrides without a 10+ character reason are refused.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import yaml

from .decisions import log_decision

ACTIONS = ("pin", "drop", "swap_rank")


def load_overrides(cfg: Any) -> list[dict[str, Any]]:
    p = cfg.path("overrides")
    if not p.exists():
        return []
    return list((yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("overrides") or [])


def validate(o: dict[str, Any], rated: dict[str, pd.DataFrame], cfg: Any) -> str | None:
    if o.get("role") not in ("batter", "bowler"):
        return "role must be batter or bowler"
    if o.get("action") not in ACTIONS:
        return f"action must be one of {ACTIONS}"
    if len(str(o.get("reason") or "").strip()) < int(cfg.get("analyst.min_reason_chars", 10)):
        return "a reason of at least 10 characters is required"
    r = rated[o["role"]]
    if int(o.get("player_id", -1)) not in set(r["player_id"].astype(int)):
        return f"player_id {o.get('player_id')} is not a rated {o['role']}"
    if o["action"] == "swap_rank" and not (1 <= int(o.get("rank", 0)) <= int(cfg.get("rating.top_n", 10))):
        return "swap_rank needs rank between 1 and 10"
    return None


def apply(rated: dict[str, pd.DataFrame], cfg: Any, overrides: list[dict] | None = None,
          log: bool = True) -> dict[str, Any]:
    """Return final lists, model lists, provenance and refused overrides."""
    top_n = int(cfg.get("rating.top_n", 10))
    ov = load_overrides(cfg) if overrides is None else overrides
    out: dict[str, Any] = {"refused": []}
    for role in ("batter", "bowler"):
        r = rated[role]
        use_target = bool(cfg.get("rr_target.use_for_final", False)) and "target_rank" in r and r["target_rank"].notna().any()
        e = r[r["target_rank"].notna()].sort_values("target_rank") if use_target else r[r["eligible"]].sort_values("rank")
        order = e["player_id"].astype(int).tolist()
        pure = r[r["eligible"]].sort_values("rank")["player_id"].astype(int).tolist()[:top_n]
        model = order[:top_n]
        mine = [o for o in ov if o.get("role") == role]
        ok = []
        for o in mine:
            err = validate(o, rated, cfg)
            (out["refused"].append({**o, "error": err}) if err else ok.append(o))
        dropped = {int(o["player_id"]) for o in ok if o["action"] == "drop"}
        final = [p for p in order if p not in dropped][:top_n]
        prov = {p: "model" for p in final}
        reasons: dict[int, str] = {}
        for o in ok:
            pid = int(o["player_id"])
            if o["action"] == "pin":
                if pid in final and "rank" not in o:
                    prov[pid] = "analyst-confirmed"
                else:
                    if pid in final:
                        final.remove(pid)
                    pos = int(o.get("rank", top_n)) - 1
                    final.insert(min(pos, len(final)), pid)
                    final = final[:top_n]
                    prov[pid] = "analyst-add" if pid not in model else "analyst-moved"
            elif o["action"] == "swap_rank" and pid in final:
                final.remove(pid)
                final.insert(int(o["rank"]) - 1, pid)
                prov[pid] = "analyst-moved"
            reasons[pid] = str(o["reason"]).strip()
            if log:
                log_decision(cfg, f"override.{role}.{o['action']}", "-", f"{pid} rank={o.get('rank', '-')}",
                             o["reason"], source="analyst", dedupe=True)
        for p in model:
            if p not in final:
                reasons.setdefault(p, next((str(o["reason"]) for o in ok if int(o["player_id"]) == p), ""))
        names = r.set_index("player_id")
        def rows(ids, final_list=False):
            res = []
            for i, p in enumerate(ids, start=1):
                x = names.loc[p]
                res.append({"rank": i, "player_id": int(p), "name": str(x["name"]), "display": str(x.get("display", x["name"])),
                            "model_rank": int(x["rank"]) if pd.notna(x["rank"]) else None, "prs": float(x["prs"]),
                            "target_rank": int(x["target_rank"]) if "target_rank" in x and pd.notna(x["target_rank"]) else None,
                            "provenance": prov.get(p, "model") if final_list else "model",
                            "reason": reasons.get(p, "")})
            return res
        out[f"{role}s_model"] = rows(model)
        out[f"{role}s_pure_pressure"] = rows(pure)
        out["list_basis"] = "RR auction targets" if use_target else "pure pressure rating"
        out[f"{role}s_final"] = rows(final, True)
        out[f"{role}s_changes"] = [{"player_id": int(p), "name": str(names.loc[p, "name"]),
                                    "change": "dropped" if p in dropped else "moved out",
                                    "reason": reasons.get(p, "")} for p in model if p not in final]
    return out


def add_override(cfg: Any, o: dict[str, Any], rated: dict[str, pd.DataFrame]) -> str | None:
    """Validate and append one override to config/analyst_overrides.yaml. Returns an error or None."""
    err = validate(o, rated, cfg)
    if err:
        return err
    p = cfg.path("overrides")
    doc = (yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else None) or {}
    lst = list(doc.get("overrides") or [])
    clean = {k: (int(v) if k in ("player_id", "rank") else str(v).strip()) for k, v in o.items() if v not in (None, "")}
    lst.append(clean)
    header = p.read_text(encoding="utf-8").split("overrides:")[0] if p.exists() else ""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(header + yaml.safe_dump({"overrides": lst}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return None


def remove_override(cfg: Any, index: int, reason: str) -> str | None:
    if len((reason or "").strip()) < int(cfg.get("analyst.min_reason_chars", 10)):
        return "a reason of at least 10 characters is required"
    p = cfg.path("overrides")
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    lst = list(doc.get("overrides") or [])
    if not 0 <= index < len(lst):
        return "no such override"
    gone = lst.pop(index)
    header = p.read_text(encoding="utf-8").split("overrides:")[0]
    p.write_text(header + yaml.safe_dump({"overrides": lst}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    log_decision(cfg, f"override.{gone.get('role')}.{gone.get('action')}.removed", gone, "-", reason)
    return None


def refresh_final(cfg: Any) -> dict[str, Any]:
    """Re-apply overrides to the stored ratings and rewrite final_lists / model_vs_final / provenance."""
    from . import store

    base = cfg.path("processed")
    rated = {r: store.read_table(f"{r}s_rated.parquet", base) for r in ("batter", "bowler")}
    finals = apply(rated, cfg)
    for role in ("batter", "bowler"):
        prov = {x["player_id"]: x["provenance"] for x in finals[f"{role}s_final"]}
        rated[role]["provenance"] = rated[role]["player_id"].map(prov)
        rated[role]["in_final"] = rated[role]["player_id"].isin(prov)
        store.write_table(rated[role], f"{role}s_rated.parquet", base)
    store.write_json(finals, "final_lists.json", base)
    store.write_json({"model": {r: finals[f"{r}s_model"] for r in ("batter", "bowler")},
                      "final": {r: finals[f"{r}s_final"] for r in ("batter", "bowler")},
                      "changes": {r: finals[f"{r}s_changes"] for r in ("batter", "bowler")},
                      "refused_overrides": finals["refused"]}, "model_vs_final.json", base)
    return finals
