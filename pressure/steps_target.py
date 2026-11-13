"""Pipeline step: RR auction targets -> rated parquets, final lists (analyst layer), RR lens, figures."""
from __future__ import annotations

import logging
from typing import Any

from . import analyst, rr_lens, rr_target, store

log = logging.getLogger(__name__)


def run(ctx: Any) -> None:
    cfg, t = ctx.cfg, ctx.tables
    res = rr_target.build(cfg)
    rr_target.apply_to_rated(cfg, res)
    finals = analyst.refresh_final(cfg)
    base = cfg.path("processed")
    rated = {r: store.read_table(f"{r}s_rated.parquet", base) for r in ("batter", "bowler")}
    if "state" in t and "splits" in t:
        store.write_json(rr_lens.build(t["state"], t["splits"], rated, finals, t["raw"]["teams"], cfg), "rr_lens.json", base)
    store.write_json({k: v for k, v in res.items() if k != "status_all"}, "rr_targets.json", base)
    from . import figures as F
    F.fig_rr_needs(cfg, res)
    F.fig_rr_targets(cfg, res)
    F.export_guard(cfg)
    rr_target.log_defaults(cfg)
    t["batters"], t["bowlers"], t["finals"] = rated["batter"], rated["bowler"], finals
    ctx.metrics.update(target_pool={r: len(res[r]["table"]) for r in ("batter", "bowler")},
                       rr_needs=res["needs"], availability_errors=len(res["availability_errors"]))
    log.info("RR targets: pool %s; needs %s", ctx.metrics["target_pool"], res["needs"])
