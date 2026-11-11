"""Win-probability model: the trained ML core.

* Separate models for innings 1 and innings 2; target = batting team wins (no-results excluded,
  ties = 0.5 via two half-weighted rows, configurable).
* Candidates: logistic regression on spline features vs HistGradientBoosting; picked by grouped-by-match
  CV log-loss (a match is never split across folds). Out-of-fold predictions are calibrated with
  isotonic regression (cross-fitted for the reported metrics).
* Time split check: train <= 2024, test >= 2025.
* wp_before / wp_after per ball (batting team's view), WPA = wp_after - wp_before; terminal balls of
  innings 2 take the actual result so a match's WPA sums to its outcome.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from .state import outcome_label

log = logging.getLogger(__name__)
CAT = ["competition_group", "bat_tier", "bowl_tier"]
NUM = {1: ["score", "wickets", "balls_remaining", "phase_num"],
       2: ["runs_needed", "balls_remaining", "wickets_in_hand", "rrr"]}
EPS = 1e-4


def features(df: pd.DataFrame, innings: int, after: bool = False) -> pd.DataFrame:
    """Feature frame for a state. ``after=True`` describes the state the ball leaves."""
    out = pd.DataFrame(index=df.index)
    for c in CAT:
        out[c] = df[c].astype(str)
    br = df["balls_remaining_after" if after else "balls_remaining"].astype(float)
    if innings == 1:
        out["score"] = df["score_after" if after else "score_before"].astype(float)
        out["wickets"] = df["wickets_after" if after else "wickets_before"].astype(float)
        out["balls_remaining"] = br
        legal = df["allotted"] - br
        out["phase_num"] = np.select([legal < 36, legal < 90], [0.0, 1.0], 2.0)
    else:
        rn = df["runs_needed_after" if after else "runs_needed"].astype(float)
        wk = df["wickets_after" if after else "wickets_before"].astype(float)
        out["runs_needed"] = rn
        out["balls_remaining"] = br
        out["wickets_in_hand"] = 10 - wk
        out["rrr"] = (6 * rn / br.clip(lower=1)).clip(-6, 36)
    return out


def _models(innings: int, cfg: Any) -> dict[str, Any]:
    num, h = NUM[innings], cfg.get("wp_model.hgb")
    lr = make_pipeline(ColumnTransformer([
        ("spl", make_pipeline(StandardScaler(), SplineTransformer(n_knots=int(cfg.get("wp_model.spline_knots", 6)),
                                                                  degree=3)), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)]), LogisticRegression(max_iter=2000, C=1.0))
    hgb = make_pipeline(ColumnTransformer([("num", "passthrough", num),
                                           ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)]),
                        HistGradientBoostingClassifier(max_iter=h["max_iter"], learning_rate=h["learning_rate"],
                                                       max_leaf_nodes=h["max_leaf_nodes"],
                                                       min_samples_leaf=h["min_samples_leaf"],
                                                       random_state=int(cfg.get("seed", 42))))
    return {"logistic_splines": lr, "hist_gradient_boosting": hgb}


def training_rows(state: pd.DataFrame, innings: int, cfg: Any) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """X, y, sample weight, groups. Ties become one 1-row and one 0-row at weight 0.5 (or are dropped)."""
    s = state[(state["innings"] == innings)].copy()
    if cfg.get("wp_model.exclude_reduced_overs", True) and "reduced_overs" in s:
        s = s[~s["reduced_overs"]]
    s["y"] = outcome_label(s)
    s = s[s["y"].notna()]
    if innings == 2:
        s = s[s["runs_needed"].notna()]
    tie = s["y"] == 0.5
    if cfg.get("wp_model.tie_handling", "half") == "drop":
        s = s[~tie]
        w = np.ones(len(s))
        y = s["y"].to_numpy()
        frame = s
    else:
        win, lose = s[tie].assign(y=1.0), s[tie].assign(y=0.0)
        frame = pd.concat([s[~tie], win, lose])
        w = np.where(frame.index.isin(s[tie].index), 0.5, 1.0)
        y = frame["y"].to_numpy()
    return features(frame, innings), y.astype(int), w, frame["match_id"].to_numpy()


def _metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict[str, float]:
    p = np.clip(p, EPS, 1 - EPS)
    return {"log_loss": round(float(log_loss(y, p, sample_weight=w)), 4),
            "brier": round(float(brier_score_loss(y, p, sample_weight=w)), 4)}


def _calibration_bins(y: np.ndarray, p: np.ndarray, w: np.ndarray, n: int) -> list[dict[str, float]]:
    edges = np.linspace(0, 1, n + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n - 1)
    out = []
    for i in range(n):
        k = idx == i
        if k.sum():
            out.append({"bin": i, "p_mean": round(float(np.average(p[k], weights=w[k])), 4),
                        "observed": round(float(np.average(y[k], weights=w[k])), 4), "n": int(k.sum())})
    return out


class _Calibrator:
    """Picklable calibration map p_raw -> p: none | isotonic | sigmoid (Platt on the logit)."""

    def __init__(self, method: str, p: np.ndarray, y: np.ndarray, w: np.ndarray):
        self.method = method
        if method == "isotonic":
            self.f = IsotonicRegression(out_of_bounds="clip", y_min=0.002, y_max=0.998).fit(p, y, sample_weight=w)
        elif method == "sigmoid":
            self.f = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p), y, sample_weight=w)

    def __call__(self, q: np.ndarray) -> np.ndarray:
        if self.method == "isotonic":
            return np.clip(self.f.predict(q), EPS, 1 - EPS)
        if self.method == "sigmoid":
            return self.f.predict_proba(_logit(q))[:, 1]
        return np.clip(q, EPS, 1 - EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


def _calibrator(method: str, p: np.ndarray, y: np.ndarray, w: np.ndarray) -> _Calibrator:
    return _Calibrator(method, p, y, w)


@dataclass
class InningsModel:
    innings: int
    name: str
    model: Any
    iso: Any        # calibration function p_raw -> p

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(X)[:, 1]
        return np.clip(self.iso(raw), EPS, 1 - EPS)


def fit_innings(state: pd.DataFrame, innings: int, cfg: Any) -> tuple[InningsModel, dict[str, Any]]:
    X, y, w, groups = training_rows(state, innings, cfg)
    gkf = GroupKFold(n_splits=int(cfg.get("wp_model.cv_folds", 5)))
    folds = list(gkf.split(X, y, groups))
    # leakage guard: no match appears in both train and test of any fold
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te]), "a match was split across CV folds"
    report: dict[str, Any] = {"rows": int(len(X)), "matches": int(len(set(groups))), "candidates": {}}
    oof: dict[str, np.ndarray] = {}
    for name, mdl in _models(innings, cfg).items():
        p = np.zeros(len(X))
        for tr, te in folds:
            mdl.fit(X.iloc[tr], y[tr], **{mdl.steps[-1][0] + "__sample_weight": w[tr]})
            p[te] = mdl.predict_proba(X.iloc[te])[:, 1]
        oof[name] = p
        report["candidates"][name] = _metrics(y, p, w)
    best = min(report["candidates"], key=lambda n: report["candidates"][n]["log_loss"])
    # cross-fitted calibration of the chosen model's OOF predictions: none vs isotonic vs Platt, keep the best
    cal_scores, cal_preds = {}, {}
    for method in ("none", "isotonic", "sigmoid"):
        pc = np.zeros(len(X))
        for tr, te in folds:
            pc[te] = _calibrator(method, oof[best][tr], y[tr], w[tr])(oof[best][te])
        cal_scores[method], cal_preds[method] = _metrics(y, pc, w), pc
    cal = min(cal_scores, key=lambda k: cal_scores[k]["log_loss"])
    pc = cal_preds[cal]
    report["chosen"] = best
    report["calibration"] = {"chosen": cal, "candidates": cal_scores, "configured": cfg.get("wp_model.calibration")}
    report["cv_raw"] = report["candidates"][best]
    report["cv_calibrated"] = cal_scores[cal]
    report["calibration_bins"] = _calibration_bins(y, pc, w, int(cfg.get("wp_model.calibration_bins", 10)))
    report["base_rate"] = round(float(np.average(y, weights=w)), 4)
    report["folds_disjoint_by_match"] = True
    # final: fit chosen model on all rows; isotonic on its OOF predictions
    final = _models(innings, cfg)[best]
    final.fit(X, y, **{final.steps[-1][0] + "__sample_weight": w})
    iso = _calibrator(cal, oof[best], y, w)
    # time split: train <= 2024, test >= 2025 (chosen model, calibrated on its own train OOF)
    ts = cfg.get("wp_model.time_split")
    season = state["season"].reindex(X.index).to_numpy()
    tr, te = season <= ts["train_until"], season >= ts["test_from"]
    if tr.sum() and te.sum():
        m2 = _models(innings, cfg)[best]
        m2.fit(X[tr], y[tr], **{m2.steps[-1][0] + "__sample_weight": w[tr]})
        iso2 = _calibrator(cal, m2.predict_proba(X[tr])[:, 1], y[tr], w[tr])
        pt = iso2(m2.predict_proba(X[te])[:, 1])
        report["time_split"] = {"train_seasons": f"<= {ts['train_until']}", "test_seasons": f">= {ts['test_from']}",
                                "train_rows": int(tr.sum()), "test_rows": int(te.sum()),
                                **_metrics(y[te], pt, w[te]),
                                "test_base_rate_brier": _metrics(y[te], np.full(te.sum(), np.average(y[tr], weights=w[tr])),
                                                                 w[te])["brier"]}
    report["no_skill_brier"] = round(float(np.average((y - report["base_rate"]) ** 2, weights=w)), 4)
    return InningsModel(innings, best, final, iso), report


def fit(state: pd.DataFrame, cfg: Any) -> tuple[dict[int, InningsModel], dict[str, Any]]:
    models, rep = {}, {}
    for inn in (1, 2):
        models[inn], rep[f"innings_{inn}"] = fit_innings(state, inn, cfg)
        r = rep[f"innings_{inn}"]
        log.info("WP innings %d: %s chosen, CV log-loss %.4f, Brier %.4f (calibrated)", inn, r["chosen"],
                 r["cv_calibrated"]["log_loss"], r["cv_calibrated"]["brier"])
    return models, rep


def _inn2_start(df: pd.DataFrame, models: dict[int, InningsModel], total_after: pd.Series) -> np.ndarray:
    """Chasing side's WP at the start of innings 2, for a first innings ending at ``total_after``."""
    s = pd.DataFrame({"competition_group": df["competition_group"].values, "bat_tier": df["bowl_tier"].values,
                      "bowl_tier": df["bat_tier"].values, "runs_needed_after": (total_after + 1).values,
                      "balls_remaining_after": df["allotted2"].values if "allotted2" in df else 120.0,
                      "wickets_after": 0.0, "allotted": df["allotted2"].values if "allotted2" in df else 120.0})
    return models[2].predict(features(s, 2, after=True))


def wp_after_state(df: pd.DataFrame, models: dict[int, InningsModel], score_after: pd.Series | None = None,
                   wickets_after: pd.Series | None = None, legal_after: pd.Series | None = None,
                   final_result: pd.Series | None = None) -> np.ndarray:
    """Batting side's WP in the state after a (real or hypothetical) ball."""
    d = df.copy()
    if score_after is not None:
        d["score_after"] = score_after
    if wickets_after is not None:
        d["wickets_after"] = wickets_after
    if legal_after is not None:
        d["balls_remaining_after"] = (d["allotted"] - legal_after).clip(lower=0)
    out = np.full(len(d), np.nan)
    i1 = (d["innings"] == 1).to_numpy()
    i2 = ~i1
    if i1.any():
        a = d[i1]
        over = ((a["wickets_after"] >= 10) | (a["balls_remaining_after"] <= 0)).to_numpy()
        p = models[1].predict(features(a, 1, after=True))
        if over.any():
            p[over] = 1 - _inn2_start(a[over], models, a.loc[over, "score_after"])
        out[i1] = p
    if i2.any():
        a = d[i2].copy()
        a["runs_needed_after"] = a["target"] - a["score_after"]
        p = models[2].predict(features(a, 2, after=True))
        won = (a["runs_needed_after"] <= 0).to_numpy()
        done = ((a["wickets_after"] >= 10) | (a["balls_remaining_after"] <= 0)).to_numpy() & ~won
        p[won] = 1.0
        p[done] = np.where(a.loc[done, "runs_needed_after"].to_numpy() == 1, 0.5, 0.0)
        out[i2] = p
    if final_result is not None:  # terminal ball of innings 2: the actual result
        fr = final_result.to_numpy()
        k = ~np.isnan(fr)
        out[k] = fr[k]
    return out


def add_wp(state: pd.DataFrame, models: dict[int, InningsModel]) -> pd.DataFrame:
    s = state.copy()
    wb = np.full(len(s), np.nan)
    for inn in (1, 2):
        k = (s["innings"] == inn).to_numpy().copy()
        if inn == 2:
            k &= s["runs_needed"].notna().to_numpy()
        wb[k] = models[inn].predict(features(s[k], inn))
    s["wp_before"] = wb
    res = outcome_label(s)
    terminal = (s["innings"] == 2) & s["innings_end"]
    s["wp_after"] = wp_after_state(s, models, final_result=res.where(terminal))
    s["wpa"] = s["wp_after"] - s["wp_before"]
    return s
