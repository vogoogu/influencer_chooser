"""Reach prediction with learned correction, plus calibration analytics.

The historic campaigns were predicted by a naive model (average views). Comparing those
predictions with the logged results gives a per-tier bias and spread, which this module
applies to future predictions ("learning loop")."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, db
from .taxonomy import CATEGORY_NICHES, PARENT_OF

TIER_BINS = [0, 10_000, 100_000, 500_000, 1_000_000, 10**12]
TIER_NAMES = ["nano", "micro", "mid", "macro", "mega"]
Z90 = 1.2816        # z value for the 10th / 90th percentile
DEFAULT_SIGMA = 0.35


def tier_of(followers) -> pd.Series | str:
    if np.isscalar(followers):
        return str(pd.cut([followers], TIER_BINS, labels=TIER_NAMES, right=False)[0])
    return pd.cut(followers, TIER_BINS, labels=TIER_NAMES, right=False).astype(str)


def load_history(path=None) -> pd.DataFrame:
    """One row per participation with a logged result, with the account stats that were known
    when the campaign started."""
    sql = """
    SELECT p.id AS participation_id, p.influencer_id, p.account_id, p.deliverable, p.fee_eur,
           p.reach_p10, p.reach_p50, p.reach_p90, COALESCE(p.reach_naive, p.reach_p50) AS reach_naive, r.actual_reach, r.actual_engagement, r.conversions,
           r.delivered_on_time, r.client_rating, r.measured_at, c.id AS campaign_id, c.start_date, c.goal, b.category,
           a.platform, i.relationship_score, i.country,
           (SELECT s.followers FROM account_snapshot s WHERE s.account_id = p.account_id
             AND s.captured_at < c.start_date ORDER BY s.captured_at DESC LIMIT 1) AS followers_then,
           (SELECT s.engagement_rate FROM account_snapshot s WHERE s.account_id = p.account_id
             AND s.captured_at < c.start_date ORDER BY s.captured_at DESC LIMIT 1) AS er_then,
           (SELECT s.fake_follower_pct FROM account_snapshot s WHERE s.account_id = p.account_id
             AND s.captured_at < c.start_date ORDER BY s.captured_at DESC LIMIT 1) AS fake_then
    FROM participation p
    JOIN result r     ON r.participation_id = p.id
    JOIN campaign c   ON c.id = p.campaign_id
    JOIN brand b      ON b.id = c.brand_id
    JOIN account a    ON a.id = p.account_id
    JOIN influencer i ON i.id = p.influencer_id
    """
    df = db.read(sql, path=path)
    if df.empty:
        return df
    df["followers_then"] = df["followers_then"].fillna(df["followers_then"].median())
    df["er_then"] = df["er_then"].fillna(df["er_then"].median())
    df["fake_then"] = df["fake_then"].fillna(df["fake_then"].median())
    df["tier"] = tier_of(df["followers_then"])
    df["ratio"] = df["actual_reach"] / df["reach_naive"].clip(lower=1)      # always vs the plain average-views forecast
    df["log_ratio"] = np.log(df["ratio"].clip(lower=1e-3))
    df["inside"] = ((df.actual_reach >= df.reach_p10) & (df.actual_reach <= df.reach_p90)).astype(int)
    return df


def fit_correction(hist: pd.DataFrame, shrink_k: float = 5.0) -> dict:
    """Per-tier bias (mu, mean log ratio) and spread (sigma), shrunk towards the global value."""
    out = {"_global": {"mu": 0.0, "sigma": DEFAULT_SIGMA, "n": 0}}
    if hist is None or hist.empty:
        for t in TIER_NAMES:
            out[t] = {"mu": 0.0, "sigma": DEFAULT_SIGMA, "n": 0}
        return out
    g_mu, g_sd, n_all = float(hist.log_ratio.mean()), float(hist.log_ratio.std(ddof=1) or DEFAULT_SIGMA), len(hist)
    out["_global"] = {"mu": g_mu, "sigma": g_sd, "n": n_all}
    for t in TIER_NAMES:
        sub = hist[hist.tier == t].log_ratio
        n = len(sub)
        mu_t = float(sub.mean()) if n else g_mu
        sd_t = float(sub.std(ddof=1)) if n > 2 else g_sd
        out[t] = {"mu": (n * mu_t + shrink_k * g_mu) / (n + shrink_k),
                  "sigma": (n * sd_t + shrink_k * g_sd) / (n + shrink_k), "n": n}
    return out


def predict_reach(raw_views, tier: str, corr: dict, confidence: float = 1.0, apply_correction: bool = True):
    """Return (p10, p50, p90). `raw_views` may be a float or a numpy array.
    Low data confidence widens the interval."""
    c = corr.get(tier) or corr["_global"]
    mu = c["mu"] if apply_correction else 0.0
    sigma = max(c["sigma"], 0.15) * (1 + 0.5 * (1 - np.clip(confidence, 0, 1)))
    p50 = np.asarray(raw_views, dtype=float) * np.exp(mu)
    return p50 * np.exp(-Z90 * sigma), p50, p50 * np.exp(Z90 * sigma), sigma


def coverage_by_tier(hist: pd.DataFrame) -> pd.DataFrame:
    if hist.empty:
        return pd.DataFrame(columns=["tier", "n", "bias_pct", "inside_p10_p90"])
    g = hist.groupby("tier").agg(n=("ratio", "size"), mean_ratio=("ratio", "mean"),
                                 inside=("inside", "mean")).reindex(TIER_NAMES).dropna()
    g["bias_pct"] = (g.mean_ratio - 1) * 100
    return g.reset_index().rename(columns={"inside": "inside_p10_p90"})[["tier", "n", "bias_pct", "inside_p10_p90"]]


# ------------------------------------------------------------------ driver analysis
def _niche_fit(row_primary: str, secondary: list[str], category: str) -> float:
    niches = CATEGORY_NICHES.get(category, [])
    if row_primary in niches:
        return 1.0
    if any(n in niches for n in secondary):
        return 0.5
    if any(PARENT_OF.get(n) == PARENT_OF.get(row_primary) for n in niches):
        return 0.2
    return 0.0


def driver_analysis(hist: pd.DataFrame, outcome: str = "reach_vs_prediction", n_boot: int = 300, seed: int = 0,
                    path=None) -> pd.DataFrame:
    """Which known-at-decision-time features predict success? Standardised OLS with bootstrap CI.
    outcome: 'reach_vs_prediction' (log actual/predicted reach) or 'conversion_rate'
    (log conversions per 1,000 reached, demeaned by campaign goal)."""
    if hist is None or len(hist) < 30:
        return pd.DataFrame(columns=["feature", "coef", "lo", "hi"])
    niches = db.read("""SELECT i.id AS influencer_id, n.name AS niche, inn.is_primary
                        FROM influencer_niche inn JOIN niche n ON n.id = inn.niche_id
                        JOIN influencer i ON i.id = inn.influencer_id""", path=path)
    prim = niches[niches.is_primary == 1].set_index("influencer_id").niche.to_dict()
    sec = niches[niches.is_primary == 0].groupby("influencer_id").niche.apply(list).to_dict()
    d = hist.copy()
    d["niche_fit"] = [_niche_fit(prim.get(i, ""), sec.get(i, []), c) for i, c in zip(d.influencer_id, d.category)]
    peer = d.groupby(["platform", "tier"]).er_then.transform("median")
    d["engagement_vs_peers"] = np.log(d.er_then.clip(lower=0.01) / peer.clip(lower=0.01))
    d["fake_followers_pct"] = d.fake_then
    d["relationship_score"] = d.relationship_score.astype(float)
    d["log_followers"] = np.log(d.followers_then.clip(lower=1))
    feats = ["niche_fit", "engagement_vs_peers", "fake_followers_pct", "relationship_score", "log_followers"]
    if outcome == "conversion_rate":
        y = np.log((d.conversions + 1) / (d.actual_reach / 1000).clip(lower=0.05))
        y = y - y.groupby(d.goal).transform("mean")
    else:
        y = d.log_ratio
    X = d[feats].astype(float)
    X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    Xm, ym = np.column_stack([np.ones(len(X)), X.values]), y.values.astype(float)

    def fit(idx):
        beta, *_ = np.linalg.lstsq(Xm[idx], ym[idx], rcond=None)
        return beta[1:]

    rng = np.random.default_rng(seed)
    base = fit(np.arange(len(ym)))
    boots = np.array([fit(rng.integers(0, len(ym), len(ym))) for _ in range(n_boot)])
    return pd.DataFrame({"feature": feats, "coef": base, "lo": np.percentile(boots, 5, axis=0),
                         "hi": np.percentile(boots, 95, axis=0)})
