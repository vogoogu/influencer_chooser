"""Deterministic decision engine: load evidence, apply hard filters (with reasons), score seven
criteria, rank, and test how stable the ranking is. No AI is involved in any number here."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import config, db, embeddings, forest, predict
from .taxonomy import CATEGORY_NICHES, PARENT_OF

AGE_BUCKETS = [("age_18_24", 18, 24), ("age_25_34", 25, 34), ("age_35_44", 35, 44), ("age_45_plus", 45, 64)]


@dataclass
class FilterSettings:
    countries: list[str] = field(default_factory=lambda: ["AT"])
    platforms: list[str] = field(default_factory=lambda: list(config.PLATFORMS))
    budget_eur: float = 20000.0
    max_fee_share: float = 0.40                 # one influencer may cost at most this share of the budget
    allow_limited: bool = True
    exclude_conflicts: bool = True
    exclude_data_warnings: bool = False
    excluded_flags: list[str] = field(default_factory=list)
    max_severity: int = 2                       # exclude influencers with an active flag above this severity
    min_followers: int = 0
    max_followers: int = 10**9
    ig_format: str = "reel"
    client_brand_id: int | None = None
    category: str | None = None
    start: date | None = None
    end: date | None = None
    apply_correction: bool = True
    forecast_model: str = "tier"                # "forest", "tier" or "naive" (naive wins if apply_correction is False)


# =========================================================================== loading
def _parse_countries(js: str) -> dict[str, float]:
    try:
        return {d["country"]: float(d["share"]) for d in json.loads(js)}
    except Exception:
        return {}


def load_universe(as_of: date, path=None) -> pd.DataFrame:
    """One row per account with everything known at `as_of` (latest snapshot, audience, niches)."""
    inf = db.read("SELECT * FROM influencer", path=path).rename(columns={"id": "influencer_id"})
    acc = db.read("SELECT id AS account_id, influencer_id, platform, handle, verified FROM account", path=path)
    snap_all = db.read("SELECT * FROM account_snapshot WHERE captured_at <= ? ORDER BY account_id, captured_at",
                       (as_of.isoformat(),), path=path)
    latest = snap_all.groupby("account_id").tail(1).rename(columns={"captured_at": "snapshot_date"})
    snap_all["growth"] = snap_all.groupby("account_id").followers.pct_change()
    extra = snap_all.groupby("account_id").apply(
        lambda g: pd.Series({"max_growth_6m": g.growth.tail(6).max(),
                             "conf_recent": g.confidence.tail(3).mean(),
                             "growth_3m": (g.followers.iloc[-1] / g.followers.iloc[-4] - 1) if len(g) >= 4 else np.nan}),
        include_groups=False).reset_index()
    aud_all = db.read("SELECT * FROM audience_snapshot WHERE captured_at <= ?", (as_of.isoformat(),), path=path)
    aud = aud_all.sort_values("captured_at").groupby("account_id").tail(1).drop(columns=["id", "captured_at"])
    aud["countries"] = aud.top_countries.map(_parse_countries)
    niches = db.read("""SELECT inn.influencer_id, n.name, inn.is_primary FROM influencer_niche inn
                        JOIN niche n ON n.id = inn.niche_id""", path=path)
    primary = niches[niches.is_primary == 1].set_index("influencer_id").name.rename("primary_niche")
    secondary = niches[niches.is_primary == 0].groupby("influencer_id").name.apply(list).rename("secondary_niches")
    df = (acc.merge(inf, on="influencer_id")
             .merge(latest.drop(columns=["id"]), on="account_id", how="inner")
             .merge(extra, on="account_id", how="left")
             .merge(aud, on="account_id", how="left")
             .merge(primary, left_on="influencer_id", right_index=True, how="left")
             .merge(secondary, left_on="influencer_id", right_index=True, how="left"))
    df["secondary_niches"] = df.secondary_niches.map(lambda x: x if isinstance(x, list) else [])
    df["days_old"] = (pd.Timestamp(as_of) - pd.to_datetime(df.snapshot_date)).dt.days
    df["tier"] = predict.tier_of(df.followers)
    return df.rename(columns={"display_name": "name", "engagement_rate": "er", "fake_follower_pct": "fake"})


def load_rates(path=None) -> pd.DataFrame:
    return db.read("SELECT influencer_id, platform, deliverable, price_eur, valid_to FROM rate_card", path=path)


def load_flags(as_of: date, path=None) -> pd.DataFrame:
    return db.read("""SELECT influencer_id, flag_type, severity, flagged_at FROM safety_flag
                      WHERE expires_at IS NULL OR expires_at >= ?""", (as_of.isoformat(),), path=path)


def find_conflicts(settings: FilterSettings, as_of: date, path=None) -> pd.DataFrame:
    """Competitor promotion in the 90 days before the campaign and exclusivity overlaps."""
    cols = ["influencer_id", "type", "detail"]
    start = settings.start or (as_of + timedelta(days=14))
    end = settings.end or (start + timedelta(days=28))
    if settings.client_brand_id:
        comp = db.read("SELECT competitor_id AS id FROM brand_competitor WHERE brand_id = ?",
                       (int(settings.client_brand_id),), path=path).id.tolist()
    elif settings.category:
        comp = db.read("SELECT id FROM brand WHERE category = ?", (settings.category,), path=path).id.tolist()
    else:
        comp = []
    if not comp:
        return pd.DataFrame(columns=cols)
    ph = ",".join("?" * len(comp))
    promo = db.read(f"""SELECT a.influencer_id, 'competitor_promo' AS type,
                               b.name || ' on ' || ci.posted_at AS detail
                        FROM content_item ci JOIN account a ON a.id = ci.account_id JOIN brand b ON b.id = ci.brand_id
                        WHERE ci.is_sponsored = 1 AND ci.brand_id IN ({ph})
                          AND julianday(?) - julianday(ci.posted_at) BETWEEN 0 AND 90""",
                    tuple(comp) + (start.isoformat(),), path=path)
    excl = db.read(f"""SELECT e.influencer_id, 'exclusivity' AS type,
                              'exclusive to ' || b.name || ' until ' || e.valid_to AS detail
                       FROM exclusivity e JOIN brand b ON b.id = e.brand_id
                       WHERE e.brand_id IN ({ph}) AND e.valid_from <= ? AND e.valid_to >= ?""",
                   tuple(comp) + (end.isoformat(), start.isoformat()), path=path)
    return pd.concat([promo, excl], ignore_index=True)[cols]


def load_experience(category: str | None, path=None) -> pd.DataFrame:
    """Per influencer: reliability and category-specific history from logged results."""
    h = predict.load_history(path)
    cols = ["influencer_id", "n", "on_time", "rating", "n_cat", "rating_cat"]
    if h.empty:
        return pd.DataFrame(columns=cols)
    g = h.groupby("influencer_id").agg(n=("ratio", "size"), on_time=("delivered_on_time", "mean"),
                                       rating=("client_rating", "mean"))
    if category:
        c = h[h.category == category].groupby("influencer_id").agg(n_cat=("ratio", "size"), rating_cat=("client_rating", "mean"))
        g = g.join(c)
    for col in ("n_cat", "rating_cat"):
        if col not in g:
            g[col] = np.nan
    g["n_cat"] = g.n_cat.fillna(0)
    return g.reset_index()[cols]


# =========================================================================== preparation
def prepare(as_of: date, settings: FilterSettings, profile: dict, brief_text: str = "", path=None
            ) -> tuple[pd.DataFrame, dict]:
    """Universe with derived columns (prices, predictions, flags, similarity). No filtering yet."""
    df = load_universe(as_of, path)
    corr = predict.fit_correction(predict.load_history(path))
    method = "naive" if (not settings.apply_correction or settings.forecast_model == "naive") else settings.forecast_model

    # deliverable and price
    fmt = {"instagram": settings.ig_format, "tiktok": "video", "youtube": "integration"}
    df["deliverable"] = df.platform.map(fmt)
    rates = load_rates(path)
    df = df.merge(rates, on=["influencer_id", "platform", "deliverable"], how="left").rename(columns={"price_eur": "price"})
    df["rate_expired"] = df.valid_to.notna() & (pd.to_datetime(df.valid_to) < pd.Timestamp(as_of))

    # reach prediction (naive views x deliverable multiplier, corrected by learned tier bias)
    raw = df.avg_views * df.deliverable.map(config.DELIVERABLE_REACH_MULT).fillna(1.0)
    df["reach_naive"] = raw
    conf = df.conf_recent.fillna(0.7).clip(0, 1)
    p10 = np.zeros(len(df)); p50 = np.zeros(len(df)); p90 = np.zeros(len(df)); sig = np.zeros(len(df))
    for t in predict.TIER_NAMES:
        m = (df.tier == t).values
        if m.any():
            a, b, c, s = predict.predict_reach(raw[m].values, t, corr, conf[m].values, method != "naive")
            p10[m], p50[m], p90[m], sig[m] = a, b, c, s
    df["reach_p10"], df["reach_p50"], df["reach_p90"], df["reach_sigma"] = p10, p50, p90, sig
    df["cpm"] = df.price / (df.reach_p50.clip(lower=1) / 1000)

    # flags
    fl = load_flags(as_of, path)
    df["safety_flags"] = df.influencer_id.map(
        lambda i: [(r.flag_type, int(r.severity)) for r in fl[fl.influencer_id == i].itertuples()])
    df["safety_penalty"] = df.safety_flags.map(lambda x: min(1.0, 0.18 * sum(s for _, s in x)))
    df["max_severity"] = df.safety_flags.map(lambda x: max([s for _, s in x], default=0))
    df["consent_ok"] = pd.to_datetime(df.consent_until) >= pd.Timestamp(as_of)

    # data warnings
    warn = []
    for r in df.itertuples():
        w = []
        if r.days_old > config.STALE_DAYS:
            w.append(f"stale stats ({int(r.days_old)} days old)")
        if pd.notna(r.max_growth_6m) and r.max_growth_6m >= config.SPIKE_GROWTH:
            w.append(f"follower spike (+{r.max_growth_6m * 100:.0f}% in one month)")
        if r.fake >= config.FAKE_PCT_WARN:
            w.append(f"fake followers {r.fake:.0f}%")
        if r.rate_expired:
            w.append("rate card expired")
        if pd.notna(r.conf_recent) and r.conf_recent < config.LOW_CONFIDENCE:
            w.append("low data confidence")
        warn.append(w)
    df["data_warnings"] = warn

    # conflicts
    cf = find_conflicts(settings, as_of, path)
    df["conflicts"] = df.influencer_id.map(lambda i: cf[cf.influencer_id == i].detail.tolist())

    # semantic similarity over ALL influencers, then percentile (stable under filtering)
    sims, backend = embeddings.similarity(embeddings.brief_document(profile, brief_text), embeddings.influencer_documents(path))
    df["sim_raw"] = df.influencer_id.map(sims).fillna(0.0)
    df["sim_pct"] = df.sim_raw.rank(pct=True)
    rel = set(profile.get("relevant_niches", []))
    rel_parents = {PARENT_OF.get(n) for n in rel}

    def niche_match(r) -> float:
        if r.primary_niche in rel:
            return 1.0
        if any(n in rel for n in r.secondary_niches):
            return 0.6
        if PARENT_OF.get(r.primary_niche) in rel_parents and rel:
            return 0.3
        return 0.0
    df["niche_match"] = df.apply(niche_match, axis=1)
    # default conversion rate per reached person: assumed rate for the goal, scaled by niche match
    goal = profile.get("goal", "awareness")
    df["conv_rate"] = config.GOAL_CONV_RATE.get(goal, 0.004) * (0.7 + 0.7 * df.niche_match)
    fmeta = {"requested": method, "reach": "naive" if method == "naive" else "tier correction", "conv": "assumed rates", "notes": []}
    if method == "forest":
        _apply_forest(df, settings, profile, conf, fmeta, path)
    df["cpm"] = df.price / (df.reach_p50.clip(lower=1) / 1000)
    meta = {"embedding_backend": backend, "correction": corr, "as_of": as_of, "forecast": fmeta}
    return df, meta


def _apply_forest(df: pd.DataFrame, settings: FilterSettings, profile: dict, conf: pd.Series, fmeta: dict, path=None) -> None:
    """Replace the tier-corrected reach forecast (and the assumed conversion rate) by the random forest's,
    but only for the parts where the forest beat its baseline in cross-validation."""
    bundle = forest.get_bundle(path)
    if bundle is None:
        fmeta["notes"].append(f"Random forest needs at least {forest.MIN_ROWS} logged results; tier correction used instead.")
        return
    fc, _ = forest.forecast(bundle, df, settings.category or profile.get("category"), profile.get("goal", "awareness"), conf, path)
    r, c = bundle["reach"], bundle["conv"]
    if r.beats_baseline:
        raw = df.reach_naive.values
        df["reach_p50"] = raw * np.exp(fc.rf_mu.values)
        df["reach_p10"] = raw * np.exp(fc.rf_lo.values)
        df["reach_p90"] = raw * np.exp(fc.rf_hi.values)
        df["reach_sigma"] = (fc.rf_hi.values - fc.rf_lo.values) / (2 * predict.Z90)
        fmeta["reach"] = "random forest"
    else:
        fmeta["notes"].append(f"Reach forest did not beat '{r.best_baseline}' in cross-validation ({r.gain_vs_best:+.0%}); tier correction used.")
    if c.beats_baseline:
        df["conv_rate"] = fc.rf_conv_rate.values
        fmeta["conv"] = "random forest"
    else:
        fmeta["notes"].append(f"Conversion forest did not beat '{c.best_baseline}' ({c.gain_vs_best:+.0%}); assumed rates used.")
    fmeta["reach_gain"], fmeta["conv_gain"] = r.gain_vs_best, c.gain_vs_best
    fmeta["n"] = bundle["n"]


# =========================================================================== filters
def apply_filters(df: pd.DataFrame, s: FilterSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (passed, excluded). Each excluded row carries the first reason it failed."""
    reasons = pd.Series("", index=df.index, dtype=object)

    def mark(mask, text):
        nonlocal reasons
        m = mask & (reasons == "")
        reasons[m] = text

    mark(df.status != "active", "not active")
    mark(~df.consent_ok, "GDPR consent expired")
    mark(df.availability == "unavailable", "unavailable")
    if not s.allow_limited:
        mark(df.availability == "limited", "only limited availability")
    mark(~df.platform.isin(s.platforms), "platform not selected")
    geo = df.countries.map(lambda c: sum(c.get(x, 0) for x in s.countries) if isinstance(c, dict) else 0)
    df = df.assign(geo_share=geo)
    if s.countries:
        mark(~(df.country.isin(s.countries) | (geo >= 0.35)), "not in target country, audience elsewhere")
    mark((df.followers < s.min_followers) | (df.followers > s.max_followers), "outside follower range")
    mark(df.price.isna(), "no rate card for requested format")
    if s.budget_eur and s.max_fee_share:
        mark(df.price > s.budget_eur * s.max_fee_share, f"fee above {s.max_fee_share:.0%} of budget")
    if s.excluded_flags:
        mark(df.safety_flags.map(lambda f: any(t in s.excluded_flags for t, _ in f)), "excluded safety flag (brief restriction)")
    mark(df.max_severity > s.max_severity, f"safety flag severity above {s.max_severity}")
    if s.exclude_conflicts:
        mark(df.conflicts.map(len) > 0, "brand conflict (competitor)")
    if s.exclude_data_warnings:
        mark(df.data_warnings.map(len) > 0, "data quality warning")
    out = df.assign(exclude_reason=reasons)
    return out[reasons == ""].copy(), out[reasons != ""].copy()


# =========================================================================== criteria
def _logistic(x):
    return 1 / (1 + np.exp(-x))


def audience_scores(df: pd.DataFrame, profile: dict, countries: list[str]) -> pd.Series:
    ta = profile.get("target_audience") or {}
    tgt_f, a_min, a_max = ta.get("female_pct"), ta.get("age_min"), ta.get("age_max")
    out = []
    for r in df.itertuples():
        comps = []
        if tgt_f is not None and pd.notna(r.female_pct):
            comps.append(float(np.clip(1 - abs(r.female_pct - tgt_f) / 60.0, 0, 1)))
        if a_min is not None and a_max is not None and pd.notna(r.age_18_24):
            share = 0.0
            for col, lo, hi in AGE_BUCKETS:
                ov = max(0, min(hi, a_max) - max(lo, a_min) + 1) / (hi - lo + 1)
                share += getattr(r, col) / 100.0 * ov
            comps.append(min(1.0, share / 0.7))
        if countries and isinstance(r.countries, dict):
            comps.append(min(1.0, sum(r.countries.get(c, 0) for c in countries) / 0.8))
        out.append(float(np.mean(comps)) if comps else 0.5)
    return pd.Series(out, index=df.index)


def score_candidates(passed: pd.DataFrame, universe: pd.DataFrame, profile: dict, s: FilterSettings,
                     experience: pd.DataFrame) -> pd.DataFrame:
    """Add the seven criterion scores (0-1) to the candidates."""
    d = passed.copy()
    # topic fit
    d["topic_fit"] = 0.5 * d.sim_pct + 0.5 * d.niche_match
    # audience
    d["audience_match"] = audience_scores(d, profile, s.countries)
    # engagement quality, relative to peers of the same platform and tier (peers from the full universe)
    u = universe.assign(adj=universe.er * (1 - universe.fake / 100))
    peer = u.groupby(["platform", "tier"]).adj.median().rename("peer_er")
    d = d.join(peer, on=["platform", "tier"])
    d["adj_er"] = d.er * (1 - d.fake / 100)
    ratio = (d.adj_er / d.peer_er.clip(lower=0.01)).clip(lower=0.05)
    d["engagement_quality"] = _logistic(1.5 * np.log(ratio))
    # cost efficiency: percentile of cpm among all priced accounts
    priced = universe.assign(cpm=universe.cpm).dropna(subset=["cpm"]).cpm.values
    d["cost_efficiency"] = d.cpm.map(lambda c: 1 - (priced < c).mean() if pd.notna(c) else 0.0)
    # brand safety
    d["brand_safety"] = 1 - d.safety_penalty
    # reliability and past performance
    d = d.merge(experience, on="influencer_id", how="left")
    prior = d.relationship_score / 5.0
    n = d.n.fillna(0)
    hist = 0.5 * d.on_time.fillna(0.5) + 0.5 * ((d.rating.fillna(3.0) - 1) / 4.0)
    d["reliability"] = (n * hist + 2 * prior) / (n + 2)
    n_cat = d.n_cat.fillna(0)
    obs = (d.rating_cat.fillna(3.0) - 1) / 4.0
    d["past_performance"] = (n_cat * obs + 1.5 * 0.5) / (n_cat + 1.5)
    d["has_category_history"] = n_cat > 0
    return d


def normalize_weights(w: dict[str, float]) -> dict[str, float]:
    tot = float(sum(max(v, 0) for v in w.values()))
    if tot <= 0:
        return {k: 1 / len(config.CRITERIA) for k in config.CRITERIA}
    return {k: max(w.get(k, 0), 0) / tot for k in config.CRITERIA}


def apply_weights(d: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    w = normalize_weights(weights)
    d = d.copy()
    for c in config.CRITERIA:
        d[f"w_{c}"] = d[c] * w[c]
    d["total"] = sum(d[f"w_{c}"] for c in config.CRITERIA)
    d = d.sort_values("total", ascending=False).reset_index(drop=True)
    d["rank"] = np.arange(1, len(d) + 1)
    return d


# =========================================================================== robustness
def rank_stability(d: pd.DataFrame, weights: dict[str, float], n_top: int = 5, n_sim: int = 400,
                   concentration: float = 25.0, seed: int = 0) -> dict:
    """Perturb the weights (Dirichlet around the chosen weights) and see who stays in the top n."""
    if len(d) == 0:
        return {"stability": 1.0, "top_freq": pd.Series(dtype=float)}
    n_top = min(n_top, len(d))
    w = np.array([normalize_weights(weights)[c] for c in config.CRITERIA])
    S = d[config.CRITERIA].values
    rng = np.random.default_rng(seed)
    base_top = set(np.argsort(-(S @ w))[:n_top])
    counts = np.zeros(len(d))
    jac = []
    for _ in range(n_sim):
        ws = rng.dirichlet(w * concentration + 1e-3)
        top = np.argsort(-(S @ ws))[:n_top]
        counts[top] += 1
        jac.append(len(base_top & set(top)) / len(base_top | set(top)))
    return {"stability": float(np.mean(jac)), "top_freq": pd.Series(counts / n_sim, index=d.index)}


def weight_sensitivity(d: pd.DataFrame, weights: dict[str, float], n_top: int = 5) -> pd.DataFrame:
    """For each criterion: how many of the current top n change if its weight doubles or drops to zero?"""
    if len(d) == 0:
        return pd.DataFrame(columns=["criterion", "double", "zero"])
    n_top = min(n_top, len(d))
    S = d[config.CRITERIA].values
    base = normalize_weights(weights)
    base_top = set(np.argsort(-(S @ np.array([base[c] for c in config.CRITERIA])))[:n_top])
    rows = []
    for c in config.CRITERIA:
        res = {}
        for label, factor in (("double", 2.0), ("zero", 0.0)):
            w2 = dict(weights)
            w2[c] = weights[c] * factor
            nw = normalize_weights(w2)
            top = set(np.argsort(-(S @ np.array([nw[k] for k in config.CRITERIA])))[:n_top])
            res[label] = n_top - len(base_top & top)
        rows.append({"criterion": c, **res})
    return pd.DataFrame(rows)


def expected_conversions(d: pd.DataFrame, goal: str) -> pd.Series:
    if "conv_rate" in d:
        return d.reach_p50 * d.conv_rate
    rate = config.GOAL_CONV_RATE.get(goal, 0.004)
    return d.reach_p50 * rate * (0.7 + 0.7 * d.niche_match)


def run_scoring(as_of: date, settings: FilterSettings, profile: dict, brief_text: str = "", path=None) -> dict:
    """Prepare, filter and score. Weights are NOT applied yet, so the UI can re-rank instantly."""
    universe, meta = prepare(as_of, settings, profile, brief_text, path)
    passed, excluded = apply_filters(universe, settings)
    exp = load_experience(settings.category or profile.get("category"), path)
    scored = score_candidates(passed, universe, profile, settings, exp) if len(passed) else passed
    return {"scored": scored, "excluded": excluded, "universe": universe, "meta": meta}


def rank(scored: pd.DataFrame, weights: dict, goal: str) -> pd.DataFrame:
    """Apply weights, sort, add rank and expected conversions."""
    if scored.empty:
        return scored
    ranked = apply_weights(scored, weights)
    ranked["exp_conversions"] = expected_conversions(ranked, goal)
    return ranked


def run_pipeline(as_of: date, settings: FilterSettings, profile: dict, weights: dict, brief_text: str = "",
                 path=None) -> dict:
    """Everything in one call (used by tests and scripts)."""
    out = run_scoring(as_of, settings, profile, brief_text, path)
    out["ranked"] = rank(out["scored"], weights, profile.get("goal", "awareness"))
    return out
