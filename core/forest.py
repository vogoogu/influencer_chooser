"""Random forest forecasts learned from logged campaign results.

Two forests are trained on every past creator-campaign result:

* reach model: log(actual reach / naive forecast). exp(prediction) is the multiplier applied to the naive forecast.
* conversion model: log(conversions per 1,000 reached people, floor 0.5 conversions), replacing the assumed rates in config.py.

Features are things known BEFORE a campaign starts (audience size, engagement vs peers, fake follower share, recent
follower jumps, rate, relationship, niche fit, the creator's earlier results ...). The target never leaks into the
features: "earlier results" only use campaigns measured before the campaign in question.

Quality is judged by grouped cross-validation (all rows of one campaign stay together) against two baselines:
"naive" (multiplier 1) and "tier correction" (average bias per audience size). If the forest does not beat the
tier correction, the app says so. Intervals come from out-of-bag residuals. Everything is deterministic (fixed seed).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeRegressor

from . import config, db, predict

SEED = 7
MIN_ROWS = 40
Z90 = predict.Z90
GOALS = config.GOALS

REACH_FEATURES = ["log_followers", "er_vs_peers", "fake_pct", "max_growth_6m", "days_old", "relationship_score", "niche_fit",
                  "prior_log_ratio", "prior_n", "log_views_per_follower", "log_fee", "is_instagram", "is_tiktok", "deliver_mult"]
CONV_FEATURES = REACH_FEATURES + [f"goal_{g}" for g in GOALS]

FEATURE_LABELS = {
    "log_followers": "Audience size",
    "er_vs_peers": "Engagement vs peers",
    "fake_pct": "Fake follower share",
    "max_growth_6m": "Biggest monthly follower jump (6 months)",
    "days_old": "Age of the latest stats",
    "relationship_score": "Relationship score",
    "niche_fit": "Niche fit with the product",
    "prior_log_ratio": "Creator's earlier delivery vs forecast",
    "prior_n": "Number of earlier campaigns",
    "log_views_per_follower": "Views per follower",
    "log_fee": "Fee",
    "is_instagram": "Platform: Instagram",
    "is_tiktok": "Platform: TikTok",
    "deliver_mult": "Format reach factor",
    **{f"goal_{g}": f"Goal: {g.replace('_', ' ')}" for g in GOALS},
}


def display_value(feature: str, v: float) -> str:
    """Human-readable value for a (possibly log-transformed) feature."""
    if feature in ("log_followers", "log_fee", "log_views_per_follower"):
        x = float(np.exp(v))
        return f"{x:,.0f}" if feature != "log_views_per_follower" else f"{x:.2f}"
    if feature == "er_vs_peers":
        return f"{np.exp(v):.2f}x peers"
    if feature == "prior_log_ratio":
        return f"x{np.exp(v):.2f}"
    if feature.startswith(("is_", "goal_")):
        return "yes" if v >= 0.5 else "no"
    if feature in ("fake_pct",):
        return f"{v:.0f}%"
    if feature == "max_growth_6m":
        return f"{v:+.0%}"
    if feature == "days_old":
        return f"{v:.0f} days"
    if feature in ("prior_n", "relationship_score"):
        return f"{v:.0f}"
    return f"{v:.2f}"


def display_threshold(feature: str, thr: float) -> str:
    if feature.startswith(("is_", "goal_")):
        return "no" if thr < 0.5 else "yes"
    return display_value(feature, thr)


# ================================================================ feature building
def _peer_table(hist: pd.DataFrame) -> pd.Series:
    return hist.groupby(["platform", "tier"]).er_then.median()


def _niche_maps(path=None) -> tuple[dict, dict]:
    n = db.read("""SELECT inn.influencer_id, n.name, inn.is_primary FROM influencer_niche inn
                   JOIN niche n ON n.id = inn.niche_id""", path=path)
    prim = n[n.is_primary == 1].set_index("influencer_id").name.to_dict()
    sec = n[n.is_primary == 0].groupby("influencer_id").name.apply(list).to_dict()
    return prim, sec


def _prior_stats(hist: pd.DataFrame, influencer_id: int, before: str) -> tuple[float, int]:
    """Shrunk mean log ratio of the creator's EARLIER measured campaigns (no leakage)."""
    m = hist[(hist.influencer_id == influencer_id) & (hist.measured_at < before)]
    n = len(m)
    return (float(m.log_ratio.sum() / (n + 2)), n) if n else (0.0, 0)


def _snapshot_arrays(path=None) -> dict:
    s = db.read("SELECT account_id, captured_at, followers FROM account_snapshot ORDER BY account_id, captured_at", path=path)
    return {int(a): (g.captured_at.values.astype(str), g.followers.values.astype(float)) for a, g in s.groupby("account_id")}


def _growth_and_age(snaps: dict, account_id: int, start: str) -> tuple[float, float]:
    dates, f = snaps.get(int(account_id), (np.array([]), np.array([])))
    n = int(np.searchsorted(dates, start, side="left"))
    if n == 0:
        return 0.0, 365.0
    window = f[max(0, n - 7):n]
    growth = float(np.max(window[1:] / window[:-1] - 1)) if len(window) > 1 else 0.0
    age = float((pd.Timestamp(start) - pd.Timestamp(dates[n - 1])).days)
    return growth, age


def build_training(hist: pd.DataFrame, path=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (X for all CONV features, targets, peer table). Rows align with `hist`."""
    prim, sec = _niche_maps(path)
    snaps = _snapshot_arrays(path)
    peers = _peer_table(hist)
    rows = []
    for r in hist.itertuples():
        growth, age = _growth_and_age(snaps, r.account_id, r.start_date)
        prior, n = _prior_stats(hist, r.influencer_id, r.start_date)
        mult = config.DELIVERABLE_REACH_MULT.get(r.deliverable, 1.0)
        peer = peers.get((r.platform, r.tier), hist.er_then.median())
        row = {
            "log_followers": np.log(max(r.followers_then, 1)),
            "er_vs_peers": np.log(max(r.er_then, 0.01) / max(peer, 0.01)),
            "fake_pct": r.fake_then, "max_growth_6m": growth, "days_old": age,
            "relationship_score": float(r.relationship_score),
            "niche_fit": predict._niche_fit(prim.get(r.influencer_id, ""), sec.get(r.influencer_id, []), r.category),
            "prior_log_ratio": prior, "prior_n": float(n),
            "log_views_per_follower": np.log(max(r.reach_naive / max(mult, 1e-6), 1) / max(r.followers_then, 1)),
            "log_fee": np.log(max(r.fee_eur, 1.0)),
            "is_instagram": float(r.platform == "instagram"), "is_tiktok": float(r.platform == "tiktok"), "deliver_mult": mult,
        }
        for g in GOALS:
            row[f"goal_{g}"] = float(r.goal == g)
        rows.append(row)
    X = pd.DataFrame(rows, index=hist.index)[CONV_FEATURES]
    y = pd.DataFrame({"reach": hist.log_ratio.values,
                      "conv": np.log(hist.conversions.clip(lower=0.5) / (hist.actual_reach / 1000).clip(lower=0.05)).values}, index=hist.index)
    return X, y, peers


def build_prediction_frame(universe: pd.DataFrame, hist: pd.DataFrame, peers: pd.Series, category: str | None, goal: str,
                           path=None) -> pd.DataFrame:
    """Features for the current candidates, computed exactly like the training features."""
    prim, sec = _niche_maps(path)
    med_er = hist.er_then.median()
    med_fee = float(np.log(max(universe.price.median(), 1.0))) if universe.price.notna().any() else 6.0
    rows = []
    for r in universe.itertuples():
        mult = config.DELIVERABLE_REACH_MULT.get(r.deliverable, 1.0)
        prior, n = _prior_stats(hist.assign(measured_at=hist.measured_at), r.influencer_id, "9999-12-31")
        peer = peers.get((r.platform, r.tier), med_er)
        row = {
            "log_followers": np.log(max(r.followers, 1)),
            "er_vs_peers": np.log(max(r.er, 0.01) / max(peer, 0.01)),
            "fake_pct": r.fake, "max_growth_6m": 0.0 if pd.isna(r.max_growth_6m) else float(r.max_growth_6m),
            "days_old": float(r.days_old), "relationship_score": float(r.relationship_score),
            "niche_fit": predict._niche_fit(r.primary_niche, list(r.secondary_niches), category or ""),
            "prior_log_ratio": prior, "prior_n": float(n),
            "log_views_per_follower": np.log(max(r.avg_views, 1) / max(r.followers, 1)),
            "log_fee": med_fee if pd.isna(r.price) else np.log(max(r.price, 1.0)),
            "is_instagram": float(r.platform == "instagram"), "is_tiktok": float(r.platform == "tiktok"), "deliver_mult": mult,
        }
        for g in GOALS:
            row[f"goal_{g}"] = float(goal == g)
        rows.append(row)
    return pd.DataFrame(rows, index=universe.index)[CONV_FEATURES]


# ================================================================ model
def _forest(n_estimators: int) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=4, max_features=0.6, oob_score=True,
                                 bootstrap=True, random_state=SEED, n_jobs=1)


@dataclass
class ForestModel:
    target: str                        # "reach" or "conv"
    features: list[str]
    model: RandomForestRegressor
    q_lo: float                        # 10th percentile of out-of-bag residuals
    q_hi: float                        # 90th percentile
    resid_var: float
    n: int
    metrics: pd.DataFrame              # cross-validated comparison with the baselines
    importance: pd.DataFrame
    oof: pd.DataFrame                  # out-of-fold predictions (for the scatter plot)
    best_baseline: str = ""
    gain_vs_best: float = 0.0          # 1 - RMSE(forest) / RMSE(best baseline), cross-validated
    beats_baseline: bool = False
    X: pd.DataFrame = field(repr=False, default=None)
    y: pd.Series = field(repr=False, default=None)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.features])

    def interval(self, X: pd.DataFrame, widen: np.ndarray | float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu = self.predict(X)
        return mu, mu + self.q_lo * widen, mu + self.q_hi * widen

    def tree_votes(self, x_row: pd.DataFrame) -> np.ndarray:
        """Prediction of every single tree for one row (the forest's 'votes')."""
        xr = x_row[self.features].values
        return np.array([t.predict(xr)[0] for t in self.model.estimators_])


MIN_GAIN = 0.03        # the forest must cut the cross-validated RMSE by at least 3% versus the best baseline


def _group_mean_baseline(train_groups: pd.Series, train_y: pd.Series, test_groups: pd.Series) -> np.ndarray:
    return test_groups.map(train_y.groupby(train_groups).mean()).fillna(train_y.mean()).values


def _tier_baseline(train_tiers: pd.Series, train_y: pd.Series, test_tiers: pd.Series, k: float = 5.0) -> np.ndarray:
    g = train_y.mean()
    mu = {}
    for t in predict.TIER_NAMES:
        m = train_y[train_tiers == t]
        mu[t] = (len(m) * (m.mean() if len(m) else g) + k * g) / (len(m) + k)
    return test_tiers.map(mu).fillna(g).values


def train(target: str, X: pd.DataFrame, y: pd.Series, groups: pd.Series, tiers: pd.Series, naive_inside: float | None = None,
          goals: pd.Series | None = None, rule_pred: np.ndarray | None = None,
          n_folds: int = 5, cv_trees: int = 150, final_trees: int = 300) -> ForestModel:
    """Fit a forest and judge it by cross-validation, folds grouped by campaign.

    Baselines for the reach target: naive (multiplier 1) and tier correction. For the conversion target:
    the average per goal and the assumed rates in config.py (`rule_pred`)."""
    feats = REACH_FEATURES if target == "reach" else CONV_FEATURES
    X = X[feats]
    yv = y.values
    folds = min(n_folds, groups.nunique())
    oof = np.zeros(len(y))
    base_a = np.zeros(len(y))                    # reach: tier correction, conv: goal average
    cover_rf, cover_a = 0.0, 0.0
    imp_runs = []
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        m = _forest(cv_trees).fit(X.iloc[tr], y.iloc[tr])
        lo_q, hi_q = np.quantile(yv[tr] - m.oob_prediction_, [0.10, 0.90])
        pred = m.predict(X.iloc[te])
        oof[te] = pred
        cover_rf += ((yv[te] >= pred + lo_q) & (yv[te] <= pred + hi_q)).sum()
        if target == "reach":
            bt = _tier_baseline(tiers.iloc[tr], y.iloc[tr], tiers.iloc[te])
            sd = float(np.std(yv[tr] - _tier_baseline(tiers.iloc[tr], y.iloc[tr], tiers.iloc[tr])))
        else:
            bt = _group_mean_baseline(goals.iloc[tr], y.iloc[tr], goals.iloc[te])
            sd = float(np.std(yv[tr] - _group_mean_baseline(goals.iloc[tr], y.iloc[tr], goals.iloc[tr])))
        base_a[te] = bt
        cover_a += ((yv[te] >= bt - Z90 * sd) & (yv[te] <= bt + Z90 * sd)).sum()
        imp_runs.append(permutation_importance(m, X.iloc[te], y.iloc[te], n_repeats=3, random_state=SEED,
                                               scoring="neg_mean_squared_error").importances_mean)

    def row(name, pred, cover, kind):
        err = yv - pred
        return {"method": name, "kind": kind, "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))), "coverage": cover}

    if target == "reach":
        rows = [row("Naive forecast (no correction)", np.zeros(len(yv)), naive_inside, "baseline"),
                row("Tier correction (average bias per audience size)", base_a, cover_a / len(yv), "baseline")]
    else:
        rows = [row("Average per campaign goal", base_a, cover_a / len(yv), "baseline"),
                row("Assumed rates in config.py (goal x niche fit)", rule_pred, None, "baseline")]
    rows.append(row("Random forest", oof, cover_rf / len(yv), "forest"))
    metrics = pd.DataFrame(rows)
    base_rows = metrics[metrics.kind == "baseline"]
    best = base_rows.loc[base_rows.rmse.idxmin()]
    forest_rmse = float(metrics[metrics.kind == "forest"].rmse.iloc[0])
    gain = 1 - forest_rmse / float(best.rmse)
    metrics["rmse_gain_vs_best_baseline"] = 1 - metrics.rmse / float(best.rmse)

    final = _forest(final_trees).fit(X, y)
    resid = yv - final.oob_prediction_
    q_lo, q_hi = np.quantile(resid, [0.10, 0.90])
    imp = pd.DataFrame({"feature": feats, "importance": np.mean(imp_runs, axis=0), "spread": np.std(imp_runs, axis=0)}) \
            .sort_values("importance", ascending=False).reset_index(drop=True)
    imp["label"] = imp.feature.map(FEATURE_LABELS)
    return ForestModel(target=target, features=feats, model=final, q_lo=float(q_lo), q_hi=float(q_hi), resid_var=float(np.var(resid)),
                       n=len(y), metrics=metrics, importance=imp,
                       oof=pd.DataFrame({"actual": yv, "predicted": oof, "tier": tiers.values}),
                       best_baseline=str(best.method), gain_vs_best=float(gain), beats_baseline=bool(gain >= MIN_GAIN), X=X, y=y)


# ================================================================ bundle (cached per database state)
_CACHE: dict = {}


def get_bundle(path=None, force: bool = False) -> dict | None:
    """Train (or reuse) both forests for the current database. Returns None when there is too little data."""
    key = (str(path or config.DB_WORKING), db.db_mtime(path))
    if key in _CACHE and not force:
        return _CACHE[key]
    hist = predict.load_history(path)
    if len(hist) < MIN_ROWS:
        return None
    X, Y, peers = build_training(hist, path)
    groups = hist.campaign_id
    rule = np.log(hist.goal.map(config.GOAL_CONV_RATE).values * (0.7 + 0.7 * X.niche_fit.values) * 1000)
    bundle = {
        "hist": hist, "peers": peers, "X": X, "n": len(hist),
        "reach": train("reach", X, Y.reach, groups, hist.tier, naive_inside=float(hist.inside.mean())),
        "conv": train("conv", X, Y.conv, groups, hist.tier, goals=hist.goal, rule_pred=rule),
    }
    _CACHE.clear()
    _CACHE[key] = bundle
    return bundle


def forecast(bundle: dict, universe: pd.DataFrame, category: str | None, goal: str, confidence: pd.Series | None = None,
             path=None) -> pd.DataFrame:
    """Forecast for candidate rows: log multipliers with interval, and the conversion rate per reached person."""
    Xp = build_prediction_frame(universe, bundle["hist"], bundle["peers"], category, goal, path)
    widen = (1 + 0.5 * (1 - (confidence if confidence is not None else pd.Series(1.0, index=universe.index)).clip(0, 1))).values
    mu, lo, hi = bundle["reach"].interval(Xp, widen)
    conv = bundle["conv"].predict(Xp)
    rate = np.exp(conv + 0.5 * bundle["conv"].resid_var) / 1000.0         # mean (not median) conversions per reached person
    return pd.DataFrame({"rf_mu": mu, "rf_lo": lo, "rf_hi": hi, "rf_conv_rate": rate}, index=universe.index), Xp


# ================================================================ explanation helpers
def local_effects(fm: ForestModel, x_row: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Replace one feature at a time by the typical (median) value and see how the prediction moves.
    Positive effect = this creator's value of the feature raises the forecast versus a typical creator."""
    base = float(fm.predict(x_row)[0])
    rows = []
    for f in fm.features:
        alt = x_row.copy()
        alt[f] = float(ref[f].median())
        rows.append({"feature": f, "label": FEATURE_LABELS[f], "value": float(x_row[f].iloc[0]),
                     "typical": float(ref[f].median()), "effect": base - float(fm.predict(alt)[0])})
    out = pd.DataFrame(rows)
    out["abs"] = out.effect.abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def partial_dependence(fm: ForestModel, feature: str, n_grid: int = 25) -> pd.DataFrame:
    """Average prediction when `feature` is forced to each value on a grid while the other features stay as observed."""
    X = fm.X.copy()
    lo, hi = np.percentile(X[feature], [2, 98])
    grid = np.unique(np.round(np.linspace(lo, hi, n_grid), 4)) if X[feature].nunique() > 2 else np.array([0.0, 1.0])
    out = []
    for v in grid:
        X2 = X.copy()
        X2[feature] = v
        out.append({"value": v, "prediction": float(fm.predict(X2).mean())})
    return pd.DataFrame(out)


def small_tree(fm: ForestModel, max_depth: int = 3, min_leaf: int = 12) -> DecisionTreeRegressor:
    """A single shallow tree on the same data, to show how one tree reasons."""
    return DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=SEED).fit(fm.X[fm.features], fm.y)


def tree_nodes(tree: DecisionTreeRegressor, features: list[str], target: str) -> tuple[list[dict], list[tuple]]:
    """Layout of a fitted tree as nodes (x, y, text) and edges, for plotting."""
    t = tree.tree_
    nodes, edges, leaf_counter = [], [], [0]

    def rec(i: int, depth: int) -> float:
        left, right = t.children_left[i], t.children_right[i]
        if left == -1:
            x = leaf_counter[0]
            leaf_counter[0] += 1
            val = float(t.value[i][0][0])
            txt = (f"x{np.exp(val):.2f}" if target == "reach" else f"{np.exp(val):.1f} per 1k") + f"<br>n={int(t.n_node_samples[i])}"
            nodes.append({"x": x, "y": -depth, "text": txt, "leaf": True, "value": val})
            return x
        xl, xr = rec(left, depth + 1), rec(right, depth + 1)
        x = (xl + xr) / 2
        f = features[t.feature[i]]
        thr = float(t.threshold[i])
        nodes.append({"x": x, "y": -depth, "text": f"{FEATURE_LABELS[f]}<br>&lt;= {display_threshold(f, thr)}?", "leaf": False, "value": 0})
        edges.append((x, -depth, xl, -depth - 1, "yes"))
        edges.append((x, -depth, xr, -depth - 1, "no"))
        return x

    rec(0, 0)
    return nodes, edges
