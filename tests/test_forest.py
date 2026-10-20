from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from core import ai, config, db, forest, predict, scoring


@pytest.fixture(scope="module")
def bundle():
    b = forest.get_bundle(force=True)
    assert b is not None
    return b


# ------------------------------------------------------------------ data and leakage
def test_features_are_finite_and_complete(bundle):
    X = bundle["X"]
    assert list(X.columns) == forest.CONV_FEATURES and len(X) == bundle["n"]
    assert np.isfinite(X.values).all()
    assert X.is_instagram.add(X.is_tiktok).le(1).all()


def test_prior_stats_only_use_earlier_measured_campaigns(bundle):
    hist = bundle["hist"]
    multi = hist.groupby("influencer_id").size().loc[lambda s: s >= 3].index[0]
    rows = hist[hist.influencer_id == multi].sort_values("start_date")
    first = rows.iloc[0]
    prior, n = forest._prior_stats(hist, int(multi), first.start_date)
    earlier = rows[rows.measured_at < first.start_date]
    assert n == len(earlier)
    last = rows.iloc[-1]
    prior_l, n_l = forest._prior_stats(hist, int(multi), last.start_date)
    assert n_l == len(rows[rows.measured_at < last.start_date]) <= len(rows) - 1
    # the creator's own current result is never part of its prior
    assert not (rows[rows.measured_at >= last.start_date].index.isin(rows[rows.measured_at < last.start_date].index)).any()
    if n_l:
        expected = rows[rows.measured_at < last.start_date].log_ratio.sum() / (n_l + 2)
        assert prior_l == pytest.approx(expected)


def test_history_uses_naive_baseline_not_the_corrected_forecast(fresh_db):
    acc = db.read("SELECT id AS account_id, influencer_id FROM account LIMIT 2")
    items = [{"influencer_id": r.influencer_id, "account_id": r.account_id, "rank": i, "total_score": 0.5,
              "criterion_scores": {c: 0.5 for c in config.CRITERIA}, "selected": 1, "deliverable": "reel", "fee_eur": 300.0,
              "reach_p10": 800, "reach_p50": 1000, "reach_p90": 1500, "reach_naive": 2000} for i, r in enumerate(acc.itertuples())]
    run = db.save_run(title="t", brand_id=1, objective="score", budget_eur=1000, weights={}, profile={}, brief_text="", notes="", items=items,
                      model_version="scoring-v1+random-forest")
    cid = db.convert_run_to_campaign(run, name="c", goal="sales", start=date(2026, 11, 1), end=date(2026, 11, 20), country="AT")
    pid = int(db.read("SELECT id FROM participation WHERE campaign_id = ? LIMIT 1", (cid,)).id[0])
    db.log_result(participation_id=pid, actual_reach=1000, actual_engagement=10, conversions=5, delivered_on_time=True,
                  client_rating=4, measured_at=date(2026, 12, 1))
    h = predict.load_history()
    row = h[h.participation_id == pid].iloc[0]
    assert row.reach_naive == 2000 and row.ratio == pytest.approx(0.5)          # vs naive 2000, not vs corrected 1000
    assert db.read("SELECT model_version FROM participation WHERE id = ?", (pid,)).model_version[0] == "scoring-v1+random-forest"


# ------------------------------------------------------------------ honest evaluation
def test_forest_beats_baselines_in_grouped_cross_validation(bundle):
    r, c = bundle["reach"], bundle["conv"]
    assert r.beats_baseline and r.gain_vs_best > 0.10
    assert c.beats_baseline and c.gain_vs_best >= forest.MIN_GAIN
    for fm in (r, c):
        rf = fm.metrics[fm.metrics.kind == "forest"].iloc[0]
        assert 0.70 <= rf.coverage <= 0.90                      # 80% ranges are roughly honest
        assert (fm.metrics.kind == "baseline").sum() == 2
    assert set(r.metrics.method) >= {"Random forest", "Naive forecast (no correction)"}
    assert r.importance.iloc[0].feature in ("fake_pct", "max_growth_6m")      # the planted effects dominate


def test_cross_validation_folds_never_split_a_campaign(bundle):
    from sklearn.model_selection import GroupKFold
    g = bundle["hist"].campaign_id
    for tr, te in GroupKFold(n_splits=5).split(bundle["X"], groups=g):
        assert not set(g.iloc[tr]) & set(g.iloc[te])


def test_training_is_deterministic():
    a = forest.get_bundle(force=True)
    b = forest.get_bundle(force=True)
    assert a is not b
    X = a["X"].head(25)
    assert np.allclose(a["reach"].predict(X), b["reach"].predict(X))
    assert np.allclose(a["conv"].predict(X), b["conv"].predict(X))
    assert a["reach"].metrics.rmse.tolist() == b["reach"].metrics.rmse.tolist()


def test_forest_learns_the_planted_fake_follower_pattern(bundle):
    typical = bundle["X"].median().to_frame().T
    risky = typical.copy()
    risky["fake_pct"], risky["max_growth_6m"], risky["er_vs_peers"] = 40.0, 1.0, np.log(0.45)
    m_typ = float(np.exp(bundle["reach"].predict(typical)[0]))
    m_risk = float(np.exp(bundle["reach"].predict(risky)[0]))
    assert m_risk < 0.6 < 0.7 < m_typ < 1.1


def test_forecast_interval_ordering_and_widening(bundle):
    X = bundle["X"].head(30)
    mu, lo, hi = bundle["reach"].interval(X)
    assert (lo < mu).all() and (mu < hi).all()
    _, lo2, hi2 = bundle["reach"].interval(X, widen=1.5)
    assert ((hi2 - lo2) > (hi - lo)).all()


# ------------------------------------------------------------------ integration in the pipeline
@pytest.fixture(scope="module")
def runs(bundle):
    text = open(config.BRIEF_DIR / "brief_nordstride.md", encoding="utf-8").read()
    prof = ai.rule_based_profile(text)
    brand = int(db.read("SELECT id FROM brand WHERE name = 'Nordstride'").id[0])
    out = {}
    for m in ("naive", "tier", "forest"):
        s = scoring.FilterSettings(countries=prof["target_countries"], budget_eur=25000, category=prof["category"], client_brand_id=brand,
                                   start=date(2026, 11, 2), end=date(2026, 11, 27), forecast_model=m)
        out[m] = scoring.run_pipeline(db.as_of(), s, prof, config.PRESETS["Balanced"], text)
    return out, prof


def test_three_methods_give_three_different_forecasts(runs):
    out, _ = runs
    naive, tier, rf = out["naive"]["ranked"], out["tier"]["ranked"], out["forest"]["ranked"]
    assert np.allclose(naive.reach_p50, naive.reach_naive)                       # naive really is naive
    assert (tier.reach_p50 < tier.reach_naive).mean() > 0.8                      # tier correction shrinks (bias is negative)
    assert not np.allclose(rf.reach_p50.sort_index(), tier.reach_p50.sort_index())
    assert out["naive"]["meta"]["forecast"]["reach"] == "naive"
    assert out["tier"]["meta"]["forecast"]["reach"] == "tier correction"
    assert out["forest"]["meta"]["forecast"]["reach"] == "random forest"
    assert out["forest"]["meta"]["forecast"]["conv"] == "random forest"


def test_forest_pipeline_outputs_are_sane(runs):
    rf = runs[0]["forest"]["ranked"]
    assert ((rf.reach_p10 < rf.reach_p50) & (rf.reach_p50 < rf.reach_p90)).all()
    assert (rf.conv_rate > 0).all() and (rf.exp_conversions > 0).all()
    assert np.isclose(rf.exp_conversions, rf.reach_p50 * rf.conv_rate).all()
    assert rf.cpm.notna().all()


def test_forest_pipeline_flags_suspicious_accounts_with_low_forecasts(runs):
    uni = runs[0]["forest"]["universe"]
    spike = uni[uni.data_warnings.map(lambda w: any("spike" in x for x in w))]
    clean = uni[uni.data_warnings.map(len) == 0]
    assert len(spike) >= 3
    assert (spike.reach_p50 / spike.reach_naive).median() < 0.6 * (clean.reach_p50 / clean.reach_naive).median()


def test_pipeline_falls_back_when_too_little_data(monkeypatch):
    monkeypatch.setattr(forest, "get_bundle", lambda path=None, force=False: None)
    prof = ai.rule_based_profile("# T\nBudget: 20000 EUR. Running shoe in Austria.")
    s = scoring.FilterSettings(countries=["AT"], budget_eur=20000, category="sportswear", forecast_model="forest")
    out = scoring.run_pipeline(db.as_of(), s, prof, config.PRESETS["Balanced"])
    f = out["meta"]["forecast"]
    assert f["reach"] == "tier correction" and any("needs at least" in n for n in f["notes"])


def test_pipeline_does_not_use_a_forest_that_loses_to_the_baseline(bundle, monkeypatch):
    weak = dict(bundle, reach=replace(bundle["reach"], beats_baseline=False, gain_vs_best=-0.1),
                conv=replace(bundle["conv"], beats_baseline=False, gain_vs_best=0.0))
    monkeypatch.setattr(forest, "get_bundle", lambda path=None, force=False: weak)
    prof = ai.rule_based_profile("# T\nBudget: 20000 EUR. Running shoe in Austria.")
    s = scoring.FilterSettings(countries=["AT"], budget_eur=20000, category="sportswear", forecast_model="forest")
    tier = scoring.run_pipeline(db.as_of(), replace(s, forecast_model="tier"), prof, config.PRESETS["Balanced"])["ranked"]
    out = scoring.run_pipeline(db.as_of(), s, prof, config.PRESETS["Balanced"])
    f = out["meta"]["forecast"]
    assert f["reach"] == "tier correction" and f["conv"] == "assumed rates" and len(f["notes"]) == 2
    assert np.allclose(out["ranked"].set_index("account_id").reach_p50.sort_index(), tier.set_index("account_id").reach_p50.sort_index())


# ------------------------------------------------------------------ explanation helpers
def test_explanation_helpers(bundle):
    fm = bundle["reach"]
    row = fm.X.iloc[[0]]
    eff = forest.local_effects(fm, row, fm.X)
    assert set(eff.feature) == set(fm.features) and eff.effect.abs().is_monotonic_decreasing
    votes = fm.tree_votes(row)
    assert len(votes) == fm.model.n_estimators and votes.mean() == pytest.approx(float(fm.predict(row)[0]))
    pdp = forest.partial_dependence(fm, "fake_pct")
    assert pdp.prediction.iloc[0] > pdp.prediction.iloc[-1]                     # more fake followers, lower forecast
    tree = forest.small_tree(fm, max_depth=3)
    nodes, edges = forest.tree_nodes(tree, fm.features, "reach")
    leaves = [n for n in nodes if n["leaf"]]
    assert 2 <= len(leaves) <= 8 and len(edges) == 2 * (len(nodes) - len(leaves))
    assert forest.display_value("log_followers", np.log(25000)) == "25,000"
    assert forest.display_value("er_vs_peers", 0.0) == "1.00x peers"
