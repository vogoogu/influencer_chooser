import glob
from datetime import date

import numpy as np
import pandas as pd
import pytest

from core import ai, config, db, fx, optimizer, predict, report, scoring, simulation, taxonomy

BRIEFS = {Path_.split("brief_")[1][:-3]: open(Path_, encoding="utf-8").read()
          for Path_ in sorted(glob.glob(str(config.BRIEF_DIR / "*.md")))}


# --------------------------------------------------------------------------- taxonomy
def test_taxonomy_in_sync_with_generator():
    import generate_dataset as g
    assert taxonomy.CATEGORY_NICHES == g.CATEGORY_NICHES
    assert set(taxonomy.NICHES) == set(g.NICHE_SPEC)
    assert taxonomy.PARENT_OF == g.PARENT_OF


# --------------------------------------------------------------------------- rule-based profile
@pytest.mark.parametrize("name,category,goal,budget,country", [
    ("nordstride", "sportswear", "awareness", 25000, "DE"),
    ("gruenwerk_foods", "plant_based_food", "sales", 15000, "AT"),
    ("kontoro", "fintech", "app_installs", 20000, "AT"),
])
def test_rule_based_profile_on_sample_briefs(name, category, goal, budget, country):
    p = ai.rule_based_profile(BRIEFS[name])
    assert p["category"] == category
    assert p["goal"] == goal
    assert p["budget_eur"] == budget
    assert country in p["target_countries"]
    assert p["target_audience"]["age_min"] and p["target_audience"]["age_max"]
    assert set(p["relevant_niches"]) <= set(taxonomy.NICHES)


def test_rule_based_profile_parses_number_formats():
    assert ai.rule_based_profile("Budget: 12.500 EUR")["budget_eur"] == 12500
    assert ai.rule_based_profile("Budget 30k")["budget_eur"] == 30000
    assert ai.rule_based_profile("Budget: 7,500 EUR")["budget_eur"] == 7500


def test_clean_profile_repairs_bad_model_output():
    fb = ai.rule_based_profile(BRIEFS["nordstride"])
    bad = {"category": "spaceships", "goal": "world domination", "relevant_niches": ["nonsense"], "target_countries": ["XX"],
           "tone": ["angry"], "restrictions": {"excluded_flag_types": ["made_up"]}, "target_audience": None}
    p = ai._clean_profile(bad, fb)
    assert p["category"] in taxonomy.CATEGORIES and p["goal"] in config.GOALS
    assert p["relevant_niches"] == fb["relevant_niches"] and p["target_countries"] == fb["target_countries"]
    assert p["restrictions"]["excluded_flag_types"] == [] and p["target_audience"]["description"] == ""


# --------------------------------------------------------------------------- scoring
@pytest.fixture(scope="module")
def nordstride():
    text = BRIEFS["nordstride"]
    prof = ai.rule_based_profile(text)
    brand = int(db.read("SELECT id FROM brand WHERE name = 'Nordstride'").id[0])
    camp = db.read("SELECT * FROM campaign WHERE status = 'planned' AND brand_id = ?", (brand,)).iloc[0]
    s = scoring.FilterSettings(countries=prof["target_countries"], budget_eur=prof["budget_eur"], category=prof["category"],
                               client_brand_id=brand, start=date.fromisoformat(camp.start_date), end=date.fromisoformat(camp.end_date),
                               excluded_flags=prof["restrictions"]["excluded_flag_types"])
    out = scoring.run_pipeline(db.as_of(), s, prof, config.PRESETS["Balanced"], text)
    return {"profile": prof, "settings": s, "out": out, "campaign": camp}


def test_scores_are_in_unit_interval_and_sorted(nordstride):
    r = nordstride["out"]["ranked"]
    assert len(r) > 10
    for c in config.CRITERIA:
        assert r[c].between(0, 1).all(), c
    assert r.total.is_monotonic_decreasing
    assert (r["rank"] == np.arange(1, len(r) + 1)).all()
    w = scoring.normalize_weights(config.PRESETS["Balanced"])
    assert np.isclose(sum(w.values()), 1)
    assert np.allclose(r.total, sum(r[c] * w[c] for c in config.CRITERIA))


def test_every_account_is_either_passed_or_excluded_with_reason(nordstride):
    o = nordstride["out"]
    assert len(o["ranked"]) + len(o["excluded"]) == len(o["universe"])
    assert (o["excluded"].exclude_reason != "").all()
    assert not set(o["ranked"].account_id) & set(o["excluded"].account_id)


def test_hard_filters_hold(nordstride):
    r = nordstride["out"]["ranked"]
    assert (r.status == "active").all() and r.consent_ok.all() and (r.availability != "unavailable").all()
    assert r.platform.isin(nordstride["settings"].platforms).all()
    assert (r.price <= nordstride["settings"].budget_eur * nordstride["settings"].max_fee_share + 1e-6).all()
    assert (r.max_severity <= nordstride["settings"].max_severity).all()
    assert r.conflicts.map(len).eq(0).all()
    assert not any("gambling" in t for flags in r.safety_flags for t, _ in flags)


def test_planted_conflicts_are_detected(nordstride):
    truth = pd.read_csv(config.ROOT / "data" / "ground_truth" / "planted_traps.csv")
    cid = int(nordstride["campaign"].id)
    t = truth[(truth.related_id == cid) & truth.trap_type.isin(["competitor_conflict", "exclusivity_block"])]
    assert len(t) == 3
    uni = nordstride["out"]["universe"]
    for inf_id in t.entity_id:
        assert uni[uni.influencer_id == inf_id].conflicts.map(len).max() >= 1, inf_id


def test_data_warnings_detect_planted_traps(nordstride):
    truth = pd.read_csv(config.ROOT / "data" / "ground_truth" / "planted_traps.csv")
    uni = nordstride["out"]["universe"].set_index("account_id")
    for trap, needle in [("fake_follower_spike", "spike"), ("stale_stats", "stale")]:
        for acc in truth[truth.trap_type == trap].entity_id:
            assert any(needle in w for w in uni.loc[acc].data_warnings), (trap, acc)
    exp = set(truth[truth.trap_type == "expired_rate_card"].entity_id)
    flagged = set(uni[uni.rate_expired].influencer_id)
    assert exp <= flagged
    consent = set(truth[truth.trap_type == "consent_expired"].entity_id)
    assert not set(nordstride["out"]["ranked"].influencer_id) & consent


def test_topic_fit_prefers_matching_niche(nordstride):
    r = nordstride["out"]["ranked"]
    match = r[r.niche_match >= 1.0].topic_fit.mean()
    other = r[r.niche_match == 0.0].topic_fit.mean()
    assert match > other + 0.2


def test_stability_and_sensitivity(nordstride):
    r = nordstride["out"]["ranked"]
    st_ = scoring.rank_stability(r, config.PRESETS["Balanced"], n_sim=100)
    assert 0 <= st_["stability"] <= 1 and len(st_["top_freq"]) == len(r)
    assert st_["top_freq"].sum() == pytest.approx(5, abs=1e-6)       # exactly 5 creators in each top 5
    sens = scoring.weight_sensitivity(r, config.PRESETS["Balanced"])
    assert set(sens.criterion) == set(config.CRITERIA) and (sens[["double", "zero"]] >= 0).all().all()
    # extreme weights: only brand safety matters -> a fully safe creator leads
    top = scoring.apply_weights(r, {**{c: 0 for c in config.CRITERIA}, "brand_safety": 1}).iloc[0]
    assert top.brand_safety == r.brand_safety.max()


def test_empty_country_filter_does_not_crash(nordstride):
    s = scoring.FilterSettings(countries=[], budget_eur=25000, category="sportswear")
    out = scoring.run_pipeline(db.as_of(), s, nordstride["profile"], config.PRESETS["Balanced"])
    assert len(out["ranked"]) > 0


# --------------------------------------------------------------------------- optimizer
def test_optimizer_respects_constraints_and_beats_baseline(nordstride):
    pool = nordstride["out"]["ranked"].head(25)
    ms = optimizer.MixSettings(budget=25000, n_min=3, n_max=7, max_share=0.4, min_small=2)
    res = optimizer.solve_mix(pool, ms)
    mix = res["selected"]
    assert res["status"] == "optimal"
    assert 3 <= len(mix) <= 7 and mix.price.sum() <= 25000 + 1e-6
    assert mix.influencer_id.is_unique and mix.tier.isin(["nano", "micro"]).sum() >= 2
    assert (mix.price <= 0.4 * 25000 + 1e-6).all()
    base = optimizer.baseline_top_by_score(pool, ms)
    v_mix = optimizer.candidate_values(mix, ms.objective, ms.risk_aversion).sum()
    v_base = optimizer.candidate_values(base, ms.objective, ms.risk_aversion).sum()
    assert v_mix >= v_base - 1e-6


def test_optimizer_must_include_and_infeasible(nordstride):
    pool = nordstride["out"]["ranked"].head(25)
    forced = int(pool.account_id.iloc[10])
    res = optimizer.solve_mix(pool, optimizer.MixSettings(budget=25000, n_max=6, must_include=[forced]))
    assert forced in set(res["selected"].account_id)
    bad = optimizer.solve_mix(pool, optimizer.MixSettings(budget=10, n_min=3))
    assert bad["selected"].empty or not bad["status"].startswith("optimal")


def test_optimizer_objective_changes_choice(nordstride):
    pool = nordstride["out"]["ranked"].head(25)
    a = optimizer.solve_mix(pool, optimizer.MixSettings(budget=15000, objective="qualified_reach", n_max=5))["selected"]
    b = optimizer.solve_mix(pool, optimizer.MixSettings(budget=15000, objective="score", n_max=5))["selected"]
    assert set(a.account_id) != set(b.account_id) or len(a) == len(b)


# --------------------------------------------------------------------------- simulation
def test_simulation_properties(nordstride):
    pool = nordstride["out"]["ranked"].head(25)
    mix = optimizer.solve_mix(pool, optimizer.MixSettings(budget=25000, n_max=6))["selected"]
    sim = simulation.simulate_mix(mix, "awareness", n_sim=4000, seed=1)
    sm = simulation.summarize(sim, float(mix.reach_naive.sum()))
    assert sm["reach_p10"] < sm["reach_p50"] < sm["reach_p90"]
    assert 0.5 * mix.reach_p50.sum() < sm["reach_p50"] < 1.2 * mix.reach_p50.sum()
    assert 0 <= sm["p_below_quote"] <= 1
    up = simulation.simulate_mix(mix, "awareness", n_sim=4000, seed=1, reach_mult=1.2)
    assert np.median(up["reach"]) == pytest.approx(1.2 * np.median(sim["reach"]), rel=1e-6)
    wide = simulation.simulate_mix(mix, "awareness", n_sim=4000, seed=1, sigma_mult=2.0)
    assert np.ptp(np.percentile(wide["reach"], [10, 90])) > np.ptp(np.percentile(sim["reach"], [10, 90]))
    tor = simulation.tornado(mix, "awareness", n_sim=800)
    assert len(tor) == 4 and (tor.swing >= 0).all()


# --------------------------------------------------------------------------- prediction and calibration
def test_calibration_finds_optimistic_history_and_corrects_it():
    h = predict.load_history()
    assert len(h) > 200
    assert h.ratio.mean() < 1.0                      # naive predictor was optimistic
    corr = predict.fit_correction(h)
    assert corr["_global"]["mu"] < 0
    assert corr["mega"]["mu"] < corr["nano"]["mu"]   # bigger accounts disappoint more
    p10, p50, p90, _ = predict.predict_reach(100_000, "macro", corr)
    assert p10 < p50 < p90 and p50 < 100_000
    _, p50_off, _, _ = predict.predict_reach(100_000, "macro", corr, apply_correction=False)
    assert p50_off == pytest.approx(100_000)
    _, _, wide_hi, s_wide = predict.predict_reach(100_000, "macro", corr, confidence=0.2)
    _, _, narrow_hi, s_narrow = predict.predict_reach(100_000, "macro", corr, confidence=1.0)
    assert s_wide > s_narrow and wide_hi > narrow_hi  # low confidence widens the interval


def test_corrected_forecast_improves_coverage():
    h = predict.load_history()
    corr = predict.fit_correction(h)
    inside_naive = h.inside.mean()
    ok = 0
    for r in h.itertuples():
        p10, p50, p90, _ = predict.predict_reach(r.reach_p50, r.tier, corr)
        ok += p10 <= r.actual_reach <= p90
    assert abs(ok / len(h) - 0.8) < abs(inside_naive - 0.8) + 0.05


def test_driver_analysis_recovers_planted_effects():
    h = predict.load_history()
    d = predict.driver_analysis(h, "reach_vs_prediction", n_boot=100).set_index("feature")
    assert d.loc["fake_followers_pct", "coef"] < 0          # fake followers hurt delivery
    c = predict.driver_analysis(h, "conversion_rate", n_boot=100).set_index("feature")
    assert c.loc["niche_fit", "coef"] > 0                    # niche fit lifts conversion
    assert c.loc["niche_fit", "lo"] > 0                      # and the interval excludes zero


# --------------------------------------------------------------------------- report
def test_report_exports(nordstride):
    pool = nordstride["out"]["ranked"].head(25)
    mix = optimizer.solve_mix(pool, optimizer.MixSettings(budget=25000, n_max=6))["selected"].reset_index(drop=True)
    sim = simulation.simulate_mix(mix, "awareness", n_sim=1000)
    sm = simulation.summarize(sim, float(mix.reach_naive.sum()))
    sm["budget"] = 25000
    checks = report.risk_checks(mix, nordstride["profile"], 25000, sm)
    assert all(c["severity"] in ("high", "medium", "low") for c in checks)
    sev = [report.SEV_ORDER[c["severity"]] for c in checks]
    assert sev == sorted(sev)
    md = report.memo_markdown("Test €-memo", nordstride["profile"], mix, sm, sm, checks, db.as_of(),
                              scoring.normalize_weights(config.PRESETS["Balanced"]), "tfidf")
    assert "## Recommendation" in md and mix.iloc[0]["name"] in md
    pdf = report.memo_pdf(md, "Test")
    assert pdf[:4] == b"%PDF" and len(pdf) > 1500
    assert "<table>" in report.memo_html(md, "Test")


def test_risk_checks_flag_data_problems():
    prof = {"goal": "sales"}
    mix = pd.DataFrame([{"name": "A", "platform": "tiktok", "price": 1000.0, "reach_p50": 1000.0, "primary_niche": "running",
                         "data_warnings": ["follower spike (+90% in one month)"], "safety_flags": [("controversy", 3)],
                         "conflicts": ["Rival on 2026-09-01"], "has_category_history": False}])
    titles = " ".join(c["title"] for c in report.risk_checks(mix, prof, 1000.0))
    assert "Very few creators" in titles and "Data quality" in titles and "Brand conflict" in titles and "Safety flag" in titles


# --------------------------------------------------------------------------- misc
def test_fx_offline_fallback(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("offline")))
    eur, src = fx.to_eur(1000, "CHF")
    assert src == "static fallback" and eur == pytest.approx(1000 * config.STATIC_FX_TO_EUR["CHF"])
    assert fx.to_eur(500, "EUR") == (500.0, "none needed")


def test_documents_extract(tmp_path):
    from core import documents
    assert documents.extract_text("a.md", "# Titel\nTäst".encode()).startswith("# Titel")
    with pytest.raises(ValueError):
        documents.extract_text("a.exe", b"x")
    with pytest.raises(ValueError):
        documents.extract_text("empty.txt", b"   ")
    import docx
    d = docx.Document()
    d.add_paragraph("Budget 5000 EUR for a vegan snack")
    p = tmp_path / "b.docx"
    d.save(p)
    assert "vegan snack" in documents.extract_text("b.docx", p.read_bytes())
