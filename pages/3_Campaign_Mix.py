import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from core import ai, charts, components as C, config, db, optimizer, report, scoring, simulation, tables, ui

ui.page_config("Campaign mix", ":material/tune:")
ui.chrome()
ss = st.session_state

ui.header("Campaign mix", "Choose the combination of creators that gets the most out of the budget, see the range of outcomes "
          "and stress-test the decision.", step=3)
if ui.need_profile():
    st.stop()
if "ranked" not in ss or ss.ranked is None or len(ss.ranked) == 0:
    C.empty_state("No candidates yet", "Run the shortlist first so there are creators to combine into a mix.", "list")
    ui.page_link("pages/2_Shortlist.py", label="Go to the shortlist", icon=":material/arrow_forward:")
    st.stop()

profile, ranked = ss.profile, ss.ranked
goal = profile.get("goal", "awareness")

# ------------------------------------------------------------------------------- controls (left rail)
ui.defaults(m_budget=float(ss.get("f_budget", 20000.0)), m_risk=0.3, m_nrange=(3, 8), m_maxshare=0.4, m_min_small=0,
            m_min_ig=0, m_min_tt=0, m_min_yt=0, sim_rho=0.2, sim_target=0, m_pool=min(25, len(ranked)))
ss.m_pool = int(min(max(ss.m_pool, 8), max(len(ranked), 8)))       # keep inside the slider range when filters change

rail, main = st.columns([1, 3.6], gap="medium")
with rail:
    with st.container(key="rail_optimizer", border=True):
        C.panel_title("Optimizer")
        budget = st.number_input("Budget (EUR)", min_value=0.0, step=500.0, key="m_budget")
        objective = st.selectbox("Maximise", list(optimizer.OBJECTIVES), format_func=optimizer.OBJECTIVES.get, key="m_objective")
        risk = st.slider("Risk aversion", 0.0, 1.0, step=0.05, key="m_risk",
                         help="0 plans with the median forecast, 1 plans with the pessimistic (p10) forecast.")
        n_rng = st.slider("Number of creators", 2, 15, key="m_nrange")
        with st.expander("Portfolio rules"):
            max_share = st.slider("Max budget share per creator", 0.1, 1.0, step=0.05, key="m_maxshare", format="percent")
            min_small = st.number_input("At least this many nano/micro creators", 0, 8, key="m_min_small")
            mp1, mp2, mp3 = st.columns(3)
            min_ig = mp1.number_input("Min IG", 0, 5, key="m_min_ig")
            min_tt = mp2.number_input("Min TikTok", 0, 5, key="m_min_tt")
            min_yt = mp3.number_input("Min YT", 0, 5, key="m_min_yt")
        if len(ranked) > 8:
            pool_n = st.slider("Candidate pool (top N by score)", 8, min(40, len(ranked)), key="m_pool")
        else:
            pool_n = len(ranked)
            st.caption(f"Candidate pool: all {pool_n} creators that passed the filters.")
        pool_all = ranked.head(pool_n)
        labels = {int(r.account_id): f"{r['name']} ({r.platform})" for _, r in pool_all.iterrows()}
        with st.expander("Overrides"):
            excl_ids = st.multiselect("Leave out", list(labels), format_func=labels.get, key="m_exclude")
            incl_ids = st.multiselect("Must include", list(labels), format_func=labels.get, key="m_include")

pool = pool_all[~pool_all.account_id.isin(excl_ids)]
ms = optimizer.MixSettings(budget=budget, objective=objective, risk_aversion=risk, n_min=n_rng[0], n_max=n_rng[1],
                           max_share=max_share, min_small=int(min_small),
                           min_per_platform={"instagram": int(min_ig), "tiktok": int(min_tt), "youtube": int(min_yt)},
                           must_include=list(incl_ids))
with st.spinner("Optimizing ..."):
    res = optimizer.solve_mix(pool, ms)
mix = res["selected"]
if mix.empty:
    with main:
        C.empty_state("No feasible mix", "Raise the budget, widen the number of creators, lower the minimums or enlarge the pool.", "alert")
        st.error(f"No feasible mix ({res['status']}). Raise the budget, widen the number of creators, "
                 "lower the minimums, or enlarge the pool.", icon=":material/error:")
    st.stop()
mix = mix.reset_index(drop=True)

# ------------------------------------------------------------------------------- everything on the right
with main:
    fmeta_now = ss.get("score_meta", {}).get("forecast", {"reach": "tier correction", "conv": "assumed rates"})
    base = optimizer.baseline_top_by_score(pool, ms)
    sm_mix = optimizer.mix_summary(mix, objective, risk)
    sm_base = optimizer.mix_summary(base, objective, risk)

    C.section("Recommended mix", "")
    C.chips([(f"Solved with {res['method']}", "outline"), (res["status"], "success" if res["status"] == "optimal" else "warning"),
             (f"Forecast: {fmeta_now['reach']}", "info")])
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Creators", sm_mix["n"])
    k2.metric("Spend", f"EUR {sm_mix['spend']:,.0f}", delta=f"{sm_mix['spend'] / budget:.0%} of budget" if budget else None, delta_color="off")
    k3.metric("Expected reach (p50)", f"{sm_mix['reach_p50']:,.0f}")
    k4.metric("Qualified reach", f"{sm_mix['qualified_reach']:,.0f}", help="Reach multiplied by the audience-match and topic-fit scores: reach among the right people on a relevant topic.")
    k5.metric("Expected conversions", f"{sm_mix['conversions']:,.0f}",
              help=f"Goal: {goal.replace('_', ' ')}. Conversion rate source: {fmeta_now['conv']} (assumed rates are set in core/config.py).")
    if sm_base["qualified_reach"]:
        gain = sm_mix["qualified_reach"] / sm_base["qualified_reach"] - 1
        st.info(f"Compared with simply taking the best-scored creators until the budget is used up "
                f"({sm_base['n']} creators, EUR {sm_base['spend']:,.0f}), this mix delivers {gain:+.0%} qualified reach.",
                icon=":material/compare_arrows:")

    st.dataframe(
        mix[["rank", "name", "platform", "deliverable", "price", "reach_p50", "reach_p10", "reach_p90", "exp_conversions", "total"]]
        .assign(platform=lambda d: d.platform.map(lambda p: tables.BRAND_CASE.get(p, p))),
        hide_index=True, width="stretch",
        column_config={"rank": st.column_config.NumberColumn("#", width=40), "name": st.column_config.TextColumn("Creator", width=140),
                       "platform": st.column_config.TextColumn("Platform", width=76), "deliverable": st.column_config.TextColumn("Format", width=88),
                       "price": st.column_config.NumberColumn("Fee", format="euro", width=72),
                       "reach_p50": st.column_config.NumberColumn("Reach p50", format="compact", width=80),
                       "reach_p10": st.column_config.NumberColumn("p10", format="compact", width=60),
                       "reach_p90": st.column_config.NumberColumn("p90", format="compact", width=60),
                       "exp_conversions": st.column_config.NumberColumn("Exp. conv.", format="%.0f", width=76),
                       "total": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="%.2f", width=88)})

    # --------------------------------------------------------------------------- simulation
    C.section("What range of outcomes should we expect?", "5,000 simulated campaigns using each creator's forecast range and a shared market factor.")
    s1, s2 = st.columns([1, 3], gap="medium")
    with s1:
        with ui.card():
            rho = st.slider("Correlation between creators", 0.0, 0.6, step=0.05, key="sim_rho",
                            help="Higher means creators tend to over- or under-deliver together (algorithm changes, season).")
            target = st.number_input("Reach target (optional)", min_value=0, step=10000, key="sim_target")
    sim = simulation.simulate_mix(mix, goal, n_sim=5000, rho=rho)
    naive = float(mix.reach_naive.sum())
    sm = simulation.summarize(sim, naive, float(target) or None)
    sm["budget"] = budget
    with s2:
        m1, m2 = st.columns(2)
        m1.metric("Reach P10 to P90", f"{sm['reach_p10'] / 1000:,.0f}k to {sm['reach_p90'] / 1000:,.0f}k")
        m2.metric("Conversions P10 to P90", f"{sm['conv_p10']:,.0f} to {sm['conv_p90']:,.0f}")
        m3, m4 = st.columns(2)
        m3.metric("Median EUR per 1,000 reach", f"{sm['cpm_p50']:.1f}")
        m4.metric("Chance of missing the naive forecast", f"{sm['p_below_quote']:.0%}",
                  help=f"The naive forecast is the plain sum of average views ({naive:,.0f}). A high value means that number is too optimistic.")
        if sm["p_below_target"] is not None:
            st.metric(f"Chance of missing the target of {target:,.0f}", f"{sm['p_below_target']:.0%}")
    t1, t2, t3 = st.tabs(["Reach", "Conversions", "Sensitivity"])
    t1.plotly_chart(charts.reach_hist(sim["reach"], sm, float(mix.price.sum())), theme=None, width="stretch")
    t2.plotly_chart(charts.conv_hist(sim["conversions"], sm), theme=None, width="stretch")
    with t3:
        metric = st.radio("Measure", ["Median conversions", "Chance of missing the naive forecast"], horizontal=True, key="m_tornado_metric")
        key = "conv_p50" if metric.startswith("Median") else "p_below_quote"
        tor = simulation.tornado(mix, goal, metric=key, naive_quote=naive)
        st.plotly_chart(charts.tornado_chart(tor, metric, ",.0f" if key == "conv_p50" else ".0%"), theme=None, width="stretch")
        st.caption("Each bar changes one assumption while the others stay fixed. Long bars are the assumptions worth checking before committing budget.")

    # --------------------------------------------------------------------------- devil's advocate
    C.section("Devil's advocate", "Automatic checks on the mix, ordered by severity. Add an AI review for biases, missing options and stakeholders.")
    mix_sig = tuple(sorted(int(a) for a in mix.account_id))
    checks = report.risk_checks(mix, profile, budget, sm)
    if st.button("Add AI review (biases, missing options, stakeholders)", icon=":material/psychology_alt:"):
        with st.spinner("Challenging the decision ..."):
            r = ai.challenge_selection(ui.ai_client(), profile, mix, checks)
        ss.challenges, ss.challenges_source, ss.challenges_error, ss.challenges_sig = r.data, r.source, r.error, mix_sig
    shown = checks
    if ss.get("challenges") and ss.get("challenges_sig") == mix_sig:
        ui.ai_badge(ss.challenges_source, ss.get("challenges_error"))
        shown = ss.challenges
    for c in shown:
        text = f"**{c['title']}**: {c['detail']}"
        {"high": st.error, "medium": st.warning, "low": st.info}[c["severity"]](text, icon=":material/flag:" if c["severity"] != "low" else ":material/info:")
    if not shown:
        st.success("No automatic risk flags for this mix.")

    # --------------------------------------------------------------------------- memo and export
    C.section("Decision memo and export", "A one-page summary for the client, with ranges instead of single numbers.")
    title_default = f"{profile.get('product_name', 'Campaign')} - influencer mix"
    if ss.get("_title_for") != ss.profile_version:
        ss.m_title, ss._title_for = title_default, ss.profile_version
    memo_title = st.text_input("Title", key="m_title")
    wn = scoring.normalize_weights(ui.weights())
    fmeta = fmeta_now
    forecast_note = ("predicted by a random forest trained on past campaign results" if fmeta["reach"] == "random forest"
                     else "bias-corrected from past campaign results (tier correction)" if fmeta["reach"] == "tier correction"
                     else "the naive average-views numbers")
    fallback_md = report.memo_markdown(memo_title, profile, mix, {**sm, "budget": budget}, sm, shown, db.as_of(), wn,
                                       ss.get("score_meta", {}).get("embedding_backend", "tfidf"), forecast_note)
    if st.button("Write the memo with AI", icon=":material/edit_note:"):
        with st.spinner("Writing ..."):
            r = ai.write_memo(ui.ai_client(), profile, mix, sm, shown, memo_title, fallback_md)
        ss.memo_text, ss.memo_source, ss.memo_error, ss.memo_sig = r.data, r.source, r.error, mix_sig
    memo = fallback_md
    if ss.get("memo_text") and ss.get("memo_sig") == mix_sig:
        memo = ss.memo_text
        ui.ai_badge(ss.memo_source, ss.get("memo_error"))
    with st.expander("Memo preview", expanded=True):
        st.markdown(memo)
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Download PDF", report.memo_pdf(memo, memo_title), file_name="decision_memo.pdf", mime="application/pdf", icon=":material/picture_as_pdf:", width="stretch")
    d2.download_button("Download HTML", report.memo_html(memo, memo_title), file_name="decision_memo.html", mime="text/html", icon=":material/html:", width="stretch")
    d3.download_button("Download Markdown", memo, file_name="decision_memo.md", mime="text/markdown", icon=":material/markdown:", width="stretch")
    d4.download_button("Download mix (CSV)", mix[["name", "platform", "handle", "deliverable", "price", "reach_p10", "reach_p50", "reach_p90", "exp_conversions", "total"]].to_csv(index=False),
                       file_name="mix.csv", mime="text/csv", icon=":material/table:", width="stretch")

    # --------------------------------------------------------------------------- save
    C.section("Save this decision", "Stores brief, weights, scores, forecasts and the chosen mix as an audit trail. You can later convert it into a campaign and log real results.")
    with ui.card():
        with st.form("save_run", border=False):
            notes = st.text_area("Notes (why this mix, who approved, open questions)")
            submitted = st.form_submit_button("Save decision", icon=":material/save:", type="primary")
    if submitted:
        selected = set(mix.account_id.astype(int))
        items = []
        for _, r in pool_all.iterrows():
            exp = ss.explanations.get(int(r.account_id)) if ss.get("explanations") else None
            items.append({"influencer_id": r.influencer_id, "account_id": r.account_id, "rank": r["rank"], "total_score": r.total,
                          "criterion_scores": {c: round(float(r[c]), 4) for c in config.CRITERIA},
                          "selected": int(int(r.account_id) in selected), "deliverable": r.deliverable, "fee_eur": r.price,
                          "reach_p10": r.reach_p10, "reach_p50": r.reach_p50, "reach_p90": r.reach_p90, "reach_naive": r.reach_naive,
                          "ai_explanation": json.dumps(exp) if exp else None})
        run_id = db.save_run(title=memo_title, brand_id=ss.client_brand_id, objective=objective, budget_eur=budget, weights=wn,
                             profile=profile, brief_text=ss.brief_text, notes=notes, items=items,
                             model_version=f"{config.MODEL_VERSION_TAG}+{fmeta['reach'].replace(' ', '-')}")
        st.success(f"Saved as decision #{run_id}. Open the Decision log to convert it into a campaign.", icon=":material/check_circle:")
        ui.page_link("pages/5_Decision_Log.py", label="Open the decision log", icon=":material/arrow_forward:")
