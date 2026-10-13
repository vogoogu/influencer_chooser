import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from core import ai, charts, config, db, scoring, ui

st.set_page_config(page_title="Shortlist", page_icon=":material/list_alt:", layout="wide")
ui.sidebar()
ss = st.session_state

st.title("Shortlist")
st.caption("Step 2. Hard filters remove creators that cannot be used. The rest are scored on seven criteria you can re-weight live.")
if ui.need_profile():
    st.stop()
profile = ss.profile
ui.sync_filters(profile)


@st.cache_data(show_spinner="Scoring candidates ...", max_entries=8)
def cached_scoring(settings_key: str, profile_key: str, brief_text: str, mtime: float):
    return scoring.run_scoring(db.as_of(), ui.settings_from_key(settings_key), json.loads(profile_key), brief_text)


# ----------------------------------------------------------------------------- filters
with st.expander("Hard filters", expanded=False):
    f1, f2, f3 = st.columns(3)
    with f1:
        st.multiselect("Target countries", config.COUNTRIES, key="f_countries",
                       help="A creator passes if based there, or if 35%+ of the audience is in these countries.")
        st.multiselect("Platforms", config.PLATFORMS, key="f_platforms")
        st.selectbox("Instagram format to buy", ["reel", "feed_post", "story_set"], key="f_ig_format")
        st.number_input("Budget (EUR)", min_value=0.0, step=500.0, key="f_budget")
        st.slider("Max fee per creator, share of budget", 0.05, 1.0, step=0.05, key="f_max_share", format="percent")
    with f2:
        st.checkbox("Allow creators with limited availability", key="f_allow_limited")
        st.checkbox("Exclude brand conflicts (competitor promo in last 90 days, exclusivity)", key="f_excl_conflicts")
        st.checkbox("Exclude accounts with data-quality warnings", key="f_excl_warn",
                    help="Stale stats, follower spikes, fake followers, expired rate card, low confidence.")
        st.multiselect("Exclude safety flags", config.FLAG_TYPES, key="f_flags", format_func=lambda x: x.replace("_", " "))
        st.selectbox("Exclude creators with a flag above severity", [1, 2, 3], key="f_max_sev",
                     help="3 = only the most severe flags exclude a creator. 1 = any flag above 'minor' excludes.")
    with f3:
        st.date_input("Campaign start", key="f_start")
        st.date_input("Campaign end", key="f_end")
        st.number_input("Min followers", min_value=0, step=1000, key="f_min_followers")
        st.number_input("Max followers", min_value=0, step=10000, key="f_max_followers")
        st.checkbox("Correct reach forecasts with the bias learned from past campaigns", key="f_apply_corr")

settings = ui.filter_settings()
pack = cached_scoring(ui.settings_key(settings), json.dumps(profile, sort_keys=True, default=str), ss.brief_text,
                      db.db_mtime())
scored, excluded, universe, meta = pack["scored"], pack["excluded"], pack["universe"], pack["meta"]

# ----------------------------------------------------------------------------- layout
left, right = st.columns([1, 3], gap="large")
with left:
    st.subheader("Weights")
    st.selectbox("Preset", list(config.PRESETS), key="preset", on_change=ui.set_preset)
    for c in config.CRITERIA:
        st.slider(config.CRITERIA_LABELS[c], 0, 10, key=f"w_{c}", help=config.CRITERIA_HELP[c])
    wn = scoring.normalize_weights(ui.weights())
    st.caption("Effective weights: " + ", ".join(f"{config.CRITERIA_LABELS[c].lower()} {wn[c]:.0%}" for c in config.CRITERIA))

with right:
    if scored.empty:
        st.warning("No creators pass the hard filters. Relax the filters above (see reasons below).")
        if len(excluded):
            st.dataframe(excluded.exclude_reason.value_counts().rename("creators").to_frame(), width="stretch")
        st.stop()

    ranked = scoring.rank(scored, ui.weights(), profile.get("goal", "awareness"))
    stab = scoring.rank_stability(ranked, ui.weights())
    ss.ranked, ss.score_meta, ss.excluded = ranked, meta, excluded

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accounts in database", len(universe))
    k2.metric("Passed hard filters", len(ranked), delta=f"-{len(excluded)} filtered out", delta_color="off")
    k3.metric("Top-5 stability", f"{stab['stability']:.0%}",
              help="How much the top 5 stays the same when the weights are varied randomly around your setting. "
                   "Below about 60% means the ranking depends strongly on your weights.")
    k4.metric("Topic similarity", meta["embedding_backend"].split(" ")[0])
    if stab["stability"] < 0.6:
        st.warning("The top 5 changes a lot when weights move slightly. Treat close ranks as a tie and check the Stability tab.", icon=":material/balance:")
    corr_g = meta["correction"]["_global"]
    if settings.apply_correction and corr_g["n"]:
        st.caption(f"Reach forecasts are corrected using {corr_g['n']} past campaign results "
                   f"(average bias {(np.exp(corr_g['mu']) - 1) * 100:+.0f}%).")

    t_rank, t_break, t_cmp, t_stab, t_out = st.tabs(["Ranking", "Score breakdown", "Compare", "Stability", "Filtered out"])

    # ---------------------------------------------------------------- ranking
    with t_rank:
        def notes(r) -> str:
            parts = list(r.conflicts) + list(r.data_warnings) + [f"flag: {t.replace('_', ' ')} ({s})" for t, s in r.safety_flags]
            return "; ".join(parts)
        show = ranked.assign(notes=ranked.apply(notes, axis=1))
        table = show[["rank", "name", "platform", "handle", "tier", "country", "followers", "er", "price", "deliverable",
                      "reach_p10", "reach_p50", "reach_p90", "total", "notes"]]
        ev = st.dataframe(
            table, hide_index=True, width="stretch", height=420, on_select="rerun", selection_mode="multi-row", key="rank_table",
            column_config={
                "rank": st.column_config.NumberColumn("#", width="small"),
                "name": "Creator", "platform": "Platform", "handle": "Handle", "tier": "Tier", "country": "Country",
                "followers": st.column_config.NumberColumn("Followers", format="%d"),
                "er": st.column_config.NumberColumn("Engagement %", format="%.2f"),
                "price": st.column_config.NumberColumn("Fee (EUR)", format="%d"),
                "deliverable": "Format",
                "reach_p10": st.column_config.NumberColumn("Reach p10", format="%d"),
                "reach_p50": st.column_config.NumberColumn("Reach p50", format="%d"),
                "reach_p90": st.column_config.NumberColumn("Reach p90", format="%d"),
                "total": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="%.2f"),
                "notes": st.column_config.TextColumn("Warnings and flags", width="large"),
            })
        sel_rows = list(ev.selection.rows) if ev and ev.selection else []
        st.caption("Select rows to compare them in the Compare tab. Forecast columns are ranges (p10 pessimistic, p90 optimistic).")

        st.markdown("##### Explanations")
        e1, e2 = st.columns([1, 3])
        ui.defaults(m_explain_n=8)
        top_n = e1.slider("Explain top N", 3, 12, key="m_explain_n")
        if e2.button("Explain the top creators", icon=":material/smart_toy:"):
            with st.spinner("Writing explanations ..."):
                res = ai.explain_candidates(ui.ai_client(), ranked, profile, top_n)
            ss.explanations, ss.explanations_source, ss.explanations_error = res.data, res.source, res.error
        if ss.explanations:
            ui.ai_badge(ss.explanations_source, ss.get("explanations_error"))
            by_acc = ranked.set_index("account_id")
            for acc_id, e in ss.explanations.items():
                if acc_id not in by_acc.index:
                    continue
                r = by_acc.loc[acc_id]
                with st.expander(f"#{int(r['rank'])}  {r['name']} on {r['platform']}: {e['headline']}"):
                    c_a, c_b = st.columns(2)
                    c_a.markdown("**Strengths**\n" + "\n".join(f"- {x}" for x in e["strengths"]))
                    c_b.markdown("**Risks**\n" + ("\n".join(f"- {x}" for x in e["risks"]) or "- none identified"))
                    st.caption(f"{int(r.followers):,} followers, {r.er:.2f}% engagement, fee EUR {r.price:,.0f} for a {r.deliverable}, "
                               f"forecast reach {int(r.reach_p10):,} to {int(r.reach_p90):,}.")

    # ---------------------------------------------------------------- breakdown
    with t_break:
        st.plotly_chart(charts.contributions_bar(ranked), width="stretch")
        cc = {c: st.column_config.ProgressColumn(config.CRITERIA_LABELS[c], min_value=0.0, max_value=1.0, format="%.2f")
              for c in config.CRITERIA}
        st.dataframe(ranked[["rank", "name", "platform"] + config.CRITERIA].head(30), hide_index=True, width="stretch",
                     column_config={"rank": "#", "name": "Creator", "platform": "Platform", **cc})
        with st.expander("What do the criteria mean?"):
            for c in config.CRITERIA:
                st.markdown(f"**{config.CRITERIA_LABELS[c]}**: {config.CRITERIA_HELP[c]}")

    # ---------------------------------------------------------------- compare
    with t_cmp:
        rows = ranked.iloc[sel_rows] if sel_rows else ranked.head(3)
        rows = rows.head(6)
        st.caption("Showing the selected creators." if sel_rows else "Showing the top 3. Select rows in the Ranking tab to choose others (max 6).")
        st.plotly_chart(charts.radar(rows), width="stretch")
        st.dataframe(rows[["name", "platform", "followers", "er", "fake", "price", "cpm", "female_pct", "age_25_34", "relationship_score"]],
                     hide_index=True, width="stretch",
                     column_config={"name": "Creator", "er": "Engagement %", "fake": "Fake followers %", "price": "Fee (EUR)",
                                    "cpm": st.column_config.NumberColumn("EUR per 1,000 reach", format="%.0f"),
                                    "female_pct": "Women %", "age_25_34": "Age 25-34 %", "relationship_score": "Relationship"})

    # ---------------------------------------------------------------- stability
    with t_stab:
        st.markdown(f"**Top-5 stability: {stab['stability']:.0%}.** "
                    "Weights are an opinion, so the tool tests whether the shortlist survives different opinions. "
                    "It draws 400 random weight sets around yours and counts how often each creator stays in the top 5.")
        st.plotly_chart(charts.stability_bar(ranked, stab["top_freq"]), width="stretch")
        st.markdown("**Which weight matters most?** How many of the current top 5 change if one weight is doubled or set to zero:")
        st.plotly_chart(charts.sensitivity_bar(scoring.weight_sensitivity(ranked, ui.weights())), width="stretch")

    # ---------------------------------------------------------------- filtered out
    with t_out:
        counts = excluded.exclude_reason.value_counts()
        st.markdown(f"**{len(excluded)} accounts were filtered out.** Each is listed with the first rule it failed.")
        st.bar_chart(counts, horizontal=True)
        st.dataframe(excluded[["name", "platform", "country", "followers", "price", "exclude_reason"]].sort_values("exclude_reason"),
                     hide_index=True, width="stretch",
                     column_config={"name": "Creator", "price": "Fee (EUR)", "exclude_reason": "Reason"})

ui.page_link("pages/3_Campaign_Mix.py", label="Continue to the campaign mix", icon=":material/arrow_forward:")
