import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from core import components as C, config, db, ui

ui.page_config("Decision log", ":material/history:")
ui.sidebar()

ui.header("Decision log", "Every saved decision is an audit trail. Convert it into a campaign, then log what really happened. "
          "Those results feed the calibration page and make future forecasts better.", step=4)

t_runs, t_results = st.tabs(["Saved decisions", "Log campaign results"])

# ------------------------------------------------------------------------------------------ decisions
with t_runs:
    runs = db.read("""SELECT r.id, r.created_at, r.title, b.name AS brand, r.objective, r.budget_eur, r.campaign_id,
                             (SELECT COUNT(*) FROM run_item i WHERE i.run_id = r.id AND i.selected = 1) AS creators
                      FROM run r LEFT JOIN brand b ON b.id = r.brand_id ORDER BY r.id DESC""")
    if runs.empty:
        C.empty_state("No decisions saved yet", "Build a mix on the Campaign mix page and press Save decision. Saved decisions keep the brief, the weights and the forecasts.", "history")
        ui.page_link("pages/3_Campaign_Mix.py", label="Go to the campaign mix", icon=":material/arrow_forward:")
    else:
        st.dataframe(runs, hide_index=True, width="stretch",
                     column_config={"id": "#", "created_at": "Saved", "title": "Title", "brand": "Client brand",
                                    "budget_eur": st.column_config.NumberColumn("Budget (EUR)", format="%d"),
                                    "campaign_id": "Campaign id", "creators": "Creators"})
        run_id = st.selectbox("Open a decision", runs.id.tolist(), format_func=lambda i: f"#{i}  {runs.set_index('id').title[i]}")
        run = db.read("SELECT * FROM run WHERE id = ?", (int(run_id),)).iloc[0]
        items = db.read("""SELECT i.rank, inf.display_name AS name, a.platform, i.deliverable, i.fee_eur, i.reach_p10, i.reach_p50, i.reach_p90,
                                  i.total_score, i.selected, i.criterion_scores, i.ai_explanation
                           FROM run_item i JOIN influencer inf ON inf.id = i.influencer_id LEFT JOIN account a ON a.id = i.account_id
                           WHERE i.run_id = ? ORDER BY i.selected DESC, i.rank""", (int(run_id),))
        c1, c2 = st.columns([2, 1])
        with c1:
            C.panel_title("Chosen mix")
            st.dataframe(items[items.selected == 1].drop(columns=["selected", "criterion_scores", "ai_explanation"]), hide_index=True, width="stretch")
            if run.notes:
                st.info(f"Notes: {run.notes}")
        with c2:
            C.panel_title("Weights used")
            w = json.loads(run.weights_json)
            st.dataframe(pd.DataFrame({"criterion": [config.CRITERIA_LABELS[k] for k in w], "weight": [round(v, 3) for v in w.values()]}),
                         hide_index=True, width="stretch")
            st.caption(f"Objective: {run.objective}. Model: {run.model_version}.")
        with st.expander("Brief and profile stored with this decision"):
            st.code(run.brief_text or "(none)", language="markdown")
            st.json(json.loads(run.profile_json) if run.profile_json else {})
        with st.expander(f"All {len(items)} scored candidates in this decision"):
            st.dataframe(items.drop(columns=["criterion_scores", "ai_explanation"]), hide_index=True, width="stretch")

        C.section("Convert into a campaign", "Creates a planned campaign with the chosen creators and their forecasts, ready for results.")
        if run.campaign_id:
            st.success(f"Already converted into campaign #{int(run.campaign_id)}.", icon=":material/check_circle:")
        elif run.brand_id is None or pd.isna(run.brand_id):
            st.warning("This decision has no client brand, so it cannot become a campaign. Pick a client brand on the Product brief page before saving.")
        else:
            prof = json.loads(run.profile_json) if run.profile_json else {}
            tl = prof.get("timeline") or {}
            with st.form("convert"):
                name = st.text_input("Campaign name", value=run.title)
                goal = st.selectbox("Goal", config.GOALS, index=config.GOALS.index(prof.get("goal", "awareness")) if prof.get("goal") in config.GOALS else 0)
                d1, d2 = st.columns(2)
                try:
                    s0 = date.fromisoformat(tl["start"]) if tl.get("start") else db.as_of() + timedelta(days=14)
                    e0 = date.fromisoformat(tl["end"]) if tl.get("end") else s0 + timedelta(days=28)
                except ValueError:
                    s0 = db.as_of() + timedelta(days=14)
                    e0 = s0 + timedelta(days=28)
                start = d1.date_input("Start", value=s0)
                end = d2.date_input("End", value=max(e0, s0))
                country = st.selectbox("Target country", config.COUNTRIES)
                if st.form_submit_button("Create campaign with these creators", icon=":material/rocket_launch:"):
                    try:
                        cid = db.convert_run_to_campaign(int(run_id), name=name, goal=goal, start=start, end=end, country=country)
                        st.cache_data.clear()
                        st.success(f"Created campaign #{cid}. Log results for it in the next tab once it has run.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

# ------------------------------------------------------------------------------------------ results
with t_results:
    pending = db.read("""SELECT p.id AS participation_id, c.id AS campaign_id, c.name AS campaign, c.status, inf.display_name AS creator,
                                a.platform, p.deliverable, p.fee_eur, p.reach_p10, p.reach_p50, p.reach_p90, c.goal, c.end_date
                         FROM participation p JOIN campaign c ON c.id = p.campaign_id JOIN influencer inf ON inf.id = p.influencer_id
                         JOIN account a ON a.id = p.account_id LEFT JOIN result r ON r.participation_id = p.id
                         WHERE r.id IS NULL AND c.status IN ('planned', 'running') ORDER BY c.start_date""")
    if pending.empty:
        C.empty_state("No campaigns waiting for results", "Convert a saved decision into a campaign first. Its creators then appear here so you can log what really happened.", "target")
    else:
        camps = pending.drop_duplicates("campaign_id")
        cid = st.selectbox("Campaign", camps.campaign_id.tolist(),
                           format_func=lambda i: f"#{i}  {camps.set_index('campaign_id').campaign[i]}  ({camps.set_index('campaign_id').status[i]})")
        sub = pending[pending.campaign_id == cid].reset_index(drop=True)
        st.caption("Enter the measured numbers per creator. Rows with reach 0 are skipped.")
        with st.form("log_results"):
            measured = st.date_input("Measured on", value=db.as_of())
            inputs = []
            for i, r in sub.iterrows():
                st.markdown(f"**{r.creator}** on {r.platform}, {r.deliverable}. Forecast reach {int(r.reach_p10):,} to {int(r.reach_p90):,} (median {int(r.reach_p50):,}).")
                a, b, c, d, e = st.columns([1.2, 1.2, 1.2, 0.8, 1.3], vertical_alignment="bottom")
                reach = a.number_input("Actual reach", 0, step=1000, key=f"res_reach_{r.participation_id}")
                eng = b.number_input("Engagements", 0, step=100, key=f"res_eng_{r.participation_id}")
                conv = c.number_input(f"Conversions ({r.goal.replace('_', ' ')})", 0, step=10, key=f"res_conv_{r.participation_id}")
                ontime = d.checkbox("On time", value=True, key=f"res_ot_{r.participation_id}")
                rating = e.slider("Client rating", 1, 5, 4, key=f"res_rt_{r.participation_id}")
                inputs.append((int(r.participation_id), reach, eng, conv, ontime, rating))
            if st.form_submit_button("Save results", icon=":material/save:"):
                n = 0
                for pid, reach, eng, conv, ontime, rating in inputs:
                    if reach > 0:
                        db.log_result(participation_id=pid, actual_reach=int(reach), actual_engagement=int(eng), conversions=int(conv),
                                      delivered_on_time=bool(ontime), client_rating=int(rating), measured_at=measured)
                        n += 1
                left = db.read("""SELECT COUNT(*) n FROM participation p LEFT JOIN result r ON r.participation_id = p.id
                                  WHERE p.campaign_id = ? AND r.id IS NULL""", (int(cid),)).n[0]
                if left == 0:
                    db.execute("UPDATE campaign SET status = 'completed' WHERE id = ?", (int(cid),))
                st.cache_data.clear()
                st.success(f"Logged {n} result(s)." + (" Campaign marked as completed." if left == 0 else ""))
                st.rerun()

        with st.expander("Demo helper: simulate outcomes for this campaign"):
            st.caption("For demos and tests only. Draws plausible results around the forecast (slightly below on average, like the historic data) "
                       "and stores them, so you can watch the calibration page change. Do not use with real campaigns.")
            if st.button("Simulate and save results", icon=":material/casino:"):
                rng = np.random.default_rng()
                for _, r in sub.iterrows():
                    reach = max(int(r.reach_p50 * np.exp(rng.normal(-0.12, 0.30))), 100)
                    rate = config.GOAL_CONV_RATE.get(r.goal, 0.004)
                    db.log_result(participation_id=int(r.participation_id), actual_reach=reach,
                                  actual_engagement=int(reach * rng.uniform(0.02, 0.08)),
                                  conversions=int(reach * rate * np.exp(rng.normal(0, 0.4))),
                                  delivered_on_time=bool(rng.random() < 0.85), client_rating=int(rng.integers(3, 6)), measured_at=db.as_of())
                db.execute("UPDATE campaign SET status = 'completed' WHERE id = ?", (int(cid),))
                st.cache_data.clear()
                st.rerun()
