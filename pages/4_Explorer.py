import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import charts, db, ui

st.set_page_config(page_title="Explorer", page_icon=":material/search:", layout="wide")
ui.sidebar()

st.title("Influencer explorer")
st.caption("Browse the evidence behind the scores: trends over time, audience, rates, flags and campaign history.")

inf = db.read("SELECT id, display_name, country, city, status FROM influencer ORDER BY display_name")
choice = st.selectbox("Creator", inf.id.tolist(), format_func=lambda i: f"{inf.set_index('id').display_name[i]}  ({inf.set_index('id').country[i]})")
row = db.read("SELECT * FROM influencer WHERE id = ?", (int(choice),)).iloc[0]
accounts = db.read("SELECT * FROM account WHERE influencer_id = ?", (int(choice),))
niches = db.read("""SELECT n.name, inn.is_primary FROM influencer_niche inn JOIN niche n ON n.id = inn.niche_id
                    WHERE inn.influencer_id = ?""", (int(choice),))
as_of = db.as_of().isoformat()

h1, h2, h3, h4 = st.columns(4)
h1.metric("Status", row.status)
h2.metric("Availability", row.availability)
h3.metric("Relationship score", f"{row.relationship_score} / 5")
h4.metric("GDPR consent until", row.consent_until, delta="expired" if row.consent_until < as_of else None, delta_color="inverse")

t_prof, t_trend, t_aud, t_rate, t_hist, t_content = st.tabs(["Profile", "Trends", "Audience", "Rates and flags", "Campaign history", "Content"])

with t_prof:
    st.markdown(f"**{row.display_name}**, {row.city}, {row.country}. Languages: {row.languages}. Tone: {row.tone}.")
    st.write(row.bio_text)
    prim = ", ".join(n.replace("_", " ") for n, p in zip(niches.name, niches.is_primary) if p)
    sec = ", ".join(n.replace("_", " ") for n, p in zip(niches.name, niches.is_primary) if not p)
    st.markdown(f"Primary niche: **{prim}**" + (f". Also: {sec}." if sec else "."))
    if row.agency_notes:
        st.info(f"Agency notes: {row.agency_notes}", icon=":material/sticky_note_2:")
    st.dataframe(accounts[["platform", "handle", "url", "verified"]], hide_index=True, width="stretch",
                 column_config={"url": st.column_config.LinkColumn("Profile link")})

with t_trend:
    for _, a in accounts.iterrows():
        snap = db.read("SELECT * FROM account_snapshot WHERE account_id = ? ORDER BY captured_at", (int(a.id),))
        st.plotly_chart(charts.trend_lines(snap, f"{a.platform} @{a.handle}"), width="stretch")
        last = snap.iloc[-1]
        age = (pd.Timestamp(as_of) - pd.Timestamp(last.captured_at)).days
        growth = snap.followers.pct_change().tail(6).max()
        notes = []
        if age > 90:
            notes.append(f"stale: last snapshot {age} days old")
        if growth >= 0.5:
            notes.append(f"follower jump of {growth:.0%} in one month")
        if last.fake_follower_pct >= 30:
            notes.append(f"{last.fake_follower_pct:.0f}% estimated fake followers")
        if notes:
            st.warning("Data warnings: " + "; ".join(notes))
        st.caption(f"Source of the latest snapshot: {last.source} (confidence {last.confidence:.2f}).")

with t_aud:
    cols = st.columns(max(len(accounts), 1))
    for col, (_, a) in zip(cols, accounts.iterrows()):
        aud = db.read("SELECT * FROM audience_snapshot WHERE account_id = ? ORDER BY captured_at DESC LIMIT 1", (int(a.id),))
        with col:
            st.markdown(f"**{a.platform}**")
            if aud.empty:
                st.caption("No audience data")
                continue
            r = aud.iloc[0]
            st.metric("Women", f"{r.female_pct:.0f}%")
            fig = go.Figure(go.Bar(x=["18-24", "25-34", "35-44", "45+"], y=[r.age_18_24, r.age_25_34, r.age_35_44, r.age_45_plus],
                                   marker_color=charts.COLORS[0]))
            fig.update_layout(height=240, yaxis_title="% of audience", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch", key=f"age_{a.id}")
            tc = pd.DataFrame(json.loads(r.top_countries))
            st.dataframe(tc.assign(share=(tc.share * 100).round(1)).rename(columns={"share": "share %"}), hide_index=True, width="stretch")
            st.caption(f"Snapshot {r.captured_at}")

with t_rate:
    rates = db.read("SELECT platform, deliverable, price_eur, usage_rights_months, exclusivity_premium_pct, valid_from, valid_to FROM rate_card WHERE influencer_id = ?", (int(choice),))
    rates["expired"] = rates.valid_to < as_of
    st.dataframe(rates, hide_index=True, width="stretch")
    if rates.expired.any():
        st.warning("Some rate cards have expired. Renegotiate before booking.")
    flags = db.read("SELECT flag_type, severity, source, evidence_url, flagged_at, expires_at FROM safety_flag WHERE influencer_id = ?", (int(choice),))
    excl = db.read("""SELECT b.name AS brand, e.valid_from, e.valid_to FROM exclusivity e JOIN brand b ON b.id = e.brand_id
                      WHERE e.influencer_id = ?""", (int(choice),))
    st.markdown("**Safety flags**")
    if len(flags):
        st.dataframe(flags, hide_index=True, width="stretch")
    else:
        st.caption("None")
    st.markdown("**Exclusivity contracts**")
    if len(excl):
        st.dataframe(excl, hide_index=True, width="stretch")
    else:
        st.caption("None")

with t_hist:
    hist = db.read("""SELECT c.name AS campaign, b.name AS brand, b.category, p.deliverable, p.fee_eur, p.reach_p50,
                             r.actual_reach, r.conversions, r.delivered_on_time, r.client_rating
                      FROM participation p JOIN campaign c ON c.id = p.campaign_id JOIN brand b ON b.id = c.brand_id
                      LEFT JOIN result r ON r.participation_id = p.id WHERE p.influencer_id = ? ORDER BY c.start_date DESC""", (int(choice),))
    if hist.empty:
        st.caption("No campaigns yet.")
    else:
        hist["actual_vs_predicted"] = hist.actual_reach / hist.reach_p50
        st.dataframe(hist, hide_index=True, width="stretch",
                     column_config={"actual_vs_predicted": st.column_config.NumberColumn("Actual / predicted", format="%.2f")})

with t_content:
    content = db.read("""SELECT c.posted_at, a.platform, c.format, c.caption, c.views, c.likes, c.comments, c.is_sponsored, b.name AS brand, c.is_disclosed
                         FROM content_item c JOIN account a ON a.id = c.account_id LEFT JOIN brand b ON b.id = c.brand_id
                         WHERE a.influencer_id = ? ORDER BY c.posted_at DESC""", (int(choice),))
    st.dataframe(content, hide_index=True, width="stretch", height=400)
    undisclosed = content[(content.is_sponsored == 1) & (content.is_disclosed == 0)]
    if len(undisclosed):
        st.warning(f"{len(undisclosed)} sponsored post(s) without an advertising label. Check compliance before working with this creator.")
