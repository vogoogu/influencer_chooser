"""Influencer Chooser: entry page. Run with `streamlit run app.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from core import config, db, scoring, ui

st.set_page_config(page_title="Influencer Chooser", page_icon=":material/groups:", layout="wide")
ui.sidebar()

st.title("Influencer Chooser")
st.markdown(
    "A decision tool for a marketing agency: upload a product brief, and the tool filters, scores and "
    "ranks your partner influencers, builds the best campaign mix for a budget, and learns from real "
    "campaign results. **AI reads the brief and explains the result. All scores, rankings and "
    "forecasts are calculated by deterministic code from your database.**")

as_of = db.as_of()
n_inf = int(db.read("SELECT COUNT(*) n FROM influencer").n[0])
n_acc = int(db.read("SELECT COUNT(*) n FROM account").n[0])
n_res = int(db.read("SELECT COUNT(*) n FROM result").n[0])
n_run = int(db.read("SELECT COUNT(*) n FROM run").n[0])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Influencers", n_inf)
c2.metric("Accounts", n_acc)
c3.metric("Campaign results logged", n_res)
c4.metric("Saved decisions", n_run)

st.subheader("Workflow")
cols = st.columns(5)
steps = [
    ("pages/1_Product_Brief.py", "1. Product brief", "Upload or paste a brief. The tool builds a structured profile you can edit.", ":material/description:"),
    ("pages/2_Shortlist.py", "2. Shortlist", "Hard filters, seven weighted criteria, live re-ranking and a stability check.", ":material/list_alt:"),
    ("pages/3_Campaign_Mix.py", "3. Campaign mix", "Optimal combination for the budget, Monte Carlo ranges, risk review, memo.", ":material/tune:"),
    ("pages/5_Decision_Log.py", "4. Decision log", "Save decisions, turn them into campaigns, log real results.", ":material/history:"),
    ("pages/6_Calibration.py", "5. Calibration", "How wrong were past forecasts, and what really predicts success?", ":material/target:"),
]
for col, (page, title, text, icon) in zip(cols, steps):
    with col:
        with st.container(border=True):
            ui.page_link(page, label=title, icon=icon)
            st.caption(text)
ui.page_link("pages/4_Explorer.py", label="Influencer explorer: browse profiles, trends and audiences", icon=":material/search:")

st.subheader("Data health")
st.caption(f"Checks on the latest data as of {as_of.isoformat()}. These are the same signals the shortlist uses to flag or exclude creators.")
uni = scoring.load_universe(as_of)
rates = scoring.load_rates()
stale = int((uni.days_old > config.STALE_DAYS).sum())
spike = int((uni.max_growth_6m >= config.SPIKE_GROWTH).sum())
fake = int((uni.fake >= config.FAKE_PCT_WARN).sum())
expired_rate = int(rates.groupby("influencer_id").valid_to.max().lt(as_of.isoformat()).sum())
consent = int((uni.drop_duplicates("influencer_id").consent_until < as_of.isoformat()).sum())
flags = scoring.load_flags(as_of)
h1, h2, h3, h4, h5, h6 = st.columns(6)
h1.metric("Stale accounts", stale, help=f"Latest snapshot older than {config.STALE_DAYS} days")
h2.metric("Follower spikes", spike, help="Followers grew 50%+ within one month in the last six months")
h3.metric("High fake follower share", fake, help=f"{config.FAKE_PCT_WARN:.0f}% or more estimated fake followers")
h4.metric("Influencers with expired rates", expired_rate)
h5.metric("Expired GDPR consent", consent)
h6.metric("Active safety flags", len(flags))

with st.expander("About this app and its limits"):
    st.markdown(
        """
- **Synthetic data.** The bundled database is fictional but calibrated to published 2026 benchmarks (see `DATA_SOURCES.md`).
- **AI is optional.** Without an API key the brief is analysed by keyword rules and explanations are template-based.
- **Forecasts are estimates.** Reach is predicted from average views, corrected by the bias found in past campaigns, and shown as ranges.
- **Topic fit** uses TF-IDF text similarity by default. A multilingual sentence-transformer model can be enabled (see README).
- **Write access.** Saved decisions and results go to a working copy of the database. Use *Reset demo database* in the sidebar to start over.
""")
