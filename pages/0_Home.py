import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from core import components as C, config, db, scoring, ui

ui.page_config("Overview", ":material/home:")
ui.sidebar()
ss = st.session_state

ui.header("Choose the right creators, backed by evidence",
          "Turn a product brief into a ranked shortlist, an optimal campaign mix and a forecast you can defend. "
          "AI reads the brief and explains the result; deterministic code calculates every score, ranking and forecast.",
          eyebrow="Overview")

as_of = db.as_of()
n_inf = int(db.read("SELECT COUNT(*) n FROM influencer").n[0])
n_acc = int(db.read("SELECT COUNT(*) n FROM account").n[0])
n_res = int(db.read("SELECT COUNT(*) n FROM result").n[0])
runs = db.read("""SELECT r.id, r.created_at, r.title, b.name AS brand, r.budget_eur, r.campaign_id
                  FROM run r LEFT JOIN brand b ON b.id = r.brand_id ORDER BY r.id DESC""")

# ------------------------------------------------------------------------------------------ hero + numbers
left, right = st.columns([3, 2], gap="large")
with left:
    with st.container(key="hero_main"):
        profile = ss.get("profile")
        done = ui.workflow_done()
        if not profile:
            C.panel_title("Start here")
            C.hero_title("Begin with a product brief")
            st.markdown("Upload a PDF, Word file or text. The tool extracts category, goal, budget, markets and audience, "
                        "and you review it before anything is scored.")
            if st.button("Start with a product brief", type="primary", icon=":material/arrow_forward:", key="home_cta"):
                st.switch_page("pages/1_Product_Brief.py")
        else:
            C.panel_title("Continue where you left off")
            C.hero_title(profile.get("product_name") or "Untitled product")
            C.chips([(profile.get("category", "").replace("_", " "), "primary"), (profile.get("goal", "").replace("_", " "), "outline"),
                     (f"EUR {profile['budget_eur']:,.0f}" if profile.get("budget_eur") else "no budget", "outline"),
                     (", ".join(profile.get("target_countries", [])), "outline")])
            if 2 not in done:
                label, target = "Continue to the shortlist", "pages/2_Shortlist.py"
            elif 3 not in done:
                label, target = "Continue to the campaign mix", "pages/3_Campaign_Mix.py"
            else:
                label, target = "Open the decision log", "pages/5_Decision_Log.py"
            if st.button(label, type="primary", icon=":material/arrow_forward:", key="home_cta"):
                st.switch_page(target)
with right:
    a, b = st.columns(2)
    a.metric("Influencers", n_inf)
    b.metric("Accounts", n_acc)
    c, d = st.columns(2)
    c.metric("Campaign results logged", n_res)
    d.metric("Saved decisions", len(runs))

C.section("Your workflow", "Five steps from brief to a forecast that learns from real results.")
C.stepper(0, ui.workflow_done())
C.tiles([
    ("file", "1. Read the brief", "Upload a brief and review the structured profile. Everything downstream follows from it."),
    ("list", "2. Filter and rank", "Hard filters with reasons, seven weighted criteria, live re-ranking and a stability check."),
    ("sliders", "3. Build the mix", "The best combination for the budget, with outcome ranges, a risk review and a memo."),
    ("target", "4. Log and learn", "Save decisions, log real results, and let the forecast model improve."),
])

# ------------------------------------------------------------------------------------------ data health
C.section("Data health", f"Signals the shortlist uses to flag or exclude creators, as of {as_of.isoformat()}.")
uni = scoring.load_universe(as_of)
rates = scoring.load_rates()
flags = scoring.load_flags(as_of)
inf_u = uni.drop_duplicates("influencer_id")
checks = [
    ("Stale accounts", int((uni.days_old > config.STALE_DAYS).sum()), len(uni), f"Latest stats older than {config.STALE_DAYS} days"),
    ("Follower spikes", int((uni.max_growth_6m >= config.SPIKE_GROWTH).sum()), len(uni), "Followers grew 50%+ in a month, last six months"),
    ("Suspected fake followers", int((uni.fake >= config.FAKE_PCT_WARN).sum()), len(uni), f"{config.FAKE_PCT_WARN:.0f}%+ estimated fake followers"),
    ("Expired rate cards", int(rates.groupby("influencer_id").valid_to.max().lt(as_of.isoformat()).sum()), n_inf, "Creators with no valid rate card"),
    ("Expired GDPR consent", int((inf_u.consent_until < as_of.isoformat()).sum()), n_inf, "Retention date passed: exclude and review"),
    ("Active safety flags", len(flags), n_inf, "Open brand-safety flags across creators"),
]
cards = [C.health_card_html(str(v), label, f"{help_}. {v} of {tot}.", "success" if v == 0 else ("danger" if label in ("Expired GDPR consent", "Suspected fake followers") else "warning"), v / tot if tot else 0)
         for label, v, tot, help_ in checks]
C.health_grid(cards)
ui.page_link("pages/4_Explorer.py", label="Review creators in the explorer", icon=":material/arrow_forward:")

# ------------------------------------------------------------------------------------------ recent decisions
C.section("Recent decisions", "Every saved decision keeps its brief, weights and forecasts as an audit trail.")
if runs.empty:
    C.empty_state("No decisions saved yet", "Build a campaign mix and press Save decision. It will appear here.", "history")
else:
    st.dataframe(runs.head(6).assign(status=lambda d: d.campaign_id.map(lambda x: "Converted to campaign" if x == x and x else "Draft")).drop(columns=["campaign_id"]),
                 hide_index=True, width="stretch",
                 column_config={"id": st.column_config.NumberColumn("#", width="small"), "created_at": "Saved", "title": "Decision", "brand": "Client brand",
                                "budget_eur": st.column_config.NumberColumn("Budget", format="euro"), "status": "Status"})
    ui.page_link("pages/5_Decision_Log.py", label="Open the decision log", icon=":material/arrow_forward:")

with st.expander("About this app and its limits"):
    st.markdown(
        """
- **Synthetic data.** The bundled database is fictional but calibrated to published 2026 benchmarks (see `DATA_SOURCES.md`).
- **AI is optional.** Without an API key the brief is analysed by keyword rules and explanations are template-based.
- **Forecasts are estimates.** Reach is predicted from average views, corrected by what past campaigns showed (tier correction or a random forest), and shown as ranges.
- **Topic fit** uses TF-IDF text similarity by default. A multilingual sentence-transformer model can be enabled (see README).
- **Write access.** Saved decisions and results go to a working copy of the database. Use *Reset demo database* in the sidebar to start over.
""")
