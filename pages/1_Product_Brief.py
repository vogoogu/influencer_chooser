import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from core import ai, config, db, documents, fx, ui
from core.taxonomy import CATEGORIES, NICHES

st.set_page_config(page_title="Product brief", page_icon=":material/description:", layout="wide")
ui.sidebar()
ss = st.session_state

st.title("Product brief")
st.caption("Step 1. Give the tool a brief. It extracts a structured profile that drives filters, scoring and the mix. You can edit everything.")

brands = db.read("SELECT id, name, category FROM brand ORDER BY name")
brand_by_id = {int(r.id): r for r in brands.itertuples()}


# --------------------------------------------------------------------------- input
def load_sample() -> None:
    f = config.BRIEF_DIR / st.session_state.sample_choice
    text = f.read_text(encoding="utf-8")
    st.session_state.brief_input = text
    for bid, r in brand_by_id.items():                      # preselect the client brand named in the brief
        if r.name.lower() in text.lower():
            st.session_state.client_brand_id = bid
            break


samples = sorted(p.name for p in config.BRIEF_DIR.glob("*.md")) if config.BRIEF_DIR.exists() else []
left, right = st.columns([3, 2])
with left:
    up = st.file_uploader("Upload a brief (PDF, Word, text or markdown)", type=["pdf", "docx", "txt", "md"])
    if up is not None and ss.get("_last_upload") != (up.name, up.size):
        try:
            ss.brief_input = documents.extract_text(up.name, up.getvalue())
            ss._last_upload = (up.name, up.size)
        except Exception as e:
            st.error(str(e))
    st.text_area("...or paste / edit the brief text", key="brief_input", height=220,
                 placeholder="Product, goal, budget, target audience, countries, timeline, restrictions ...")
with right:
    if samples:
        st.selectbox("Sample briefs", samples, key="sample_choice")
        st.button("Load sample brief", on_click=load_sample, icon=":material/upload_file:")
    options = [None] + list(brand_by_id)
    ss.client_brand_id = st.selectbox(
        "Client brand", options, index=options.index(ss.client_brand_id) if ss.client_brand_id in options else 0,
        format_func=lambda i: "(not in database / other)" if i is None else f"{brand_by_id[i].name}  [{brand_by_id[i].category.replace('_', ' ')}]",
        help="Used to detect conflicts: creators who recently promoted a competitor or are bound by exclusivity.")
    do_research = st.checkbox("Research the market first (AI + web search)", value=False, disabled=ui.ai_client() is None,
                              help="Adds a short note on competitors, creator trends and brand-safety topics. Needs an API key.")

# --------------------------------------------------------------------------- editor state
EDITOR_KEYS = ["p_name", "p_category", "p_goal", "p_budget", "p_countries", "p_platforms", "p_niches", "p_tone", "p_female_on",
               "p_female", "p_age_on", "p_age", "p_flags", "p_notes", "p_start", "p_end", "p_summary", "p_selling"]


def load_editor(p: dict) -> None:
    ta = p.get("target_audience") or {}
    tl = p.get("timeline") or {}
    ss.p_name = p.get("product_name", "")
    ss.p_category = p.get("category", CATEGORIES[0])
    ss.p_goal = p.get("goal", "awareness")
    ss.p_budget = float(p.get("budget_eur") or 0)
    ss.p_countries = p.get("target_countries", ["AT"])
    ss.p_platforms = p.get("preferred_platforms", list(config.PLATFORMS))
    ss.p_niches = p.get("relevant_niches", [])
    ss.p_tone = p.get("tone", [])
    ss.p_female_on = ta.get("female_pct") is not None
    ss.p_female = float(ta.get("female_pct") if ta.get("female_pct") is not None else 50)
    ss.p_age_on = ta.get("age_min") is not None and ta.get("age_max") is not None
    ss.p_age = (int(ta.get("age_min") or 18), int(ta.get("age_max") or 45))
    ss.p_flags = p.get("restrictions", {}).get("excluded_flag_types", [])
    ss.p_notes = p.get("restrictions", {}).get("notes", "")
    base = db.as_of() + timedelta(days=14)
    for key, fallback in (("p_start", base), ("p_end", base + timedelta(days=28))):
        raw = tl.get("start" if key == "p_start" else "end")
        try:
            ss[key] = date.fromisoformat(raw) if raw else fallback
        except ValueError:
            ss[key] = fallback
    ss.p_summary = p.get("summary", "")
    ss.p_selling = "\n".join(p.get("selling_points", []))


def profile_from_editor() -> dict:
    return {
        "product_name": ss.p_name, "category": ss.p_category, "goal": ss.p_goal, "budget_eur": float(ss.p_budget) or None,
        "target_countries": list(ss.p_countries), "preferred_platforms": list(ss.p_platforms),
        "relevant_niches": list(ss.p_niches), "tone": list(ss.p_tone), "summary": ss.p_summary,
        "selling_points": [x.strip() for x in ss.p_selling.splitlines() if x.strip()],
        "languages": ["de"] if set(ss.p_countries) & {"AT", "DE", "CH"} else ["en"],
        "target_audience": {"female_pct": float(ss.p_female) if ss.p_female_on else None,
                            "age_min": int(ss.p_age[0]) if ss.p_age_on else None,
                            "age_max": int(ss.p_age[1]) if ss.p_age_on else None,
                            "description": (ss.get("profile") or {}).get("target_audience", {}).get("description", "")},
        "restrictions": {"excluded_flag_types": list(ss.p_flags), "notes": ss.p_notes},
        "timeline": {"start": ss.p_start.isoformat(), "end": ss.p_end.isoformat()},
    }


# --------------------------------------------------------------------------- analysis
if st.button("Analyze brief", type="primary", icon=":material/auto_awesome:", disabled=not ss.brief_input.strip()):
    client = ui.ai_client()
    with st.status("Analyzing the brief ...", expanded=True) as status:
        research = ""
        if do_research and client:
            st.write("Researching the market with web search ...")
            r = ai.research_market(client, ss.brief_input)
            research = r.data
            if r.error:
                st.write(f"Web research skipped: {r.error}")
        st.write("Building the structured profile ...")
        result = ai.build_profile(client, ss.brief_input, research)
        profile = result.data
        if ss.client_brand_id:
            profile["category"] = brand_by_id[ss.client_brand_id].category
        ss.brief_text, ss.profile, ss.profile_source, ss.profile_error = ss.brief_input, profile, result.source, result.error
        ss.research_text = research
        ss.profile_version += 1
        for k in ("explanations", "memo_text", "challenges"):
            ss[k] = {} if k == "explanations" else None
        load_editor(profile)
        status.update(label="Brief analyzed", state="complete", expanded=False)
    st.rerun()

if ss.profile is None:
    st.info("Load a sample brief or paste your own, then press **Analyze brief**.")
    st.stop()

# --------------------------------------------------------------------------- editor
if "p_name" not in ss:
    load_editor(ss.profile)

st.divider()
st.subheader("Campaign profile")
ui.ai_badge(ss.profile_source, ss.profile_error)
if ss.profile_source == "fallback" and ss.profile_error and ss.profile_error != "no API key":
    st.warning(f"AI analysis failed, rule-based extraction was used instead. Details: {ss.profile_error}")
if ss.research_text:
    with st.expander("Market research notes (AI with web search)"):
        st.markdown(ss.research_text)

c1, c2, c3 = st.columns(3)
with c1:
    st.text_input("Product name", key="p_name")
    st.selectbox("Product category", CATEGORIES, key="p_category", format_func=lambda c: c.replace("_", " "))
    st.selectbox("Campaign goal", config.GOALS, key="p_goal", format_func=lambda g: g.replace("_", " "))
    st.number_input("Budget (EUR)", min_value=0.0, step=500.0, key="p_budget")
    with st.expander("Budget in another currency?"):
        amt = st.number_input("Amount", min_value=0.0, step=500.0, key="fx_amount")
        cur = st.selectbox("Currency", ["CHF", "USD", "GBP"], key="fx_cur")
        if st.button("Convert to EUR with ECB rate"):
            eur, src = fx.to_eur(amt, cur)
            ss.p_budget = round(eur, 0)
            st.toast(f"{amt:,.0f} {cur} = {eur:,.0f} EUR ({src})")
            st.rerun()
with c2:
    st.multiselect("Target countries", config.COUNTRIES, key="p_countries")
    st.multiselect("Platforms", config.PLATFORMS, key="p_platforms")
    st.multiselect("Relevant creator niches", NICHES, key="p_niches", format_func=lambda n: n.replace("_", " "),
                   help="Creators in these niches get the full niche-match score.")
    st.multiselect("Desired tone", config.TONES, key="p_tone", format_func=lambda t: t.replace("_", " "))
with c3:
    st.checkbox("Specify target gender split", key="p_female_on")
    st.slider("Target share of women (%)", 0, 100, key="p_female", disabled=not ss.p_female_on)
    st.checkbox("Specify target age range", key="p_age_on")
    st.slider("Target age range", 14, 70, key="p_age", disabled=not ss.p_age_on)
    st.multiselect("Exclude creators with these safety flags", config.FLAG_TYPES, key="p_flags",
                   format_func=lambda f: f.replace("_", " "))
d1, d2 = st.columns(2)
d1.date_input("Campaign start", key="p_start")
d2.date_input("Campaign end", key="p_end")
st.text_area("Summary", key="p_summary", height=80)
s1, s2 = st.columns(2)
s1.text_area("Selling points (one per line)", key="p_selling", height=100)
s2.text_area("Restriction notes (not machine-enforced)", key="p_notes", height=100)

ss.profile = profile_from_editor()

b1, b2 = st.columns([1, 3])
if b1.button("Use these values in the shortlist filters", icon=":material/sync:"):
    ss.profile_version += 1
    st.toast("Shortlist filters will start from these values")
ui.page_link("pages/2_Shortlist.py", label="Continue to the shortlist", icon=":material/arrow_forward:")
