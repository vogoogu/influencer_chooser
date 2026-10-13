"""Streamlit helpers shared by all pages: session state, sidebar, settings objects."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, timedelta

import streamlit as st

from . import ai, config, db
from .scoring import FilterSettings

DEFAULT_WEIGHT_PRESET = "Balanced"


# ----------------------------------------------------------------------------- state
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("brief_text", "")
    ss.setdefault("profile", None)
    ss.setdefault("profile_source", None)
    ss.setdefault("profile_error", None)
    ss.setdefault("research_text", "")
    ss.setdefault("profile_version", 0)
    ss.setdefault("filters_version", -1)
    ss.setdefault("client_brand_id", None)
    ss.setdefault("explanations", {})
    ss.setdefault("explanations_source", None)
    ss.setdefault("memo_text", None)
    ss.setdefault("memo_source", None)
    ss.setdefault("challenges", None)
    ss.setdefault("challenges_source", None)
    for c, v in config.PRESETS[DEFAULT_WEIGHT_PRESET].items():
        ss.setdefault(f"w_{c}", v)
    ss.setdefault("preset", DEFAULT_WEIGHT_PRESET)
    for k, v in {"f_max_share": 0.4, "f_allow_limited": True, "f_excl_conflicts": True, "f_excl_warn": False,
                 "f_max_sev": 2, "f_ig_format": "reel", "f_min_followers": 0, "f_max_followers": 10_000_000,
                 "f_apply_corr": True, "use_ai": True}.items():
        ss.setdefault(k, v)
    keep_alive()


KEEP_PREFIXES = ("f_", "w_", "p_", "m_", "sim_", "fx_")
KEEP_KEYS = ("brief_input", "preset", "use_ai", "ui_api_key", "sample_choice")


def keep_alive() -> None:
    """Streamlit drops widget state when a page does not render the widget. Re-assigning the value on
    every page keeps filters, weights and editor fields when the user moves between pages."""
    ss = st.session_state
    for k in list(ss.keys()):
        if k.startswith(KEEP_PREFIXES) or k in KEEP_KEYS:
            ss[k] = ss[k]


def defaults(**kv) -> None:
    """Initialise widget state without passing `value=` to the widget (avoids Streamlit's default/state conflict warning)."""
    for k, v in kv.items():
        st.session_state.setdefault(k, v)


def set_preset() -> None:
    for c, v in config.PRESETS[st.session_state.preset].items():
        st.session_state[f"w_{c}"] = v


def weights() -> dict[str, float]:
    return {c: float(st.session_state.get(f"w_{c}", 1)) for c in config.CRITERIA}


def sync_filters(profile: dict) -> None:
    """Fill the filter widgets from the profile when a new profile was created."""
    ss = st.session_state
    if ss.filters_version == ss.profile_version:
        return
    ss.f_countries = [c for c in profile.get("target_countries", ["AT"]) if c in config.COUNTRIES] or ["AT"]
    ss.f_platforms = [p for p in profile.get("preferred_platforms", config.PLATFORMS) if p in config.PLATFORMS] or list(config.PLATFORMS)
    ss.f_budget = float(profile.get("budget_eur") or 20000)
    ss.f_flags = list(profile.get("restrictions", {}).get("excluded_flag_types", []))
    tl = profile.get("timeline") or {}
    try:
        start = date.fromisoformat(tl.get("start")) if tl.get("start") else db.as_of() + timedelta(days=14)
    except ValueError:
        start = db.as_of() + timedelta(days=14)
    try:
        end = date.fromisoformat(tl.get("end")) if tl.get("end") else start + timedelta(days=28)
    except ValueError:
        end = start + timedelta(days=28)
    ss.f_start, ss.f_end = start, max(end, start)
    ss.filters_version = ss.profile_version


def filter_settings() -> FilterSettings:
    ss = st.session_state
    return FilterSettings(
        countries=list(ss.get("f_countries", ["AT"])), platforms=list(ss.get("f_platforms", config.PLATFORMS)),
        budget_eur=float(ss.get("f_budget", 20000)), max_fee_share=float(ss.get("f_max_share", 0.4)),
        allow_limited=bool(ss.get("f_allow_limited", True)), exclude_conflicts=bool(ss.get("f_excl_conflicts", True)),
        exclude_data_warnings=bool(ss.get("f_excl_warn", False)), excluded_flags=list(ss.get("f_flags", [])),
        max_severity=int(ss.get("f_max_sev", 2)), min_followers=int(ss.get("f_min_followers", 0)),
        max_followers=int(ss.get("f_max_followers", 10**9)), ig_format=ss.get("f_ig_format", "reel"),
        client_brand_id=ss.get("client_brand_id"), category=(ss.profile or {}).get("category"),
        start=ss.get("f_start"), end=ss.get("f_end"), apply_correction=bool(ss.get("f_apply_corr", True)))


def settings_key(s: FilterSettings) -> str:
    return json.dumps(asdict(s), default=str, sort_keys=True)


def settings_from_key(key: str) -> FilterSettings:
    d = json.loads(key)
    for k in ("start", "end"):
        d[k] = date.fromisoformat(d[k]) if d.get(k) else None
    return FilterSettings(**d)


# ----------------------------------------------------------------------------- AI client
def ai_client():
    """Anthropic client from secrets/env or the sidebar field, or None (rule-based mode)."""
    ss = st.session_state
    if not ss.get("use_ai", True):
        return None
    key = config.get_secret("ANTHROPIC_API_KEY") or ss.get("ui_api_key") or None
    if not key:
        return None
    if ss.get("_client_for") != key:
        ss._client = ai.make_client(key)
        ss._client_for = key
    return ss.get("_client")


def ai_badge(source: str | None, error: str | None = None) -> None:
    if source == "ai":
        st.caption(":material/smart_toy: Written by Claude from the database facts")
    elif source == "fallback":
        msg = " (" + error + ")" if error and error != "no API key" else ""
        st.caption(f":material/rule: Rule-based text, no AI used{msg}")


# ----------------------------------------------------------------------------- sidebar
def sidebar() -> None:
    init_state()
    with st.sidebar:
        st.markdown("### Influencer Chooser")
        st.caption("Decision support for matching creators to a product brief.")
        st.markdown("**AI assistant**")
        has_secret = bool(config.get_secret("ANTHROPIC_API_KEY"))
        st.checkbox("Use AI (Claude)", key="use_ai", help="Off = rule-based mode, nothing leaves your machine.")
        if st.session_state.use_ai and not has_secret:
            st.text_input("Anthropic API key", type="password", key="ui_api_key",
                          help="Kept in this browser session only. For deployment use Streamlit secrets.")
        client = ai_client()
        if client:
            st.success(f"Claude active ({config.MODEL})", icon=":material/check_circle:")
        else:
            st.info("Rule-based mode (no API key)", icon=":material/rule:")
        st.markdown("**Data**")
        st.caption(f"Database as of {db.as_of().isoformat()}. Synthetic demo data." if _is_synthetic() else f"Database as of {db.as_of().isoformat()}.")
        with st.expander("Reset demo database"):
            st.caption("Deletes saved decisions and logged results and restores the original dataset.")
            if st.button("Reset now", key="reset_db_btn"):
                db.reset_db()
                st.cache_data.clear()
                for k in ("explanations", "memo_text", "challenges"):
                    st.session_state[k] = {} if k == "explanations" else None
                st.toast("Database reset")
                st.rerun()


def _is_synthetic() -> bool:
    try:
        return db.read("SELECT value FROM meta WHERE key='synthetic'").value.iloc[0] == "true"
    except Exception:
        return False


def page_link(path: str, label: str, icon: str | None = None) -> None:
    """st.page_link that degrades to plain text if the page cannot be resolved (e.g. other entry point)."""
    try:
        st.page_link(path, label=label, icon=icon)
    except Exception:
        st.caption(f"{label} (use the sidebar navigation)")


def need_profile() -> bool:
    """Show a hint and return True if no brief has been analysed yet."""
    if not st.session_state.get("profile"):
        st.info("Start with the **Product brief** page: upload or paste a brief and let the tool build the profile.", icon=":material/arrow_back:")
        page_link("pages/1_Product_Brief.py", label="Go to Product brief", icon=":material/description:")
        return True
    return False


def fmt_eur(x: float) -> str:
    return f"EUR {x:,.0f}"
