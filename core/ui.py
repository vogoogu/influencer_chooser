"""Streamlit helpers shared by all pages: session state, header chrome, settings objects."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, timedelta

import streamlit as st

from . import ai, components as C, config, db, theme
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
                 "f_forecast": "forest", "use_ai": True, "ui_model": config.DEFAULT_MODEL}.items():
        ss.setdefault(k, v)
    keep_alive()


KEEP_PREFIXES = ("f_", "w_", "p_", "m_", "sim_", "fx_")
KEEP_KEYS = ("brief_input", "preset", "use_ai", "ui_api_key", "ui_model", "sample_choice")


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
        start=ss.get("f_start"), end=ss.get("f_end"), apply_correction=ss.get("f_forecast", "forest") != "naive",
        forecast_model=ss.get("f_forecast", "forest"))


def settings_key(s: FilterSettings) -> str:
    return json.dumps(asdict(s), default=str, sort_keys=True)


def settings_from_key(key: str) -> FilterSettings:
    d = json.loads(key)
    for k in ("start", "end"):
        d[k] = date.fromisoformat(d[k]) if d.get(k) else None
    return FilterSettings(**d)


# ----------------------------------------------------------------------------- AI client
def active_model() -> str:
    """Model slug: secret or env var OPENROUTER_MODEL, else the Menu field, else the default."""
    return (config.get_secret("OPENROUTER_MODEL") or st.session_state.get("ui_model") or config.DEFAULT_MODEL).strip()


def ai_client():
    """OpenRouter client from secrets/env or the Menu field, or None (rule-based mode)."""
    ss = st.session_state
    if not ss.get("use_ai", True):
        return None
    key = config.get_secret("OPENROUTER_API_KEY") or ss.get("ui_api_key") or None
    if not key:
        return None
    sig = (key, active_model())
    if ss.get("_client_for") != sig:
        ss._client = ai.make_client(key, sig[1])
        ss._client_for = sig
    return ss.get("_client")


def ai_badge(source: str | None, error: str | None = None) -> None:
    """Small provenance chip: who wrote this text, the AI model or the rule-based fallback."""
    if source == "ai":
        C.chips([("Written by the AI model from database facts", "primary", "sparkles")])
    elif source == "fallback":
        msg = f" ({error})" if error and error != "no API key" else ""
        C.chips([(f"Rule-based text, no AI used{msg}", "neutral", "info")])


# ----------------------------------------------------------------------------- chrome
def page_config(title: str, icon: str) -> None:
    """set_page_config that is safe under st.navigation (the router already set it) and when run standalone."""
    try:
        st.set_page_config(page_title=f"{title} | Influencer Chooser", page_icon=icon, layout="wide", initial_sidebar_state="collapsed")
    except Exception:
        pass


def apply_theme() -> None:
    theme.register_plotly_template()
    # st.html sanitises a leading <style> away; markdown with unsafe_allow_html keeps it
    st.markdown(f"<style>{theme.css()}</style>", unsafe_allow_html=True)


def workflow_done() -> set[int]:
    """Which of the five workflow steps have been completed (shown in the stepper)."""
    ss, done = st.session_state, set()
    if ss.get("profile"):
        done.add(1)
    if ss.get("ranked") is not None and len(ss.ranked):
        done.add(2)
    try:
        runs = db.read("SELECT COUNT(*) AS n, COALESCE(SUM(campaign_id IS NOT NULL), 0) AS c FROM run")
        if int(runs.n[0]) > 0:
            done.add(3)
        if int(runs.c[0]) > 0:
            done.add(4)
    except Exception:
        pass
    return done


def header(title: str, subtitle: str = "", step: int | None = None, eyebrow: str | None = None) -> None:
    """Page title block, with the workflow stepper on the five workflow pages."""
    C.page_header(title, subtitle, eyebrow or (f"Step {step} of {len(C.WORKFLOW)}" if step else None))
    if step:
        C.stepper(step, workflow_done())


def card(key: str | None = None):
    """A bordered white surface for grouping related content. Keys are deterministic per run (card-1, card-2 ...)
    so the stylesheet can address them and widget state is never affected."""
    if key is None:
        st.session_state["_card_i"] = st.session_state.get("_card_i", 0) + 1
        key = f"card-{st.session_state['_card_i']}"
    return st.container(border=True, key=key)


def chrome() -> None:
    """Theme plus the header utilities: AI status and a Menu popover (AI settings, data, reset).
    Navigation itself lives in the header too (see app.py)."""
    init_state()
    st.session_state["_card_i"] = 0                    # restart the card key counter on every run
    apply_theme()
    client = ai_client()
    with st.container(key="topbar", horizontal=True, vertical_alignment="center", gap="small"):
        if client:
            st.html(C.chips_html([("AI active", "success", "check")]))
        else:
            st.html(C.chips_html([("Rule-based mode", "neutral", "info")]))
        with st.popover("Menu", icon=":material/tune:"):
            C.panel_title("AI assistant")
            if client:
                C.chips([("Active via OpenRouter", "success", "check"), (active_model(), "outline")])
                if getattr(client, "tool_mode", None):
                    st.caption(f"Structured answers use: {client.tool_mode.replace('json', 'plain JSON')} mode.")
            else:
                st.caption("No API key set: keyword rules and templates are used instead of AI.")
            has_secret = bool(config.get_secret("OPENROUTER_API_KEY"))
            st.checkbox("Use AI", key="use_ai", help="Off = rule-based mode, nothing leaves your machine.")
            if st.session_state.use_ai and not has_secret:
                st.text_input("OpenRouter API key", type="password", key="ui_api_key",
                              help="Kept in this browser session only. For deployment add OPENROUTER_API_KEY to Streamlit secrets. "
                                   "Create a key at openrouter.ai/keys.")
            elif has_secret:
                st.caption("OpenRouter API key loaded from secrets.")
            if st.session_state.use_ai and not config.get_secret("OPENROUTER_MODEL"):
                st.text_input("Model", key="ui_model",
                              help="Any OpenRouter model slug that supports tool calling, for example anthropic/claude-sonnet-5.5 "
                                   "or openai/gpt-5.2. See openrouter.ai/models.")
            st.divider()
            C.panel_title("Data")
            C.chips([(f"As of {db.as_of().isoformat()}", "outline"), *([("Synthetic", "warning")] if _is_synthetic() else [])])
            if st.button("Reset now", key="reset_db_btn", icon=":material/restart_alt:",
                         help="Deletes saved decisions and logged results and restores the original dataset."):
                db.reset_db()
                st.cache_data.clear()
                for k in ("explanations", "memo_text", "challenges"):
                    st.session_state[k] = {} if k == "explanations" else None
                st.toast("Database reset")
                st.rerun()


sidebar = chrome        # old name, kept for compatibility


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
        st.caption(f"{label} (use the menu in the header)")


def need_profile() -> bool:
    """Show an empty state and return True if no brief has been analysed yet."""
    if not st.session_state.get("profile"):
        C.empty_state("No brief analysed yet", "Upload or paste a product brief first. The tool turns it into a structured profile "
                      "that drives the filters, scoring and the campaign mix.", "file")
        page_link("pages/1_Product_Brief.py", label="Go to the product brief", icon=":material/arrow_forward:")
        return True
    return False


def fmt_eur(x: float) -> str:
    return f"EUR {x:,.0f}"
