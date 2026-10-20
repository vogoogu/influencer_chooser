"""Reusable UI building blocks (HTML via st.html) that complement the restyled native widgets.

All text passed in is HTML-escaped. Functions named `*_html` return strings, the others render directly.
Native Streamlit widgets (inputs, buttons, tables, metrics) stay native so behaviour, accessibility and tests are unchanged.
"""
from __future__ import annotations

from html import escape

import streamlit as st

def icon(name: str, size: int = 16) -> str:
    """Inline icon rendered as a CSS mask (st.html strips inline <svg>). Colour follows the text colour."""
    return f'<span class="ic-ico ic-ico-{escape(name)}" style="width:{size}px;height:{size}px" aria-hidden="true"></span>'


def _render(html: str) -> None:
    st.html(html)


# ----------------------------------------------------------------------------- header and workflow
WORKFLOW = [("Brief", "Product brief"), ("Shortlist", "Shortlist"), ("Mix", "Campaign mix"), ("Log", "Decision log"), ("Calibrate", "Calibration")]


def page_header(title: str, subtitle: str = "", eyebrow: str | None = None) -> None:
    eb = f'<div class="ic-eyebrow">{escape(eyebrow)}</div>' if eyebrow else ""
    sub = f'<p class="ic-subtitle">{escape(subtitle)}</p>' if subtitle else ""
    _render(f'<div class="ic-header">{eb}<h1 class="ic-title">{escape(title)}</h1>{sub}</div>')


def stepper_html(current: int, done: set[int]) -> str:
    """current: 1-based index of the page being viewed; done: indices of completed steps."""
    parts = []
    for i, (short, full) in enumerate(WORKFLOW, start=1):
        state = "current" if i == current else ("done" if i in done else "")
        dot = icon("check", 13) if (i in done and i != current) else str(i)
        parts.append(f'<div class="ic-step {state}" {"aria-current=step" if i == current else ""}><span class="ic-step-dot">{dot}</span>'
                     f'<span class="ic-step-label">{escape(full)}</span></div>')
        if i < len(WORKFLOW):
            parts.append(f'<div class="ic-step-line {"done" if i in done and i != current else ""}"></div>')
    return f'<nav class="ic-stepper" aria-label="Workflow progress">{"".join(parts)}</nav>'


def stepper(current: int, done: set[int]) -> None:
    _render(stepper_html(current, done))


# ----------------------------------------------------------------------------- small elements
def chip_html(text: str, tone: str = "neutral", icon_name: str | None = None) -> str:
    ic = icon(icon_name, 12) if icon_name else ""
    return f'<span class="ic-chip {escape(tone)}">{ic}{escape(str(text))}</span>'


def chips(items: list[tuple[str, str] | str], label: str | None = None, tone: str = "outline") -> None:
    """Render a row of chips. Items are text or (text, tone)."""
    html = "".join(chip_html(*(it if isinstance(it, tuple) else (it, tone))) for it in items)
    lab = f'<span class="ic-chips-label">{escape(label)}</span>' if label else ""
    _render(f'<div class="ic-chips">{lab}{html}</div>')


def chips_html(items: list[tuple[str, str] | str], tone: str = "outline") -> str:
    return '<div class="ic-chips">' + "".join(chip_html(*(it if isinstance(it, tuple) else (it, tone))) for it in items) + "</div>"


def section(title: str, desc: str = "") -> None:
    d = f"<p>{escape(desc)}</p>" if desc else ""
    _render(f'<div class="ic-section"><h2>{escape(title)}</h2>{d}</div>')


def step_header(n: int, title: str, desc: str = "") -> None:
    d = f"<p>{escape(desc)}</p>" if desc else ""
    _render(f'<div class="ic-stephead"><div class="ic-stephead-n">{n}</div><div><h2>{escape(title)}</h2>{d}</div></div>')


def empty_state(title: str, text: str, icon_name: str = "inbox") -> None:
    _render(f'<div class="ic-empty"><div class="ic-empty-icon">{icon(icon_name, 22)}</div><h3>{escape(title)}</h3><p>{escape(text)}</p></div>')


def kv_html(rows: list[tuple[str, str]]) -> str:
    return '<dl class="ic-kv">' + "".join(f"<dt>{escape(k)}</dt><dd>{escape(str(v))}</dd>" for k, v in rows) + "</dl>"


def hero_title(text: str) -> None:
    """Large card title without the anchor-link icon that markdown headings get."""
    _render(f'<div class="ic-hero-title">{escape(text)}</div>')


def panel_title(text: str) -> None:
    _render(f'<div class="ic-panel-title">{escape(text)}</div>')


def initials(name: str) -> str:
    parts = [p for p in name.replace("-", " ").split() if p]
    return (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper() if parts else "?"


def profile_header(name: str, meta: str, chips_markup: str = "") -> None:
    _render(f'<div class="ic-profile"><div class="ic-avatar">{escape(initials(name))}</div>'
            f'<div><h2>{escape(name)}</h2><p>{escape(meta)}</p>{chips_markup}</div></div>')


def health_card_html(value: str, label: str, help_text: str, tone: str = "neutral", share: float | None = None) -> str:
    bar = f'<div class="ic-bar"><span style="width:{max(0, min(share, 1)) * 100:.0f}%"></span></div>' if share is not None else ""
    return (f'<div class="ic-health-card {escape(tone)}"><div class="ic-health-val">{escape(str(value))}</div>'
            f'<div class="ic-health-label">{escape(label)}</div><div class="ic-health-help">{escape(help_text)}</div>{bar}</div>')


def health_grid(cards: list[str]) -> None:
    _render(f'<div class="ic-health">{"".join(cards)}</div>')


def tile_html(icon_name: str, title: str, text: str) -> str:
    return f'<div class="ic-tile"><div class="ic-tile-icon">{icon(icon_name, 18)}</div><h3>{escape(title)}</h3><p>{escape(text)}</p></div>'


def tiles(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (ic, title, text) in zip(cols, items):
        col.html(tile_html(ic, title, text))


def bullet_card(title: str, items: list[str], kind: str = "good") -> str:
    li = "".join(f"<li>{escape(i)}</li>" for i in items) or "<li>None identified</li>"
    return f'<div class="ic-card-title">{escape(title)}</div><ul class="ic-list {kind}">{li}</ul>'
