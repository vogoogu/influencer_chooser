"""One formatter for every plain data table in the app.

Why: raw dataframes show database names (`price_eur`), bare numbers (`15000`), ISO timestamps, `None`, `1`/`0` and snake_case
values. `format_table` turns a dataframe into display data plus a `column_config`, so every table reads the same way:

* headers are sentence case with units ("Fee (EUR)", "Usage rights (months)")
* numbers use thousands separators, percentages and multipliers say so, scores get a bar
* dates are real dates, 1/0 flags are Yes/No, empty cells show a dash, snake_case values become words

Rules are by column name, so most calls need no arguments. Pass overrides when a name is ambiguous.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import streamlit as st

LABELS = {
    "id": "#", "rank": "#", "name": "Creator", "display_name": "Creator", "creator": "Creator", "platform": "Platform", "handle": "Handle",
    "url": "Profile link", "verified": "Verified", "deliverable": "Format", "format": "Format", "tier": "Tier", "country": "Country",
    "followers": "Followers", "er": "Engagement", "engagement_rate": "Engagement", "fake": "Fake followers", "cpm": "EUR per 1,000 reach",
    "price": "Fee (EUR)", "price_eur": "Fee (EUR)", "fee_eur": "Fee (EUR)", "budget_eur": "Budget (EUR)", "budget": "Budget (EUR)",
    "reach_p10": "Reach p10", "reach_p50": "Reach p50", "reach_p90": "Reach p90", "actual_reach": "Actual reach", "conversions": "Conversions",
    "exp_conversions": "Exp. conversions", "delivered_on_time": "On time", "client_rating": "Client rating",
    "actual_vs_predicted": "Actual vs forecast", "avg_actual_vs_predicted": "Actual vs forecast", "on_time_rate": "On time",
    "avg_client_rating": "Avg rating", "n_campaigns": "Campaigns", "usage_rights_months": "Usage rights (months)",
    "exclusivity_premium_pct": "Exclusivity premium", "valid_from": "Valid from", "valid_to": "Valid to", "expired": "Expired",
    "flag_type": "Flag", "severity": "Severity", "source": "Source", "evidence_url": "Evidence", "flagged_at": "Flagged", "expires_at": "Expires",
    "brand": "Brand", "category": "Category", "campaign": "Campaign", "campaign_id": "Campaign", "total_score": "Score", "total": "Score",
    "created_at": "Saved", "title": "Decision", "objective": "Objective", "creators": "Creators", "status": "Status", "posted_at": "Posted",
    "caption": "Caption", "views": "Views", "likes": "Likes", "comments": "Comments", "is_sponsored": "Sponsored", "is_disclosed": "Ad label",
    "share": "Share", "criterion": "Criterion", "weight": "Weight", "exclude_reason": "Reason", "female_pct": "Women", "age_25_34": "Age 25-34",
    "relationship_score": "Relationship (1-5)", "selected": "Chosen", "is_primary": "Primary", "consent_until": "Consent until",
    "captured_at": "Captured", "results": "Results", "bias multiplier": "Bias multiplier", "rmse": "Typical error (RMSE)", "mae": "Average error (MAE)",
    "coverage": "80% range hit rate", "method": "Method", "rmse_gain_vs_best_baseline": "Better than best baseline", "pool": "Pool",
}
MONEY = {"price", "price_eur", "fee_eur", "budget", "budget_eur"}
PCT_FORMATS = {"er": "%.2f%%", "engagement_rate": "%.2f%%", "fake": "%.0f%%", "female_pct": "%.0f%%", "age_25_34": "%.0f%%", "bias_pct": "%+.0f%%"}
RATIO = {"weight", "share", "coverage", "on_time_rate", "inside_p10_p90", "rmse_gain_vs_best_baseline"}
MULT = {"actual_vs_predicted", "avg_actual_vs_predicted", "ratio", "bias multiplier"}
SCORE = {"total", "total_score", "score"}
YESNO = {"verified", "expired", "delivered_on_time", "is_sponsored", "is_disclosed", "selected", "is_primary", "has_category_history"}
NO_HUMANIZE = {"handle", "url", "evidence_url", "hashtags", "caption", "bio_text", "agency_notes", "title", "name", "display_name", "creator",
               "campaign", "brand", "notes", "exclude_reason", "method", "criterion"}
BRAND_CASE = {"tiktok": "TikTok", "instagram": "Instagram", "youtube": "YouTube"}
SMALL = {"id", "rank", "verified", "expired", "delivered_on_time", "is_sponsored", "is_disclosed", "selected", "is_primary", "tier", "country",
         "platform", "severity", "client_rating", "campaign_id", "creators", "n_campaigns", "relationship_score"}
LARGE = {"caption", "title", "exclude_reason", "url", "evidence_url", "campaign", "method"}
_SNAKE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)+$")


def label_for(col: str) -> str:
    if col in LABELS:
        return LABELS[col]
    words = str(col).replace("_", " ").strip()
    words = re.sub(r"\beur\b", "EUR", words)
    words = re.sub(r"\bpct\b", "%", words)
    return words[:1].upper() + words[1:]


def _is_date_name(col: str) -> bool:
    return col.endswith("_at") or col.endswith("_date") or col in {"valid_from", "valid_to", "consent_until", "start_date", "end_date"}


def _yesno(s: pd.Series) -> pd.Series:
    def f(v):
        if pd.isna(v):
            return "–"
        return "Yes" if bool(v) and str(v).lower() not in ("0", "false", "no") else "No"
    return s.map(f)


def format_table(df: pd.DataFrame, *, labels: dict | None = None, hide: tuple | list = (), order: list | None = None,
                 money: tuple | list = (), pct: dict | None = None, ratio: tuple | list = (), mult: tuple | list = (),
                 score: tuple | list = (), yesno: tuple | list = (), dates: tuple | list = (), count: tuple | list = (),
                 formats: dict | None = None, widths: dict | None = None, links: tuple | list = ("url", "evidence_url")) -> tuple[pd.DataFrame, dict]:
    """Return (display dataframe, column_config). The input is never modified."""
    d = df.drop(columns=[c for c in hide if c in df.columns]).copy()
    if order:
        d = d[[c for c in order if c in d.columns] + [c for c in d.columns if c not in order]]
    labels = {**LABELS, **(labels or {})}
    money_cols, ratio_cols, mult_cols, score_cols = set(MONEY) | set(money), set(RATIO) | set(ratio), set(MULT) | set(mult), set(SCORE) | set(score)
    yes_cols, pct_cols = set(YESNO) | set(yesno), {**PCT_FORMATS, **(pct or {})}
    cfg: dict = {}
    for col in list(d.columns):
        lab = labels.get(col) or label_for(col)
        s = d[col]
        width = (widths or {}).get(col) or ("small" if col in SMALL else ("large" if col in LARGE else None))
        kw = {"width": width} if width else {}
        if col in links:
            cfg[col] = st.column_config.LinkColumn(lab, **kw)
        elif col in yes_cols or s.dtype == bool:
            d[col] = _yesno(s)
            cfg[col] = st.column_config.TextColumn(lab, width="small")
        elif col in dates or _is_date_name(col):
            parsed = pd.to_datetime(s, errors="coerce")
            d[col] = parsed
            has_time = parsed.notna().any() and bool((parsed.dropna().dt.time != pd.Timestamp(0).time()).any())
            cfg[col] = (st.column_config.DatetimeColumn(lab, format="YYYY-MM-DD HH:mm", **kw) if has_time
                        else st.column_config.DateColumn(lab, format="YYYY-MM-DD", **kw))
        elif pd.api.types.is_numeric_dtype(s):
            if formats and col in formats:
                cfg[col] = st.column_config.NumberColumn(lab if col not in money_cols or "EUR" in lab else f"{lab} (EUR)", format=formats[col], **kw)
            elif col in score_cols:
                cfg[col] = st.column_config.ProgressColumn(lab, min_value=0.0, max_value=1.0, format="%.2f", **kw)
            elif col in ratio_cols:                       # 0-1 shares shown as whole percentages
                d[col] = s * 100
                cfg[col] = st.column_config.NumberColumn(lab, format="%.0f%%", **kw)
            elif col in mult_cols:
                cfg[col] = st.column_config.NumberColumn(lab, format="%.2f×", **kw)
            elif col in pct_cols or col.endswith("_pct"):
                cfg[col] = st.column_config.NumberColumn(lab, format=pct_cols.get(col, "%.0f%%"), **kw)
            elif col in money_cols:
                cfg[col] = st.column_config.NumberColumn(lab if "EUR" in lab else f"{lab} (EUR)", format="localized", **kw)
            else:
                vals = s.dropna()
                whole = len(vals) > 0 and bool((vals % 1 == 0).all()) if pd.api.types.is_float_dtype(s) else True
                if col in count or whole or pd.api.types.is_integer_dtype(s):
                    cfg[col] = st.column_config.NumberColumn(lab, format="localized" if col not in ("id", "rank", "severity") else "%d", **kw)
                else:
                    cfg[col] = st.column_config.NumberColumn(lab, format="%.2f", **kw)
        else:
            if col == "platform":
                d[col] = s.map(lambda v: BRAND_CASE.get(v, v) if isinstance(v, str) else v)
            elif col not in NO_HUMANIZE:                      # per value: "qualified_reach" -> "qualified reach", "score" stays
                d[col] = s.map(lambda v: v.replace("_", " ") if isinstance(v, str) and _SNAKE.match(v) else v)
            d[col] = d[col].where(d[col].notna(), "–").replace({"None": "–", "": "–"})
            cfg[col] = st.column_config.TextColumn(lab, **kw)
    return d, cfg


def show(df: pd.DataFrame, *, height: int | None = None, key: str | None = None, **kw) -> None:
    """Render a formatted table (see format_table for the options)."""
    d, cfg = format_table(df, **kw)
    st.dataframe(d, column_config=cfg, hide_index=True, width="stretch", height=height if height else "auto", key=key)
