"""Automatic risk checks ("devil's advocate", rule-based part) and the decision memo exports."""
from __future__ import annotations

import io
import re
from datetime import date

import pandas as pd

from . import config

SEV_ORDER = {"high": 0, "medium": 1, "low": 2}


def risk_checks(mix: pd.DataFrame, profile: dict, budget: float, sim: dict | None = None) -> list[dict]:
    """Deterministic checks on the chosen mix. Each item: title, severity, detail, source."""
    out: list[dict] = []

    def add(title, severity, detail):
        out.append({"title": title, "severity": severity, "detail": detail, "source": "rule"})

    if mix.empty:
        return [{"title": "No creators selected", "severity": "high", "detail": "The optimizer found no feasible mix. Relax constraints or widen the pool.", "source": "rule"}]
    spend = float(mix.price.sum())
    if len(mix) < 3:
        add("Very few creators", "high", f"Only {len(mix)} creator(s): one weak delivery changes the whole campaign.")
    top_share = float(mix.price.max() / spend) if spend else 0
    if top_share > 0.35:
        r = mix.loc[mix.price.idxmax()]
        add("Budget concentration", "medium", f"{r['name']} takes {top_share:.0%} of the spend.")
    reach_share = float(mix.reach_p50.max() / mix.reach_p50.sum()) if mix.reach_p50.sum() else 0
    if reach_share > 0.5:
        add("Reach depends on one creator", "medium", f"{reach_share:.0%} of expected reach comes from a single creator.")
    plat = mix.groupby("platform").price.sum() / spend if spend else pd.Series(dtype=float)
    if len(plat) and plat.max() > 0.8:
        add("Single-platform dependence", "low", f"{plat.idxmax()} carries {plat.max():.0%} of spend; an algorithm change hits everything at once.")
    if mix.primary_niche.nunique() == 1 and len(mix) > 2:
        add("No niche diversification", "low", f"All creators are in {mix.primary_niche.iloc[0].replace('_', ' ')}: audiences may overlap heavily.")
    warn = mix[mix.data_warnings.map(len) > 0]
    for _, r in warn.iterrows():
        spike = any("spike" in w or "fake" in w for w in r.data_warnings)
        add(f"Data quality: {r['name']} ({r.platform})", "high" if spike else "medium", "; ".join(r.data_warnings) + ". Verify the numbers before signing.")
    flagged = mix[mix.safety_flags.map(len) > 0]
    for _, r in flagged.iterrows():
        add(f"Safety flag: {r['name']}", "medium", ", ".join(f"{t.replace('_', ' ')} (severity {s})" for t, s in r.safety_flags))
    conf = mix[mix.conflicts.map(len) > 0]
    for _, r in conf.iterrows():
        add(f"Brand conflict: {r['name']}", "high", "; ".join(r.conflicts))
    nocat = (~mix.has_category_history).mean() if "has_category_history" in mix else 0
    if nocat > 0.6:
        add("Little category experience", "low", f"{nocat:.0%} of the mix has no past campaign in this product category.")
    if sim and sim.get("p_below_quote", 0) > 0.6:
        add("Naive forecast looks optimistic", "medium",
            f"A plain average-views forecast would be {sim['naive_quote']:,.0f} reach, but the corrected simulation falls below it in "
            f"{sim['p_below_quote']:.0%} of runs. Quote the client a range, not the naive number.")
    if budget and spend < 0.7 * budget:
        add("Budget not used", "low", f"Mix uses {spend / budget:.0%} of the budget. Consider more creators or higher-reach formats.")
    return sorted(out, key=lambda x: SEV_ORDER[x["severity"]])


def memo_markdown(title: str, profile: dict, mix: pd.DataFrame, summary: dict, sim: dict, checks: list[dict],
                  as_of: date, weights: dict, backend: str, forecast_note: str = "bias-corrected from past campaign results (tier correction)") -> str:
    L = [f"# Decision memo: {title}", "",
         f"*Prepared {as_of.isoformat()}. Product: {profile.get('product_name', '')}. Goal: {profile.get('goal', '')}. "
         f"Budget: EUR {summary.get('budget', 0):,.0f}.*", "", "## Recommendation", ""]
    if mix.empty:
        L.append("No feasible mix found under the current constraints.")
    else:
        L += ["| Creator | Platform | Format | Fee (EUR) | Reach p10 / p50 / p90 | Score |", "|---|---|---|---:|---|---:|"]
        for _, r in mix.iterrows():
            L.append(f"| {r['name']} | {r.platform} | {r.deliverable} | {r.price:,.0f} | {r.reach_p10:,.0f} / {r.reach_p50:,.0f} / {r.reach_p90:,.0f} | {r.total:.2f} |")
        L += ["", f"Total spend EUR {mix.price.sum():,.0f} for {len(mix)} creators.", "", "## Expected results", "",
              f"- Reach: {sim['reach_p10']:,.0f} to {sim['reach_p90']:,.0f} (median {sim['reach_p50']:,.0f}).",
              f"- Conversions ({profile.get('goal', '')}): {sim['conv_p10']:,.0f} to {sim['conv_p90']:,.0f} (median {sim['conv_p50']:,.0f}).",
              f"- Median cost per 1,000 reach: EUR {sim['cpm_p50']:.1f}.",
              f"- Probability of falling short of the naive average-views forecast ({sim['naive_quote']:,.0f}): {sim['p_below_quote']:.0%}."]
    L += ["", "## Risks to review", ""]
    L += [f"- **{c['severity'].upper()}** {c['title']}: {c['detail']}" for c in checks] or ["- No automatic risk flags."]
    w = {k: v for k, v in weights.items()}
    L += ["", "## How this was decided", "",
          "Hard filters removed unavailable, conflicted or non-compliant creators. The rest were scored on seven weighted criteria: "
          + ", ".join(f"{config.CRITERIA_LABELS[k].lower()} {w[k]:.0%}" for k in config.CRITERIA) + ". "
          f"Topic similarity used {backend}. Reach forecasts are {forecast_note}; "
          "the mix was chosen by integer programming under the budget and portfolio constraints. "
          "Numbers come from deterministic code and the agency database; AI was used for reading the brief and wording."]
    return "\n".join(L)


def memo_html(markdown_text: str, title: str) -> str:
    import markdown as md
    body = md.markdown(markdown_text, extensions=["tables"])
    css = ("body{font-family:Helvetica,Arial,sans-serif;max-width:820px;margin:32px auto;padding:0 20px;color:#222;line-height:1.5}"
           "table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ccc;padding:6px 8px;text-align:left}"
           "th{background:#f3f3f3}h1{font-size:24px}h2{font-size:18px;margin-top:24px}@media print{body{margin:0}}")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title><style>{css}</style></head><body>{body}</body></html>"


def _latin(s: str) -> str:
    s = (s.replace("€", "EUR").replace("–", "-").replace("—", "-").replace("’", "'").replace("“", '"').replace("”", '"')
          .replace("≥", ">=").replace("≤", "<="))
    s = re.sub(r"\*\*|__|`", "", s)
    return s.encode("latin-1", "replace").decode("latin-1")


def memo_pdf(markdown_text: str, title: str) -> bytes:
    """Render the memo markdown as a simple one-or-two page PDF (core fonts, latin-1)."""
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    w = pdf.w - 30
    rows: list[list[str]] = []

    def flush_table():
        nonlocal rows
        if not rows:
            return
        n = len(rows[0])
        widths = [w * f for f in ([0.24, 0.12, 0.10, 0.12, 0.32, 0.10] if n == 6 else [1 / n] * n)]
        for i, row in enumerate(rows):
            pdf.set_font("Helvetica", "B" if i == 0 else "", 8)
            h = 6
            for cell, cw in zip(row, widths):
                pdf.cell(cw, h, _latin(cell)[:48], border=1, fill=(i == 0))
            pdf.ln(h)
        pdf.ln(2)
        rows = []

    pdf.set_fill_color(240, 240, 240)
    for line in markdown_text.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            rows.append(cells)
            continue
        flush_table()
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.multi_cell(w, 8, _latin(line[2:]), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif line.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(w, 7, _latin(line[3:]), new_x="LMARGIN", new_y="NEXT")
        elif line.strip():
            pdf.set_font("Helvetica", "", 9.5)
            txt = _latin(line.lstrip("*-").strip() if line.startswith(("- ", "* ")) else line)
            pdf.multi_cell(w, 5, ("  - " + txt) if line.startswith(("- ", "* ")) else txt, new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.ln(1.5)
    flush_table()
    return bytes(pdf.output())
