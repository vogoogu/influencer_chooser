"""Plotly figures. Pure functions: data in, figure out."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import config

from .theme import PALETTE as COLORS  # noqa: E402  (data palette lives with the design system)

GOOD, BAD, WARN, BASE = "#2E9E5B", "#D2553F", "#E0932B", "#3F5BD0"      # semantic colours, independent of palette order
LABELS = config.CRITERIA_LABELS


def _label(r) -> str:
    return f"{r['name']} ({r['platform'][:2]})"


def contributions_bar(ranked: pd.DataFrame, n: int = 15) -> go.Figure:
    top = ranked.head(n).iloc[::-1]
    fig = go.Figure()
    for c, col in zip(config.CRITERIA, COLORS):
        fig.add_bar(y=[_label(r) for _, r in top.iterrows()], x=top[f"w_{c}"], name=LABELS[c], orientation="h",
                    marker_color=col, hovertemplate="%{y}<br>" + LABELS[c] + ": %{x:.3f}<extra></extra>")
    fig.update_layout(barmode="stack", height=max(320, 28 * len(top) + 120), xaxis_title="Weighted score",
                      legend=dict(orientation="h", y=-0.15), margin=dict(l=10, r=10, t=10, b=10))
    return fig


def radar(rows: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    cats = [LABELS[c] for c in config.CRITERIA]
    for (_, r), col in zip(rows.iterrows(), COLORS):
        vals = [float(r[c]) for c in config.CRITERIA]
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], name=_label(r), line_color=col, fill="toself", opacity=0.35))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], tickvals=[0.25, 0.5, 0.75, 1])), height=420,
                      margin=dict(l=40, r=40, t=20, b=20), legend=dict(orientation="h", y=-0.1))
    return fig


def stability_bar(ranked: pd.DataFrame, top_freq: pd.Series, n: int = 10) -> go.Figure:
    d = ranked.head(n).assign(freq=top_freq.reindex(ranked.head(n).index).values).iloc[::-1]
    fig = go.Figure(go.Bar(y=[_label(r) for _, r in d.iterrows()], x=d.freq, orientation="h", marker_color=COLORS[0],
                           hovertemplate="%{y}: in top 5 in %{x:.0%} of weight scenarios<extra></extra>"))
    fig.update_layout(xaxis=dict(range=[0, 1], tickformat=".0%", title="Share of weight scenarios in the top 5"), height=max(280, 28 * len(d) + 90),
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig


def sensitivity_bar(sens: pd.DataFrame) -> go.Figure:
    s = sens.assign(label=sens.criterion.map(LABELS)).iloc[::-1]
    fig = go.Figure()
    fig.add_bar(y=s.label, x=s["double"], orientation="h", name="Weight doubled", marker_color=WARN)
    fig.add_bar(y=s.label, x=s["zero"], orientation="h", name="Weight set to zero", marker_color=BAD)
    fig.update_layout(barmode="group", xaxis=dict(title="Creators in the top 5 that change", dtick=1), height=380,
                      legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=10, b=10))
    return fig


def reach_hist(sim_reach: np.ndarray, summary: dict, spend: float) -> go.Figure:
    fig = go.Figure(go.Histogram(x=sim_reach, nbinsx=50, marker_color=COLORS[0], opacity=0.85,
                                 hovertemplate="reach %{x:,.0f}<br>%{y} runs<extra></extra>"))
    for key, label, col in (("reach_p10", "P10", BAD), ("reach_p50", "P50", "#15191E"), ("reach_p90", "P90", GOOD)):
        fig.add_vline(x=summary[key], line_dash="dash", line_color=col, annotation_text=f"{label} {summary[key]:,.0f}", annotation_position="top")
    fig.add_vline(x=summary["naive_quote"], line_color=WARN, line_width=3,
                  annotation_text="Naive forecast", annotation_position="bottom right")
    fig.update_layout(xaxis_title="Total reach of the mix", yaxis_title="Simulation runs", height=360, showlegend=False,
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig


def conv_hist(sim_conv: np.ndarray, summary: dict) -> go.Figure:
    fig = go.Figure(go.Histogram(x=sim_conv, nbinsx=50, marker_color=COLORS[1], opacity=0.85))
    for key, label in (("conv_p10", "P10"), ("conv_p50", "P50"), ("conv_p90", "P90")):
        fig.add_vline(x=summary[key], line_dash="dash", annotation_text=f"{label} {summary[key]:,.0f}", annotation_position="top")
    fig.update_layout(xaxis_title="Total conversions", yaxis_title="Simulation runs", height=300, showlegend=False,
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig


def tornado_chart(df: pd.DataFrame, metric_label: str, fmt: str = ",.0f") -> go.Figure:
    base = float(df.base.iloc[0])
    fig = go.Figure()
    fig.add_bar(y=df.assumption, x=df.low - base, base=base, orientation="h", name="Pessimistic setting", marker_color=BAD)
    fig.add_bar(y=df.assumption, x=df.high - base, base=base, orientation="h", name="Optimistic setting", marker_color=GOOD)
    fig.add_vline(x=base, line_color="#333")
    fig.update_layout(barmode="overlay", xaxis_title=metric_label, height=300, legend=dict(orientation="h", y=-0.25),
                      margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(tickformat=fmt)
    return fig


def calibration_scatter(hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for tier, col in zip(["nano", "micro", "mid", "macro", "mega"], COLORS):
        d = hist[hist.tier == tier]
        if len(d):
            fig.add_trace(go.Scatter(x=d.reach_p50, y=d.actual_reach, mode="markers", name=tier, marker=dict(color=col, size=7, opacity=0.7),
                                     hovertemplate="predicted %{x:,.0f}<br>actual %{y:,.0f}<extra>" + tier + "</extra>"))
    lo, hi = max(hist.reach_p50.min(), 1), hist.reach_p50.max()
    xs = np.array([lo, hi])
    fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", line=dict(color="#333", dash="dash"), name="perfect forecast"))
    fig.update_xaxes(type="log", title="Predicted reach (p50)")
    fig.update_yaxes(type="log", title="Actual reach")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2))
    return fig


def bias_bar(cov: pd.DataFrame) -> go.Figure:
    colors = [BAD if v < 0 else GOOD for v in cov.bias_pct]
    fig = go.Figure(go.Bar(x=cov.tier, y=cov.bias_pct, marker_color=colors, text=[f"{v:+.0f}%" for v in cov.bias_pct], textposition="outside"))
    fig.update_layout(yaxis_title="Actual vs predicted reach (%)", height=320, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def coverage_bar(cov: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=cov.tier, y=cov.inside_p10_p90, marker_color=COLORS[0], text=[f"{v:.0%}" for v in cov.inside_p10_p90], textposition="outside"))
    fig.add_hline(y=0.8, line_dash="dash", annotation_text="nominal 80%")
    fig.update_layout(yaxis=dict(range=[0, 1.05], tickformat=".0%", title="Actual inside predicted p10-p90"), height=320,
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig


def driver_bar(df: pd.DataFrame) -> go.Figure:
    names = {"niche_fit": "Niche fit", "engagement_vs_peers": "Engagement vs peers", "fake_followers_pct": "Fake followers %",
             "relationship_score": "Relationship score", "log_followers": "Audience size (log)"}
    d = df.assign(label=df.feature.map(names)).iloc[::-1]
    fig = go.Figure(go.Bar(y=d.label, x=d.coef, orientation="h", marker_color=[GOOD if v > 0 else BAD for v in d.coef],
                           error_x=dict(type="data", symmetric=False, array=(d.hi - d.coef).clip(lower=0), arrayminus=(d.coef - d.lo).clip(lower=0))))
    fig.add_vline(x=0, line_color="#333")
    fig.update_layout(xaxis_title="Effect of +1 standard deviation (log scale outcome), 90% interval", height=320, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def trend_lines(snap: pd.DataFrame, platform: str) -> go.Figure:
    from plotly.subplots import make_subplots
    d = snap.sort_values("captured_at")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=d.captured_at, y=d.followers, name="Followers", line_color=COLORS[0]), secondary_y=False)
    fig.add_trace(go.Scatter(x=d.captured_at, y=d.engagement_rate, name="Engagement %", line_color=COLORS[1]), secondary_y=True)
    fig.add_trace(go.Scatter(x=d.captured_at, y=d.fake_follower_pct, name="Fake followers %", line=dict(color=COLORS[3], dash="dot")), secondary_y=True)
    fig.update_yaxes(title_text="Followers", secondary_y=False)
    fig.update_yaxes(title_text="Percent", secondary_y=True)
    fig.update_layout(height=320, title=platform, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
    return fig


# ============================================================================ random forest explanation
def tree_figure(nodes: list[dict], edges: list[tuple]) -> go.Figure:
    """A shallow decision tree as a diagram. Leaves show the predicted result."""
    fig = go.Figure()
    for x0, y0, x1, y1, label in edges:
        fig.add_shape(type="line", x0=x0, y0=y0 - 0.12, x1=x1, y1=y1 + 0.12, line=dict(color="#999", width=1.2))
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=label, showarrow=False, font=dict(size=10, color="#777"), bgcolor="rgba(255,255,255,0.7)")
    vals = [n["value"] for n in nodes if n["leaf"]]
    lo, hi = (min(vals), max(vals)) if vals else (0, 1)
    for n in nodes:
        if n["leaf"]:
            t = 0.5 if hi == lo else (n["value"] - lo) / (hi - lo)
            bg = f"rgba({int(228 - 120 * t)},{int(120 + 90 * t)},{int(110 + 10 * t)},0.35)"       # low = red-ish, high = green-ish
            border = GOOD if t > 0.5 else BAD
        else:
            bg, border = "rgba(76,120,168,0.15)", "#4C78A8"
        fig.add_annotation(x=n["x"], y=n["y"], text=n["text"], showarrow=False, align="center", font=dict(size=11),
                           bgcolor=bg, bordercolor=border, borderwidth=1.5, borderpad=5)
    xs = [n["x"] for n in nodes]
    fig.update_xaxes(visible=False, range=[min(xs) - 0.7, max(xs) + 0.7])
    fig.update_yaxes(visible=False, range=[min(n["y"] for n in nodes) - 0.5, 0.5])
    fig.update_layout(height=430, margin=dict(l=5, r=5, t=5, b=5), plot_bgcolor="white")
    return fig


def votes_hist(votes: np.ndarray, mu: float, lo: float, hi: float, target: str) -> go.Figure:
    """Every tree's prediction for one creator, as multipliers (reach) or rate per 1,000 (conversions)."""
    v = np.exp(votes)
    fig = go.Figure(go.Histogram(x=v, nbinsx=30, marker_color=COLORS[0], opacity=0.8,
                                 hovertemplate="%{x:.2f}: %{y} trees<extra></extra>"))
    unit = "x" if target == "reach" else ""
    fig.add_vline(x=float(np.exp(mu)), line_color="#222", line_width=3, annotation_text=f"forest average {np.exp(mu):.2f}", annotation_position="top")
    fig.add_vrect(x0=float(np.exp(lo)), x1=float(np.exp(hi)), fillcolor=GOOD, opacity=0.12, line_width=0,
                  annotation_text="80% range", annotation_position="bottom left")
    fig.update_layout(xaxis_title=("Reach multiplier vs naive forecast" if target == "reach" else "Conversions per 1,000 reached"),
                      yaxis_title="Number of trees", height=320, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def importance_bar(imp: pd.DataFrame, n: int = 10) -> go.Figure:
    d = imp.head(n).iloc[::-1]
    fig = go.Figure(go.Bar(y=d.label, x=d.importance.clip(lower=0), orientation="h", marker_color=COLORS[0],
                           error_x=dict(type="data", array=d.spread, color="#999")))
    fig.update_layout(xaxis_title="Error that increases when this feature is scrambled (higher = more important)", height=max(300, 30 * len(d) + 80),
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig


def oof_scatter(oof: pd.DataFrame, target: str) -> go.Figure:
    """Out-of-fold predictions vs actual outcomes, shown as multipliers / rates."""
    fig = go.Figure()
    for tier, col in zip(["nano", "micro", "mid", "macro", "mega"], COLORS):
        d = oof[oof.tier == tier]
        if len(d):
            fig.add_trace(go.Scatter(x=np.exp(d.predicted), y=np.exp(d.actual), mode="markers", name=tier,
                                     marker=dict(color=col, size=6, opacity=0.7),
                                     hovertemplate="predicted %{x:.2f}<br>actual %{y:.2f}<extra>" + tier + "</extra>"))
    lo, hi = float(np.exp(min(oof.predicted.min(), oof.actual.min()))), float(np.exp(max(oof.predicted.max(), oof.actual.max())))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#333", dash="dash"), name="perfect"))
    label = "multiplier" if target == "reach" else "conversions per 1,000"
    fig.update_xaxes(type="log", title=f"Predicted {label} (model never saw this campaign)")
    fig.update_yaxes(type="log", title=f"Actual {label}")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2))
    return fig


def metrics_bar(metrics: pd.DataFrame) -> go.Figure:
    colors = ["#A9B0BC" if k == "baseline" else COLORS[0] for k in metrics.kind]
    fig = go.Figure(go.Bar(x=metrics.rmse, y=metrics.method, orientation="h", marker_color=colors,
                           text=[f"{v:.2f}" for v in metrics.rmse], textposition="outside"))
    fig.update_layout(xaxis_title="Typical forecast error on unseen campaigns (RMSE of the log multiplier, lower is better)",
                      yaxis=dict(autorange="reversed"), height=240, margin=dict(l=10, r=40, t=10, b=10))
    return fig


def pdp_line(pdp: pd.DataFrame, feature: str, target: str, display) -> go.Figure:
    y = np.exp(pdp.prediction)
    fig = go.Figure(go.Scatter(x=pdp.value, y=y, mode="lines+markers", line_color=COLORS[0]))
    step = max(len(pdp) // 6, 1)
    fig.update_xaxes(tickvals=pdp.value.iloc[::step], ticktext=[display(feature, v) for v in pdp.value.iloc[::step]])
    fig.add_hline(y=1.0 if target == "reach" else float(y.mean()), line_dash="dot", line_color="#999")
    fig.update_layout(yaxis_title="Average predicted " + ("reach multiplier" if target == "reach" else "conversions per 1,000"),
                      height=300, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def effects_bar(effects: pd.DataFrame, n: int = 8) -> go.Figure:
    d = effects.head(n).iloc[::-1]
    fig = go.Figure(go.Bar(y=d.label, x=np.exp(d.effect) - 1, orientation="h",
                           marker_color=[GOOD if v > 0 else BAD for v in d.effect],
                           hovertemplate="%{y}: %{x:+.0%} vs a typical creator<extra></extra>"))
    fig.add_vline(x=0, line_color="#333")
    fig.update_layout(xaxis=dict(tickformat="+.0%", title="Effect on the forecast versus a typical creator's value"), height=max(280, 30 * len(d) + 80),
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig


# ---- every chart gets an explicit white canvas: figure-level layout always beats frontend or template defaults, so charts
# look identical inside the white cards regardless of Streamlit's plotly handling.
def _white_canvas(fn):
    import functools

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        fig = fn(*args, **kwargs)
        fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF")
        return fig
    return inner


for _name in ["contributions_bar", "radar", "stability_bar", "sensitivity_bar", "reach_hist", "conv_hist", "tornado_chart",
              "calibration_scatter", "bias_bar", "coverage_bar", "driver_bar", "trend_lines", "tree_figure", "votes_hist",
              "importance_bar", "oof_scatter", "metrics_bar", "pdp_line", "effects_bar"]:
    globals()[_name] = _white_canvas(globals()[_name])
