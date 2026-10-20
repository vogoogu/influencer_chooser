"""Design system for Influencer Chooser.

One place for the visual language: colour, type, spacing, radii and elevation tokens, plus a CSS layer that restyles
Streamlit's native widgets (so every page stays consistent without touching the widgets' behaviour) and the
`ic-*` classes used by core/components.py.

Principles
* Warm paper canvas, white working surfaces, one confident green for actions, semantic colours only for status.
* Numbers are first-class: tabular figures, generous KPI cards, no truncated labels.
* Every interactive element has hover, focus-visible, active and disabled states.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

TOKENS = {
    "ink": "#15191E", "ink2": "#3A424D", "muted": "#667085", "faint": "#98A2B3",
    "canvas": "#F5F4F0", "surface": "#FFFFFF", "surface2": "#FAF9F6", "line": "#E6E3DC", "line_strong": "#D3CFC5",
    "primary": "#165B4C", "primary_hover": "#0F4A3D", "primary_soft": "#E3F1EC", "primary_ring": "rgba(22,91,76,0.28)",
    "accent": "#F2B84B",
    "success": "#17803D", "success_bg": "#E6F4EA", "warning": "#8F5B00", "warning_bg": "#FFF3D6",
    "danger": "#B42318", "danger_bg": "#FDE8E6", "info": "#1D4ED8", "info_bg": "#E8EEFD",
    "neutral": "#475467", "neutral_bg": "#EEF0F3",
}
# Data palette (colour-blind safe ordering): used by charts and the seven scoring criteria
PALETTE = ["#1B7F6B", "#3F5BD0", "#E0932B", "#D2553F", "#7A5AA6", "#2A9DB5", "#8A8F98"]
FONT = "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"


ICONS = {
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "alert": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    "inbox": '<polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="8" y1="13" x2="16" y2="13"/><line x1="8" y1="17" x2="14" y2="17"/>',
    "list": '<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>',
    "sliders": '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
    "history": '<path d="M3 3v5h5"/><path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"/><polyline points="12 7 12 12 15 14"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "sparkles": '<path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3z"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/>',
    "users": '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "refresh": '<polyline points="23 4 23 10 17 10"/><path d="M20.5 15a9 9 0 1 1-2.1-9.4L23 10"/>',
}


def icon_css() -> str:
    """CSS classes `ic-ico-<name>`: the glyph is a mask over currentColor, so it inherits the text colour."""
    from urllib.parse import quote
    rules = []
    for name, body in ICONS.items():
        svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' "
               f"stroke-linecap='round' stroke-linejoin='round'>{body.replace(chr(34), chr(39))}</svg>")
        uri = "url(\"data:image/svg+xml," + quote(svg, safe="/:=' ") + "\")"
        rules.append(f".ic-ico-{name}{{-webkit-mask-image:{uri};mask-image:{uri};}}")
    return (".ic-ico{display:inline-block;flex:none;background-color:currentColor;-webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;"
            "-webkit-mask-position:center;mask-position:center;-webkit-mask-size:contain;mask-size:contain;vertical-align:-0.15em;}"
            + "".join(rules))


def register_plotly_template() -> None:
    """One look for every chart: Inter, soft grid, no chart junk, unified hover labels."""
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, size=12.5, color=TOKENS["ink2"]),
        colorway=PALETTE, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        xaxis=dict(gridcolor="#ECE9E2", zerolinecolor="#D3CFC5", linecolor="#D3CFC5", ticks="outside", tickcolor="#D3CFC5",
                   title=dict(font=dict(size=12, color=TOKENS["muted"]), standoff=10)),
        yaxis=dict(gridcolor="#ECE9E2", zerolinecolor="#D3CFC5", linecolor="rgba(0,0,0,0)", ticks="",
                   title=dict(font=dict(size=12, color=TOKENS["muted"]), standoff=10)),
        legend=dict(font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#15191E", font=dict(family=FONT, size=12.5, color="#FFFFFF"), bordercolor="#15191E"),
        margin=dict(l=10, r=10, t=24, b=10), bargap=0.28,
    )
    pio.templates["ic"] = t
    pio.templates.default = "ic"


CSS = """
:root{
  --ic-ink:%(ink)s; --ic-ink2:%(ink2)s; --ic-muted:%(muted)s; --ic-faint:%(faint)s;
  --ic-canvas:%(canvas)s; --ic-surface:%(surface)s; --ic-surface2:%(surface2)s; --ic-line:%(line)s; --ic-line-strong:%(line_strong)s;
  --ic-primary:%(primary)s; --ic-primary-hover:%(primary_hover)s; --ic-primary-soft:%(primary_soft)s; --ic-ring:%(primary_ring)s;
  --ic-accent:%(accent)s;
  --ic-success:%(success)s; --ic-success-bg:%(success_bg)s; --ic-warning:%(warning)s; --ic-warning-bg:%(warning_bg)s;
  --ic-danger:%(danger)s; --ic-danger-bg:%(danger_bg)s; --ic-info:%(info)s; --ic-info-bg:%(info_bg)s;
  --ic-neutral:%(neutral)s; --ic-neutral-bg:%(neutral_bg)s;
  --ic-r:10px; --ic-r-lg:14px;
  --ic-shadow-xs:0 1px 2px rgba(16,24,40,.05); --ic-shadow-sm:0 2px 10px rgba(16,24,40,.07); --ic-shadow-md:0 8px 24px rgba(16,24,40,.10);
}

/* ---------- base ---------- */
.stApp{background:var(--ic-canvas); color:var(--ic-ink);}
html,body,.stApp,button,input,textarea,select{font-feature-settings:"cv11","ss03"; -webkit-font-smoothing:antialiased;}
[data-testid="stMainBlockContainer"]{max-width:1280px; padding:1.4rem 2.2rem 4.5rem;}
[data-testid="stHeader"]{background:transparent;}
/* helper elements that only carry <style> must not add a gap */
[data-testid="stElementContainer"]:has(> [data-testid="stHtml"] > style:only-child),
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] > div > style:only-child){display:none;}
hr{border-color:var(--ic-line)!important; margin:1.4rem 0!important;}
a{color:var(--ic-primary);}

/* ---------- typography ---------- */
[data-testid="stMarkdown"] [data-testid="stMarkdownContainer"] p,[data-testid="stMarkdown"] [data-testid="stMarkdownContainer"] li{font-size:14.5px; line-height:1.6; color:var(--ic-ink2);}
[data-testid="stMarkdown"] [data-testid="stMarkdownContainer"] strong{color:var(--ic-ink); font-weight:620;}
h1,h2,h3,h4{letter-spacing:-0.015em; color:var(--ic-ink);}
[data-testid="stHeading"] h1,h1{font-size:30px; font-weight:680; line-height:1.15; padding:0;}
[data-testid="stHeading"] h2,h2{font-size:19px; font-weight:650; line-height:1.3; padding:1.4rem 0 .3rem;}
[data-testid="stHeading"] h3,h3{font-size:15.5px; font-weight:650; line-height:1.35; padding:.9rem 0 .2rem;}
h4{font-size:14px; font-weight:650; padding:.6rem 0 .2rem;}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p{font-size:12.5px!important; color:var(--ic-muted)!important; line-height:1.5;}
[data-testid="stMarkdownContainer"] code{background:var(--ic-neutral-bg); color:var(--ic-ink); border-radius:6px; padding:.1em .38em; font-size:.88em;}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{border-right:1px solid var(--ic-line);}
[data-testid="stSidebarHeader"]{padding:1.1rem 1rem .4rem 1.1rem;}
[data-testid="stSidebarHeader"] img{height:2rem;}
[data-testid="stSidebarNav"] ul{gap:2px;}
[data-testid="stSidebarNavLink"]{border-radius:9px; padding:.45rem .7rem; color:var(--ic-ink2); font-weight:520; transition:background .12s,color .12s;}
[data-testid="stSidebarNavLink"]:hover{background:var(--ic-neutral-bg); color:var(--ic-ink);}
[data-testid="stSidebarNavLink"][aria-current="page"]{background:var(--ic-primary-soft); color:var(--ic-primary); font-weight:640;}
[data-testid="stSidebarNavLink"][aria-current="page"] span{color:var(--ic-primary);}
[data-testid="stNavSectionHeader"]{padding:.9rem .7rem .2rem;}
[data-testid="stNavSectionHeader"] p,[data-testid="stNavSectionHeader"] span{font-size:11px!important; font-weight:650!important; letter-spacing:.08em; text-transform:uppercase; color:var(--ic-faint)!important;}
[data-testid="stSidebarNavLink"] > span:last-child,[data-testid="stSidebarNavLink"] p{font-size:14px!important;}
[data-testid="stSidebar"] [data-testid="stExpander"] summary{font-size:13.5px; padding:.55rem .8rem;}
[data-testid="stSidebarUserContent"]{padding:0 1rem 1.5rem;}
[data-testid="stSidebar"] [data-testid="stExpander"]{background:transparent; border-color:var(--ic-line);}

/* ---------- surfaces: bordered containers become cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]){}
/* a bordered st.container is the stVerticalBlock itself: cards are identified by their key (see ui.card) */
[class*="st-key-card"],[class*="st-key-rail"]{background:var(--ic-surface)!important; border-radius:var(--ic-r-lg)!important; border-color:var(--ic-line)!important; box-shadow:var(--ic-shadow-xs); padding:.7rem .8rem;}
[class*="st-key-card"] [class*="st-key-card"]{box-shadow:none; background:var(--ic-surface2)!important;}
[class*="st-key-rail"]{position:sticky; top:3.4rem; align-self:flex-start; max-height:calc(100vh - 4.4rem); overflow-y:auto; scrollbar-width:thin;}
[class*="st-key-hero"]{background:linear-gradient(135deg,#EEF6F2 0%%,#FFFFFF 65%%); border:1px solid #CFE3DB; border-radius:var(--ic-r-lg); padding:1.4rem 1.6rem 1.5rem; box-shadow:var(--ic-shadow-sm); height:100%%;}
[class*="st-key-hero"] h4{padding-top:.1rem; font-size:20px; letter-spacing:-.02em;}

/* ---------- KPI cards (native st.metric) ---------- */
[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]){align-items:stretch;}
[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"] > div,
[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) [data-testid="stElementContainer"]{height:100%%;}
[data-testid="stMetric"]{background:var(--ic-surface); border:1px solid var(--ic-line); border-radius:var(--ic-r-lg); padding:14px 16px 13px; box-shadow:var(--ic-shadow-xs); height:100%%; min-height:96px;}
[data-testid="stMetricLabel"]{width:100%%;}
[data-testid="stMetricLabel"] p,[data-testid="stMetricLabel"] label,[data-testid="stMetricLabel"] div{font-size:12.5px!important; font-weight:560; color:var(--ic-muted)!important; white-space:normal!important; line-height:1.3;}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] p{font-size:26px!important; font-weight:670!important; letter-spacing:-.025em; color:var(--ic-ink)!important; font-variant-numeric:tabular-nums; line-height:1.25!important;}
[data-testid="stMetricValue"]{padding-top:2px;} [data-testid="stMetricValue"] > div{white-space:normal!important; overflow:visible!important; text-overflow:clip!important;}
[data-testid="stMetricDelta"]{font-size:12px; font-weight:600; font-variant-numeric:tabular-nums;}

/* ---------- buttons ---------- */
.stButton>button,.stDownloadButton>button,[data-testid="stBaseButton-secondary"],[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-tertiary"],[data-testid="stPageLink"] a{
  border-radius:var(--ic-r)!important; font-weight:600; font-size:14px; min-height:2.45rem; transition:background .12s,border-color .12s,box-shadow .12s,transform .06s; }
[data-testid="stBaseButton-primary"] p,[data-testid="stBaseButton-primary"] span{color:#fff!important;}
[data-testid="stBaseButton-secondary"] p,.stDownloadButton>button p{color:var(--ic-ink)!important;}
[data-testid="stBaseButton-primary"]{background:var(--ic-primary)!important; border:1px solid var(--ic-primary)!important; color:#fff!important; box-shadow:var(--ic-shadow-xs);}
[data-testid="stBaseButton-primary"]:hover{background:var(--ic-primary-hover)!important; border-color:var(--ic-primary-hover)!important; box-shadow:var(--ic-shadow-sm);}
[data-testid="stBaseButton-primary"]:active{transform:translateY(1px);}
[data-testid="stBaseButton-secondary"],.stDownloadButton>button{background:var(--ic-surface)!important; border:1px solid var(--ic-line-strong)!important; color:var(--ic-ink)!important; box-shadow:var(--ic-shadow-xs);}
[data-testid="stBaseButton-secondary"]:hover,.stDownloadButton>button:hover{background:var(--ic-surface2)!important; border-color:var(--ic-faint)!important;}
[data-testid="stBaseButton-secondary"]:active,.stDownloadButton>button:active{transform:translateY(1px);}
[data-testid="stBaseButton-tertiary"]{color:var(--ic-primary)!important;} [data-testid="stBaseButton-tertiary"]:hover{background:var(--ic-primary-soft)!important;}
button:focus-visible,a:focus-visible,[data-baseweb="tab"]:focus-visible,summary:focus-visible{outline:3px solid var(--ic-ring)!important; outline-offset:2px!important;}
button:disabled,[data-testid="stBaseButton-primary"]:disabled,[data-testid="stBaseButton-secondary"]:disabled{opacity:.55!important; cursor:not-allowed!important; box-shadow:none!important;}
[data-testid="stPageLink"] a{border:1px solid transparent; padding:.3rem .5rem; color:var(--ic-primary); font-weight:600;}
[data-testid="stPageLink"] a:hover{background:var(--ic-primary-soft);}

/* ---------- inputs ---------- */
[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] label{font-size:13px!important; font-weight:600; color:var(--ic-ink2)!important;}
/* input controls: Streamlit renders them as plain divs identified by data-testid, not baseweb */
[data-testid="stTextInputRootElement"],[data-testid="stTextAreaRootElement"],[data-testid="stNumberInputContainer"],
[data-testid="stSelectbox"] div:has(> input),[data-testid="stMultiSelect"] div:has(> [data-testid="stMultiSelectTagsContainer"]),
[data-testid="stDateInput"] div:has(> [data-testid="stDateInputField"]){
  border-radius:var(--ic-r)!important; background:var(--ic-surface)!important; border:1px solid var(--ic-line-strong)!important; transition:border-color .12s,box-shadow .12s;}
[data-testid="stTextInputRootElement"]:hover,[data-testid="stTextAreaRootElement"]:hover,[data-testid="stNumberInputContainer"]:hover,
[data-testid="stSelectbox"] div:has(> input):hover,[data-testid="stMultiSelect"] div:has(> [data-testid="stMultiSelectTagsContainer"]):hover{border-color:var(--ic-faint)!important;}
[data-testid="stTextInputRootElement"]:focus-within,[data-testid="stTextAreaRootElement"]:focus-within,[data-testid="stNumberInputContainer"]:focus-within,
[data-testid="stSelectbox"] div:has(> input):focus-within,[data-testid="stMultiSelect"] div:has(> [data-testid="stMultiSelectTagsContainer"]):focus-within,
[data-testid="stDateInput"] div:has(> [data-testid="stDateInputField"]):focus-within{border-color:var(--ic-primary)!important; box-shadow:0 0 0 3px var(--ic-ring)!important;}
[data-testid="stTextInputField"],[data-testid="stNumberInputField"],[data-testid="stTextAreaRootElement"] textarea{font-size:14px; color:var(--ic-ink);}
[data-testid="stCheckbox"] label{align-items:flex-start;}
[data-testid="stCheckbox"] [data-testid="stMarkdownContainer"] p{white-space:normal!important; overflow:visible!important; text-overflow:clip!important; line-height:1.4;}
[data-baseweb="tag"]{background:var(--ic-primary-soft)!important; border-radius:7px!important;} [data-baseweb="tag"] span{color:var(--ic-primary)!important; font-weight:560;}
[data-testid="stFileUploaderDropzone"]{border:1.5px dashed var(--ic-line-strong); border-radius:var(--ic-r-lg); background:var(--ic-surface2); transition:border-color .12s,background .12s;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--ic-primary); background:var(--ic-primary-soft);}
[data-testid="stCheckbox"] label,[data-testid="stToggle"] label{font-size:14px;}
[data-testid="stSlider"] [data-testid="stThumbValue"]{font-weight:650; font-variant-numeric:tabular-nums; color:var(--ic-primary);}
[data-testid="stSlider"]{padding-bottom:.2rem;}

/* ---------- tabs ---------- */
[data-baseweb="tab-list"]{gap:.25rem; border-bottom:1px solid var(--ic-line);}
button[data-baseweb="tab"]{font-weight:560; font-size:14px; color:var(--ic-muted); padding:.65rem .95rem; border-radius:8px 8px 0 0; transition:color .12s,background .12s;}
button[data-baseweb="tab"]:hover{color:var(--ic-ink); background:var(--ic-neutral-bg);}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--ic-primary); font-weight:650;}
[data-baseweb="tab-highlight"]{background:var(--ic-primary)!important; height:2.5px!important; border-radius:2px;}
[data-baseweb="tab-border"]{display:none;}
[data-testid="stTabPanel"]{padding-top:1.1rem;}

/* ---------- expanders / popovers ---------- */
[data-testid="stExpander"]{border:1px solid var(--ic-line)!important; border-radius:var(--ic-r-lg)!important; background:var(--ic-surface); box-shadow:var(--ic-shadow-xs);}
[data-testid="stExpander"] summary{padding:.7rem 1rem; font-weight:600; color:var(--ic-ink); border-radius:var(--ic-r-lg);}
[data-testid="stExpander"] summary:hover{background:var(--ic-surface2);}
[data-testid="stPopoverBody"]{border-radius:var(--ic-r-lg); border:1px solid var(--ic-line); box-shadow:var(--ic-shadow-md); padding:1.1rem; background:var(--ic-surface);}

/* ---------- alerts: softer, status-coloured, readable ---------- */
[data-testid="stAlert"]{border-radius:var(--ic-r-lg); border:1px solid var(--ic-line); box-shadow:none; padding:.1rem .1rem;}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p{font-size:14px; color:inherit; line-height:1.5;}
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]){background:var(--ic-info-bg); border-color:#C9D7F8; color:#1B3FA6;}
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]){background:var(--ic-success-bg); border-color:#BFE3CB; color:#14623A;}
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]){background:var(--ic-warning-bg); border-color:#F2DDA6; color:#7A4E00;}
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]){background:var(--ic-danger-bg); border-color:#F5C4BF; color:#9A1C12;}

/* ---------- tables / charts ---------- */
[data-testid="stDataFrame"]{border:1px solid var(--ic-line); border-radius:var(--ic-r-lg); overflow:hidden; box-shadow:var(--ic-shadow-xs); background:var(--ic-surface);}
[data-testid="stPlotlyChart"]{background:var(--ic-surface); border:1px solid var(--ic-line); border-radius:var(--ic-r-lg); padding:.6rem .6rem .2rem; box-shadow:var(--ic-shadow-xs);}
[data-testid="stStatusWidget"],[data-testid="stSpinner"]{color:var(--ic-muted);}
[data-testid="stStatus"]{border-radius:var(--ic-r-lg)!important; border-color:var(--ic-line)!important;}
[data-testid="stToast"]{border-radius:var(--ic-r); box-shadow:var(--ic-shadow-md);}

/* ================== components (core/components.py) ================== */
.ic-header{margin:.2rem 0 1.15rem;}
.ic-eyebrow{font-size:11.5px; font-weight:650; letter-spacing:.09em; text-transform:uppercase; color:var(--ic-primary); margin-bottom:.35rem;}
.ic-title{font-size:30px; font-weight:690; letter-spacing:-.025em; line-height:1.12; margin:0; color:var(--ic-ink);}
.ic-subtitle{font-size:15px; color:var(--ic-muted); line-height:1.55; margin:.45rem 0 0; max-width:62rem;}

.ic-stepper{display:flex; align-items:center; gap:0; margin:.2rem 0 1.5rem; padding:.7rem .9rem; background:var(--ic-surface); border:1px solid var(--ic-line); border-radius:var(--ic-r-lg); box-shadow:var(--ic-shadow-xs); overflow-x:auto;}
.ic-step{display:flex; align-items:center; gap:.55rem; white-space:nowrap; font-size:13px; font-weight:560; color:var(--ic-muted);}
.ic-step-dot{width:24px; height:24px; border-radius:50%%; display:inline-flex; align-items:center; justify-content:center; font-size:12px; font-weight:700; background:var(--ic-neutral-bg); color:var(--ic-neutral); border:1.5px solid transparent; flex:none;}
.ic-step.done .ic-step-dot{background:var(--ic-primary); color:#fff;}
.ic-step.done{color:var(--ic-ink2);}
.ic-step.current .ic-step-dot{background:#fff; color:var(--ic-primary); border-color:var(--ic-primary); box-shadow:0 0 0 4px var(--ic-ring);}
.ic-step.current{color:var(--ic-ink); font-weight:680;}
.ic-step-line{flex:1 1 20px; min-width:16px; height:2px; background:var(--ic-line); margin:0 .7rem; border-radius:2px;}
.ic-step-line.done{background:var(--ic-primary);}

.ic-chip{display:inline-flex; align-items:center; gap:.35rem; padding:.18rem .6rem; border-radius:999px; font-size:12px; font-weight:600; line-height:1.4; background:var(--ic-neutral-bg); color:var(--ic-neutral); border:1px solid transparent; white-space:nowrap; margin:0 .3rem .3rem 0;}
.ic-chip svg{width:12px; height:12px; flex:none;}
.ic-chip.success{background:var(--ic-success-bg); color:var(--ic-success);} .ic-chip.warning{background:var(--ic-warning-bg); color:var(--ic-warning);}
.ic-chip.danger{background:var(--ic-danger-bg); color:var(--ic-danger);} .ic-chip.info{background:var(--ic-info-bg); color:var(--ic-info);}
.ic-chip.primary{background:var(--ic-primary-soft); color:var(--ic-primary);} .ic-chip.outline{background:#fff; border-color:var(--ic-line-strong); color:var(--ic-ink2);}
.ic-chips{display:flex; flex-wrap:wrap; align-items:center; margin:.1rem 0 .6rem;}
.ic-chips-label{font-size:12px; font-weight:600; color:var(--ic-muted); margin-right:.55rem; margin-bottom:.3rem;}

.ic-empty{text-align:center; padding:2.4rem 1.4rem; background:var(--ic-surface); border:1.5px dashed var(--ic-line-strong); border-radius:var(--ic-r-lg); }
.ic-empty-icon{width:46px; height:46px; margin:0 auto .8rem; border-radius:14px; background:var(--ic-primary-soft); color:var(--ic-primary); display:flex; align-items:center; justify-content:center;}
.ic-empty-icon svg{width:22px; height:22px;}
.ic-empty h3{margin:0 0 .3rem; padding:0; font-size:16px;} .ic-empty p{margin:0 auto; max-width:30rem; color:var(--ic-muted); font-size:14px; line-height:1.55;}

.ic-section{margin:1.5rem 0 .6rem;} .ic-section h2{margin:0; padding:0; font-size:19px; font-weight:660; letter-spacing:-.015em;}
.ic-section p{margin:.25rem 0 0; color:var(--ic-muted); font-size:13.5px; line-height:1.5; max-width:60rem;}
.ic-stephead{display:flex; gap:.85rem; align-items:flex-start; margin:1.6rem 0 .7rem;}
.ic-stephead-n{flex:none; width:30px; height:30px; border-radius:10px; background:var(--ic-primary); color:#fff; font-weight:700; font-size:14px; display:flex; align-items:center; justify-content:center;}
.ic-stephead h2{margin:0; padding:0; font-size:18px; line-height:1.3;} .ic-stephead p{margin:.2rem 0 0; color:var(--ic-muted); font-size:13.5px;}

.ic-kv{display:grid; grid-template-columns:auto 1fr; gap:.45rem 1rem; font-size:13.5px; margin:.2rem 0;}
.ic-kv{padding:0;} .ic-kv dt{color:var(--ic-muted); font-weight:560; margin:0; padding:0;} .ic-kv dd{margin:0; padding:0; color:var(--ic-ink); font-weight:600; text-align:right; font-variant-numeric:tabular-nums;}
.ic-hero-title{font-size:20px; font-weight:670; letter-spacing:-.02em; line-height:1.25; margin:.1rem 0 .5rem; color:var(--ic-ink);}
.ic-panel-title{font-size:11.5px; font-weight:650; letter-spacing:.08em; text-transform:uppercase; color:var(--ic-muted); margin:.1rem 0 .6rem;}
.ic-profile{display:flex; gap:1rem; align-items:center; padding:1.1rem 1.2rem; background:var(--ic-surface); border:1px solid var(--ic-line); border-radius:var(--ic-r-lg); box-shadow:var(--ic-shadow-xs); margin-bottom:1rem;}
.ic-avatar{flex:none; width:56px; height:56px; border-radius:16px; display:flex; align-items:center; justify-content:center; font-weight:700; font-size:20px; letter-spacing:.02em; color:var(--ic-primary); background:var(--ic-primary-soft);}
.ic-profile h2{margin:0; padding:0; font-size:21px;} .ic-profile p{margin:.15rem 0 0; color:var(--ic-muted); font-size:13.5px;}
.ic-health{display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:.8rem;}
.ic-health-card{background:var(--ic-surface); border:1px solid var(--ic-line); border-radius:var(--ic-r-lg); padding:.95rem 1.05rem; box-shadow:var(--ic-shadow-xs); position:relative; overflow:hidden;}
.ic-health-card::before{content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--ic-line-strong);}
.ic-health-card.warning::before{background:#E0932B;} .ic-health-card.danger::before{background:#D2553F;} .ic-health-card.success::before{background:#2E9E5B;}
.ic-health-val{font-size:26px; font-weight:680; letter-spacing:-.02em; font-variant-numeric:tabular-nums; line-height:1.2;}
.ic-health-label{font-size:13px; font-weight:620; color:var(--ic-ink); margin-top:.1rem;} .ic-health-help{font-size:12px; color:var(--ic-muted); margin-top:.2rem; line-height:1.4;}
.ic-bar{height:6px; background:var(--ic-neutral-bg); border-radius:99px; overflow:hidden; margin-top:.55rem;} .ic-bar>span{display:block; height:100%%; border-radius:99px; background:var(--ic-primary);}
.ic-tile{background:var(--ic-surface); border:1px solid var(--ic-line); border-radius:var(--ic-r-lg); padding:1.05rem 1.15rem; box-shadow:var(--ic-shadow-xs); height:100%%;}
.ic-tile h3{margin:.2rem 0 .3rem; padding:0; font-size:15px;} .ic-tile p{margin:0; color:var(--ic-muted); font-size:13.5px; line-height:1.55;}
.ic-tile-icon{width:34px; height:34px; border-radius:10px; background:var(--ic-primary-soft); color:var(--ic-primary); display:flex; align-items:center; justify-content:center;}
.ic-tile-icon svg{width:18px; height:18px;}
.ic-card-title{font-size:14.5px; font-weight:650; color:var(--ic-ink); margin:0 0 .15rem;} .ic-card-sub{font-size:12.5px; color:var(--ic-muted); margin:0 0 .7rem;}
.ic-list{margin:.2rem 0 0; padding:0; list-style:none;} .ic-list li{position:relative; padding:.18rem 0 .18rem 1.1rem; font-size:13.5px; color:var(--ic-ink2); line-height:1.45;}
.ic-list li::before{content:""; position:absolute; left:.1rem; top:.62rem; width:6px; height:6px; border-radius:50%%; background:var(--ic-faint);}
.ic-list.good li::before{background:#2E9E5B;} .ic-list.risk li::before{background:#E0932B;}

/* ---------- responsive ---------- */
@media (max-width:1000px){
  /* rail + content layouts stack: the sidebar already takes a third of the screen */
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [class*="st-key-rail"]) > [data-testid="stColumn"]{min-width:100%%!important; flex:1 1 100%%!important;}
}
@media (max-width:1100px){ [data-testid="stMainBlockContainer"]{padding:1.2rem 1.4rem 4rem;} [class*="st-key-rail"]{position:static;} }
@media (max-width:900px){
  [data-testid="stHorizontalBlock"]{flex-wrap:wrap!important; row-gap:.8rem;}
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{min-width:calc(50%% - 1rem)!important; flex:1 1 calc(50%% - 1rem)!important;}
  .ic-title{font-size:25px;} [data-testid="stHeading"] h1,h1{font-size:25px;}
  .ic-step-label{display:none;} .ic-step.current .ic-step-label{display:inline;}
}
@media (max-width:640px){
  [data-testid="stMainBlockContainer"]{padding:1rem .9rem 3.5rem;}
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{min-width:100%%!important; flex:1 1 100%%!important;}
  [data-testid="stMetricValue"]{font-size:22px;} .ic-profile{flex-direction:column; align-items:flex-start;}
}
@media (max-width:1000px){ [data-testid="stMainBlockContainer"]{padding-top:3.4rem;} }
@media (max-width:640px){
  /* KPI cards stay two per row on phones */
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > [data-testid="stColumn"]{min-width:calc(50%% - .6rem)!important; flex:1 1 calc(50%% - .6rem)!important;}
  [data-testid="stMetric"]{padding:11px 12px; min-height:84px;} [data-testid="stMetricValue"],[data-testid="stMetricValue"] p{font-size:21px!important;}
}
@media (prefers-reduced-motion:reduce){ *{transition:none!important; animation:none!important;} }
"""


def css() -> str:
    return CSS % TOKENS + icon_css()
