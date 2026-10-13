import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from core import charts, db, predict, ui

st.set_page_config(page_title="Calibration", page_icon=":material/target:", layout="wide")
ui.sidebar()

st.title("Calibration")
st.caption("Step 5. How good were past forecasts, and what really predicts success? "
           "This is where the tool learns: its forecasts are corrected with what it finds here.")

hist = predict.load_history()
if len(hist) < 10:
    st.info("At least 10 logged campaign results are needed. Log results on the Decision log page.", icon=":material/info:")
    st.stop()

cov = predict.coverage_by_tier(hist)
corr = predict.fit_correction(hist)
g = corr["_global"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Results analysed", len(hist))
k2.metric("Average actual vs predicted reach", f"{(hist.ratio.mean() - 1) * 100:+.0f}%",
          help="Negative means forecasts were too optimistic on average.")
k3.metric("Actual inside predicted p10-p90", f"{hist.inside.mean():.0%}", delta=f"{hist.inside.mean() - 0.8:+.0%} vs 80% nominal", delta_color="off",
          help="A well-calibrated 80% interval would contain 80% of outcomes.")
k4.metric("Typical forecast error (log sd)", f"{g['sigma']:.2f}")

if hist.ratio.mean() < 0.95:
    st.warning("Past forecasts were optimistic. The shortlist and mix now correct reach by tier (table below), "
               "and the simulation uses the measured spread.", icon=":material/trending_down:")

t1, t2, t3 = st.tabs(["Forecast accuracy", "Learned correction", "What predicts success?"])

with t1:
    a, b = st.columns(2)
    with a:
        st.markdown("**Predicted vs actual reach**")
        st.plotly_chart(charts.calibration_scatter(hist), width="stretch")
        st.caption("Points below the dashed line were over-predicted.")
    with b:
        st.markdown("**Average bias by audience size**")
        st.plotly_chart(charts.bias_bar(cov), width="stretch")
        st.markdown("**Interval coverage by audience size**")
        st.plotly_chart(charts.coverage_bar(cov), width="stretch")

with t2:
    rows = [{"tier": t, "results": corr[t]["n"], "bias multiplier": float(np.exp(corr[t]["mu"])),
             "uncertainty (log sd)": corr[t]["sigma"]} for t in predict.TIER_NAMES]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"bias multiplier": st.column_config.NumberColumn(format="%.2f"),
                                "uncertainty (log sd)": st.column_config.NumberColumn(format="%.2f")})
    st.markdown(
        "A multiplier of 0.85 means: multiply the naive forecast by 0.85. Tiers with few results are pulled towards the overall "
        "average (shrinkage) so that two unlucky campaigns do not distort a tier. Example: a macro creator with a naive forecast of "
        f"100,000 reach is now forecast at **{100000 * np.exp(corr['macro']['mu']):,.0f}** (p10 to p90: "
        f"{predict.predict_reach(100000, 'macro', corr)[0]:,.0f} to {predict.predict_reach(100000, 'macro', corr)[2]:,.0f}).")
    rel = db.read("""SELECT inf.display_name AS creator, v.n_campaigns, v.on_time_rate, v.avg_client_rating, v.avg_actual_vs_predicted
                     FROM v_influencer_reliability v JOIN influencer inf ON inf.id = v.influencer_id
                     WHERE v.n_campaigns >= 3 ORDER BY v.avg_actual_vs_predicted""")
    st.markdown("**Creators who most often under- or over-deliver** (at least 3 campaigns)")
    st.dataframe(rel, hide_index=True, width="stretch",
                 column_config={"on_time_rate": st.column_config.NumberColumn("On time", format="percent"),
                                "avg_client_rating": st.column_config.NumberColumn("Avg rating", format="%.1f"),
                                "avg_actual_vs_predicted": st.column_config.NumberColumn("Actual / predicted", format="%.2f"),
                                "n_campaigns": "Campaigns"})

with t3:
    outcome = st.radio("Outcome to explain", ["Reach vs prediction", "Conversion rate (conversions per 1,000 reach)"], horizontal=True)
    key = "reach_vs_prediction" if outcome.startswith("Reach") else "conversion_rate"
    drv = predict.driver_analysis(hist, outcome=key)
    if drv.empty:
        st.info("Not enough data for a driver analysis (30 results needed).")
    else:
        st.plotly_chart(charts.driver_bar(drv), width="stretch")
        sig = drv[(drv.lo > 0) | (drv.hi < 0)]
        if len(sig):
            names = {"niche_fit": "niche fit", "engagement_vs_peers": "engagement vs peers", "fake_followers_pct": "fake follower share",
                     "relationship_score": "relationship score", "log_followers": "audience size"}
            st.success("Clear drivers (interval does not include zero): " + ", ".join(
                f"{names[r.feature]} ({'raises' if r.coef > 0 else 'lowers'} the outcome)" for r in sig.itertuples()), icon=":material/insights:")
        else:
            st.info("No feature shows a clear effect yet. More campaign results will sharpen this.")
        st.caption("Method: standardised linear regression on information known before each campaign, with a bootstrap 90% interval. "
                   "It shows association in your data, not proof of cause. Use it to check whether the scoring weights point at what actually matters.")
