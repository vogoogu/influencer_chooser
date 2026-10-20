import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from core import charts, components as C, config, forest, predict, ui

ui.page_config("Forecast model", ":material/forest:")
ui.sidebar()
ss = st.session_state

ui.header("Forecast model: random forest", "How the app predicts a creator's real reach and conversions, how you can check it, and when it is not used.", eyebrow="Insights")

with st.spinner("Training the random forest on past campaign results (about 10 seconds the first time) ..."):
    bundle = forest.get_bundle()
if bundle is None:
    st.info(f"At least {forest.MIN_ROWS} logged campaign results are needed. Log results on the Decision log page.", icon=":material/info:")
    st.stop()

hist, X, reach, conv = bundle["hist"], bundle["X"], bundle["reach"], bundle["conv"]
corr = predict.fit_correction(hist)

# ------------------------------------------------------------------------------ verdict
c1, c2, c3 = st.columns(3)
c1.metric("Past results used for training", bundle["n"])
c2.metric("Reach: error vs best simple method", f"{-reach.gain_vs_best:+.0%}", help=f"Best simple method: {reach.best_baseline}. Negative = the forest makes smaller errors.",
          delta="forest wins" if reach.beats_baseline else "forest not used", delta_color="normal" if reach.beats_baseline else "inverse")
c3.metric("Conversions: error vs best simple method", f"{-conv.gain_vs_best:+.0%}", help=f"Best simple method: {conv.best_baseline}.",
          delta="forest wins" if conv.beats_baseline else "forest not used", delta_color="normal" if conv.beats_baseline else "inverse")
if reach.beats_baseline:
    st.success(f"The forest predicts reach on campaigns it has never seen with {reach.gain_vs_best:.0%} smaller error than the best simple method, "
               f"so the app uses it when *Random forest* is chosen on the Shortlist page.", icon=":material/check_circle:")
else:
    st.warning("The forest does not beat the simple tier correction on this data, so the app keeps using the tier correction for reach.", icon=":material/info:")

t_how, t_drive, t_try, t_limits = st.tabs(["How it works", "What drives the forecast", "Explain a creator and what-if", "Settings and limits"])

# ================================================================================== HOW IT WORKS
with t_how:
    C.step_header(1, "The problem", "The naive forecast is optimistic.")
    naive_gap = hist.ratio.mean() - 1
    st.markdown(
        f"A **naive forecast** says: a creator's reach will be about their average views. Across **{len(hist)} past creator-campaign results** "
        f"the real reach was on average **{naive_gap:+.0%}** versus that forecast, and **{(hist.ratio < 1).mean():.0%}** of the time it came in lower. "
        "Some creators are reliable, some deliver far less. The forest learns **which kinds of creators miss the naive forecast, and by how much**.")

    C.step_header(2, "The training data", "One row per past result, using only facts known before the campaign.")
    st.markdown("Each past result becomes one row. The columns are facts that were **known before the campaign started** "
                "(never anything from after it, so the model cannot cheat). The last column is what we want to predict.")
    show = pd.DataFrame({
        "Platform": np.where(X.is_instagram > 0.5, "instagram", np.where(X.is_tiktok > 0.5, "tiktok", "youtube")),
        "Audience size": [forest.display_value("log_followers", v) for v in X.log_followers],
        "Engagement vs peers": [forest.display_value("er_vs_peers", v) for v in X.er_vs_peers],
        "Fake followers": [forest.display_value("fake_pct", v) for v in X.fake_pct],
        "Biggest follower jump": [forest.display_value("max_growth_6m", v) for v in X.max_growth_6m],
        "Relationship (1-5)": X.relationship_score.astype(int),
        "Niche fit (0-1)": X.niche_fit,
        "Earlier delivery": [forest.display_value("prior_log_ratio", v) if n else "none yet" for v, n in zip(X.prior_log_ratio, X.prior_n)],
        "Outcome: real reach / naive forecast": hist.ratio.round(2),
    })
    st.dataframe(show.sample(10, random_state=3), hide_index=True, width="stretch")
    with st.expander("What does every input mean?"):
        help_text = {
            "log_followers": "Number of followers. Bigger audiences tend to disappoint more.",
            "er_vs_peers": "Engagement rate divided by the median of creators with the same platform and size. 1.0x = typical.",
            "fake_pct": "Estimated share of fake or inactive followers.",
            "max_growth_6m": "Largest month-over-month follower growth in the last six months. A huge jump can mean bought followers.",
            "days_old": "How old the latest stats were when the forecast was made.",
            "relationship_score": "Account manager's 1-5 rating of the relationship.",
            "niche_fit": "1 = the creator's main niche is one of the product's niches, 0.5 = secondary niche, 0 = unrelated.",
            "prior_log_ratio": "How the same creator delivered in earlier, already measured campaigns (shrunk towards neutral when there are few).",
            "prior_n": "Number of those earlier campaigns.",
            "log_views_per_follower": "Average views divided by followers: how much of the audience a post usually reaches.",
            "log_fee": "What the creator costs.",
            "is_instagram": "Platform is Instagram.", "is_tiktok": "Platform is TikTok (YouTube is the remaining case).",
            "deliver_mult": "Reach factor of the format (reel 1.0, feed post 0.7, story set 0.45).",
        }
        st.dataframe(pd.DataFrame({"Input": [forest.FEATURE_LABELS[f] for f in forest.REACH_FEATURES],
                                   "Meaning": [help_text[f] for f in forest.REACH_FEATURES]}), hide_index=True, width="stretch")
        st.caption("The conversion model additionally knows the campaign goal.")

    C.step_header(3, "One decision tree", "A flowchart of yes/no questions.")
    st.markdown("A tree is a flowchart of yes/no questions. Each question splits the past results into two groups that behave more alike. "
                "At the end of every branch sits a prediction: the average outcome of the results that landed there. "
                "Green leaves forecast more reach, red leaves less.")
    tc1, tc2 = st.columns([1, 3])
    with tc1:
        which = st.radio("Show the tree for", ["Reach", "Conversions"], key="m_tree_target")
        depth = st.slider("Tree depth", 1, 4, 3, key="m_tree_depth_ui")
    fm_show = reach if which == "Reach" else conv
    tree = forest.small_tree(fm_show, max_depth=depth)
    nodes, edges = forest.tree_nodes(tree, fm_show.features, fm_show.target)
    with tc2:
        st.plotly_chart(charts.tree_figure(nodes, edges), width="stretch")
    st.caption(("Leaves show the reach multiplier: x0.80 means 20% less reach than the naive forecast. " if which == "Reach"
                else "Leaves show expected conversions per 1,000 reached people. ") +
               "n is the number of past results in that leaf. A single tree like this is easy to read but unstable: a few different "
               "results would give a different tree. That is why we grow many.")

    C.step_header(4, "Many trees vote: the forest", "Averaging hundreds of slightly different trees.")
    st.markdown(
        f"The model grows **{reach.model.n_estimators} trees**. Each tree sees a **random sample** of the past results and, at every question, "
        "only a **random subset of the inputs**. So the trees disagree a little, and their disagreement is useful: "
        "the **average** of all trees is the forecast, and the typical error of such averages (measured on rows each tree did not see) gives the **80% range**. "
        "Below, the same forest judges two creators. Every bar is a group of trees giving the same answer.")
    typical = X.median().to_frame().T
    risky = typical.copy()
    risky["fake_pct"], risky["max_growth_6m"], risky["er_vs_peers"] = 40.0, 1.0, np.log(0.45)
    v1, v2 = st.columns(2)
    for col, name, row in ((v1, "A typical creator", typical), (v2, "Same creator, but 40% fake followers, followers doubled in a month, engagement halved", risky)):
        votes = reach.tree_votes(row)
        mu, lo, hi = (float(a[0]) for a in reach.interval(row))
        with col:
            st.markdown(f"**{name}**")
            st.plotly_chart(charts.votes_hist(votes, mu, lo, hi, "reach"), width="stretch", key=f"votes_{name[:6]}_{len(name)}")
            st.caption(f"Forecast: **x{np.exp(mu):.2f}** the naive reach (80% range x{np.exp(lo):.2f} to x{np.exp(hi):.2f}).")

    C.step_header(5, "Is it any good? An honest test", "Every number below comes from campaigns the model had not seen.")
    st.markdown(
        "A model that is only tested on the data it learned from always looks great. So the past results are split into 5 groups "
        "**by campaign** (all creators of one campaign stay together). The model is trained on 4 groups and must predict the 5th, "
        "five times over. Every number below comes from predictions on campaigns the model had **not seen**.")
    sel = st.radio("Target", ["Reach", "Conversions"], horizontal=True, key="m_eval_target")
    fm_eval = reach if sel == "Reach" else conv
    m = fm_eval.metrics.copy()
    st.plotly_chart(charts.metrics_bar(m), width="stretch")
    st.dataframe(m[["method", "rmse", "mae", "coverage", "rmse_gain_vs_best_baseline"]], hide_index=True, width="stretch",
                 column_config={"method": "Method", "rmse": st.column_config.NumberColumn("Typical error (RMSE)", format="%.3f"),
                                "mae": st.column_config.NumberColumn("Average error (MAE)", format="%.3f"),
                                "coverage": st.column_config.NumberColumn("80% range hit rate", format="percent"),
                                "rmse_gain_vs_best_baseline": st.column_config.NumberColumn("Better than best baseline", format="percent")})
    st.caption("Errors are measured on the log scale of the multiplier, so 0.30 is roughly a 35% typical miss. "
               "The hit rate should be close to 80%: much lower means the range is too narrow, much higher means it is too wide.")
    st.plotly_chart(charts.oof_scatter(fm_eval.oof, fm_eval.target), width="stretch")
    st.caption("Each dot is one past result. Dots near the dashed line were predicted well. The model never saw these campaigns when predicting them.")
    rule = (f"**Decision rule in the app:** the forest is only used where it cuts the cross-validated error by at least {forest.MIN_GAIN:.0%} versus the "
            "best simple method. Right now: "
            f"reach **{'uses the forest' if reach.beats_baseline else 'keeps the tier correction'}** ({reach.gain_vs_best:+.0%}), "
            f"conversions **{'use the forest' if conv.beats_baseline else 'keep the assumed rates'}** ({conv.gain_vs_best:+.0%}).")
    st.info(rule, icon=":material/rule:")

# ================================================================================== WHAT DRIVES IT
with t_drive:
    st.subheader("Which inputs matter?")
    st.markdown("**Permutation importance:** scramble one input at random and see how much worse the predictions on unseen campaigns get. "
                "A big increase means the model relies on that input.")
    tgt = st.radio("Model", ["Reach", "Conversions"], horizontal=True, key="m_drive_target")
    fm = reach if tgt == "Reach" else conv
    st.plotly_chart(charts.importance_bar(fm.importance), width="stretch")
    top = fm.importance.head(3).label.tolist()
    st.markdown(f"For this model the most important inputs are **{top[0]}**, **{top[1]}** and **{top[2]}**.")
    st.subheader("What does the model do with one input?")
    feat = st.selectbox("Input", fm.importance.feature.tolist(), format_func=lambda f: forest.FEATURE_LABELS[f], key="m_pdp_feature")
    pdp = forest.partial_dependence(fm, feat)
    st.plotly_chart(charts.pdp_line(pdp, feat, fm.target, forest.display_value), width="stretch")
    st.caption("Partial dependence: the average forecast if every past creator had this value, other inputs unchanged. "
               + ("The dotted line is 'no change versus the naive forecast'." if fm.target == "reach" else "The dotted line is the average over all values."))
    st.warning("These are patterns in your data, not proof of cause. For example, fake followers may simply be a symptom of other problems. "
               "Use this page to check whether the scoring weights point at what actually predicts delivery.", icon=":material/psychology:")

# ================================================================================== TRY IT
with t_try:
    st.subheader("Explain a creator from your shortlist")
    ranked = ss.get("ranked")
    profile = ss.get("profile") or {}
    if ranked is None or len(ranked) == 0:
        st.info("Run the Shortlist first to explain real candidates. The what-if tool below works without it.", icon=":material/info:")
    else:
        top_cands = ranked.head(15)
        pick = st.selectbox("Creator", top_cands.account_id.tolist(),
                            format_func=lambda a: f"#{int(top_cands.set_index('account_id').loc[a, 'rank'])} {top_cands.set_index('account_id').loc[a, 'name']} ({top_cands.set_index('account_id').loc[a, 'platform']})")
        row = top_cands[top_cands.account_id == pick]
        fc, Xp = forest.forecast(bundle, row, profile.get("category"), profile.get("goal", "awareness"),
                                 row.conf_recent.fillna(0.7))
        naive = float(row.reach_naive.iloc[0])
        tier_p10, tier_p50, tier_p90, _ = predict.predict_reach(naive, row.tier.iloc[0], corr, float(row.conf_recent.fillna(0.7).iloc[0]))
        mu, lo, hi = float(fc.rf_mu.iloc[0]), float(fc.rf_lo.iloc[0]), float(fc.rf_hi.iloc[0])
        k1, k2, k3 = st.columns(3)
        k1.metric("Naive forecast", f"{naive:,.0f}")
        k2.metric("Tier correction", f"{float(tier_p50):,.0f}", delta=f"{float(tier_p50) / naive - 1:+.0%}")
        k3.metric("Random forest", f"{naive * np.exp(mu):,.0f}", delta=f"{np.exp(mu) - 1:+.0%}",
                  help=f"80% range {naive * np.exp(lo):,.0f} to {naive * np.exp(hi):,.0f}")
        a, b = st.columns(2)
        with a:
            st.markdown("**What all trees say**")
            st.plotly_chart(charts.votes_hist(reach.tree_votes(Xp.iloc[[0]]), mu, lo, hi, "reach"), width="stretch", key="cand_votes")
        with b:
            st.markdown("**Why? Each input versus a typical creator**")
            eff = forest.local_effects(reach, Xp.iloc[[0]], reach.X)
            st.plotly_chart(charts.effects_bar(eff), width="stretch", key="cand_eff")
        st.dataframe(eff.head(6).assign(value=[forest.display_value(f, v) for f, v in zip(eff.head(6).feature, eff.head(6).value)],
                                        typical=[forest.display_value(f, v) for f, v in zip(eff.head(6).feature, eff.head(6).typical)],
                                        effect=lambda d: (np.exp(d.effect) - 1))[["label", "value", "typical", "effect"]],
                     hide_index=True, width="stretch",
                     column_config={"label": "Input", "value": "This creator", "typical": "Typical creator",
                                    "effect": st.column_config.NumberColumn("Effect on forecast", format="percent")})
        st.caption("Method: replace one input at a time by the typical value and see how the forecast changes (an 'occlusion' test). "
                   "It is a simple explanation, not an exact decomposition.")

    st.subheader("What-if: build a creator and see the forecast")
    med = X.median()
    w1, w2, w3 = st.columns(3)
    with w1:
        followers = st.select_slider("Followers", options=[2_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000],
                                     value=25_000, format_func=lambda v: f"{v:,}")
        platform = st.selectbox("Platform", ["instagram", "tiktok", "youtube"])
        fmt = st.selectbox("Format", ["reel", "feed_post", "story_set", "video", "integration"])
        naive_in = st.number_input("Naive forecast (reach)", 1000, 5_000_000, 10_000, step=1000)
    with w2:
        er_rel = st.slider("Engagement vs peers (x)", 0.3, 3.0, 1.0, 0.05)
        fake = st.slider("Fake followers (%)", 0, 50, 6)
        jump = st.slider("Biggest monthly follower jump (%)", 0, 200, 3)
        fit = st.select_slider("Niche fit", options=[0.0, 0.2, 0.5, 1.0], value=1.0)
    with w3:
        rel = st.slider("Relationship score", 1, 5, 4)
        prior_n = st.slider("Earlier measured campaigns", 0, 10, 2)
        prior_x = st.slider("Earlier delivery vs forecast (x)", 0.3, 1.5, 0.9, 0.05, disabled=prior_n == 0)
        goal_in = st.selectbox("Campaign goal (conversions)", config.GOALS, format_func=lambda g: g.replace("_", " "))
    x = med.copy()
    x["log_followers"] = np.log(followers)
    x["er_vs_peers"], x["fake_pct"], x["max_growth_6m"] = np.log(er_rel), fake, jump / 100
    x["niche_fit"], x["relationship_score"], x["prior_n"] = fit, rel, prior_n
    x["prior_log_ratio"] = np.log(prior_x) * prior_n / (prior_n + 2) if prior_n else 0.0
    x["is_instagram"], x["is_tiktok"] = float(platform == "instagram"), float(platform == "tiktok")
    x["deliver_mult"] = config.DELIVERABLE_REACH_MULT.get(fmt, 1.0)
    for g in config.GOALS:
        x[f"goal_{g}"] = float(g == goal_in)
    xr = x.to_frame().T[forest.CONV_FEATURES]
    mu, lo, hi = (float(a[0]) for a in reach.interval(xr))
    rate = float(np.exp(float(conv.predict(xr)[0]) + 0.5 * conv.resid_var))
    r1, r2, r3 = st.columns(3)
    r1.metric("Forecast reach", f"{naive_in * np.exp(mu):,.0f}", delta=f"x{np.exp(mu):.2f} the naive forecast")
    r2.metric("80% range", f"{naive_in * np.exp(lo):,.0f} to {naive_in * np.exp(hi):,.0f}")
    r3.metric("Expected conversions", f"{naive_in * np.exp(mu) * rate / 1000:,.0f}", delta=f"{rate:.1f} per 1,000 reached", delta_color="off")
    st.caption("Medium-sized inputs not shown (stats age, views per follower, fee) are set to the typical past value. "
               "Move the sliders to see which inputs the forest reacts to. Try 40% fake followers with a 100% follower jump.")

# ================================================================================== SETTINGS AND LIMITS
with t_limits:
    st.subheader("How it is used in the app")
    st.markdown(
        "- On the **Shortlist** page choose the forecast method: *Random forest*, *Tier correction* or *Naive*.\n"
        "- The forecast sets each creator's **reach range** and **expected conversions**. These feed cost efficiency, the **campaign mix optimizer** and the **Monte Carlo simulation**.\n"
        "- The forest is **retrained automatically** whenever new results are logged, so forecasts improve as you use the tool.\n"
        "- The forest is only used where it beat the simple baseline in the honest test above. Otherwise the app falls back and tells you.\n"
        "- It changes numbers, not the ranking weights: the seven scoring criteria stay transparent and adjustable.")
    st.subheader("Model settings")
    mdl = reach.model
    st.dataframe(pd.DataFrame({
        "Setting": ["Trees", "Minimum results per leaf", "Inputs tried per question", "Sampling", "Random seed", "Training rows", "Evaluation"],
        "Value": [mdl.n_estimators, mdl.min_samples_leaf, f"{mdl.max_features:.0%} of the inputs", "bootstrap (random sample with replacement)", forest.SEED,
                  bundle["n"], "5-fold cross-validation grouped by campaign; 80% ranges from out-of-bag residuals"]}),
        hide_index=True, width="stretch")
    st.caption("A fixed seed makes results repeatable: the same data always gives the same forecast.")
    if st.button("Retrain now", icon=":material/refresh:"):
        with st.spinner("Retraining ..."):
            forest.get_bundle(force=True)
        st.cache_data.clear()
        st.rerun()
    st.subheader("Limits you should know")
    st.markdown(
        f"- **Small data.** {bundle['n']} results is little for a forest. The honest test above accounts for that, but expect forecasts to improve with more logged campaigns.\n"
        "- **Past is not future.** The forest cannot predict situations it has never seen, such as a brand-new platform or a viral event.\n"
        "- **Synthetic data.** The bundled results were generated with known effects (for example fake followers lower delivery), so the forest finding them shows the method works, not that the real market behaves this way.\n"
        "- **Association, not cause.** Importance and partial dependence describe patterns in the data.\n"
        "- **Ranges are approximate.** The 80% range is based on typical past errors and is widened when a creator's data has low confidence.\n"
        "- **The AI does not predict anything.** The AI model never produces these numbers; the forest is plain scikit-learn code.")
