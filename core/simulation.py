"""Monte Carlo simulation of a campaign mix.

Each creator's reach is log-normal around the (bias-corrected) median with the spread learned from
history, widened when data confidence is low. A shared market factor (correlation rho) models that
creators tend to over- or under-deliver together (algorithm changes, seasonality)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

CONV_NOISE = 0.40


def simulate_mix(mix: pd.DataFrame, goal: str, n_sim: int = 5000, rho: float = 0.2, reach_mult: float = 1.0,
                 conv_mult: float = 1.0, sigma_mult: float = 1.0, seed: int = 0) -> dict:
    n = len(mix)
    if n == 0:
        z = np.zeros(n_sim)
        return {"reach": z, "conversions": z, "cpm": z, "cost_per_conversion": z}
    rng = np.random.default_rng(seed)
    mu = np.log(mix.reach_p50.clip(lower=1).values)
    sigma = mix.reach_sigma.values * sigma_mult
    shared = rng.standard_normal((n_sim, 1))
    own = rng.standard_normal((n_sim, n))
    z = np.sqrt(rho) * shared + np.sqrt(1 - rho) * own
    reach_i = np.exp(mu + sigma * z) * reach_mult
    if "conv_rate" in mix:
        rate = mix.conv_rate.values * conv_mult
    else:
        rate = config.GOAL_CONV_RATE.get(goal, 0.004) * (0.7 + 0.7 * mix.niche_match.values) * conv_mult
    conv_i = reach_i * rate * np.exp(rng.normal(0, CONV_NOISE, (n_sim, n)) - 0.5 * CONV_NOISE ** 2)
    reach, conv = reach_i.sum(axis=1), conv_i.sum(axis=1)
    spend = float(mix.price.sum())
    return {"reach": reach, "conversions": conv, "cpm": spend / (reach / 1000.0),
            "cost_per_conversion": spend / np.maximum(conv, 1e-9)}


def summarize(sim: dict, naive_quote: float, reach_target: float | None = None) -> dict:
    r, c = sim["reach"], sim["conversions"]
    out = {
        "reach_p10": float(np.percentile(r, 10)), "reach_p50": float(np.percentile(r, 50)), "reach_p90": float(np.percentile(r, 90)),
        "conv_p10": float(np.percentile(c, 10)), "conv_p50": float(np.percentile(c, 50)), "conv_p90": float(np.percentile(c, 90)),
        "cpm_p50": float(np.percentile(sim["cpm"], 50)),
        "naive_quote": float(naive_quote),
        "p_below_quote": float((r < naive_quote).mean()),
    }
    out["p_below_target"] = float((r < reach_target).mean()) if reach_target else None
    return out


def tornado(mix: pd.DataFrame, goal: str, metric: str = "conv_p50", naive_quote: float | None = None,
            n_sim: int = 3000, seed: int = 0) -> pd.DataFrame:
    """One-at-a-time sensitivity of a metric to the modelling assumptions."""
    quote = naive_quote if naive_quote is not None else float(mix.reach_naive.sum())

    def run(**kw):
        return summarize(simulate_mix(mix, goal, n_sim=n_sim, seed=seed, **kw), quote)[metric]

    base = run()
    params = [("Reach vs forecast (-15% / +15%)", dict(reach_mult=0.85), dict(reach_mult=1.15)),
              ("Conversion rate (-30% / +30%)", dict(conv_mult=0.7), dict(conv_mult=1.3)),
              ("Forecast uncertainty (x1.5 / x0.5)", dict(sigma_mult=1.5), dict(sigma_mult=0.5)),
              ("Correlation between creators (0.5 / 0)", dict(rho=0.5), dict(rho=0.0))]
    rows = [{"assumption": label, "low": run(**lo), "high": run(**hi), "base": base} for label, lo, hi in params]
    df = pd.DataFrame(rows)
    df["swing"] = (df[["low", "high"]].max(axis=1) - df[["low", "high"]].min(axis=1))
    return df.sort_values("swing", ascending=True).reset_index(drop=True)
