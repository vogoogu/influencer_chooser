"""Campaign mix: choose the combination of influencers that maximises an objective under a budget
and portfolio constraints (integer programming with PuLP/CBC, greedy fallback)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

OBJECTIVES = {
    "qualified_reach": "Qualified reach (reach x audience match)",
    "conversions": "Expected conversions",
    "score": "Total fit score",
}


@dataclass
class MixSettings:
    budget: float = 20000.0
    objective: str = "qualified_reach"
    risk_aversion: float = 0.3          # 0 = plan with the median (p50), 1 = plan with the pessimistic p10
    n_min: int = 3
    n_max: int = 8
    max_share: float = 0.40             # max share of budget on one influencer
    min_small: int = 0                  # at least this many nano/micro creators
    min_per_platform: dict = field(default_factory=dict)
    must_include: list = field(default_factory=list)       # account_ids that have to be in the mix


def candidate_values(pool: pd.DataFrame, objective: str, risk_aversion: float) -> pd.Series:
    reach = (1 - risk_aversion) * pool.reach_p50 + risk_aversion * pool.reach_p10
    if objective == "qualified_reach":
        return reach * pool.audience_match
    if objective == "conversions":
        return pool.exp_conversions * (reach / pool.reach_p50.clip(lower=1))
    if objective == "score":
        return pool.total * 1000.0
    raise ValueError(f"unknown objective {objective}")


def _constraints_ok(pool: pd.DataFrame, chosen: list[int], s: MixSettings) -> bool:
    sub = pool.iloc[chosen]
    return (len(sub) >= s.n_min and len(sub) <= s.n_max and sub.price.sum() <= s.budget + 1e-6
            and sub.influencer_id.is_unique and (sub.tier.isin(["nano", "micro"]).sum() >= s.min_small)
            and all((sub.platform == p).sum() >= k for p, k in s.min_per_platform.items()))


def _greedy(pool: pd.DataFrame, vals: pd.Series, s: MixSettings) -> list[int]:
    order = (vals / pool.price.clip(lower=1)).sort_values(ascending=False).index.tolist()
    chosen, spend, seen = [], 0.0, set()
    for i in order:
        r = pool.loc[i]
        if len(chosen) >= s.n_max or r.influencer_id in seen or spend + r.price > s.budget or r.price > s.max_share * s.budget:
            continue
        chosen.append(i)
        seen.add(r.influencer_id)
        spend += r.price
    return chosen


def solve_mix(pool: pd.DataFrame, s: MixSettings) -> dict:
    """Return {'status', 'method', 'selected' (DataFrame), 'value'}.
    `pool` needs: influencer_id, price, tier, platform, reach_p10, reach_p50, audience_match, exp_conversions, total."""
    pool = pool.reset_index(drop=True)
    if pool.empty:
        return {"status": "empty pool", "method": "none", "selected": pool, "value": 0.0}
    vals = candidate_values(pool, s.objective, s.risk_aversion)
    try:
        import pulp
        prob = pulp.LpProblem("mix", pulp.LpMaximize)
        x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(len(pool))]
        prob += pulp.lpSum(vals[i] * x[i] for i in range(len(pool)))
        prob += pulp.lpSum(pool.price[i] * x[i] for i in range(len(pool))) <= s.budget
        prob += pulp.lpSum(x) >= s.n_min
        prob += pulp.lpSum(x) <= s.n_max
        for _, idx in pool.groupby("influencer_id").groups.items():       # one account per influencer
            prob += pulp.lpSum(x[i] for i in idx) <= 1
        for i in range(len(pool)):                                        # concentration limit
            if pool.price[i] > s.max_share * s.budget:
                prob += x[i] == 0
        if s.min_small:
            prob += pulp.lpSum(x[i] for i in range(len(pool)) if pool.tier[i] in ("nano", "micro")) >= s.min_small
        for plat, k in s.min_per_platform.items():
            if k:
                prob += pulp.lpSum(x[i] for i in range(len(pool)) if pool.platform[i] == plat) >= k
        for i in range(len(pool)):
            if int(pool.account_id[i]) in set(int(a) for a in s.must_include):
                prob += x[i] == 1
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=20))
        status = pulp.LpStatus[prob.status]
        if status == "Optimal":
            chosen = [i for i in range(len(pool)) if x[i].value() and x[i].value() > 0.5]
            return {"status": "optimal", "method": "integer programming (CBC)", "value": float(vals[chosen].sum()),
                    "selected": pool.loc[chosen].assign(value=vals[chosen]).sort_values("value", ascending=False)}
        method = "integer programming (CBC)"
    except Exception:
        status, method = "solver unavailable", "greedy fallback"
    chosen = _greedy(pool, vals, s)
    ok = bool(chosen) and _constraints_ok(pool, chosen, s)
    return {"status": ("greedy, feasible" if ok else f"no feasible mix ({status})"), "method": "greedy fallback",
            "value": float(vals[chosen].sum()) if chosen else 0.0,
            "selected": pool.loc[chosen].assign(value=vals[chosen]) if chosen else pool.iloc[0:0]}


def baseline_top_by_score(pool: pd.DataFrame, s: MixSettings) -> pd.DataFrame:
    """What an account manager would do without the optimizer: take the best-scored creators until the budget is gone."""
    pool = pool.sort_values("total", ascending=False)
    chosen, spend, seen = [], 0.0, set()
    for i, r in pool.iterrows():
        if len(chosen) >= s.n_max or r.influencer_id in seen or spend + r.price > s.budget:
            continue
        chosen.append(i)
        seen.add(r.influencer_id)
        spend += r.price
    return pool.loc[chosen]


def mix_summary(mix: pd.DataFrame, objective: str, risk_aversion: float) -> dict:
    if mix.empty:
        return {"n": 0, "spend": 0.0, "reach_p50": 0.0, "reach_p10_sum": 0.0, "qualified_reach": 0.0, "conversions": 0.0, "cpm": np.nan}
    return {"n": int(len(mix)), "spend": float(mix.price.sum()), "reach_p50": float(mix.reach_p50.sum()),
            "reach_p10_sum": float(mix.reach_p10.sum()),
            "qualified_reach": float((mix.reach_p50 * mix.audience_match).sum()),
            "conversions": float(mix.exp_conversions.sum()),
            "cpm": float(mix.price.sum() / (mix.reach_p50.sum() / 1000)) if mix.reach_p50.sum() else np.nan}
