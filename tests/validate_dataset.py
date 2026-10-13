#!/usr/bin/env python3
"""
Validates data/influencer_db.sqlite.

Checks: referential integrity, value ranges, calibration against the benchmark ordering,
prediction bias, and that every planted trap (data/ground_truth/planted_traps.csv) can be
found with plain SQL. Exit code 1 if any check fails.

    python tests/validate_dataset.py
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "influencer_db.sqlite"
TODAY = "2026-10-09"
results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str):
    results.append((bool(ok), msg))
    print(("PASS  " if ok else "FAIL  ") + msg)


def q(con, sql, params=()):
    return pd.read_sql_query(sql, con, params=params)


def main() -> int:
    if not DB.exists():
        print("Database missing. Run: python generate_dataset.py")
        return 1
    con = sqlite3.connect(DB)
    con.execute("PRAGMA foreign_keys = ON")
    truth = pd.read_csv(ROOT / "data" / "ground_truth" / "planted_traps.csv")

    # ---------------------------------------------------------- integrity
    check(len(con.execute("PRAGMA foreign_key_check").fetchall()) == 0, "no foreign key violations")
    n = lambda t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    check(n("influencer") >= 50 and n("account") >= n("influencer"), f"{n('influencer')} influencers, {n('account')} accounts")
    check(n("account_snapshot") > 1500, f"{n('account_snapshot')} monthly snapshots")
    check(n("result") > 200, f"{n('result')} campaign results")

    # ------------------------------------------------------------- ranges
    s = q(con, "SELECT * FROM account_snapshot")
    check(s.engagement_rate.between(0.01, 20).all(), "engagement rates within 0.01-20 %")
    check(s.fake_follower_pct.between(0.5, 60).all(), "fake follower share within 0.5-60 %")
    a = q(con, "SELECT * FROM audience_snapshot")
    ages = a[["age_18_24", "age_25_34", "age_35_44", "age_45_plus"]].sum(axis=1)
    check((ages - 100).abs().max() < 0.2, "age buckets sum to 100%% (max deviation %.2f)" % (ages - 100).abs().max())

    # -------------------------------------------------------- calibration
    er = q(con, """SELECT a.platform, v.followers, v.engagement_rate
                   FROM v_latest_snapshot v JOIN account a ON a.id = v.account_id""")
    bins = [0, 10_000, 100_000, 500_000, 1_000_000, 10**9]
    er["tier"] = pd.cut(er.followers, bins, labels=["nano", "micro", "mid", "macro", "mega"])
    med = er[er.platform == "instagram"].groupby("tier", observed=True).engagement_rate.median()
    print("      Instagram median engagement by tier:", {k: round(v, 2) for k, v in med.items()})
    check(med.is_monotonic_decreasing, "Instagram engagement decreases with audience size")
    rc = q(con, """SELECT r.price_eur, v.followers FROM rate_card r
                   JOIN account a ON a.influencer_id = r.influencer_id AND a.platform = r.platform
                   JOIN v_latest_snapshot v ON v.account_id = a.id
                   WHERE r.deliverable = 'feed_post'""")
    rc["tier"] = pd.cut(rc.followers, bins, labels=["nano", "micro", "mid", "macro", "mega"])
    pm = rc.groupby("tier", observed=True).price_eur.median()
    print("      median feed post price (EUR) by tier:", {k: int(v) for k, v in pm.items()})
    check(pm.is_monotonic_increasing, "prices rise with audience size")
    check(40 <= pm.get("nano", 0) <= 300, "nano feed post median in plausible DACH range (40-300 EUR)")

    # ------------------------------------------------- prediction quality
    pe = q(con, "SELECT * FROM v_prediction_error")
    print("      mean actual/predicted reach: %.2f, inside p10-p90: %.0f %%" % (pe.ratio.mean(), pe.inside_p10_p90.mean() * 100))
    check(pe.ratio.mean() < 1.0, "historic predictions are optimistic on average (calibration page has a story)")
    check(0.35 <= pe.inside_p10_p90.mean() <= 0.9, "p10-p90 coverage is imperfect but not absurd")

    # ------------------------------------------------------ planted traps
    spike = q(con, """SELECT DISTINCT cur.account_id FROM account_snapshot cur
                      JOIN account_snapshot prev ON prev.account_id = cur.account_id
                       AND prev.captured_at = date(cur.captured_at, 'start of month', '-1 day')
                      WHERE cur.followers > prev.followers * 1.5 AND cur.fake_follower_pct > 30""")
    want = set(truth[truth.trap_type == "fake_follower_spike"].entity_id)
    check(want <= set(spike.account_id), f"all {len(want)} fake-follower spikes detectable (found {len(spike)})")

    stale = q(con, "SELECT account_id FROM v_latest_snapshot WHERE captured_at < '2026-06-01'")
    want = set(truth[truth.trap_type == "stale_stats"].entity_id)
    check(want <= set(stale.account_id), f"all {len(want)} stale accounts detectable")

    exp = q(con, f"""SELECT influencer_id FROM rate_card GROUP BY influencer_id HAVING MAX(valid_to) < '{TODAY}'""")
    want = set(truth[truth.trap_type == "expired_rate_card"].entity_id)
    check(want <= set(exp.influencer_id), f"all {len(want)} expired rate cards detectable")

    cons = q(con, f"SELECT id FROM influencer WHERE consent_until < '{TODAY}'")
    want = set(truth[truth.trap_type == "consent_expired"].entity_id)
    check(want <= set(cons.id), f"all {len(want)} expired GDPR consents detectable")

    conf = q(con, "SELECT campaign_id, influencer_id, conflict_type FROM v_campaign_conflicts")
    for ttype, ctype in [("competitor_conflict", "competitor_promo_90d"), ("exclusivity_block", "exclusivity_contract")]:
        t = truth[truth.trap_type == ttype]
        found = {(int(r.related_id), int(r.entity_id)) for r in t.itertuples()}
        have = {(r.campaign_id, r.influencer_id) for r in conf[conf.conflict_type == ctype].itertuples()}
        check(found <= have, f"all {len(found)} '{ttype}' cases detectable via v_campaign_conflicts")

    pending = q(con, "SELECT COUNT(*) AS n FROM campaign WHERE status = 'planned'").n[0]
    check(pending >= 3, f"{pending} planned campaigns available as demo targets")

    failed = [m for ok, m in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
