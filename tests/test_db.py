from datetime import date

import pandas as pd
import pytest

from core import config, db, predict


def _items(n=3):
    acc = db.read("SELECT a.id AS account_id, a.influencer_id FROM account a LIMIT ?", (n,))
    return [{"influencer_id": r.influencer_id, "account_id": r.account_id, "rank": i + 1, "total_score": 0.8 - i * 0.1,
             "criterion_scores": {c: 0.5 for c in config.CRITERIA}, "selected": 1, "deliverable": "reel", "fee_eur": 500.0 + i,
             "reach_p10": 1000, "reach_p50": 2000, "reach_p90": 4000, "ai_explanation": None} for i, r in enumerate(acc.itertuples())]


def test_baseline_is_never_modified(fresh_db):
    before = config.DB_BASELINE.stat().st_mtime
    db.save_run(title="t", brand_id=1, objective="score", budget_eur=1000, weights={"a": 1}, profile={}, brief_text="b", notes="", items=_items())
    assert config.DB_BASELINE.stat().st_mtime == before
    assert config.DB_WORKING != config.DB_BASELINE


def test_save_convert_log_and_calibration_grow(fresh_db):
    n_hist = len(predict.load_history())
    run_id = db.save_run(title="Decision A", brand_id=1, objective="qualified_reach", budget_eur=5000,
                         weights={"topic_fit": 0.5}, profile={"goal": "sales"}, brief_text="brief", notes="n", items=_items(3))
    assert db.read("SELECT COUNT(*) n FROM run_item WHERE run_id = ?", (run_id,)).n[0] == 3
    cid = db.convert_run_to_campaign(run_id, name="Camp", goal="sales", start=date(2026, 11, 1), end=date(2026, 11, 20), country="AT")
    parts = db.read("SELECT * FROM participation WHERE campaign_id = ?", (cid,))
    assert len(parts) == 3 and (parts.reach_p50 == 2000).all() and (parts.model_version == config.MODEL_VERSION_TAG).all()
    with pytest.raises(ValueError, match="already converted"):
        db.convert_run_to_campaign(run_id, name="again", goal="sales", start=date(2026, 11, 1), end=date(2026, 11, 2), country="AT")
    for pid in parts.id:
        db.log_result(participation_id=int(pid), actual_reach=1500, actual_engagement=90, conversions=12, delivered_on_time=True,
                      client_rating=4, measured_at=date(2026, 12, 1))
    assert len(predict.load_history()) == n_hist + 3
    with pytest.raises(Exception):                       # a participation can only have one result
        db.log_result(participation_id=int(parts.id[0]), actual_reach=1, actual_engagement=1, conversions=0,
                      delivered_on_time=True, client_rating=3, measured_at=date(2026, 12, 2))


def test_convert_requires_brand(fresh_db):
    run_id = db.save_run(title="No brand", brand_id=None, objective="score", budget_eur=1, weights={}, profile={}, brief_text="", notes="", items=_items(1))
    with pytest.raises(ValueError, match="brand"):
        db.convert_run_to_campaign(run_id, name="x", goal="sales", start=date(2026, 11, 1), end=date(2026, 11, 2), country="AT")


def test_reset_restores_baseline(fresh_db):
    db.save_run(title="x", brand_id=1, objective="score", budget_eur=1, weights={}, profile={}, brief_text="", notes="", items=_items(1))
    assert db.read("SELECT COUNT(*) n FROM run").n[0] == 1
    db.reset_db()
    assert db.read("SELECT COUNT(*) n FROM run").n[0] == 0


def test_as_of_comes_from_meta():
    assert db.as_of() == date(2026, 10, 9)
