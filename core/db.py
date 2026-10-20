"""SQLite access. The app never writes to the committed baseline database: on first start it
copies it to a working database, and "Reset" restores that copy."""
from __future__ import annotations

import shutil
import sqlite3
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from . import config


def ensure_db(path: Path | None = None) -> Path:
    """Make sure the working database exists (copy baseline, or generate it)."""
    path = Path(path or config.DB_WORKING)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not config.DB_BASELINE.exists():
        subprocess.run([sys.executable, str(config.ROOT / "generate_dataset.py")], check=True, cwd=config.ROOT)
    shutil.copy(config.DB_BASELINE, path)
    return path


def reset_db(path: Path | None = None) -> Path:
    path = Path(path or config.DB_WORKING)
    if path.exists():
        path.unlink()
    return ensure_db(path)


def connect(path: Path | None = None) -> sqlite3.Connection:
    con = sqlite3.connect(ensure_db(path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def read(sql: str, params: tuple | dict = (), path: Path | None = None) -> pd.DataFrame:
    con = sqlite3.connect(ensure_db(path))      # plain connection: pandas dislikes Row factories
    try:
        return pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()


def execute(sql: str, params: tuple | dict = (), path: Path | None = None) -> int:
    con = connect(path)
    try:
        cur = con.execute(sql, params)
        con.commit()
        return cur.lastrowid
    finally:
        con.close()


def as_of(path: Path | None = None) -> date:
    """The 'today' of the dataset (synthetic data is frozen at 2026-10-09). Falls back to today."""
    try:
        df = read("SELECT value FROM meta WHERE key = 'as_of'", path=path)
        if len(df):
            return date.fromisoformat(df.value[0])
    except Exception:
        pass
    return date.today()


def db_mtime(path: Path | None = None) -> float:
    return Path(ensure_db(path)).stat().st_mtime


def save_run(*, title: str, brand_id: int | None, objective: str, budget_eur: float, weights: dict,
             profile: dict, brief_text: str, notes: str, items: list[dict], model_version: str | None = None,
             path: Path | None = None) -> int:
    """Store a decision (audit trail). `items` need influencer_id, account_id, rank, total_score,
    criterion_scores (dict), selected (0/1), deliverable, fee_eur, reach_p10/p50/p90, ai_explanation."""
    import json
    con = connect(path)
    try:
        cur = con.execute(
            """INSERT INTO run (created_at, title, brand_id, objective, budget_eur, weights_json, profile_json,
                                brief_text, notes, model_version) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (datetime.now().isoformat(timespec="seconds"), title, brand_id, objective, budget_eur,
             json.dumps(weights), json.dumps(profile, default=str), brief_text[:20000], notes, model_version or config.MODEL_VERSION_TAG))
        run_id = cur.lastrowid
        for it in items:
            con.execute(
                """INSERT INTO run_item (run_id, influencer_id, account_id, rank, total_score, criterion_scores, selected,
                                         deliverable, fee_eur, reach_p10, reach_p50, reach_p90, reach_naive, ai_explanation)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, int(it["influencer_id"]), int(it["account_id"]), int(it.get("rank", 0)), float(it["total_score"]),
                 json.dumps(it["criterion_scores"]), int(it.get("selected", 0)), it.get("deliverable"),
                 None if it.get("fee_eur") is None else float(it["fee_eur"]),
                 None if it.get("reach_p10") is None else int(it["reach_p10"]),
                 None if it.get("reach_p50") is None else int(it["reach_p50"]),
                 None if it.get("reach_p90") is None else int(it["reach_p90"]),
                 None if it.get("reach_naive") is None else int(it["reach_naive"]), it.get("ai_explanation")))
        con.commit()
        return run_id
    finally:
        con.close()


def convert_run_to_campaign(run_id: int, *, name: str, goal: str, start: date, end: date, country: str,
                            path: Path | None = None) -> int:
    """Turn a saved decision into a planned campaign with participations, so real results can
    be logged later and feed the calibration page."""
    con = connect(path)
    try:
        run = con.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            raise ValueError(f"run {run_id} not found")
        if run["campaign_id"]:
            raise ValueError("this decision was already converted into a campaign")
        if run["brand_id"] is None:
            raise ValueError("pick a client brand when saving the decision to convert it into a campaign")
        cur = con.execute(
            """INSERT INTO campaign (brand_id, name, goal, budget_eur, start_date, end_date, target_country, status)
               VALUES (?,?,?,?,?,?,?, 'planned')""",
            (run["brand_id"], name, goal, run["budget_eur"] or 0, start.isoformat(), end.isoformat(), country))
        cid = cur.lastrowid
        items = con.execute("SELECT * FROM run_item WHERE run_id = ? AND selected = 1", (run_id,)).fetchall()
        for it in items:
            con.execute(
                """INSERT INTO participation (campaign_id, influencer_id, account_id, deliverable, fee_eur,
                                              reach_p10, reach_p50, reach_p90, model_version, reach_naive)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (cid, it["influencer_id"], it["account_id"], it["deliverable"] or "reel", it["fee_eur"] or 0,
                 it["reach_p10"] or 0, it["reach_p50"] or 0, it["reach_p90"] or 0, run["model_version"],
                 it["reach_naive"] if it["reach_naive"] is not None else it["reach_p50"]))
        con.execute("UPDATE run SET campaign_id = ? WHERE id = ?", (cid, run_id))
        con.commit()
        return cid
    finally:
        con.close()


def log_result(*, participation_id: int, actual_reach: int, actual_engagement: int, conversions: int,
               delivered_on_time: bool, client_rating: int, measured_at: date, path: Path | None = None) -> None:
    con = connect(path)
    try:
        con.execute(
            """INSERT INTO result (participation_id, actual_reach, actual_engagement, conversions,
                                   delivered_on_time, client_rating, measured_at) VALUES (?,?,?,?,?,?,?)""",
            (participation_id, actual_reach, actual_engagement, conversions, int(delivered_on_time),
             client_rating, measured_at.isoformat()))
        con.commit()
    finally:
        con.close()
