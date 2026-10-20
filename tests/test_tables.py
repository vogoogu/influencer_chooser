"""The shared table formatter: readable labels, formats, dates, yes/no, dashes."""
import numpy as np
import pandas as pd
import pytest

from core import tables


def fmt(df, **kw):
    return tables.format_table(df, **kw)


def test_labels_are_human_and_have_units():
    d, cfg = fmt(pd.DataFrame({"price_eur": [1500], "usage_rights_months": [3], "exclusivity_premium_pct": [25.0], "reach_p50": [8436],
                               "some_new_column": [1], "budget_eur": [15000]}))
    assert cfg["price_eur"]["label"] == "Fee (EUR)" and cfg["budget_eur"]["label"] == "Budget (EUR)"
    assert cfg["usage_rights_months"]["label"] == "Usage rights (months)" and cfg["exclusivity_premium_pct"]["label"] == "Exclusivity premium"
    assert cfg["some_new_column"]["label"] == "Some new column"          # unknown names still become sentence case


def test_number_formats():
    d, cfg = fmt(pd.DataFrame({"followers": [1500, 25000], "er": [1.234, 4.0], "weight": [0.192, 0.5], "ratio": [0.87, 1.1],
                               "total_score": [0.7185, 0.5], "cpm": [12.34, 5.6], "exclusivity_premium_pct": [25.0, 30.0]}),
                     formats={"cpm": "%.0f"})
    assert cfg["followers"]["type_config"]["format"] == "localized"
    assert cfg["er"]["type_config"]["format"] == "%.2f%%"
    assert cfg["weight"]["type_config"]["format"] == "%.0f%%" and d.weight.tolist() == [19.2, 50.0]          # 0.192 -> 19.2 shown as 19%
    assert cfg["ratio"]["type_config"]["format"] == "%.2f×"
    assert cfg["total_score"]["type_config"]["type"] == "progress"
    assert cfg["cpm"]["type_config"]["format"] == "%.0f"
    assert cfg["exclusivity_premium_pct"]["type_config"]["format"] == "%.0f%%"


def test_whole_floats_are_counts_but_fractions_keep_decimals():
    d, cfg = fmt(pd.DataFrame({"reach_p50": [8436.0, 2131.0], "avg_thing": [1.25, 2.5]}))
    assert cfg["reach_p50"]["type_config"]["format"] == "localized"
    assert cfg["avg_thing"]["type_config"]["format"] == "%.2f"


def test_flags_become_yes_no_and_empty_becomes_dash():
    d, _ = fmt(pd.DataFrame({"verified": [1, 0], "delivered_on_time": [1, np.nan], "expired": [True, False],
                             "brand": ["Kleinstern", None], "is_disclosed": [np.nan, 1.0]}))
    assert d.verified.tolist() == ["Yes", "No"] and d.expired.tolist() == ["Yes", "No"]
    assert d.delivered_on_time.tolist() == ["Yes", "–"] and d.is_disclosed.tolist() == ["–", "Yes"]
    assert d.brand.tolist() == ["Kleinstern", "–"]


def test_dates_and_timestamps():
    d, cfg = fmt(pd.DataFrame({"valid_from": ["2026-01-01", "2026-02-01"], "created_at": ["2026-10-20T07:26:57", "2026-10-21T09:00:00"],
                               "posted_at": ["2026-09-01", None]}))
    assert pd.api.types.is_datetime64_any_dtype(d.valid_from) and pd.api.types.is_datetime64_any_dtype(d.created_at)
    assert cfg["valid_from"]["type_config"]["format"] == "YYYY-MM-DD"
    assert cfg["created_at"]["type_config"]["format"] == "YYYY-MM-DD HH:mm" and cfg["created_at"]["label"] == "Saved"
    assert d.posted_at.isna().sum() == 1                                   # missing dates stay missing, no crash


def test_snake_case_values_become_words_but_handles_do_not():
    d, _ = fmt(pd.DataFrame({"category": ["outdoor_gear", "plant_based_food"], "handle": ["anna_auer_style", "x_y_z"],
                             "platform": ["tiktok", "youtube"], "objective": ["qualified_reach", "score"]}))
    assert d.category.tolist() == ["outdoor gear", "plant based food"]
    assert d.handle.tolist() == ["anna_auer_style", "x_y_z"]
    assert d.platform.tolist() == ["TikTok", "YouTube"]
    assert d.objective.tolist() == ["qualified reach", "score"]            # converted per value, single words stay as they are


def test_hide_order_and_input_not_modified():
    src = pd.DataFrame({"a_col": [1], "b_col": [2], "c_col": [3]})
    d, cfg = fmt(src, hide=["b_col"], order=["c_col", "a_col"])
    assert list(d.columns) == ["c_col", "a_col"] and set(cfg) == {"c_col", "a_col"}
    assert list(src.columns) == ["a_col", "b_col", "c_col"] and src.a_col.tolist() == [1]


def test_empty_and_odd_frames_do_not_crash():
    for df in (pd.DataFrame(), pd.DataFrame({"price": pd.Series([], dtype=float), "name": pd.Series([], dtype=object)}),
               pd.DataFrame({"x": [np.nan, np.nan]})):
        d, cfg = fmt(df)
        assert set(cfg) == set(d.columns)


def test_explicit_widths_win():
    _, cfg = fmt(pd.DataFrame({"name": ["a"], "total_score": [0.5]}), widths={"total_score": 110})
    assert cfg["total_score"]["width"] == 110


def test_links_use_link_columns():
    _, cfg = fmt(pd.DataFrame({"url": ["https://x"], "evidence_url": ["https://y"]}))
    assert cfg["url"]["type_config"]["type"] == "link" and cfg["evidence_url"]["label"] == "Evidence"
