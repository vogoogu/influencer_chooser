"""Headless page tests with Streamlit's AppTest: every page renders, and the main flow works end to end."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core import db, predict

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["1_Product_Brief", "2_Shortlist", "3_Campaign_Mix", "4_Explorer", "5_Decision_Log", "6_Calibration"]
STATE_KEYS = ["brief_text", "profile", "profile_source", "profile_version", "client_brand_id", "research_text"]


def page(name: str) -> AppTest:
    return AppTest.from_file(str(ROOT / "pages" / f"{name}.py"), default_timeout=180)


def click(at: AppTest, label: str, starts: bool = False) -> AppTest:
    btn = [b for b in at.button if (b.label.startswith(label) if starts else b.label == label)]
    assert btn, f"button '{label}' not found in {[b.label for b in at.button]}"
    return btn[0].click().run()


def assert_clean(at: AppTest):
    assert not at.exception, [e.value for e in at.exception]


def test_home_renders(fresh_db):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert_clean(at)
    assert {"Influencers", "Accounts"} <= {m.label for m in at.metric}


@pytest.mark.parametrize("name", PAGES)
def test_pages_render_without_a_brief(name, fresh_db):
    assert_clean(page(name).run())


def test_full_flow(fresh_db):
    # 1. brief
    b = page("1_Product_Brief").run()
    b.selectbox(key="sample_choice").select("brief_nordstride.md").run()
    click(b, "Load sample brief")
    assert "Nordstride" in b.session_state["brief_input"] and b.session_state["client_brand_id"]
    click(b, "Analyze brief")
    assert_clean(b)
    profile = b.session_state["profile"]
    assert profile["category"] == "sportswear" and profile["budget_eur"] == 25000
    state = {k: b.session_state[k] for k in STATE_KEYS}

    # 2. shortlist + explanations
    s = page("2_Shortlist")
    for k, v in state.items():
        s.session_state[k] = v
    s.run()
    assert_clean(s)
    assert any(m.label == "Passed hard filters" and int(m.value) > 10 for m in s.metric)
    click(s, "Explain the top creators")
    assert_clean(s)
    assert len(s.session_state["explanations"]) == 8
    state.update({k: s.session_state[k] for k in ["ranked", "score_meta", "excluded", "explanations"]})

    # 3. mix, AI-free review and memo, save
    m = page("3_Campaign_Mix")
    for k, v in state.items():
        m.session_state[k] = v
    m.run()
    assert_clean(m)
    assert {"Creators", "Spend", "Expected conversions"} <= {x.label for x in m.metric}
    click(m, "Add AI review", starts=True)
    click(m, "Write the memo with AI")
    assert_clean(m)
    assert {x.label for x in m.get("download_button")} == {"Download PDF", "Download HTML", "Download Markdown", "Download mix (CSV)"}
    click(m, "Save decision")
    assert_clean(m)
    assert any("Saved as decision" in x.value for x in m.success)
    assert db.read("SELECT COUNT(*) n FROM run").n[0] == 1

    # 4. decision log: convert, then simulate outcomes for that campaign
    d = page("5_Decision_Log").run()
    click(d, "Create campaign with these creators")
    assert_clean(d)
    cid = int(db.read("SELECT MAX(id) id FROM campaign").id[0])
    assert db.read("SELECT status FROM campaign WHERE id = ?", (cid,)).status[0] == "planned"
    n_before = len(predict.load_history())
    d2 = page("5_Decision_Log").run()
    d2.selectbox[1 if len(d2.selectbox) > 1 else 0].select(cid).run()      # pick our campaign in the results tab
    click(d2, "Simulate and save results")
    assert_clean(d2)
    assert len(predict.load_history()) > n_before

    # 5. calibration reflects the new data
    c = page("6_Calibration").run()
    assert_clean(c)
    assert any(x.label == "Results analysed" and int(x.value) == len(predict.load_history()) for x in c.metric)


def test_shortlist_with_impossible_filters_shows_message(fresh_db):
    from core import ai
    s = page("2_Shortlist")
    prof = ai.rule_based_profile("# T\nBudget: 10 EUR. Running shoe in Austria.")
    s.session_state["profile"], s.session_state["brief_text"] = prof, "x"
    s.session_state["profile_version"], s.session_state["client_brand_id"] = 1, None
    s.run()
    assert_clean(s)
    assert any("No creators pass" in w.value for w in s.warning)
