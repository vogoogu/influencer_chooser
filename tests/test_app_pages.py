"""Headless page tests with Streamlit's AppTest: every page renders, and the main flow works end to end."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core import db, predict

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["1_Product_Brief", "2_Shortlist", "3_Campaign_Mix", "4_Explorer", "5_Decision_Log", "6_Calibration", "7_Forecast_Model"]
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


def test_forecast_page_explains_forest_and_a_shortlisted_creator(fresh_db):
    from core import ai
    prof = ai.rule_based_profile(open(ROOT / "data" / "test_briefs" / "brief_nordstride.md", encoding="utf-8").read())
    s = page("2_Shortlist")
    s.session_state["profile"], s.session_state["brief_text"] = prof, "brief"
    s.session_state["profile_version"], s.session_state["client_brand_id"] = 1, None
    s.run()
    assert_clean(s)
    assert any("random forest" in c.value.lower() for c in s.caption)                    # default method is the forest
    f = page("7_Forecast_Model")
    for k in ["profile", "brief_text", "profile_version", "client_brand_id", "ranked"]:
        f.session_state[k] = s.session_state[k]
    f.run()
    assert_clean(f)
    assert {"Past results used for training", "Naive forecast", "Tier correction", "Random forest", "Forecast reach"} <= {m.label for m in f.metric}
    assert len(f.tabs) == 4 and any("forest wins" in str(m.delta) for m in f.metric)


def test_shortlist_forecast_selector_switches_method(fresh_db):
    from core import ai
    prof = ai.rule_based_profile("# T\nBudget: 20000 EUR. Running shoe for runners in Austria.")
    seen = {}
    for method in ("naive", "tier", "forest"):
        s = page("2_Shortlist")
        s.session_state["profile"], s.session_state["brief_text"] = prof, "x"
        s.session_state["profile_version"], s.session_state["client_brand_id"] = 1, None
        s.session_state["f_forecast"] = method
        s.run()
        assert_clean(s)
        seen[method] = s.session_state["score_meta"]["forecast"]["reach"]
    assert seen == {"naive": "naive", "tier": "tier correction", "forest": "random forest"}


# ------------------------------------------------------------------ OpenRouter wiring through the real pages
def _fake_openrouter(monkeypatch):
    import json as _json
    import requests
    calls = []
    profile = {"product_name": "Fake Shoe", "category": "sportswear", "summary": "AI summary.", "goal": "sales", "target_countries": ["AT"],
               "target_audience": {"description": "runners", "age_min": 25, "age_max": 40}, "relevant_niches": ["running"],
               "tone": ["casual"], "restrictions": {"excluded_flag_types": []}}

    class R:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "submit_profile", "arguments": _json.dumps(profile)}}]}}]}

    def post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json})
        return R()
    monkeypatch.setattr(requests, "post", post)
    return calls


def _analyze_sample(at):
    at.run()
    at.selectbox(key="sample_choice").select("brief_nordstride.md").run()
    click(at, "Load sample brief")
    click(at, "Analyze brief")
    return at


def test_key_in_streamlit_secrets_is_used_for_openrouter(fresh_db, monkeypatch):
    calls = _fake_openrouter(monkeypatch)
    at = page("1_Product_Brief")
    at.secrets["OPENROUTER_API_KEY"] = "sk-or-from-secrets"
    at.secrets["OPENROUTER_MODEL"] = "openai/gpt-5.2"
    _analyze_sample(at)
    assert_clean(at)
    assert at.session_state["profile_source"] == "ai" and at.session_state["profile"]["product_name"] == "Fake Shoe"
    assert calls[0]["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-or-from-secrets"
    assert calls[0]["json"]["model"] == "openai/gpt-5.2"
    assert calls[0]["json"]["tool_choice"]["function"]["name"] == "submit_profile"


def test_sidebar_key_and_model_work_without_secrets(fresh_db, monkeypatch):
    calls = _fake_openrouter(monkeypatch)
    at = page("1_Product_Brief")
    at.session_state["ui_api_key"] = "sk-or-from-sidebar"
    at.session_state["ui_model"] = "anthropic/claude-sonnet-5.5"
    _analyze_sample(at)
    assert_clean(at)
    assert at.session_state["profile_source"] == "ai"
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-or-from-sidebar" and calls[0]["json"]["model"] == "anthropic/claude-sonnet-5.5"


def test_bad_key_falls_back_to_rules_and_says_why(fresh_db, monkeypatch):
    import requests

    class R:
        status_code = 401

        def json(self):
            return {"error": {"message": "No auth credentials found", "code": 401}}
    monkeypatch.setattr(requests, "post", lambda *a, **k: R())
    at = page("1_Product_Brief")
    at.session_state["ui_api_key"] = "sk-or-bad"
    _analyze_sample(at)
    assert_clean(at)
    assert at.session_state["profile_source"] == "fallback" and "OPENROUTER_API_KEY" in at.session_state["profile_error"]
    assert at.session_state["profile"]["category"] == "sportswear"          # rule-based result is still usable
    assert any("AI analysis failed" in w.value for w in at.warning)


def test_openrouter_404_on_forced_tool_choice_still_gives_an_ai_profile(fresh_db, monkeypatch):
    """The real-world failure: 'No endpoints found that support the provided tool_choice value'. The app must recover."""
    import json as _json
    import requests
    calls = []
    profile = {"product_name": "Fallback Shoe", "category": "sportswear", "summary": "AI summary.", "goal": "sales", "target_countries": ["AT"],
               "target_audience": {"description": "runners"}, "relevant_niches": ["running"], "tone": ["casual"],
               "restrictions": {"excluded_flag_types": []}}

    class R:
        def __init__(self, status, data):
            self.status_code, self._d = status, data

        def json(self):
            return self._d

    def post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        if isinstance(json.get("tool_choice"), dict):           # forced function choice: rejected like OpenRouter did
            return R(404, {"error": {"message": "No endpoints found that support the provided 'tool_choice' value.", "code": 404}})
        return R(200, {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "submit_profile", "arguments": _json.dumps(profile)}}]}}]})
    monkeypatch.setattr(requests, "post", post)
    at = page("1_Product_Brief")
    at.secrets["OPENROUTER_API_KEY"] = "sk-or-test"
    _analyze_sample(at)
    assert_clean(at)
    assert at.session_state["profile_source"] == "ai" and at.session_state["profile"]["product_name"] == "Fallback Shoe"
    assert [c.get("tool_choice") for c in calls][:2] == [{"type": "function", "function": {"name": "submit_profile"}}, "required"]
    assert not any("AI analysis failed" in w.value for w in at.warning)
